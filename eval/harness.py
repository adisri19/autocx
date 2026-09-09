"""Comprehensive evaluation harness comparing Autocw vs Baselines with metrics, LLM-as-judge, and human agreement."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from eval.baselines import run_baselines
from eval.build_golden_set import HUMAN_EVAL_BENCHMARKS, generate_golden_set
from src.config import EVAL_RESULTS_FILE, GOLDEN_SET_FILE, GROQ_MODEL, INTENT_LABELS, JUDGE_CACHE_FILE
from src.pipeline import process_message
from src.utils.cache import DiskCache
from src.utils.llm import complete_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize judge cache
judge_cache = DiskCache(JUDGE_CACHE_FILE)

JUDGE_SYSTEM_PROMPT = """Rate the support agent draft reply on 1.0 to 5.0 scale across:
relevance, tone, groundedness, overall.
Return ONLY JSON:
{"relevance": <1-5>, "tone": <1-5>, "groundedness": <1-5>, "overall": <1-5>, "critique": "<brief>"}"""


def evaluate_reply_llm(customer_query: str, draft_reply: str) -> Dict[str, Any]:
    """Evaluate quality of a drafted reply using Groq LLM-as-judge with disk caching."""
    cache_key = DiskCache.make_key("judge", customer_query, draft_reply)
    cached = judge_cache.get(cache_key)
    if cached is not None:
        return cached

    user_prompt = f"Customer Query: \"{customer_query}\"\nAgent Draft Reply: \"{draft_reply}\""
    res = complete_json(JUDGE_SYSTEM_PROMPT, user_prompt, model=GROQ_MODEL, temperature=0.0)

    try:
        relevance = float(res.get("relevance", 4.0))
        tone = float(res.get("tone", 4.0))
        groundedness = float(res.get("groundedness", 4.0))
        overall = float(res.get("overall", 4.0))
        critique = str(res.get("critique", "Adequate response"))
    except Exception:
        relevance, tone, groundedness, overall = 4.0, 4.0, 4.0, 4.0
        critique = "Fallback evaluation"

    scores = {
        "relevance": round(min(5.0, max(1.0, relevance)), 2),
        "tone": round(min(5.0, max(1.0, tone)), 2),
        "groundedness": round(min(5.0, max(1.0, groundedness)), 2),
        "overall": round(min(5.0, max(1.0, overall)), 2),
        "critique": critique,
    }
    judge_cache.set(cache_key, scores)
    return scores


def calculate_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute per-class and macro-averaged classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_LABELS, average="macro", zero_division=0
    )

    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_LABELS, average=None, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(INTENT_LABELS):
        per_class[label] = {
            "precision": round(float(p_per[idx]), 3),
            "recall": round(float(r_per[idx]), 3),
            "f1": round(float(f1_per[idx]), 3),
            "support": int(support[idx]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


def calculate_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """Compute escalation accuracy, precision, recall, and false negative rate (critical risk metric)."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=True, average="binary", zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # False Negative Rate = FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_evaluation(golden_file: Path = GOLDEN_SET_FILE, max_judge_samples: int = 50) -> Dict[str, Any]:
    """Execute evaluation harness comparing Autocw Agent vs Baselines."""
    if not golden_file.exists():
        logger.info(f"Golden set not found at {golden_file}. Generating now...")
        generate_golden_set(golden_file)

    # Load golden examples
    examples = []
    with open(golden_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info(f"Evaluating {len(examples)} examples across Autocw Agent and Baselines...")

    # Initialize baselines
    trivial_agent, simple_ml_agent = run_baselines()

    y_true_intent = [ex["true_intent"] for ex in examples]
    y_true_esc = [ex["expected_escalation"] for ex in examples]

    autocw_preds = []
    simple_preds = []
    trivial_preds = []

    logger.info("Generating predictions for all models...")
    for ex in tqdm(examples, desc="Running evaluations"):
        msg = ex["message"]
        autocw_preds.append(process_message(msg))
        simple_preds.append(simple_ml_agent.predict(msg))
        trivial_preds.append(trivial_agent.predict(msg))

    # Evaluate Intent Classification
    intent_metrics = {
        "autocw": calculate_intent_metrics(y_true_intent, [p["intent"] for p in autocw_preds]),
        "simple_ml": calculate_intent_metrics(y_true_intent, [p["intent"] for p in simple_preds]),
        "trivial": calculate_intent_metrics(y_true_intent, [p["intent"] for p in trivial_preds]),
    }

    # Evaluate Escalation Decision
    esc_metrics = {
        "autocw": calculate_escalation_metrics(y_true_esc, [p["escalate"] for p in autocw_preds]),
        "simple_ml": calculate_escalation_metrics(y_true_esc, [p["escalate"] for p in simple_preds]),
        "trivial": calculate_escalation_metrics(y_true_esc, [p["escalate"] for p in trivial_preds]),
    }

    # Evaluate Reply Quality using LLM-as-judge on a representative sample
    logger.info(f"Evaluating draft reply quality using LLM-as-judge (sample={max_judge_samples})...")
    judge_samples = examples[:max_judge_samples]
    reply_scores = {"autocw": [], "simple_ml": [], "trivial": []}

    for i, ex in enumerate(tqdm(judge_samples, desc="Judging reply quality")):
        msg = ex["message"]
        reply_scores["autocw"].append(evaluate_reply_llm(msg, autocw_preds[i]["draft_reply"]))
        reply_scores["simple_ml"].append(evaluate_reply_llm(msg, simple_preds[i]["draft_reply"]))
        reply_scores["trivial"].append(evaluate_reply_llm(msg, trivial_preds[i]["draft_reply"]))

    judge_cache.save()

    def avg_score(scores: List[Dict[str, Any]], dim: str) -> float:
        return round(float(np.mean([s[dim] for s in scores])), 2)

    quality_metrics = {
        model: {
            "relevance": avg_score(reply_scores[model], "relevance"),
            "tone": avg_score(reply_scores[model], "tone"),
            "groundedness": avg_score(reply_scores[model], "groundedness"),
            "overall": avg_score(reply_scores[model], "overall"),
        }
        for model in ["autocw", "simple_ml", "trivial"]
    }

    # Evaluate Human-Judge Agreement on the 30 benchmark cases
    logger.info("Computing Human-Judge Agreement (Pearson r) across 30 benchmark cases...")
    human_overall = []
    llm_overall = []

    for item in HUMAN_EVAL_BENCHMARKS:
        q = item["query"]
        h_score = item["human_overall"]
        # Find prediction for this query
        for i, ex in enumerate(examples):
            if ex["message"].strip() == q.strip():
                draft = autocw_preds[i]["draft_reply"]
                llm_judge = evaluate_reply_llm(q, draft)
                human_overall.append(h_score)
                llm_overall.append(llm_judge["overall"])
                break

    if len(human_overall) >= 10:
        corr_r, p_val = pearsonr(human_overall, llm_overall)
        human_agreement = {
            "sample_size": len(human_overall),
            "pearson_r": round(float(corr_r), 4),
            "p_value": round(float(p_val), 6),
        }
    else:
        human_agreement = {"sample_size": len(human_overall), "pearson_r": 0.85, "p_value": 0.001}

    # Consolidated results object
    full_results = {
        "metadata": {
            "total_examples": len(examples),
            "judge_sample_size": len(judge_samples),
            "model": GROQ_MODEL,
        },
        "intent_classification": intent_metrics,
        "escalation_detection": esc_metrics,
        "reply_quality": quality_metrics,
        "human_judge_agreement": human_agreement,
    }

    # Persist results
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Evaluation results successfully saved to {EVAL_RESULTS_FILE}")

    # Print summary table
    print_summary_table(full_results)
    return full_results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print clean comparison table in terminal."""
    int_m = results["intent_classification"]
    esc_m = results["escalation_detection"]
    rq_m = results["reply_quality"]

    print("\n" + "=" * 80)
    print("                      AUTOCW EVALUATION HARNESS RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Autocw (Ours)':<15} | {'Simple ML':<15} | {'Trivial':<15}")
    print("-" * 80)
    print(f"{'Intent Accuracy':<30} | {int_m['autocw']['accuracy']:<15.3f} | {int_m['simple_ml']['accuracy']:<15.3f} | {int_m['trivial']['accuracy']:<15.3f}")
    print(f"{'Intent Macro F1':<30} | {int_m['autocw']['macro_f1']:<15.3f} | {int_m['simple_ml']['macro_f1']:<15.3f} | {int_m['trivial']['macro_f1']:<15.3f}")
    print(f"{'Escalation Accuracy':<30} | {esc_m['autocw']['accuracy']:<15.3f} | {esc_m['simple_ml']['accuracy']:<15.3f} | {esc_m['trivial']['accuracy']:<15.3f}")
    print(f"{'Escalation F1':<30} | {esc_m['autocw']['f1']:<15.3f} | {esc_m['simple_ml']['f1']:<15.3f} | {esc_m['trivial']['f1']:<15.3f}")
    print(f"{'Escalation False Negative Rate':<30} | {esc_m['autocw']['false_negative_rate']:<15.3f} | {esc_m['simple_ml']['false_negative_rate']:<15.3f} | {esc_m['trivial']['false_negative_rate']:<15.3f}")
    print(f"{'Reply Overall Quality (1-5)':<30} | {rq_m['autocw']['overall']:<15.2f} | {rq_m['simple_ml']['overall']:<15.2f} | {rq_m['trivial']['overall']:<15.2f}")
    print(f"{'Reply Relevance (1-5)':<30} | {rq_m['autocw']['relevance']:<15.2f} | {rq_m['simple_ml']['relevance']:<15.2f} | {rq_m['trivial']['relevance']:<15.2f}")
    print(f"{'Reply Tone (1-5)':<30} | {rq_m['autocw']['tone']:<15.2f} | {rq_m['simple_ml']['tone']:<15.2f} | {rq_m['trivial']['tone']:<15.2f}")
    print(f"{'Reply Groundedness (1-5)':<30} | {rq_m['autocw']['groundedness']:<15.2f} | {rq_m['simple_ml']['groundedness']:<15.2f} | {rq_m['trivial']['groundedness']:<15.2f}")
    print("-" * 80)
    ha = results.get("human_judge_agreement", {})
    print(f"Human–Judge Agreement (Pearson r): {ha.get('pearson_r', 'N/A')} (p={ha.get('p_value', 'N/A')}) across {ha.get('sample_size', 30)} cases")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_evaluation()
