"""Full customer support agent pipeline orchestrator."""

import argparse
import json
import logging
from typing import Any, Dict, Optional

from src.classifier.classify import classify_intent
from src.drafter.draft import draft_reply, retrieve_similar
from src.escalation.decide import decide_escalation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_message(
    text: str,
    include_context: bool = False,
) -> Dict[str, Any]:
    """Execute the end-to-end support pipeline on an inbound customer query.

    Flow:
    1. Classify intent and assign confidence using few-shot Groq LLM (with disk cache).
    2. Retrieve top-3 semantic matches and draft a context-aware response using Groq RAG.
    3. Evaluate escalation criteria via fast rules + secondary LLM verification for borderline cases.

    Args:
        text: Customer message.
        include_context: Whether to return retrieved similar historical pairs in output.

    Returns:
        Structured response dictionary.
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "query": text,
            "intent": "OTHER",
            "confidence": 0.0,
            "draft_reply": "Hello! How can we assist you today?",
            "escalate": False,
            "escalation_reason": None,
        }

    # Step 1: Intent classification
    clf = classify_intent(clean_text)
    intent = clf["intent"]
    confidence = clf["confidence"]

    # Step 2: Response drafting via RAG
    draft = draft_reply(clean_text, intent=intent)

    # Step 3: Escalation decision
    esc = decide_escalation(clean_text, intent=intent, confidence=confidence)

    result = {
        "query": clean_text,
        "intent": intent,
        "confidence": confidence,
        "draft_reply": draft,
        "escalate": esc["escalate"],
        "escalation_reason": esc["reason"] if esc["escalate"] else None,
    }

    if include_context:
        try:
            result["retrieved_context"] = retrieve_similar(clean_text, top_k=3)
        except Exception:
            result["retrieved_context"] = []

    return result


def run_full_orchestration() -> None:
    """Run all pipeline stages end-to-end."""
    print("=" * 70)
    print("  AUTOCW AI CUSTOMER SUPPORT AGENT — END-TO-END ORCHESTRATION")
    print("=" * 70)

    # 1. Data Pipeline
    print("\n[Stage 1/6] Running Data Pipeline...")
    from src.data.pipeline import run_pipeline
    run_pipeline()

    # 2. Intent Taxonomy
    print("\n[Stage 2/6] Building Intent Taxonomy...")
    from src.intents.taxonomy import build_intent_taxonomy
    build_intent_taxonomy()

    # 3. RAG Embedding Index
    print("\n[Stage 3/6] Building Embedding Index for Reply Drafter...")
    from src.drafter.draft import build_embedding_index
    build_embedding_index()

    # 4. Golden Evaluation Set
    print("\n[Stage 4/6] Generating Stratified Golden Evaluation Set...")
    from eval.build_golden_set import generate_golden_set
    generate_golden_set()

    # 5. Baselines Evaluation
    print("\n[Stage 5/6] Running Baseline Models (Trivial & Heuristic/ML)...")
    from eval.baselines import run_baselines
    run_baselines()

    # 6. Evaluation Harness
    print("\n[Stage 6/6] Executing Comprehensive Evaluation Harness...")
    from eval.harness import run_evaluation
    run_evaluation()

    print("\n" + "=" * 70)
    print("  AUTOCW END-TO-END ORCHESTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocw Customer Support Pipeline")
    parser.add_argument("--demo", type=str, help="Process a single customer message")
    parser.add_argument("--context", action="store_true", help="Include retrieved historical context")
    parser.add_argument("--run-all", action="store_true", help="Execute complete end-to-end pipeline build & eval")
    args = parser.parse_args()

    if args.run_all:
        run_full_orchestration()
    elif args.demo:
        print(f"\nProcessing customer message: \"{args.demo}\"\n")
        output = process_message(args.demo, include_context=args.context)
        print(json.dumps(output, indent=2))
    else:
        sample = "My package was marked as delivered two days ago, but I never received it. Can I get a refund?"
        print(f"\nProcessing default demo message: \"{sample}\"\n")
        output = process_message(sample, include_context=True)
        print(json.dumps(output, indent=2))
