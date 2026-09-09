"""Few-shot LLM intent classification module using Groq and disk caching."""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.config import (
    CLASSIFICATION_CACHE_FILE,
    CONVERSATIONS_FILE,
    GROQ_MODEL,
    INTENT_LABELS,
    INTENTS_FILE,
)
from src.utils.cache import DiskCache
from src.utils.llm import complete_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize classification cache
classification_cache = DiskCache(CLASSIFICATION_CACHE_FILE)

# System prompt defining taxonomy, formatting, and few-shot guidance
CLASSIFIER_SYSTEM_PROMPT = """Classify customer support query into EXACTLY ONE intent:
ORDER_STATUS: Tracking, shipment status, delivery date
REFUND_REQUEST: Refund, returns, billing, double charge, money back
ACCOUNT_ISSUE: Login, password, OTP, locked account, email change
SHIPPING_DELAY: Late shipment, delay grievance, missed arrival deadline
GENERAL_COMPLAINT: Damaged goods, rude service, defective item
TECHNICAL_ISSUE: App crash, website error, checkout/payment bug
COMPLIMENT: Praise, thank you, positive feedback
OTHER: Ambiguous, small talk, irrelevant

Respond ONLY with valid JSON:
{"intent": "<INTENT>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}"""

FEW_SHOT_EXAMPLES = [
    {"query": "Where is my package? Tracking not updated.", "intent": "ORDER_STATUS", "confidence": 0.98, "reasoning": "Tracking inquiry"},
    {"query": "I returned it, when will my refund arrive?", "intent": "REFUND_REQUEST", "confidence": 0.97, "reasoning": "Refund inquiry"},
    {"query": "Cannot log in, OTP code not received.", "intent": "ACCOUNT_ISSUE", "confidence": 0.96, "reasoning": "OTP login issue"},
    {"query": "Guaranteed delivery date passed, still late.", "intent": "SHIPPING_DELAY", "confidence": 0.95, "reasoning": "Late delivery grievance"},
]


def build_user_prompt(text: str) -> str:
    """Construct the few-shot prompt with the customer query."""
    prompt_lines = ["Here are reference examples:\n"]
    for ex in FEW_SHOT_EXAMPLES:
        prompt_lines.append(f"Customer: \"{ex['query']}\"")
        prompt_lines.append(
            json.dumps({"intent": ex["intent"], "confidence": ex["confidence"], "reasoning": ex["reasoning"]})
        )
        prompt_lines.append("")

    prompt_lines.append(f"Customer: \"{text}\"")
    prompt_lines.append("JSON response:")
    return "\n".join(prompt_lines)


def classify_intent(text: str) -> Dict[str, Any]:
    """Classify the intent of a customer message using Groq LLM with disk caching.

    Args:
        text: Customer message.

    Returns:
        Dict with keys: 'intent' (str), 'confidence' (float), 'reasoning' (str).
    """
    clean_text = text.strip()
    if not clean_text:
        return {"intent": "OTHER", "confidence": 0.0, "reasoning": "Empty text provided."}

    # Check disk cache
    cache_key = DiskCache.make_key("classify_intent", clean_text)
    cached = classification_cache.get(cache_key)
    if cached is not None:
        return cached

    user_prompt = build_user_prompt(clean_text)

    try:
        result = complete_json(
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=GROQ_MODEL,
            temperature=0.0,
        )

        intent = str(result.get("intent", "OTHER")).upper().strip()
        if intent not in INTENT_LABELS:
            logger.warning(f"Unrecognized intent '{intent}' returned by LLM, defaulting to OTHER")
            intent = "OTHER"

        confidence = float(result.get("confidence", 0.70))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(result.get("reasoning", "Classified by Groq LLM"))

        output = {
            "intent": intent,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
        }
    except Exception as e:
        logger.error(f"Classification failed with error: {e}. Defaulting to OTHER.")
        output = {
            "intent": "OTHER",
            "confidence": 0.0,
            "reasoning": f"Classification error fallback: {str(e)}",
        }

    # Save to disk cache
    classification_cache.set(cache_key, output)
    return output


def batch_classify(queries: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Batch classify a list of queries with disk caching and progress reporting."""
    if limit:
        queries = queries[:limit]

    results = []
    logger.info(f"Classifying {len(queries)} queries...")
    for q in tqdm(queries, desc="Classifying intents"):
        res = classify_intent(q)
        results.append(res)

    classification_cache.save()
    logger.info("Batch classification completed and cached.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocw Intent Classifier")
    parser.add_argument("--text", type=str, help="Single customer text to classify")
    parser.add_argument("--batch", action="store_true", help="Run batch classification on dataset sample")
    parser.add_argument("--limit", type=int, default=100, help="Batch limit")
    args = parser.parse_args()

    if args.text:
        res = classify_intent(args.text)
        print("\nClassification Result:")
        print(json.dumps(res, indent=2))
    elif args.batch:
        sample_queries = []
        if CONVERSATIONS_FILE.exists():
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    for turn in data.get("turns", []):
                        if turn.get("role") == "customer" and turn.get("text"):
                            sample_queries.append(turn["text"])
                            break
                    if len(sample_queries) >= args.limit:
                        break
        batch_classify(sample_queries, limit=args.limit)
    else:
        # Quick demo
        demo_text = "Where is my order? It was supposed to be delivered yesterday!"
        print(f"Classifying demo query: {demo_text}")
        print(json.dumps(classify_intent(demo_text), indent=2))
