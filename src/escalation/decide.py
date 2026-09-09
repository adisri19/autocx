"""Hybrid escalation engine: fast keyword/rule path + Groq LLM borderline verification."""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from src.config import (
    BORDERLINE_CONFIDENCE_UPPER,
    ESCALATION_CACHE_FILE,
    ESCALATION_CONFIDENCE_THRESHOLD,
    GROQ_MODEL,
)
from src.utils.cache import DiskCache
from src.utils.llm import complete_json

logger = logging.getLogger(__name__)

# Initialize escalation cache
escalation_cache = DiskCache(ESCALATION_CACHE_FILE)

# Regex rule patterns for fast-path escalation
LEGAL_KEYWORDS = re.compile(
    r"\b(lawyer|attorney|legal action|sue|suing|lawsuit|court|small claims|litigation|counsel)\b",
    re.IGNORECASE,
)
FRAUD_SAFETY_KEYWORDS = re.compile(
    r"\b(police|fraud|fraudulent|stolen|theft|scam|chargeback|unauthorized|identity theft|investigation)\b",
    re.IGNORECASE,
)
HOSTILITY_PROFANITY_KEYWORDS = re.compile(
    r"\b(fuck|fucking|bullshit|asshole|bastard|scumbag|piece of shit|disgusting|pathetic|garbage|scam artists)\b",
    re.IGNORECASE,
)
AGENT_DEMANDS = re.compile(
    r"\b(speak to a manager|talk to a manager|supervisor|talk to a human|human agent|live representative|escalate this)\b",
    re.IGNORECASE,
)


def is_rule_based_escalation(
    message: str, intent: str, confidence: float
) -> Tuple[bool, Optional[str]]:
    """Determine if a message should be escalated immediately via fast deterministic rules.

    Args:
        message: Customer message text.
        intent: Classified intent label.
        confidence: Classification confidence score [0.0, 1.0].

    Returns:
        Tuple of (should_escalate, reason_string_or_none).
    """
    text = message.strip()

    # 1. Critical Legal Threat
    if LEGAL_KEYWORDS.search(text):
        return True, "Rule-based: Legal action or attorney threat detected"

    # 2. Law enforcement, fraud, or stolen property
    if FRAUD_SAFETY_KEYWORDS.search(text):
        return True, "Rule-based: Fraud, theft, or law enforcement concern detected"

    # 3. Severe hostility or abusive profanity
    if HOSTILITY_PROFANITY_KEYWORDS.search(text):
        return True, "Rule-based: Extreme customer hostility or abusive language detected"

    # 4. Explicit demand for human representative/manager
    if AGENT_DEMANDS.search(text):
        return True, "Rule-based: Customer explicitly demanded supervisor/human agent"

    # 5. Low confidence threshold
    if confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return True, f"Rule-based: Low classification confidence ({confidence:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD})"

    # 6. Ambiguous / unclassifiable 'OTHER' intent
    if intent == "OTHER":
        return True, "Rule-based: Ambiguous or unclassifiable intent ('OTHER')"

    return False, None


def check_escalation_llm(message: str, intent: str, confidence: float) -> Dict[str, Any]:
    """Secondary LLM check for borderline messages.

    Args:
        message: Customer query.
        intent: Detected intent.
        confidence: Confidence score.

    Returns:
        Dict with 'escalate' (bool) and 'reason' (str).
    """
    cache_key = DiskCache.make_key("escalation_llm", message, intent, round(confidence, 2))
    cached = escalation_cache.get(cache_key)
    if cached is not None:
        return cached

    system_prompt = (
        "You are an expert customer support supervisor. Evaluate whether this customer message requires "
        "human escalation.\n"
        "Criteria for escalation:\n"
        "- The customer is excessively frustrated, threatening churn, or reporting high monetary loss\n"
        "- The situation is nuanced or requires manual intervention/investigation that a bot cannot resolve\n"
        "- Ambiguous safety or sensitive policy concerns\n\n"
        "Respond STRICTLY with a valid JSON object:\n"
        '{"escalate": true|false, "reason": "<brief justification>"}'
    )

    user_prompt = f'Customer Message: "{message}"\nDetected Intent: {intent}\nConfidence: {confidence:.2f}'

    result = complete_json(system_prompt, user_prompt, model=GROQ_MODEL, temperature=0.0)

    if not result or "escalate" not in result:
        # Fail-safe fallback to escalation when LLM is unreachable on borderline cases
        decision = {
            "escalate": True,
            "reason": "LLM unavailable; defaulting to safe escalation",
        }
    else:
        decision = {
            "escalate": bool(result.get("escalate", False)),
            "reason": str(result.get("reason", "LLM determined escalation status")),
        }

    escalation_cache.set(cache_key, decision)
    return decision


def decide_escalation(
    message: str,
    intent: str,
    confidence: float,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Hybrid escalation decision: Fast rule check followed by borderline LLM evaluation.

    Args:
        message: Customer message text.
        intent: Classified intent label.
        confidence: Classification confidence score [0.0, 1.0].
        use_llm: Whether to invoke LLM for borderline cases.

    Returns:
        Dict with keys 'escalate' (bool) and 'reason' (str or None).
    """
    # Check fast rule-based path first
    should_escalate, reason = is_rule_based_escalation(message, intent, confidence)
    if should_escalate:
        return {"escalate": True, "reason": reason}

    # Borderline zone: confidence is between threshold and upper bound (e.g. 0.60 to 0.75)
    is_borderline = (
        ESCALATION_CONFIDENCE_THRESHOLD <= confidence <= BORDERLINE_CONFIDENCE_UPPER
    )

    if is_borderline and use_llm:
        return check_escalation_llm(message, intent, confidence)

    # Standard non-escalated routine flow
    return {"escalate": False, "reason": None}
