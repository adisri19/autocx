"""Unit tests for escalation decision engine."""

import pytest
from src.escalation.decide import decide_escalation, is_rule_based_escalation


def test_rule_based_escalation_legal_keywords():
    msg = "If I don't get my refund today, I am going to contact my lawyer and sue you."
    escalate, reason = is_rule_based_escalation(msg, intent="REFUND_REQUEST", confidence=0.95)
    assert escalate is True
    assert "keyword" in reason.lower() or "legal" in reason.lower()


def test_rule_based_escalation_fraud_keywords():
    msg = "My credit card was charged for an order I never made, this is fraud and stolen identity!"
    escalate, reason = is_rule_based_escalation(msg, intent="REFUND_REQUEST", confidence=0.9)
    assert escalate is True
    assert "fraud" in reason.lower() or "keyword" in reason.lower()


def test_rule_based_escalation_low_confidence():
    msg = "Maybe something happened or maybe not."
    escalate, reason = is_rule_based_escalation(msg, intent="ORDER_STATUS", confidence=0.45)
    assert escalate is True
    assert "confidence" in reason.lower()


def test_rule_based_escalation_other_intent():
    msg = "Can you send a spaceship to Mars?"
    escalate, reason = is_rule_based_escalation(msg, intent="OTHER", confidence=0.85)
    assert escalate is True
    assert "intent" in reason.lower()


def test_decide_escalation_routine_query_no_escalate():
    msg = "Where can I check the tracking status of my delivery?"
    result = decide_escalation(msg, intent="ORDER_STATUS", confidence=0.95, use_llm=False)
    assert result["escalate"] is False
    assert result["reason"] is None


def test_decide_escalation_full_wrapper():
    msg = "I will file a police report against your driver who stole my package"
    result = decide_escalation(msg, intent="GENERAL_COMPLAINT", confidence=0.9)
    assert result["escalate"] is True
    assert result["reason"] is not None
