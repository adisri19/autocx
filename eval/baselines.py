"""Baseline customer support agents: Trivial (majority/canned) and Simple ML (TF-IDF + LogReg)."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.config import INTENT_LABELS, QUERY_REPLY_PAIRS_FILE
from src.escalation.decide import FRAUD_SAFETY_KEYWORDS, HOSTILITY_PROFANITY_KEYWORDS, LEGAL_KEYWORDS
from src.utils.text import clean_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TrivialBaseline:
    """Trivial baseline: predicts majority intent class, never escalates, returns static canned reply."""

    def __init__(self, default_intent: str = "ORDER_STATUS"):
        self.default_intent = default_intent
        self.canned_reply = (
            "Hello! Thank you for contacting customer support. We have received your message "
            "and a representative will look into this as soon as possible."
        )

    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.default_intent,
            "confidence": 0.50,
            "draft_reply": self.canned_reply,
            "escalate": False,
            "escalation_reason": None,
        }


class SimpleMLBaseline:
    """Heuristic / Simple ML baseline:
    - Intent: TF-IDF + LogisticRegression
    - Escalation: Regex keyword matching only (no LLM)
    - Drafter: Top-1 TF-IDF nearest neighbor reply verbatim
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.retrieval_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.retrieval_tfidf: Optional[Any] = None
        self.retrieval_replies: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        pairs_file: Path = QUERY_REPLY_PAIRS_FILE,
        max_retrieval_pairs: int = 5000,
    ) -> "SimpleMLBaseline":
        """Train the classifier and index retrieval pairs."""
        logger.info(f"Training TF-IDF + LogisticRegression classifier on {len(train_texts)} examples...")
        X = self.vectorizer.fit_transform(train_texts)
        self.classifier.fit(X, train_labels)

        # Build TF-IDF retrieval index for verbatim replies
        if pairs_file.exists():
            logger.info(f"Building TF-IDF retrieval index from {pairs_file}...")
            r_queries = []
            r_replies = []
            with open(pairs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    q = item.get("query", "").strip()
                    r = item.get("reply", "").strip()
                    if q and r:
                        r_queries.append(q)
                        r_replies.append(r)
                    if len(r_queries) >= max_retrieval_pairs:
                        break

            if r_queries:
                self.retrieval_tfidf = self.retrieval_vectorizer.fit_transform(r_queries)
                self.retrieval_replies = r_replies
                logger.info(f"Indexed {len(self.retrieval_replies)} historical replies for TF-IDF baseline.")

        self._is_fitted = True
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify intent, check keyword escalation, and retrieve top-1 verbatim reply."""
        clean_text = clean_tweet_text(text)
        if not clean_text:
            return {
                "intent": "OTHER",
                "confidence": 0.0,
                "draft_reply": "Hello! How can we help you?",
                "escalate": False,
                "escalation_reason": None,
            }

        # Intent classification
        if self._is_fitted:
            x_vec = self.vectorizer.transform([clean_text])
            intent = str(self.classifier.predict(x_vec)[0])
            probs = self.classifier.predict_proba(x_vec)[0]
            confidence = float(np.max(probs))
        else:
            intent = "ORDER_STATUS"
            confidence = 0.50

        # Escalation: Keyword matching only
        escalate = False
        reason = None
        if LEGAL_KEYWORDS.search(clean_text):
            escalate = True
            reason = "Keyword: Legal threat"
        elif FRAUD_SAFETY_KEYWORDS.search(clean_text):
            escalate = True
            reason = "Keyword: Fraud / stolen property"
        elif HOSTILITY_PROFANITY_KEYWORDS.search(clean_text):
            escalate = True
            reason = "Keyword: Hostility / profanity"
        elif intent == "OTHER":
            escalate = True
            reason = "Baseline: OTHER intent"

        # Drafter: Top-1 TF-IDF match verbatim
        reply = "Please reach out to our customer service team with your order number so we can assist."
        if self._is_fitted and self.retrieval_tfidf is not None and len(self.retrieval_replies) > 0:
            try:
                q_vec = self.retrieval_vectorizer.transform([clean_text])
                sims = cosine_similarity(q_vec, self.retrieval_tfidf).flatten()
                top_idx = int(np.argmax(sims))
                if sims[top_idx] > 0.05:
                    reply = self.retrieval_replies[top_idx]
            except Exception as e:
                logger.warning(f"Baseline retrieval error: {e}")

        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "draft_reply": reply,
            "escalate": escalate,
            "escalation_reason": reason,
        }


def run_baselines() -> Tuple[TrivialBaseline, SimpleMLBaseline]:
    """Train and return initialized baseline instances."""
    trivial = TrivialBaseline()

    # Gather training data for SimpleML from seed examples
    from eval.build_golden_set import INTENT_SEED_DATA
    train_texts = []
    train_labels = []
    for intent, items in INTENT_SEED_DATA.items():
        for msg, _, _ in items:
            train_texts.append(msg)
            train_labels.append(intent)

    simple_ml = SimpleMLBaseline().fit(train_texts, train_labels)
    logger.info("Baselines initialized successfully.")
    return trivial, simple_ml


if __name__ == "__main__":
    t_base, ml_base = run_baselines()
    sample = "Where is my package? It is 3 days late!"
    print("\nTrivial Baseline Output:")
    print(json.dumps(t_base.predict(sample), indent=2))
    print("\nSimple ML Baseline Output:")
    print(json.dumps(ml_base.predict(sample), indent=2))
