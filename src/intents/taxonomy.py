"""Unsupervised intent discovery and taxonomy definition."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import (
    CLUSTER_REPORT_FILE,
    CONVERSATIONS_FILE,
    DATA_PROCESSED_DIR,
    INTENTS_FILE,
    RANDOM_SEED,
    SRC_INTENTS_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 8-intent taxonomy definition with rich descriptions and few-shot exemplar queries
DEFAULT_INTENTS = {
    "ORDER_STATUS": {
        "description": "Inquiries regarding the current status, tracking details, shipment progress, or delivery date of an order.",
        "keywords": ["track", "tracking", "where is", "order status", "shipped", "delivery date", "package location"],
        "examples": [
            "Where is my package? Tracking says it has not moved.",
            "Can you tell me if order #123-4567 has shipped yet?",
            "I need an update on my delivery tracking.",
            "When will my order arrive? It was supposed to be here by now.",
        ],
    },
    "REFUND_REQUEST": {
        "description": "Requests for refunds, return processing, billing discrepancies, double charges, or money back.",
        "keywords": ["refund", "money back", "charged twice", "overcharged", "return", "cancelled order charge"],
        "examples": [
            "I returned my item last week and still have not received my refund.",
            "I was charged twice for the same purchase, please refund the second charge.",
            "I cancelled my order, how long until my money is back in my account?",
            "Can I get a full refund for this defective product?",
        ],
    },
    "ACCOUNT_ISSUE": {
        "description": "Problems with logging in, password reset, account lock, verification codes, OTP, or email updates.",
        "keywords": ["login", "password", "locked out", "account", "otp", "verification code", "sign in", "reset"],
        "examples": [
            "I am locked out of my account and cannot reset my password.",
            "I am not receiving the OTP code on my mobile phone to verify my account.",
            "Someone hacked my account and changed the email address.",
            "How do I update the email associated with my profile?",
        ],
    },
    "SHIPPING_DELAY": {
        "description": "Complaints specifically regarding late delivery, missed delivery windows, transit delays, or carrier hold-ups.",
        "keywords": ["late", "delayed", "delay", "not arrived", "missed delivery", "overdue", "past delivery date"],
        "examples": [
            "My guaranteed delivery date was yesterday and my package still is not here.",
            "Why is my delivery constantly getting delayed?",
            "It has been over two weeks and the package is running severely late.",
            "Carrier missed the delivery appointment.",
        ],
    },
    "GENERAL_COMPLAINT": {
        "description": "General dissatisfaction, damaged goods on arrival, rude service experience, or poor product quality.",
        "keywords": ["damaged", "broken", "unhappy", "terrible service", "disappointed", "poor quality", "complaint"],
        "examples": [
            "The box was completely crushed and the item inside is shattered.",
            "Your representative was extremely unhelpful and rude.",
            "This is the worst customer service experience I have ever had.",
            "Very disappointed with the quality of this item.",
        ],
    },
    "TECHNICAL_ISSUE": {
        "description": "Website or mobile application bugs, checkout errors, payment gateway failures, or broken links.",
        "keywords": ["app crash", "error code", "website down", "checkout failed", "payment error", "glitch", "bug"],
        "examples": [
            "The app keeps crashing whenever I click on my cart.",
            "I am getting error code 500 when attempting to complete checkout.",
            "The website payment page will not load my payment options.",
            "Your search button is completely broken on iOS.",
        ],
    },
    "COMPLIMENT": {
        "description": "Expressions of gratitude, satisfaction, positive feedback, or compliments on helpful service.",
        "keywords": ["thank you", "thanks", "great service", "appreciate", "awesome", "kudos", "wonderful"],
        "examples": [
            "Thank you so much for resolving my issue so quickly!",
            "Great service as always, really appreciate the prompt assistance.",
            "The delivery driver was so polite and helpful, kudos to the team!",
            "Thanks a lot, that worked perfectly.",
        ],
    },
    "OTHER": {
        "description": "Messages that are ambiguous, unclassifiable, multi-topic, irrelevant, or spam.",
        "keywords": ["misc", "unclear", "random", "hello", "info", "other"],
        "examples": [
            "Hello there.",
            "Can you tell me if you sell spaceships?",
            "I need info about your corporate history.",
            "Check this out!",
        ],
    },
}


def load_inbound_queries(conversations_file: Path) -> List[str]:
    """Extract initial customer query strings from conversations.jsonl."""
    queries = []
    if not conversations_file.exists():
        logger.warning(f"{conversations_file} not found.")
        return queries

    with open(conversations_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            turns = data.get("turns", [])
            for turn in turns:
                if turn.get("role") == "customer" and turn.get("text"):
                    queries.append(turn["text"])
                    break  # First customer message in thread
    return queries


def discover_clusters(queries: List[str], n_clusters: int = 8) -> Tuple[Dict[int, List[str]], str]:
    """Discover topic clusters using BERTopic if available, otherwise K-Means fallback."""
    report_lines = []
    clusters: Dict[int, List[str]] = {}

    if not queries:
        logger.warning("No queries available for clustering.")
        return clusters, "No queries found."

    logger.info(f"Clustering {len(queries):,} customer queries...")
    sample_queries = queries[:3000]  # Fast, representative subset

    try:
        from bertopic import BERTopic
        logger.info("Running BERTopic clustering...")
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=15,
            nr_topics=n_clusters + 1,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(sample_queries)
        topic_info = topic_model.get_topic_info()
        
        report_lines.append("=== BERTopic Cluster Discovery Report ===\n")
        for _, row in topic_info.iterrows():
            t_id = row["Topic"]
            t_count = row["Count"]
            t_name = row.get("Name", "")
            words = [word for word, _ in topic_model.get_topic(t_id)[:8]] if t_id != -1 else ["outlier"]
            report_lines.append(f"Cluster {t_id} (count={t_count}): {', '.join(words)}")
            clusters[t_id] = words
    except Exception as e:
        logger.warning(f"BERTopic clustering encountered an issue: {e}. Falling back to TF-IDF + KMeans.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        tfidf_mat = vectorizer.fit_transform(sample_queries)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, batch_size=256)
        labels = kmeans.fit_predict(tfidf_mat)
        
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        report_lines.append("=== TF-IDF + KMeans Cluster Discovery Report ===\n")
        for i in range(n_clusters):
            top_words = [terms[ind] for ind in order_centroids[i, :8]]
            count = (labels == i).sum()
            report_lines.append(f"Cluster {i} (count={count}): {', '.join(top_words)}")
            clusters[i] = top_words

    report_str = "\n".join(report_lines)
    return clusters, report_str


def build_intent_taxonomy() -> Dict[str, Any]:
    """Construct, validate, and persist the 8-intent taxonomy."""
    # Discover clusters from processed data if available
    queries = load_inbound_queries(CONVERSATIONS_FILE)
    if queries:
        _, report = discover_clusters(queries, n_clusters=8)
    else:
        report = "Pre-computed taxonomy used."

    # Save cluster report
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Cluster report written to {CLUSTER_REPORT_FILE}")

    # Persist intents to both data/processed/ and src/intents/
    for path in [INTENTS_FILE, SRC_INTENTS_FILE]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_INTENTS, f, indent=2, ensure_ascii=False)
        logger.info(f"Intent taxonomy persisted to {path}")

    return DEFAULT_INTENTS


if __name__ == "__main__":
    build_intent_taxonomy()
