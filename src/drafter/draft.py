"""Reply drafter using MiniLM-L6-v2 semantic retrieval and Groq LLM RAG generation."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    DRAFT_CACHE_FILE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    QUERY_REPLY_PAIRS_FILE,
    REPLY_INDEX_FILE,
)
from src.utils.cache import DiskCache
from src.utils.llm import complete_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize draft cache
draft_cache = DiskCache(DRAFT_CACHE_FILE)

# Singleton embedding model and index state
_embedder: Optional[SentenceTransformer] = None
_index_data: Optional[Dict[str, Any]] = None


def get_embedder() -> SentenceTransformer:
    """Retrieve or lazily initialize the SentenceTransformer embedding model."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model {EMBEDDING_MODEL} on CPU...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def build_embedding_index(
    pairs_file: Path = QUERY_REPLY_PAIRS_FILE,
    output_file: Path = REPLY_INDEX_FILE,
    max_items: int = 5000,
) -> Path:
    """Read query-reply pairs, compute normalized embeddings, and save to npz."""
    if not pairs_file.exists():
        raise FileNotFoundError(f"Query-reply pairs not found at {pairs_file}. Run data pipeline first.")

    logger.info(f"Reading query-reply pairs from {pairs_file}...")
    queries = []
    replies = []

    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get("query", "").strip()
            r = item.get("reply", "").strip()
            if q and r:
                queries.append(q)
                replies.append(r)
            if len(queries) >= max_items:
                break

    logger.info(f"Encoding {len(queries):,} queries with {EMBEDDING_MODEL}...")
    embedder = get_embedder()
    embeddings = embedder.encode(queries, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        queries=np.array(queries, dtype=object),
        replies=np.array(replies, dtype=object),
    )
    logger.info(f"Saved embedding index with {len(queries):,} items to {output_file}")
    return output_file


def load_embedding_index(index_file: Path = REPLY_INDEX_FILE) -> Dict[str, Any]:
    """Load precomputed embeddings and metadata from npz file."""
    global _index_data
    if _index_data is None:
        if not index_file.exists():
            logger.warning(f"Index file {index_file} not found. Building index automatically...")
            build_embedding_index(output_file=index_file)

        data = np.load(index_file, allow_pickle=True)
        _index_data = {
            "embeddings": data["embeddings"],
            "queries": data["queries"],
            "replies": data["replies"],
        }
        logger.info(f"Loaded embedding index containing {len(_index_data['queries']):,} entries.")
    return _index_data


def retrieve_similar(
    query: str,
    top_k: int = 3,
    index_file: Path = REPLY_INDEX_FILE,
) -> List[Dict[str, Any]]:
    """Retrieve top-K most similar historical customer queries and agent replies.

    Args:
        query: New customer query.
        top_k: Number of nearest neighbors to retrieve.
        index_file: Path to reply_index.npz.

    Returns:
        List of dicts with 'query', 'reply', and 'score'.
    """
    index = load_embedding_index(index_file)
    embedder = get_embedder()

    query_vec = embedder.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype=np.float32)

    # Cosine similarity via dot product on normalized vectors
    scores = np.dot(index["embeddings"], query_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "query": str(index["queries"][idx]),
            "reply": str(index["replies"][idx]),
            "score": float(scores[idx]),
        })
    return results


DRAFT_SYSTEM_PROMPT = """You are an empathetic customer support agent.
Draft a concise (2-3 sentences), professional response to the customer based on the similar reference replies.
Address the customer directly without placeholders."""


def draft_reply(
    customer_query: str,
    intent: str,
    top_k: int = 2,
    index_file: Path = REPLY_INDEX_FILE,
) -> str:
    """Generate a RAG-augmented response draft using Groq LLM and retrieved historical replies.

    Args:
        customer_query: The incoming customer query.
        intent: Detected intent category.
        top_k: Number of reference pairs to retrieve.
        index_file: Path to vector index.

    Returns:
        Drafted response string.
    """
    clean_query = customer_query.strip()
    if not clean_query:
        return "Hello! How can I assist you with your order today?"

    # Check disk cache
    cache_key = DiskCache.make_key("draft_reply", clean_query, intent)
    cached = draft_cache.get(cache_key)
    if cached:
        return cached

    # Retrieve similar historical interactions
    try:
        retrieved = retrieve_similar(clean_query, top_k=top_k, index_file=index_file)
    except Exception as e:
        logger.warning(f"Retrieval error: {e}. Proceeding without retrieval context.")
        retrieved = []

    # Format retrieved examples
    context_str = ""
    if retrieved:
        context_parts = []
        for i, item in enumerate(retrieved, 1):
            context_parts.append(
                f"Example {i}:\nCustomer: {item['query']}\nSupport Agent: {item['reply']}"
            )
        context_str = "\n\n".join(context_parts)

    user_prompt = (
        f"Customer Inquiry: \"{clean_query}\"\n"
        f"Intent Category: {intent}\n\n"
        f"Similar Historical Interactions:\n{context_str}\n\n"
        "Draft the optimal response:"
    )

    try:
        draft = complete_text(
            system_prompt=DRAFT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=GROQ_MODEL,
            temperature=0.2,
        )
        draft = draft.strip().strip('"')
    except Exception as e:
        logger.error(f"Groq drafting failed: {e}")
        draft = ""

    # Resilient fallback: use top retrieved reply if LLM fails or returns empty
    if not draft and retrieved:
        logger.info("Using top-1 retrieved reply as fallback draft.")
        draft = retrieved[0]["reply"]
    elif not draft:
        draft = "Hello! We apologize for any inconvenience. Please reach out to our support team with your order details so we can investigate right away."

    draft_cache.set(cache_key, draft)
    return draft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocw Reply Drafter")
    parser.add_argument("--build-index", action="store_true", help="Build reply embedding index")
    parser.add_argument("--demo", type=str, help="Demo query to draft reply for")
    parser.add_argument("--intent", type=str, default="ORDER_STATUS", help="Intent for demo query")
    args = parser.parse_args()

    if args.build_index:
        build_embedding_index()
    elif args.demo:
        print(f"Query: {args.demo} (Intent: {args.intent})")
        reply = draft_reply(args.demo, args.intent)
        print(f"\nDraft Reply:\n{reply}")
    else:
        demo = "My order #123-4567 hasn't arrived yet, where is it?"
        print(f"Drafting demo reply for: {demo}")
        print(draft_reply(demo, "ORDER_STATUS"))
