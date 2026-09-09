"""Data pipeline for loading, filtering, reconstructing threads, and processing twcs.csv."""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.config import (
    CONVERSATIONS_FILE,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    QUERY_REPLY_PAIRS_FILE,
    RANDOM_SEED,
    RAW_DATASET_FILE,
    SUBSAMPLE_SIZE,
    TARGET_BRAND,
)
from src.utils.text import clean_tweet_text, is_meaningful_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Download the Customer Support on Twitter dataset from Kaggle if not present."""
    if RAW_DATASET_FILE.exists() and not force:
        logger.info(f"Dataset already exists at {RAW_DATASET_FILE}, skipping download.")
        return RAW_DATASET_FILE

    logger.info("Downloading dataset from Kaggle: thoughtvector/customer-support-on-twitter...")
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "thoughtvector/customer-support-on-twitter",
            path=str(DATA_RAW_DIR),
            unzip=True,
            quiet=False,
        )
        logger.info(f"Download complete: {RAW_DATASET_FILE}")
        return RAW_DATASET_FILE
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise RuntimeError(
            f"Could not download dataset. Please place twcs.csv manually at {RAW_DATASET_FILE} or check Kaggle credentials."
        ) from e


def determine_top_brand(df: pd.DataFrame) -> str:
    """Find the brand with the highest number of outbound responses."""
    outbound = df[df["inbound"] == False]
    top_brand = outbound["author_id"].value_counts().index[0]
    logger.info(f"Top brand by outbound volume: {top_brand} ({outbound['author_id'].value_counts().iloc[0]} tweets)")
    return str(top_brand)


def reconstruct_threads_from_df(
    df: pd.DataFrame, target_brand: str = TARGET_BRAND
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reconstruct conversational threads and query-reply pairs for the target brand.
    
    Args:
        df: DataFrame with twcs schema.
        target_brand: The brand handle to filter on (e.g. 'AmazonHelp').
        
    Returns:
        Tuple of (reconstructed_threads, query_reply_pairs).
    """
    logger.info(f"Building lookup tables for target brand: {target_brand}...")

    # Fast row lookup by tweet_id
    # We only care about rows authored by target_brand or directed at target_brand
    brand_tweets = df[df["author_id"] == target_brand]
    brand_tweet_ids = set(brand_tweets["tweet_id"])

    # Inbound tweets responded to by brand
    in_response_ids = set(brand_tweets["in_response_to_tweet_id"].dropna().astype(int if df["tweet_id"].dtype == int else str))

    # Relevant tweet IDs: brand tweets and the customer tweets that triggered them or responded to them
    relevant_ids = brand_tweet_ids | in_response_ids
    
    # Filter working subset
    sub_df = df[df["tweet_id"].isin(relevant_ids)].copy()
    
    # Build fast index: tweet_id -> record dict
    records = {}
    for row in sub_df.itertuples(index=False):
        records[row.tweet_id] = {
            "tweet_id": row.tweet_id,
            "author_id": str(row.author_id),
            "inbound": bool(row.inbound),
            "created_at": str(row.created_at),
            "text": str(row.text),
            "response_tweet_id": str(row.response_tweet_id) if pd.notna(row.response_tweet_id) else None,
            "in_response_to_tweet_id": row.in_response_to_tweet_id if pd.notna(row.in_response_to_tweet_id) else None,
        }

    # Connect parents and children to find thread roots
    # A root is an inbound customer tweet that is not responding to another tweet in our set
    roots = []
    for tid, rec in records.items():
        if rec["inbound"]:
            parent_id = rec["in_response_to_tweet_id"]
            if parent_id is None or parent_id not in records:
                roots.append(tid)

    logger.info(f"Identified {len(roots)} root customer queries for brand {target_brand}.")

    threads = []
    query_reply_pairs = []

    for root_id in tqdm(roots, desc="Reconstructing threads"):
        current_id = root_id
        turns = []
        visited = set()

        # Follow the chain forward
        while current_id and current_id in records and current_id not in visited:
            visited.add(current_id)
            rec = records[current_id]
            role = "customer" if rec["inbound"] else "agent"
            cleaned = clean_tweet_text(rec["text"])
            
            turns.append({
                "tweet_id": rec["tweet_id"],
                "role": role,
                "author_id": rec["author_id"],
                "created_at": rec["created_at"],
                "text": cleaned,
                "raw_text": rec["text"],
            })

            # Check for next turn
            resp_ids_raw = rec["response_tweet_id"]
            if resp_ids_raw:
                # Can be comma-separated list of IDs in twcs
                first_resp = str(resp_ids_raw).split(",")[0].strip()
                try:
                    next_id = int(first_resp) if isinstance(root_id, int) else first_resp
                except ValueError:
                    next_id = first_resp
                current_id = next_id if next_id in records else None
            else:
                current_id = None

        if len(turns) >= 2:
            threads.append({
                "thread_id": f"thread_{root_id}",
                "brand": target_brand,
                "turns": turns,
            })

            # Extract customer query -> agent reply pairs
            for i in range(len(turns) - 1):
                if turns[i]["role"] == "customer" and turns[i + 1]["role"] == "agent":
                    q_text = turns[i]["text"]
                    r_text = turns[i + 1]["text"]
                    if is_meaningful_text(q_text) and is_meaningful_text(r_text):
                        query_reply_pairs.append({
                            "query_id": turns[i]["tweet_id"],
                            "reply_id": turns[i + 1]["tweet_id"],
                            "query": q_text,
                            "reply": r_text,
                        })

    return threads, query_reply_pairs


def run_pipeline(
    raw_csv_path: Optional[Path] = None,
    subsample_size: int = SUBSAMPLE_SIZE,
    target_brand: str = TARGET_BRAND,
    seed: int = RANDOM_SEED,
) -> None:
    """Execute complete data pipeline: load, filter, reconstruct, subsample, and write JSONL."""
    csv_file = raw_csv_path or RAW_DATASET_FILE
    if not csv_file.exists():
        raise FileNotFoundError(f"Raw CSV not found at {csv_file}. Use --download to fetch it.")

    logger.info(f"Loading raw dataset from {csv_file} (low_memory=False)...")
    df = pd.read_csv(csv_file, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {csv_file}.")

    # Verify top brand
    detected_brand = determine_top_brand(df)
    brand = target_brand if target_brand else detected_brand

    # Reconstruct threads
    threads, pairs = reconstruct_threads_from_df(df, target_brand=brand)
    logger.info(f"Reconstructed {len(threads):,} multi-turn threads and {len(pairs):,} query-reply pairs.")

    # Deterministic Subsample
    random.seed(seed)
    if len(threads) > subsample_size:
        sampled_threads = random.sample(threads, subsample_size)
        logger.info(f"Subsampled {subsample_size:,} threads from {len(threads):,}.")
    else:
        sampled_threads = threads
        logger.info(f"Retained all {len(sampled_threads):,} threads (less than cap {subsample_size:,}).")

    # Filter pairs matching sampled threads
    sampled_tweet_ids = {turn["tweet_id"] for t in sampled_threads for turn in t["turns"]}
    sampled_pairs = [p for p in pairs if p["query_id"] in sampled_tweet_ids]
    if len(sampled_pairs) > subsample_size:
        sampled_pairs = random.sample(sampled_pairs, subsample_size)

    # Save to JSONL
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(sampled_threads):,} conversations to {CONVERSATIONS_FILE}...")
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        for item in sampled_threads:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Writing {len(sampled_pairs):,} query-reply pairs to {QUERY_REPLY_PAIRS_FILE}...")
    with open(QUERY_REPLY_PAIRS_FILE, "w", encoding="utf-8") as f:
        for item in sampled_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Data pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocw Data Pipeline")
    parser.add_argument("--download", action="store_true", help="Download twcs.csv from Kaggle if missing")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE_SIZE, help="Subsample size (default: 7500)")
    parser.add_argument("--brand", type=str, default=TARGET_BRAND, help="Target brand handle")
    args = parser.parse_args()

    if args.download:
        download_dataset()

    run_pipeline(subsample_size=args.subsample, target_brand=args.brand)
