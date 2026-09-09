"""Configuration settings and global constants for Autocw."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = PROJECT_ROOT / "eval"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure runtime directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CACHE_DIR, EVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
RAW_DATASET_FILE = DATA_RAW_DIR / "twcs.csv"
CONVERSATIONS_FILE = DATA_PROCESSED_DIR / "conversations.jsonl"
QUERY_REPLY_PAIRS_FILE = DATA_PROCESSED_DIR / "query_reply_pairs.jsonl"
INTENTS_FILE = DATA_PROCESSED_DIR / "intents.json"
SRC_INTENTS_FILE = PROJECT_ROOT / "src" / "intents" / "intents.json"
CLUSTER_REPORT_FILE = DATA_PROCESSED_DIR / "cluster_report.txt"
REPLY_INDEX_FILE = DATA_PROCESSED_DIR / "reply_index.npz"

# Cache files
CLASSIFICATION_CACHE_FILE = CACHE_DIR / "classification_cache.json"
DRAFT_CACHE_FILE = CACHE_DIR / "draft_cache.json"
ESCALATION_CACHE_FILE = CACHE_DIR / "escalation_cache.json"
JUDGE_CACHE_FILE = CACHE_DIR / "judge_cache.json"

# Evaluation files
GOLDEN_SET_FILE = EVAL_DIR / "golden_set.jsonl"
EVAL_RESULTS_FILE = EVAL_DIR / "results.json"

# Target brand & dataset sampling
TARGET_BRAND = "AmazonHelp"
SUBSAMPLE_SIZE = 7500
GOLDEN_SET_SIZE = 200
RANDOM_SEED = 42

# Confidence thresholds for escalation
ESCALATION_CONFIDENCE_THRESHOLD = 0.6
BORDERLINE_CONFIDENCE_UPPER = 0.75

# Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 8 Core Intents
INTENT_LABELS = [
    "ORDER_STATUS",
    "REFUND_REQUEST",
    "ACCOUNT_ISSUE",
    "SHIPPING_DELAY",
    "GENERAL_COMPLAINT",
    "TECHNICAL_ISSUE",
    "COMPLIMENT",
    "OTHER",
]
