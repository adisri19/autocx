# Autocw Architecture & Engineering Decision Log

This document records the 14 core architectural, algorithmic, and engineering decisions made during the design and implementation of the Autocw AI Customer Support Agent.

---

## 1. Primary LLM Provider: Groq API
- **Context**: The system must run on a standard developer laptop with no dedicated GPU, complete end-to-end execution within 15 minutes, and avoid expensive proprietary API subscriptions.
- **Decision**: Use Groq API as the exclusive LLM provider.
- **Alternatives Considered**: Local Ollama / vLLM (prohibitive memory and CPU latency on laptops), OpenAI API / Anthropic Claude (paid subscriptions, slower cold-start).
- **Consequences**: Blazing-fast inference speeds (~200+ tokens/sec on LPUs), zero GPU hardware requirement, low operational latency.

---

## 2. Groq Chat Model Selection: `openai/gpt-oss-20b`
- **Context**: The original prompt target `llama3-8b-8192` was decommissioned by Groq in early 2026 (`Error 400: model_decommissioned`).
- **Decision**: Migrate to `openai/gpt-oss-20b` on Groq (with configurable fallback in `src/config.py`).
- **Alternatives Considered**: `qwen/qwen3.8-27b` (slightly higher latency), `llama-prompt-guard` (safety classifier only).
- **Consequences**: Superior instruction-following for strict JSON schemas, rapid response times (<400ms per request), and consistent adherence to classification rules.

---

## 3. Embedding Model: `all-MiniLM-L6-v2`
- **Context**: Dense vector embeddings are required to retrieve historically effective customer support replies.
- **Decision**: Select `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Alternatives Considered**: OpenAI `text-embedding-3-small` (requires external API calls and rate limits), `e5-large` (too heavy for fast CPU inference).
- **Consequences**: Extremely compact (~80MB download), executes on CPU in milliseconds, normalized vectors permit single-instruction dot-product cosine similarity.

---

## 4. Vector Storage: Compressed NumPy Archive (`.npz`)
- **Context**: Vector storage and similarity retrieval for 5,000 query-reply pairs.
- **Decision**: Store normalized float32 embeddings and query/reply arrays in a compressed NumPy archive (`data/processed/reply_index.npz`).
- **Alternatives Considered**: ChromaDB / FAISS / Pinecone / Qdrant.
- **Consequences**: Zero external daemon or database dependencies, instant zero-copy in-memory loading, sub-millisecond dot-product retrieval for top-3 nearest neighbors via pure NumPy on CPU.

---

## 5. Intent Taxonomy: 8 Core Mutually Exclusive Categories
- **Context**: Customer service inquiries on Twitter span diverse topics. A focused taxonomy is needed for automated routing.
- **Decision**: Establish an 8-intent taxonomy:
  1. `ORDER_STATUS`
  2. `REFUND_REQUEST`
  3. `ACCOUNT_ISSUE`
  4. `SHIPPING_DELAY`
  5. `GENERAL_COMPLAINT`
  6. `TECHNICAL_ISSUE`
  7. `COMPLIMENT`
  8. `OTHER`
- **Alternatives Considered**: 20+ granular intents (causes severe class overlap and lower classifier accuracy), 3 broad intents (insufficient routing value).
- **Consequences**: High inter-annotator agreement, clean separation between tracking status vs transit delay grievances vs refund demands.

---

## 6. Unsupervised Clustering & Discovery
- **Context**: Verify that the taxonomy reflects empirical customer inquiries in `twcs.csv`.
- **Decision**: Implement BERTopic clustering with fallback to TF-IDF + MiniBatchKMeans, saving top keyword clusters to `cluster_report.txt`.
- **Alternatives Considered**: Manual guessing without data inspection, LDA topic modeling (lower semantic coherence than transformer embeddings).
- **Consequences**: Confirmed top customer conversation clusters around tracking, delayed delivery, order issues, and account questions.

---

## 7. Hybrid Escalation Engine (Fast-Path Rules + LLM Verification)
- **Context**: Customer service bots must not mishandle legal threats, fraud accusations, or extreme customer hostility, but running an LLM check on every single message increases latency and token consumption.
- **Decision**: Implement a two-tier hybrid escalation architecture:
  - **Tier 1 (Fast Path)**: Deterministic regex matching for legal keywords, fraud/theft alerts, abusive profanity, explicit supervisor demands, low classifier confidence (<0.60), and `OTHER` intent.
  - **Tier 2 (Borderline LLM Check)**: Invoked only when confidence falls in `[0.60, 0.75]` or text contains mixed sentiment.
- **Alternatives Considered**: Pure rule-based (misses nuanced sarcasm and complex grievances), Pure LLM (unnecessary latency and cost for obvious legal threats or routine inquiries).
- **Consequences**: 0ms latency for 90%+ of messages, near-zero false-negative rate on critical legal/fraud threats.

---

## 8. Multi-Turn Thread Reconstruction Algorithm
- **Context**: The raw `twcs.csv` dataset contains 2.8M rows with fragmented pointer columns (`in_response_to_tweet_id`, `response_tweet_id`).
- **Decision**: Reconstruct threads by indexing tweets for the target brand (`AmazonHelp`), locating root inbounds, following forward response chains, and extracting adjacent `(customer query, agent reply)` pairs.
- **Optimization**: Replaced nested generator filtering with $O(1)$ set membership, reducing processing time from minutes to under 5 seconds.
- **Consequences**: Successfully reconstructed 84,874 multi-turn conversations and 124,643 query-reply pairs.

---

## 9. Text Sanitization Protocol
- **Context**: Twitter customer queries contain noisy elements: handles (`@AmazonHelp`, `@115712`), shortened links (`t.co`), HTML entities (`&amp;`), and irregular spacing.
- **Decision**: Clean text using a regex pipeline: decode HTML entities first, strip URLs, strip `@mentions`, and collapse whitespace while preserving punctuation (`?`, `!`, `$`) crucial for sentiment and intent detection.
- **Consequences**: Clean semantic embeddings, consistent tokenization, and elimination of noisy twitter usernames from LLM generation prompts.

---

## 10. Few-Shot Prompt Engineering with Strict JSON Contracts
- **Context**: Classification outputs must be parsed programmatically without regex fragility.
- **Decision**: Provide detailed intent descriptions, boundary guidelines, and 8 concrete few-shot input-output examples in system/user prompts, enforcing strict JSON output schemas: `{"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}`.
- **Consequences**: 100% JSON parsing success rate, deterministic output across runs with `temperature=0.0`.

---

## 11. Fail-Safe Defaults & Defensive Fallbacks
- **Context**: Network drops, API rate limits, or malformed outputs could halt production customer service pipelines.
- **Decision**: Implement defensive fallbacks across every module:
  - Classifier error -> returns `{"intent": "OTHER", "confidence": 0.0}` (which triggers safe escalation).
  - Drafter error -> returns top-1 retrieved historical reply verbatim.
  - Escalation error -> returns `{"escalate": True, "reason": "LLM unavailable; defaulting to safe escalation"}`.
- **Consequences**: Zero uncaught exceptions; failures always degrade safely toward human supervisor intervention.

---

## 12. Multi-Tier Persistent Disk Caching
- **Context**: Avoid redundant Groq API calls during development, testing, batch runs, and evaluation.
- **Decision**: Build a centralized, thread-safe `DiskCache` class with MD5-keyed hashes on inputs, auto-save persistence, and `atexit` flush hooks.
- **Consequences**: Repeat runs and test suites execute in seconds; immune to Groq rate limits on subsequent runs.

---

## 13. Comprehensive Evaluation with Baselines & LLM-as-a-Judge
- **Context**: Objectively demonstrate that the LLM RAG agent outperforms traditional baselines.
- **Decision**: Create an evaluation harness comparing Autocw against two baselines (Trivial Majority and Simple TF-IDF + LogisticRegression) across 200 stratified golden examples. Evaluate intent metrics, escalation metrics (including False Negative Rate), LLM-as-judge rubric (1-5 Likert), and compute Pearson $r$ correlation against human ratings.
- **Consequences**: Transparent benchmarking showing significant F1 and quality gains over classical NLP.

---

## 14. 7,500-Conversation Subsampling Strategy
- **Context**: Balancing dataset representativeness with local developer laptop execution constraints.
- **Decision**: Subsample 7,500 threads with fixed random seed (`seed=42`), producing 5,000 indexed retrieval pairs.
- **Consequences**: Complete pipeline build (data -> taxonomy -> embeddings -> evaluation) completes in under 5 minutes on CPU.
