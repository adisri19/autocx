"""Groq API client wrapper with exponential backoff, JSON extraction, and robust error handling."""

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Initialize client if API key is available
_client: Optional[Groq] = None


def get_client() -> Optional[Groq]:
    """Retrieve or initialize the singleton Groq client."""
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if key:
            _client = Groq(api_key=key)
        else:
            logger.warning("GROQ_API_KEY is not set.")
    return _client


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text, handling markdown code fences if present."""
    text = text.strip()
    # Check for markdown code blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return None


def complete_text(
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Send a completion request to Groq LLM with retries and automatic model fallback.

    Args:
        system_prompt: System instruction.
        user_prompt: User content.
        model: Primary Groq model name.
        temperature: Sampling temperature (0 for deterministic).
        max_retries: Number of retries per model.

    Returns:
        Generated text string or empty string on failure.
    """
    client = get_client()
    if not client:
        return ""

    candidate_models = [model]
    for alt in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    for current_model in candidate_models:
        delay = 2.0
        for attempt in range(max_retries):
            try:
                time.sleep(0.1)
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e)
                if "tpd" in err_msg.lower() or "tokens per day" in err_msg.lower():
                    logger.warning(f"Model {current_model} daily quota exhausted, falling back to next candidate model...")
                    break  # Try next candidate model immediately
                logger.warning(f"Groq API error on {current_model} (attempt {attempt+1}/{max_retries}): {err_msg[:120]}")
                if "rate_limit" in err_msg.lower() or "429" in err_msg:
                    time.sleep(delay)
                    delay *= 1.5
                elif attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break

    logger.error("All candidate Groq models failed.")
    return ""


def complete_json(
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Send a prompt requesting JSON output and parse the result.

    Args:
        system_prompt: System instruction specifying JSON schema.
        user_prompt: User content.
        model: Groq model name.
        temperature: Sampling temperature.
        max_retries: Number of retries on failure.

    Returns:
        Parsed dictionary, or empty dict on failure.
    """
    raw = complete_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )
    if not raw:
        return {}
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning(f"Failed to parse JSON from LLM output: {raw[:150]}...")
        return {}
    return parsed
