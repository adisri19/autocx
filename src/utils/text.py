"""Text processing and cleaning utilities."""

import html
import re
from typing import List

# Precompile regex patterns for performance
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_CLEANUP = re.compile(r"[\r\n\t]+")


def clean_tweet_text(text: str) -> str:
    """Clean raw tweet text by removing URLs, mentions, unescaping HTML, and normalizing whitespace.
    
    Args:
        text: Raw input tweet string.
        
    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities first (&amp; -> &, &lt; -> <, etc.)
    cleaned = html.unescape(text)
    
    # Strip URLs
    cleaned = URL_PATTERN.sub("", cleaned)
    
    # Strip user mentions (@username, @AmazonHelp, @123456)
    cleaned = MENTION_PATTERN.sub("", cleaned)
    
    # Replace newlines, carriage returns, tabs with space
    cleaned = SPECIAL_CHAR_CLEANUP.sub(" ", cleaned)
    
    # Collapse multiple whitespace characters into single space
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    
    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def is_meaningful_text(text: str, min_chars: int = 5, min_words: int = 2) -> bool:
    """Check if the text has sufficient substance for classification and retrieval."""
    if not text:
        return False
    words = text.split()
    return len(text) >= min_chars and len(words) >= min_words
