"""Unit tests for text cleaning utilities."""

import pytest
from src.utils.text import clean_tweet_text, is_meaningful_text


def test_clean_tweet_text_removes_mentions():
    raw = "@AmazonHelp @115712 I need help with my package"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@115712" not in cleaned
    assert cleaned == "I need help with my package"


def test_clean_tweet_text_removes_urls():
    raw = "Check tracking at https://t.co/xyz123 or http://amazon.com/track for updates"
    cleaned = clean_tweet_text(raw)
    assert "https://" not in cleaned
    assert "http://" not in cleaned
    assert cleaned == "Check tracking at or for updates"


def test_clean_tweet_text_unescapes_html():
    raw = "Shoes &amp; Shirts &lt;3 are back &gt; in stock"
    cleaned = clean_tweet_text(raw)
    assert "&amp;" not in cleaned
    assert "Shoes & Shirts <3 are back > in stock" in cleaned


def test_clean_tweet_text_collapses_whitespace():
    raw = "   Where \n\n is \t my   order?   "
    cleaned = clean_tweet_text(raw)
    assert cleaned == "Where is my order?"


def test_clean_tweet_text_handles_empty_and_non_string():
    assert clean_tweet_text("") == ""
    assert clean_tweet_text(None) == ""
    assert clean_tweet_text(123) == ""


def test_is_meaningful_text():
    assert is_meaningful_text("Where is my order?") is True
    assert is_meaningful_text("hi") is False
    assert is_meaningful_text("") is False
    assert is_meaningful_text("a b") is False
