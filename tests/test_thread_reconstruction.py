"""Unit tests for thread reconstruction logic."""

import pytest
import pandas as pd
from src.data.pipeline import reconstruct_threads_from_df


def test_reconstruct_single_turn_thread():
    records = [
        {
            "tweet_id": 1,
            "author_id": "cust1",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "@AmazonHelp my book arrived damaged",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": 2,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "@cust1 We are sorry to hear that! Please DM us your order number.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 1,
        },
    ]
    df = pd.DataFrame(records)
    threads, pairs = reconstruct_threads_from_df(df, target_brand="AmazonHelp")
    
    assert len(threads) == 1
    thread = threads[0]
    assert len(thread["turns"]) == 2
    assert thread["turns"][0]["role"] == "customer"
    assert "damaged" in thread["turns"][0]["text"]
    assert thread["turns"][1]["role"] == "agent"
    assert "DM" in thread["turns"][1]["text"]
    
    assert len(pairs) == 1
    assert "damaged" in pairs[0]["query"]
    assert "DM" in pairs[0]["reply"]


def test_reconstruct_multi_turn_thread():
    records = [
        {
            "tweet_id": 10,
            "author_id": "cust2",
            "inbound": True,
            "created_at": "Tue Oct 31 10:00:00 +0000 2017",
            "text": "@AmazonHelp where is my package?",
            "response_tweet_id": "11",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": 11,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Tue Oct 31 10:05:00 +0000 2017",
            "text": "@cust2 What is the tracking ID?",
            "response_tweet_id": "12",
            "in_response_to_tweet_id": 10,
        },
        {
            "tweet_id": 12,
            "author_id": "cust2",
            "inbound": True,
            "created_at": "Tue Oct 31 10:08:00 +0000 2017",
            "text": "@AmazonHelp tracking ID is 998877",
            "response_tweet_id": "13",
            "in_response_to_tweet_id": 11,
        },
        {
            "tweet_id": 13,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Tue Oct 31 10:15:00 +0000 2017",
            "text": "@cust2 Thanks! It is out for delivery today.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 12,
        },
    ]
    df = pd.DataFrame(records)
    threads, pairs = reconstruct_threads_from_df(df, target_brand="AmazonHelp")
    
    assert len(threads) == 1
    assert len(threads[0]["turns"]) == 4
    # Query-reply pairs extracted
    assert len(pairs) == 2
    assert "where is my package" in pairs[0]["query"]
    assert "tracking ID is 998877" in pairs[1]["query"]
