"""test_replace.py.

Coverage: 100%
Last Update: 2025-12-10.
"""

import pytest

from lexos.scrubber.replace import (
    currency_symbols,
    digits,
    emails,
    emojis,
    hashtags,
    pattern,
    phone_numbers,
    punctuation,
    special_characters,
    urls,
    user_handles,
)


def test_currency_symbols():
    """Test replacing currency symbols."""
    text = "The price is $100."
    expected = "The price is _CUR_100."
    assert currency_symbols(text) == expected


def test_digits():
    """Test replacing digits."""
    text = "My phone number is 123-456-7890."
    expected = "My phone number is _DIGIT_-_DIGIT_-_DIGIT_."
    assert digits(text) == expected


def test_emails():
    """Test replacing email addresses."""
    text = "Contact me at example@example.com."
    expected = "Contact me at _EMAIL_."
    assert emails(text) == expected


def test_emojis():
    """Test replacing emojis."""
    text = "I am happy 😊."
    expected = "I am happy _EMOJI_."
    assert emojis(text) == expected


def test_hashtags():
    """Test replacing hashtags."""
    text = "This is a #test."
    expected = "This is a _HASHTAG_."
    assert hashtags(text) == expected


def test_pattern():
    """Test replacing patterns."""
    text = "This is a test."
    pattern_dict = {"test": "_PATTERN_"}
    expected = "This is a _PATTERN_."
    assert pattern(text, pattern=pattern_dict) == expected


def test_phone_numbers():
    """Test replacing phone numbers."""
    text = "Call me at 123-456-7890 or 1.123.456.7890."
    expected = "Call me at _PHONE_ or _PHONE_."
    assert phone_numbers(text) == expected


def test_punctuation():
    """Test replacing punctuation."""
    text = "Hello, world!"
    assert punctuation(text) == "Hello  world "
    assert punctuation(text, only="!") == "Hello, world "
    assert punctuation(text, exclude=",") == "Hello, world "


def test_special_characters():
    """Test replacing special characters."""
    text = "This is a test & example."
    ruleset = {"&": "and"}
    expected = "This is a test and example."
    assert special_characters(text, ruleset=ruleset) == expected


def test_special_characters_html_unescape():
    """Test replacing special characters with is_html=True (HTML unescape branch)."""
    text = "This &amp; that &lt;test&gt;"
    expected = "This & that <test>"
    assert special_characters(text, is_html=True) == expected


def test_urls():
    """Test replacing URLs."""
    text = "Visit https://example.com for more info."
    expected = "Visit _URL_ for more info."
    assert urls(text) == expected


def test_user_handles():
    """Test replacing user handles."""
    text = "Follow me on Twitter @example."
    expected = "Follow me on Twitter _USER_."
    assert user_handles(text) == expected
