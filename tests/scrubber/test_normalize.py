"""test_normalize.py.

Coverage: 100%
Last Update: 2025-01-14.
"""

import pytest

from lexos.scrubber.normalize import (
    bullet_points,
    hyphenated_words,
    lower_case,
    quotation_marks,
    repeating_chars,
    unicode,
    whitespace,
)


def test_bullet_points():
    """Test normalizing bullet points."""
    text = "• Item 1\n• Item 2\n- Item 3"
    expected = "- Item 1\n- Item 2\n- Item 3"
    assert bullet_points(text) == expected


def test_hyphenated_words():
    """Test normalizing hyphenated words."""
    text = "This is a hyphen-\nated word."
    expected = "This is a hyphenated word."
    assert hyphenated_words(text) == expected


def test_lower_case():
    """Test converting text to lower case."""
    text = "This Is A TeSt to LOWeR."
    expected = "this is a test to lower."
    assert lower_case(text) == expected


def test_quotation_marks():
    """Test normalizing quotation marks."""
    text = "“This is a ‘test’.”"
    expected = "\"This is a 'test'.\""
    assert quotation_marks(text) == expected


def test_repeating_chars():
    """Test normalizing repeating characters."""
    text = "TTTThis is sooo cool!!!"
    assert repeating_chars(text, chars="o", maxn=1) == "TTTThis is so col!!!"
    assert repeating_chars(text, chars="!", maxn=1) == "TTTThis is sooo cool!"
    assert repeating_chars(text, chars="T", maxn=2) == "TThis is sooo cool!!!"


def test_unicode():
    """Test normalizing unicode characters."""
    text = "e\u0301"
    expected = "é"
    assert unicode(text, form="NFC") == expected


def test_whitespace():
    """Test normalizing whitespace."""
    text = "This is a test.\u00a0\nThis is another test.\u200b"
    expected = "This is a test. \nThis is another test."
    assert whitespace(text) == expected
