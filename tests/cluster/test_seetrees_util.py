"""Tests for the seetrees utility helpers.

Coverage: 100%

Last Updated: August 23, 2026
"""

import pytest

from lexos.cluster.seetrees.util import sanitize_label_text


def test_sanitize_label_text_returns_empty_for_none():
    """sanitize_label_text should return an empty string for None values."""
    assert sanitize_label_text(None) == ""


def test_sanitize_label_text_replaces_spaces_with_whitespace_token():
    """Spaces should be replaced with the <whitespace> token."""
    assert sanitize_label_text("hello world") == "hello<whitespace>world"


def test_sanitize_label_text_replaces_tabs_with_whitespace_token():
    """Tabs should be replaced with the <whitespace> token."""
    assert sanitize_label_text("hello\tworld") == "hello<whitespace>world"


def test_sanitize_label_text_replaces_linebreaks_with_linebreak_token():
    """Newlines and carriage returns should be replaced with <linebreak>."""
    assert sanitize_label_text("hello\nworld") == "hello<linebreak>world"
    assert sanitize_label_text("hello\rworld") == "hello<linebreak>world"
    assert sanitize_label_text("hello\r\nworld") == "hello<linebreak>world"


def test_sanitize_label_text_preserves_text_without_spaces_or_linebreaks():
    """Labels without whitespace should be returned unchanged."""
    assert sanitize_label_text("helloworld") == "helloworld"
