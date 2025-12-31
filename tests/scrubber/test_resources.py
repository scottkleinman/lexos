"""test_resources.py.

Coverage: 96%. Missing: 193, 207
Last Update: 2025-01-14.
"""

import pytest

from lexos.scrubber.resources import HTMLTextExtractor


def test_html_text_extractor_basic():
    """Test basic HTML parsing and text extraction."""
    html_content = "<html><body><p>Test</p><p>Content</p></body></html>"
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    result = extractor.get_text()
    assert result == "TestContent"


def test_html_text_extractor_with_separator():
    """Test text extraction with separator."""
    html_content = "<html><body><p>Test</p><p>Content</p></body></html>"
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    result = extractor.get_text(sep=" ")
    assert result == "Test Content"


def test_html_text_extractor_empty():
    """Test handling of empty data."""
    extractor = HTMLTextExtractor()
    result = extractor.get_text()
    assert result == ""


def test_html_text_extractor_handle_data():
    """Test handling of data elements."""
    extractor = HTMLTextExtractor()
    extractor.handle_data("Sample")
    extractor.handle_data("Text")
    result = extractor.get_text()
    assert result == "SampleText"


def test_re_emoji_narrow(monkeypatch):
    """Test RE_EMOJI regex compilation for narrow Python builds (line 152)."""
    import importlib
    import sys

    monkeypatch.setattr(sys, "maxunicode", 0xFFFF)
    import lexos.scrubber.resources as resources

    importlib.reload(resources)
    assert resources.RE_EMOJI.pattern == r"[\u2600-\u26FF\u2700-\u27BF]"
