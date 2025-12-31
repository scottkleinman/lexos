"""test_utils.py.

Coverage: 100%
Last Update: 2025-01-14.
"""

import pytest

from lexos.scrubber.utils import get_tags


def test_get_tags_well_formed_xml():
    """Test get_tags with well-formed XML."""
    text = '<root><child attr="value">Content</child></root>'
    result = get_tags(text)
    assert result["tags"] == ["child", "root"]
    assert result["attributes"] == [{"child": {"attr": "value"}}]


def test_get_tags_malformed_xml():
    """Test get_tags with malformed XML falling back to BeautifulSoup."""
    text = '<root><child attr="value">Content</child>'
    result = get_tags(text)
    assert result["tags"] == ["child", "root"]
    assert result["attributes"] == [{"root": {}}, {"child": {"attr": "value"}}]


def test_get_tags_no_tags():
    """Test get_tags with text containing no tags."""
    text = "Just plain text with no tags."
    result = get_tags(text)
    assert result["tags"] == []
    assert result["attributes"] == []


def test_get_tags_only_tags_no_attributes():
    """Test get_tags with text containing only tags and no attributes."""
    text = "<root><child>Content</child></root>"
    result = get_tags(text)
    assert result["tags"] == ["child", "root"]
    assert result["attributes"] == []


def test_get_tags_with_processing_instructions():
    """Test get_tags with text containing processing instructions."""
    text = '<?xml version="1.0"?><root><child attr="value">Content</child></root>'
    result = get_tags(text)
    assert result["tags"] == ["child", "root"]
    assert result["attributes"] == [{"child": {"attr": "value"}}]


def test_get_tags_processing_instruction_extraction():
    """Ensure get_tags triggers bs4.element.ProcessingInstruction extraction (line 48)."""
    # Place a processing instruction outside the root element
    text = '<?myproc instruction?><root><child attr="value">Content</child></root'
    # This is malformed XML (missing > at the end), so ElementTree will fail and BS4 will be used
    result = get_tags(text)
    # The test is to ensure no error and tags are extracted
    assert "root" in result["tags"]
    assert "child" in result["tags"]
