"""test_remove.py.

Coverage: 100%
Last Update: 2026-06-26.
"""

import builtins
import importlib
import sys

import pytest

from lexos.scrubber import resources
from lexos.scrubber.remove import (
    accents,
    brackets,
    digits,
    new_lines,
    pattern,
    project_gutenberg_headers,
    punctuation,
    tabs,
    tags,
)


def test_accents():
    """Test removing accents."""
    text = "Café"
    assert accents(text) == "Cafe"
    assert accents(text, fast=True) == "Cafe"
    assert accents(text, accents="COMBINING ACUTE ACCENT") == "Cafe"
    assert accents(text, accents=("COMBINING ACUTE ACCENT",)) == "Cafe"
    assert (
        accents(text, accents=("COMBINING ACUTE ACCENT", "COMBINING DIAERESIS"))
        == "Cafe"
    )


def test_brackets():
    """Test removing text within brackets."""
    text = "This is a {curly} [square] (round) test."
    assert brackets(text) == "This is a    test."
    assert brackets(text, only="curly") == "This is a  [square] (round) test."
    assert brackets(text, only=["square", "round"]) == "This is a {curly}   test."


def test_digits():
    """Test removing digits."""
    text = "This is a test with digits 123."
    assert digits(text) == "This is a test with digits ."
    assert digits(text, only="1") == "This is a test with digits 23."
    assert digits(text, only=["1", "3"]) == "This is a test with digits 2."


def test_project_gutenberg_headers():
    """Test removing Project Gutenberg headers and footers, including legalese sections."""
    content = "This is the content.\n" * 100
    # Add a legalese section to test ignore_section logic
    legalese_start = next(iter(resources.LEGALESE_START_MARKERS))
    legalese_end = next(iter(resources.LEGALESE_END_MARKERS))
    legalese_content = "This should be ignored.\n" * 5
    text = (
        "Project Gutenberg's EBook\n\n"
        "*** START OF THIS PROJECT GUTENBERG EBOOK ***\n\n"
        f"{content}"
        f"{legalese_start}\n"
        f"{legalese_content}"
        f"{legalese_end}\n"
        f"{content}"
        "\n\n*** END OF THIS PROJECT GUTENBERG EBOOK ***"
    )
    result = project_gutenberg_headers(text)
    # The legalese_content should not appear in the result
    assert "This should be ignored." not in result
    assert result.strip().startswith("This is the content.")
    assert result.strip().endswith("This is the content.")


def test_tags():
    """Test removing tags."""
    # Basic HTML
    text = "<html><body><p>This is a test.</p></body></html>"
    # The current implementation uses HTMLTextExtractor which might put spaces around joined text
    assert "This is a test." in tags(text)

    # Custom separator
    text_sep = "<p>Word1</p><p>Word2</p>"
    assert tags(text_sep, sep="|") == "Word1|Word2"

    # Multiple whitespace handling
    text_ws = "<p>Word1  </p>  <p>  Word2</p>"
    assert tags(text_ws, remove_whitespace=True, sep=" ") == "Word1 Word2"

    # Empty
    assert tags("") == ""

    # Test with comments (the implementation of HTMLTextExtractor seems to strip them)
    text_comm = "<!-- comment --><p>Text</p>"
    assert "comment" not in tags(text_comm)
    assert "Text" in tags(text_comm)


def test_tags_security():
    """Test security protections in tags function."""
    from lexos.exceptions import LexosException

    # Create a "tag bomb": a long unclosed tag with many nested start tokens INSIDE it
    # The regex matches <[a-zA-Z][^>]{1000,}
    # So we need < then a letter, then 1000 non-> chars that include many <
    tag_bomb = "<a" + ("<" * 60) + (" " * 1000)
    with pytest.raises(
        LexosException, match="Potential security risk: excessive nested"
    ):
        tags(tag_bomb)


def test_tags_fallbacks(monkeypatch):
    """Test tags function fallback mechanisms."""
    text = "<html><body>Test</body></html>"

    def raise_exc(*args, **kwargs):
        raise Exception("fail")

    # 1. Force lxml failure to trigger etree.fromstring catch
    import lexos.scrubber.remove as remove_mod

    with monkeypatch.context() as m:
        m.setattr(remove_mod.html, "fromstring", raise_exc)
        # This should trigger the second try block (etree.fromstring)
        assert "Test" in tags(text)

    # 2. Force complete lxml failure to trigger BeautifulSoup fallback
    with monkeypatch.context() as m:
        m.setattr(remove_mod.html, "fromstring", raise_exc)
        m.setattr(remove_mod.etree, "fromstring", raise_exc)

        # Now it should hit BeautifulSoup
        assert "Test" in tags(text)

    # 3. Simulate BS4 failure
    with monkeypatch.context() as m:
        # Mock _HAS_LXML to False to ensure we don't use it
        m.setattr(remove_mod, "_HAS_LXML", False)

        # We need to mock the BeautifulSoup import or the class
        import bs4

        m.setattr(bs4, "BeautifulSoup", raise_exc)

        # Should return text as-is
        res = tags(text, remove_whitespace=False)
        assert "<html><body>Test</body></html>" in res


def test_new_lines():
    """Test removing new lines."""
    text = "This is a test.\nThis is another test."
    assert new_lines(text) == "This is a test.This is another test."


def test_pattern():
    """Test removing patterns using regex."""
    text = "This is a test."
    assert pattern(text, pattern="test") == "This is a ."
    assert pattern(text, pattern=["test", "is"]) == "Th  a ."


def test_pattern_none():
    """Test that pattern(None) returns the original text."""
    text = "This is a test."
    assert pattern(text, pattern=None) == text


def test_tags_without_lxml_import(monkeypatch):
    """Test tags fallback when lxml cannot be imported."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lxml" or name.startswith("lxml."):
            raise ImportError("No lxml available")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    original_module = sys.modules.get("lexos.scrubber.remove")
    monkeypatch.delitem(sys.modules, "lexos.scrubber.remove", raising=False)

    remove_no_lxml = importlib.import_module("lexos.scrubber.remove")
    assert not remove_no_lxml._HAS_LXML
    assert "Test" in remove_no_lxml.tags("<html><body>Test</body></html>")


def test_punctuation():
    """Test removing punctuation."""
    text = "This: is a test, with punctuation!?."
    assert punctuation(text) == "This is a test with punctuation"
    assert punctuation(text, only=",") == "This: is a test with punctuation!?."
    assert (
        punctuation(text, only=[",", ":", "?", "."])
        == "This is a test with punctuation!"
    )
    assert punctuation(text, exclude="!") == "This is a test with punctuation!"
    assert punctuation(text, exclude=["!", ","]) == "This is a test, with punctuation!"


def test_tabs():
    """Test removing tabs."""
    text = "This is a\ttest."
    assert tabs(text) == "This is atest."
