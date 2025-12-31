"""test_remove.py.

Coverage: 100%
Last Update: 2025-01-14.
"""

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
    text = "<html><body><p>This is a test.</p></body></html>"
    assert tags(text) == "This is a test."
    assert tags(text, sep=" ") == "This is a test."


def test_new_lines():
    """Test removing new lines."""
    text = "This is a test.\nThis is another test."
    assert new_lines(text) == "This is a test.This is another test."


def test_pattern():
    """Test removing patterns using regex."""
    text = "This is a test."
    assert pattern(text, pattern="test") == "This is a ."
    assert pattern(text, pattern=["test", "is"]) == "Th  a ."


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
