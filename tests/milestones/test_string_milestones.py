"""test_string_milestones.py.

Coverage: 96%. Missing: 114, 127
Last Update: June 27, 2026
"""

import pytest
import spacy
from pydantic import ValidationError
from spacy.tokens import Doc

from lexos.milestones.string_milestones import (
    StringMilestones,
    case_insensitive_flags,
    case_sensitive_flags,
)

nlp = spacy.load("en_core_web_sm")

# Fixtures


@pytest.fixture
def doc():
    """Doc fixture for StringMilestones tests."""
    text = "This is a test document."
    return nlp(text)


# Tests


def test_string_milestones_init_with_string():
    """Test StringMilestones initialization with a string."""
    milestones = StringMilestones(doc="This is a test document.")
    assert isinstance(milestones.doc, str)
    assert milestones.patterns == [None]
    assert milestones.case_sensitive is True
    assert milestones.flags == case_sensitive_flags


def test_string_milestones_init_with_doc(doc):
    """Test StringMilestones initialization with a spaCy Doc."""
    milestones = StringMilestones(doc=doc)
    assert isinstance(milestones.doc, Doc)
    assert milestones.patterns == [None]
    assert milestones.case_sensitive is True
    assert milestones.flags == case_sensitive_flags


def test_string_milestones_init_with_patterns():
    """Test StringMilestones initialization with patterns."""
    patterns = ["test", "document"]
    milestones = StringMilestones(doc="This is a test document.", patterns=patterns)
    assert milestones.patterns == patterns


def test_string_milestones_init_case_insensitive():
    """Test StringMilestones initialization with case insensitive."""
    milestones = StringMilestones(doc="This is a test document.", case_sensitive=False)
    assert milestones.case_sensitive is False
    assert milestones.flags == case_insensitive_flags


def test_string_milestones_invalid_doc():
    """Test StringMilestones initialization with invalid doc type."""
    with pytest.raises(ValidationError):
        StringMilestones(doc=12345)  # Invalid doc type


def test_string_milestones_set_method_called():
    """Test StringMilestones set method is called when patterns are provided."""
    patterns = ["test"]
    milestones = StringMilestones(doc="This is a test document.", patterns=patterns)
    assert milestones.patterns == patterns
    assert len(milestones.spans) == 1
    assert milestones.spans[0].text == "test"


def test_string_milestones_repeated_set_compiles_patterns():
    """Test repeated set() calls recompile patterns correctly."""
    milestones = StringMilestones(doc="Hello test world", patterns=["test"])
    assert len(milestones._compiled_patterns) == 1
    assert milestones._compiled_patterns[0].pattern == "test"

    milestones.set(patterns=["Hello"], case_sensitive=True)
    assert len(milestones._compiled_patterns) == 1
    assert milestones._compiled_patterns[0].pattern == "Hello"
    assert len(milestones.spans) == 1
    assert milestones.spans[0].text == "Hello"


if __name__ == "__main__":
    pytest.main()
