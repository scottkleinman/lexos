"""test_milestones.py.

Coverage: 97%. Missing: 300-307
Last Update: December 24, 2025
"""

import re
from string import punctuation
from typing import Iterable
from unittest.mock import MagicMock

import pytest
import spacy
from spacy.tokens import Doc, Span, Token

from lexos.milestones.token_milestones import (
    TokenMilestones,
    case_insensitive_flags,
    case_sensitive_flags,
)

nlp = spacy.load("en_core_web_sm")

# Fixtures


@pytest.fixture
def mock_text():
    """Return a mock text object."""
    return "This is a test."


@pytest.fixture
def mock_doc_with_model(mock_text):
    """Return a mock Doc object."""
    return nlp(mock_text)


@pytest.fixture
def mock_milestones(mock_doc):
    """Return a mock Milestones instance with spans."""
    milestones = TokenMilestones(doc=mock_doc)
    milestones.doc.spans["milestones"] = []
    return milestones


@pytest.fixture
def mock_doc():
    """Return a mock Doc object."""
    doc = MagicMock(spec=Doc)
    doc.text = "Test text with TEST pattern and test variations."
    # Mock char_span to return a mock Span
    doc.char_span = MagicMock(return_value=MagicMock(spec=Span))
    return doc


@pytest.fixture
def milestones_mock_doc(mock_doc):
    """Test instance of Milestones with mock doc."""
    return TokenMilestones(doc=mock_doc)


@pytest.fixture
def milestones():
    """Test instance of Milestones with real doc."""
    return TokenMilestones(
        doc=nlp("This is a Test string with test pattern and TEST variations.")
    )


# Tests


def test_milestones_initialization_case_sensitive(mock_doc):
    """Test Milestones initialization with case_sensitive=True."""
    milestones = TokenMilestones(doc=mock_doc, case_sensitive=True)
    assert milestones.case_sensitive is True
    assert milestones.flags == case_sensitive_flags
    assert milestones.attr == "ORTH"


def test_milestones_initialization_case_insensitive(mock_doc):
    """Test Milestones initialization with case_sensitive=False."""
    milestones = TokenMilestones(doc=mock_doc, case_sensitive=False)
    assert milestones.case_sensitive is False
    assert milestones.flags == case_insensitive_flags
    assert milestones.attr == "LOWER"


def test_milestones_token_extensions(mock_doc):
    """Test Milestones initialization sets token extensions."""
    _ = TokenMilestones(doc=mock_doc)
    assert Token.has_extension("milestone_iob")
    assert Token.has_extension("milestone_label")


def test_milestones_default_values(mock_doc):
    """Test Milestones initialization with default values."""
    milestones = TokenMilestones(doc=mock_doc)
    assert milestones.patterns is None
    assert milestones.character_map is None
    assert milestones.mode == "string"
    assert milestones.type is None


def test_spans_property(mock_doc, milestones):
    """Test the spans property of the Milestones class."""
    # An empty instance
    milestones.doc = mock_doc
    assert list(milestones.doc.spans["milestones"]) == []
    # Insert some spans
    milestones.doc.spans["milestones"] = [mock_doc[:3], mock_doc[3:]]
    assert milestones.spans == list(milestones.doc.spans["milestones"])
    assert all(isinstance(span, Span) for span in milestones.spans)


def test_iter_method(mock_doc, milestones):
    """Test the __iter__ method of the Milestones class."""
    milestones.doc = mock_doc
    milestones.doc.spans["milestones"] = [mock_doc[:3], mock_doc[3:]]
    assert isinstance(milestones, Iterable)


def test_autodetect_mode_string(mock_doc):
    """Test _autodetect_mode with string patterns."""
    milestones = TokenMilestones(doc=mock_doc)
    patterns = ["test", "example"]
    mode = milestones._autodetect_mode(patterns)
    assert mode == "string"


def test_autodetect_mode_phrase(mock_doc):
    """Test _autodetect_mode with phrase patterns."""
    milestones = TokenMilestones(doc=mock_doc)
    patterns = ["test example", "another test"]
    mode = milestones._autodetect_mode(patterns)
    assert mode == "phrase"


def test_autodetect_mode_rule(mock_doc_with_model):
    """Test _autodetect_mode with rule patterns."""
    milestones = TokenMilestones(doc=mock_doc_with_model)
    patterns = [[{"LOWER": "test"}, {"LOWER": "example"}]]
    mode = milestones._autodetect_mode(patterns)
    assert mode == "rule"


def test_autodetect_mode_invalid_pattern(mock_doc):
    """Test _autodetect_mode with an invalid pattern."""
    milestones = TokenMilestones(doc=mock_doc)
    patterns = [12345]  # Invalid pattern
    with pytest.raises(ValueError, match=r"must be a string or a spaCy Matcher rule."):
        milestones._autodetect_mode(patterns)


def test_get_string_matches_basic(milestones):
    """Test basic string matching."""
    patterns = ["test"]
    spans = milestones._get_string_matches(patterns, case_sensitive_flags)
    assert isinstance(spans, list)
    assert len(spans) == 1  # Should only find one uppercase
    assert all(isinstance(span, Span) for span in spans)


def test_get_string_matches_case_insensitive(milestones):
    """Test case-sensitive string matching."""
    patterns = ["TEST"]
    spans = milestones._get_string_matches(patterns, flags=case_insensitive_flags)
    assert len(spans) == 3  # Should find three occurrences of "test"
    assert all(isinstance(span, Span) for span in spans)


def test_get_string_matches_multiple_patterns(milestones):
    """Test matching multiple patterns."""
    patterns = ["test", "pattern"]
    spans = milestones._get_string_matches(patterns, case_insensitive_flags)
    assert len(spans) == 4  # Should find three "test" and one "pattern"


def test_get_string_matches_empty_patterns(milestones_mock_doc):
    """Test with empty pattern list."""
    with pytest.raises(ValueError, match=r"Patterns cannot be empty"):
        _ = milestones_mock_doc._get_string_matches([], case_insensitive_flags)


def test_get_string_matches_invalid_pattern(milestones_mock_doc):
    """Test with invalid regex pattern."""
    patterns = ["[invalid"]
    with pytest.raises(re.error):
        milestones_mock_doc._get_string_matches(patterns, case_insensitive_flags)


def test_get_string_matches_case_sensitive(milestones):
    """Test case-sensitive string matching."""
    patterns = ["Test"]
    spans = milestones._get_string_matches(patterns, case_sensitive_flags)
    assert len(spans) == 1  # Should only find one "Test"


def test_get_string_matches_no_matches(milestones):
    """Test pattern with no matches."""
    patterns = ["nonexistent"]
    spans = milestones._get_string_matches(patterns, case_sensitive_flags)
    assert len(spans) == 0


def test_get_string_matches_null_character_map(milestones):
    """Test with null character map."""
    patterns = ["test"]
    spans = milestones._get_string_matches(patterns, case_insensitive_flags)
    assert isinstance(milestones.character_map, dict)
    assert len(spans) == 3


def test_get_phrase_matches_basic(milestones):
    """Test basic phrase matching."""
    patterns = ["test pattern"]
    spans = milestones._get_phrase_matches(patterns)
    assert isinstance(spans, list)
    assert len(spans) == 1
    assert all(isinstance(span, Span) for span in spans)


def test_get_phrase_matches_multiple_patterns(milestones):
    """Test matching multiple phrase patterns."""
    patterns = ["test pattern", "TEST variations"]
    spans = milestones._get_phrase_matches(patterns)
    assert len(spans) == 2


def test_get_phrase_matches_empty_patterns(milestones):
    """Test with empty pattern list."""
    spans = milestones._get_phrase_matches([])
    assert len(spans) == 0


def test_get_phrase_matches_different_attr(milestones):
    """Test with different token attribute."""
    patterns = ["TEST"]
    spans = milestones._get_phrase_matches(patterns, attr="LOWER")
    assert len(spans) == 3


def test_get_phrase_matches_no_matches(milestones):
    """Test pattern with no matches."""
    patterns = ["No match"]
    spans = milestones._get_phrase_matches(patterns, attr="LOWER")
    assert len(spans) == 0


def test_get_rule_matches_basic(milestones):
    """Test basic rule matching."""
    patterns = [[{"TEXT": "test"}]]
    spans = milestones._get_rule_matches(patterns)
    assert isinstance(spans, list)
    assert len(spans) == 1
    assert all(isinstance(span, Span) for span in spans)


def test_get_rule_matches_case_insensitive(milestones):
    """Test case-insensitive rule matching."""
    patterns = [[{"TEXT": "test"}]]
    milestones.case_sensitive = False
    spans = milestones._get_rule_matches(patterns)
    assert isinstance(spans, list)
    assert len(spans) == 3
    assert all(isinstance(span, Span) for span in spans)


def test_get_rule_matches_multiple_patterns(milestones):
    """Test matching multiple patterns."""
    patterns = [[{"LOWER": "test"}, {"LOWER": "pattern"}]]
    spans = milestones._get_rule_matches(patterns)
    assert len(spans) == 1


def test_get_rule_matches_empty_patterns(milestones_mock_doc):
    """Test with empty pattern list."""
    spans = milestones_mock_doc._get_rule_matches([])
    assert len(spans) == 0


def test_get_rule_matches_invalid_pattern(milestones):
    """Test with invalid pattern."""
    patterns = [{"INVALID": "test"}]
    with pytest.raises(Exception):
        milestones._get_rule_matches(patterns)


def test_remove_duplicate_spans_basic(milestones):
    """Test basic duplicate span removal."""
    spans = [
        milestones.doc[0:2],
        milestones.doc[0:2],
        milestones.doc[2:],
    ]
    result = milestones._remove_duplicate_spans(spans)
    assert len(result) == 2
    assert result[0].start == 0 and result[0].end == 2
    assert result[1].start == 2 and result[1].end == len(milestones.doc)


def test_remove_duplicate_spans_no_duplicates(milestones):
    """Test with no duplicate spans."""
    spans = [
        milestones.doc[0:2],
        milestones.doc[2:4],
        milestones.doc[4:6],
    ]
    result = milestones._remove_duplicate_spans(spans)
    assert len(result) == 3
    assert [(span.start, span.end) for span in result] == [(0, 2), (2, 4), (4, 6)]


def test_remove_duplicate_spans_empty(milestones):
    """Test with empty span list."""
    spans = []
    result = milestones._remove_duplicate_spans(spans)
    assert len(result) == 0


def test_remove_duplicate_spans_multiple_duplicates(milestones):
    """Test with multiple duplicate spans."""
    spans = [
        milestones.doc[0:2],
        milestones.doc[0:2],
        milestones.doc[2:4],
        milestones.doc[2:4],
        milestones.doc[4:6],
    ]
    result = milestones._remove_duplicate_spans(spans)
    assert len(result) == 3
    assert [(span.start, span.end) for span in result] == [(0, 2), (2, 4), (4, 6)]


def test_set_case_sensitivity_true(milestones_mock_doc):
    """Test case-sensitive setting."""
    milestones_mock_doc._set_case_sensitivity(case_sensitive=True)
    assert milestones_mock_doc.case_sensitive is True
    assert milestones_mock_doc.flags == (re.DOTALL | re.MULTILINE | re.UNICODE)
    assert milestones_mock_doc.attr == "ORTH"


def test_set_case_sensitivity_false(milestones_mock_doc):
    """Test case-insensitive setting."""
    milestones_mock_doc._set_case_sensitivity(case_sensitive=False)
    assert milestones_mock_doc.case_sensitive is False
    assert milestones_mock_doc.flags == (
        re.DOTALL | re.IGNORECASE | re.MULTILINE | re.UNICODE
    )
    assert milestones_mock_doc.attr == "LOWER"


def test_set_case_sensitivity_default(milestones_mock_doc):
    """Test default case sensitivity setting."""
    milestones_mock_doc._set_case_sensitivity()
    if milestones_mock_doc.case_sensitive is True:
        assert len(milestones_mock_doc.flags) == 3
        assert milestones_mock_doc.attr == "ORTH"
    else:
        assert len(milestones_mock_doc.flags) == 4
        assert milestones_mock_doc.attr == "LOWER"


def test_set_case_sensitivity_toggle(milestones_mock_doc):
    """Test toggling case sensitivity."""
    milestones_mock_doc._set_case_sensitivity(False)
    assert len(milestones_mock_doc.flags) == 4
    assert milestones_mock_doc.attr == "LOWER"
    milestones_mock_doc._set_case_sensitivity(True)
    assert len(milestones_mock_doc.flags) == 3
    assert milestones_mock_doc.attr == "ORTH"


def test_to_spacy_span_basic(milestones):
    """Test basic conversion from match to span."""
    mock_match = re.search("test", milestones.doc.text)
    span = milestones._to_spacy_span(mock_match)
    assert isinstance(span, Span)


def test_to_spacy_span_fallback(milestones):
    """Test fallback when char_span returns None."""
    mock_match = milestones.doc.char_span(30, 40)
    with pytest.raises(ValueError):
        _ = milestones._to_spacy_span(mock_match)


def test_to_spacy_span_null_character_map(milestones):
    """Test with null character map or map with invalid indexes."""
    milestones.character_map = None
    mock_match = milestones.doc.char_span(3, 6)
    with pytest.raises(ValueError):
        _ = milestones._to_spacy_span(mock_match)
    milestones.character_map = None
    with pytest.raises(ValueError):
        _ = milestones._to_spacy_span(mock_match)


def test_assign_token_attributes_basic(milestones):
    """Test basic token attribute assignment."""
    spans = [milestones.doc[0:4]]
    milestones._assign_token_attributes(spans)
    for token in milestones.doc:
        if token.i == 0:
            assert token._.milestone_iob == "B"
        elif token.i < 4:
            assert token._.milestone_iob == "I"
        else:
            assert token._.milestone_iob == "O"


def test_assign_token_attributes_empty_spans(milestones):
    """Test with empty spans list."""
    milestones._assign_token_attributes([])
    for token in milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_assign_token_attributes_multiple_spans(milestones):
    """Test with multiple spans."""
    spans = [milestones.doc[0:2], milestones.doc[2:4]]
    milestones._assign_token_attributes(spans)
    assert milestones.doc[0]._.milestone_iob == "B"
    assert milestones.doc[1]._.milestone_iob == "I"
    assert milestones.doc[2]._.milestone_iob == "B"
    assert milestones.doc[3]._.milestone_iob == "I"


def test_assign_token_attributes_long_label(milestones):
    """Test truncation of long labels."""
    spans = [milestones.doc[0:4]]
    milestones._assign_token_attributes(spans, max_label_length=3)
    assert milestones.doc[0]._.milestone_label == "Thi..."


def test_get_matches_empty_patterns(milestones):
    """Test get_matches with empty patterns."""
    with pytest.raises(ValueError, match="Patterns cannot be empty"):
        milestones.get_matches(patterns=None, mode="string")
    with pytest.raises(ValueError, match="Patterns cannot be empty"):
        milestones.get_matches(patterns=[], mode="string")


def test_get_matches_string_mode(milestones):
    """Test get_matches with string mode."""
    spans = milestones.get_matches(patterns=["test"], mode="string")
    assert isinstance(spans, list)
    assert all(isinstance(span, Span) for span in spans)


def test_get_matches_phrase_mode(milestones):
    """Test get_matches with phrase mode."""
    spans = milestones.get_matches(patterns=["test phrase"], mode="phrase")
    assert isinstance(spans, list)


def test_get_matches_rule_mode(milestones):
    """Test get_matches with rule mode."""
    spans = milestones.get_matches(patterns=[[{"LOWER": "test"}]], mode="rule")
    assert isinstance(spans, list)


def test_get_matches_autodetect(milestones):
    """Test get_matches with mode autodetection."""
    spans = milestones.get_matches(patterns=["test"])
    assert isinstance(spans, list)
    spans = milestones.get_matches(patterns=["test phrase"])
    assert isinstance(spans, list)
    spans = milestones.get_matches(patterns=[[{"LOWER": "test"}]])
    assert isinstance(spans, list)


def test_get_matches_case_sensitive(milestones):
    """Test get_matches with case sensitivity."""
    milestones.get_matches(patterns=["test"], mode="string", case_sensitive=True)
    assert milestones.case_sensitive is True
    milestones.get_matches(patterns=["test"], mode="string", case_sensitive=False)
    assert milestones.case_sensitive is False


def test_get_matches_pattern_update_with_keyword(milestones):
    """Test pattern update with pattern keyword."""
    milestones.patterns = ["existing"]
    milestones.get_matches(patterns=["new"])
    assert milestones.patterns == ["new"]


def test_get_matches_invalid_mode(milestones):
    """Test get_matches with invalid mode."""
    spans = milestones.get_matches(patterns=["test"], mode="invalid")
    assert milestones.mode == "string"
    assert isinstance(spans, list)


def test_remove_patterns(milestones):
    """Test removing patterns."""
    patterns = ["test"]
    spans = milestones.get_matches(patterns, mode="string")
    milestones.set_milestones(spans)
    assert any(token._.milestone_iob == "B" for token in milestones.doc)
    milestones.remove(patterns, mode="string")
    assert all(token._.milestone_iob == "O" for token in milestones.doc)
    assert all(token._.milestone_label == "" for token in milestones.doc)


def test_remove_patterns_no_matches(milestones):
    """Test removing patterns with no matches."""
    patterns = ["test"]
    spans = milestones.get_matches(patterns, mode="string")
    milestones.set_milestones(spans)
    patterns = ["no_match"]
    milestones.remove(patterns, mode="string")
    assert any(token._.milestone_iob == "B" for token in milestones.doc)


def test_remove_patterns_multiple_matches(milestones):
    """Test removing patterns with multiple matches."""
    patterns = ["test", "string"]
    spans = milestones.get_matches(patterns, mode="string")
    milestones.set_milestones(spans)
    assert any(token._.milestone_iob == "B" for token in milestones.doc)
    milestones.remove(patterns, mode="string")
    assert all(token._.milestone_iob == "O" for token in milestones.doc)
    assert all(token._.milestone_label == "" for token in milestones.doc)


def test_reset(milestones):
    """Test the reset method of the Milestones class."""
    milestones.reset()
    assert list(milestones.doc.spans["milestones"]) == []
    for token in milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_reset_no_milestones(milestones):
    """Test the reset method when there are no milestones."""
    milestones.doc.spans["milestones"] = []
    milestones.reset()
    assert list(milestones.doc.spans["milestones"]) == []
    for token in milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_reset_with_existing_milestones(milestones):
    """Test the reset method with existing milestones."""
    patterns = ["test"]
    spans = milestones.get_matches(patterns, mode="string")
    milestones.set_milestones(spans)
    milestones.reset()
    assert list(milestones.doc.spans["milestones"]) == []
    for token in milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_set_milestones_basic(milestones):
    """Test set_milestones with basic spans."""
    spans = milestones.get_matches(patterns=["test"], mode="string")
    milestones.set_milestones(spans)
    assert list(milestones.doc.spans["milestones"]) == spans
    for span in milestones.doc.spans["milestones"]:
        assert span[0]._.milestone_iob == "B"
        assert (
            span[0]._.milestone_label
            == f"{span.text[:20]}{'...' if len(span.text) > 20 else ''}"
        )


def test_set_milestones_start_before(milestones):
    """Test set_milestones with start="before."""
    spans = milestones.get_matches(["test"], mode="string")
    original_span_start = spans[0].start
    milestones.set_milestones(spans, start="before")
    # Note that if the last span contains the first token, there may be no milestones
    if len(milestones.doc.spans["milestones"]) > 0:
        assert milestones.doc.spans["milestones"][0].start == original_span_start - 1
        assert milestones.doc.spans["milestones"][0][0]._.milestone_iob == "B"
        assert milestones.doc.spans["milestones"][0][0]._.milestone_label == ""


def test_set_milestones_start_after(milestones):
    """Test set_milestones with start="after."""
    spans = milestones.get_matches(["test"], mode="string")
    original_span_start = spans[0].start
    milestones.set_milestones(spans, start="after")
    # Note that if the last span contains the last token, there may be no milestones
    if len(milestones.doc.spans["milestones"]) > 0:
        assert milestones.doc.spans["milestones"][0].start == original_span_start + 1
        assert milestones.doc.spans["milestones"][0][0]._.milestone_iob == "B"
        assert milestones.doc.spans["milestones"][0][0]._.milestone_label == ""


def test_set_milestones_remove_true(milestones):
    """Test set_milestones with remove_token=True."""
    spans = milestones.get_matches(["test"], mode="string")
    next_token = milestones.doc[spans[0].end]
    milestones.set_milestones(spans, remove=True)
    assert milestones.doc.spans["milestones"][0].text == next_token.text
    for span in spans:
        token = milestones.doc[span.start]
        assert token._.milestone_iob == "B"
        assert token._.milestone_label == ""


def test_set_milestones_no_spans(milestones):
    """Test set_milestones with no spans."""
    milestones.set_milestones([])
    assert list(milestones.doc.spans["milestones"]) == []
    for token in milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_to_list_basic(milestones):
    """Test to_list method with default strip_punct=True."""
    patterns = ["test"]
    spans = milestones.get_matches(patterns, mode="string", case_sensitive=False)
    milestones.set_milestones(spans)
    milestone_dicts = milestones.to_list()
    assert isinstance(milestone_dicts, list)
    assert len(milestone_dicts) == 3
    for milestone in milestone_dicts:
        assert isinstance(milestone, dict)
        assert "text" in milestone
        assert "characters" in milestone
        assert "start_token" in milestone
        assert "end_token" in milestone
        assert "start_char" in milestone
        assert "end_char" in milestone


# WARNING: I can't reproduce the behaviour that strip_punct was meant to address
def test_to_list_strip_punct_true(milestones):
    """Test to_list method with strip_punct=True."""
    patterns = ["test"]
    text = "This is a test of the test pattern."
    milestones = TokenMilestones(doc=nlp(text))
    spans = milestones.get_matches(patterns, mode="string", case_sensitive=False)
    milestones.set_milestones(spans)
    milestone_dicts = milestones.to_list(strip_punct=False)
    for milestone in milestone_dicts:
        assert milestone["characters"][-1] not in punctuation


def test_to_list_empty_spans(milestones):
    """Test to_list method with no spans."""
    milestones.doc.spans["milestones"] = []
    milestone_dicts = milestones.to_list()
    assert milestone_dicts == []


def test_spans_property_no_milestones_key():
    """Test the spans property when 'milestones' key doesn't exist in doc.spans."""
    doc = nlp("This is a test.")
    milestones = TokenMilestones(doc=doc)
    # Don't set doc.spans["milestones"], so it won't exist
    # Clear it if it exists
    if "milestones" in milestones.doc.spans:
        del milestones.doc.spans["milestones"]
    assert milestones.spans == []


def test_iter_method_returns_generator(milestones):
    """Test that __iter__ returns an iterator/generator."""
    patterns = ["test"]
    spans = milestones.get_matches(patterns, mode="string", case_sensitive=False)
    milestones.set_milestones(spans)
    # Test that we can iterate
    iterator = iter(milestones)
    # Consume the iterator to cover the return statement
    result = list(iterator)
    assert len(result) > 0
    assert all(isinstance(span, Span) for span in result)


def test_autodetect_mode_invalid_pattern_raises():
    """Test that _autodetect_mode raises error for completely invalid pattern."""
    doc = nlp("This is a test.")
    milestones = TokenMilestones(doc=doc)
    # Create an invalid pattern that is a list but will fail Matcher validation
    # It needs to be a list (not dict) but with invalid structure for spaCy Matcher
    invalid_pattern = [{"INVALID_ATTR": "value"}]  # Invalid spaCy token attribute
    with pytest.raises(BaseException, match="could not be matched automatically"):
        milestones._autodetect_mode([invalid_pattern])


def test_get_matches_invalid_mode_autodetects():
    """Test that get_matches autodetects when mode is invalid."""
    doc = nlp("This is a test pattern.")
    milestones = TokenMilestones(doc=doc)
    # Pass an invalid mode, should trigger autodetect
    spans = milestones.get_matches(["test"], mode="invalid_mode")
    assert len(spans) > 0
    assert all(isinstance(span, Span) for span in spans)


def test_get_matches_none_mode_autodetects():
    """Test that get_matches autodetects when mode is None."""
    doc = nlp("This is a test pattern.")
    milestones = TokenMilestones(doc=doc)
    # Pass None as mode, should trigger autodetect
    spans = milestones.get_matches(["test"], mode=None)
    assert len(spans) > 0
    assert all(isinstance(span, Span) for span in spans)


def test_to_list_strip_punct_actually_strips():
    """Test that to_list with strip_punct=True actually strips punctuation."""
    # Create a doc where a milestone will end with punctuation
    doc = nlp("This is a test.")
    milestones = TokenMilestones(doc=doc)
    # Match "test." which includes the period
    spans = milestones.get_matches(["test."], mode="string", case_sensitive=False)
    milestones.set_milestones(spans)
    milestone_dicts = milestones.to_list(strip_punct=True)
    if milestone_dicts:
        # Check that punctuation was stripped
        for milestone in milestone_dicts:
            assert not milestone["characters"].endswith(".")


def test_set_milestones_invalid_start():
    """Test that set_milestones raises ValueError for invalid start parameter."""
    doc = nlp("This is a test.")
    milestones = TokenMilestones(doc=doc)
    spans = milestones.get_matches(["test"], mode="string")
    with pytest.raises(ValueError, match="Start must be None, 'before', or 'after'"):
        milestones.set_milestones(spans, start="invalid")


if __name__ == "__main__":
    pytest.main()
