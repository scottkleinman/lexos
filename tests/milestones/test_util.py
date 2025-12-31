"""test_util.py.

Coverage: 94%. Missing: 52, 69, 162-163
Refactor: 12/27/2024
"""

import pytest
import spacy

from lexos.milestones.span_milestones import (
    LineMilestones,
    SentenceMilestones,
    SpanMilestones,
)
from lexos.milestones.string_milestones import StringMilestones
from lexos.milestones.token_milestones import TokenMilestones
from lexos.milestones.util import (
    chars_to_tokens,
    ensure_list,
    filter_doc,
    lowercase_spacy_rules,
    move_milestone,
)

nlp = spacy.load("en_core_web_sm")

# Fixtures


@pytest.fixture
def chars_to_tokens_doc():
    """Chars to tokens test doc."""
    return nlp("This is a test.")


@pytest.fixture
def doc_for_filter_doc():
    """Doc fixture for filter_doc tests."""
    return nlp("This is a test document.")


@pytest.fixture
def doc():
    """Doc fixture for move_milestone tests."""
    text = "This is a test document."
    doc = nlp(text)
    for token in doc:
        token.set_extension("milestone_iob", default="O", force=True)
        token.set_extension("milestone_label", default="", force=True)
    return doc


# Tests


def test_chars_to_tokens_basic(chars_to_tokens_doc):
    """Test chars_to_tokens with a basic example."""
    mapping = chars_to_tokens(chars_to_tokens_doc)
    expected_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,  # "This"
        5: 1,
        6: 1,  # "is"
        8: 2,  # "a"
        10: 3,
        11: 3,
        12: 3,
        13: 3,  # "test"
        14: 4,  # "."
    }
    assert mapping == expected_mapping


def test_chars_to_tokens_empty_doc():
    """Test chars_to_tokens with an empty doc."""
    empty_doc = nlp("")
    mapping = chars_to_tokens(empty_doc)
    assert mapping == {}


def test_chars_to_tokens_single_token(chars_to_tokens_doc):
    """Test chars_to_tokens with a single token."""
    single_token_doc = chars_to_tokens_doc[:1]  # "This"
    mapping = chars_to_tokens(single_token_doc)
    expected_mapping = {0: 0, 1: 0, 2: 0, 3: 0}
    assert mapping == expected_mapping


def test_ensure_list():
    """Test ensure_list with various inputs."""
    assert isinstance(ensure_list("A test string"), list)


def test_filter_doc(doc_for_filter_doc):
    """Test filter_doc with a basic example."""
    milestones = TokenMilestones(doc=doc_for_filter_doc)
    milestones.doc.spans["milestones"] = [milestones.doc[2:4]]
    spans = [milestones.doc[2:4]]
    filtered_doc = filter_doc(milestones.doc, spans)
    assert len(filtered_doc) == 4
    assert filtered_doc.text == "This is document."


def test_filter_doc_empty(doc_for_filter_doc):
    """Test filter_doc with empty remove_ids."""
    milestones = TokenMilestones(doc=doc_for_filter_doc)
    spans = []
    filtered_doc = filter_doc(milestones.doc, spans)
    assert len(filtered_doc) == 6
    assert filtered_doc.text == "This is a test document."


def test_spacy_rule_to_lower_with_dict():
    """Test lowercase_spacy_rules with a dict pattern."""
    pattern = {"ORTH": "text"}
    result = lowercase_spacy_rules(pattern)
    assert result == {"LOWER": "text"}


def test_spacy_rule_to_lower_with_list():
    """Test lowercase_spacy_rules with a list pattern."""
    pattern = [{"ORTH": "text"}]
    result = lowercase_spacy_rules(pattern)
    assert result == [{"LOWER": "text"}]


def test_spacy_rule_to_lower_with_text():
    """Test lowercase_spacy_rules with a TEXT pattern."""
    pattern = {"TEXT": "text"}
    result = lowercase_spacy_rules(pattern)
    assert result == {"LOWER": "text"}


def test_spacy_rule_to_lower_with_regex():
    """Test lowercase_spacy_rules with a regex pattern."""
    pattern = [
        {"TEXT": {"REGEX": "^[Uu](\\.?|nited)$"}},
        {"TEXT": {"REGEX": "^[Ss](\\.?|tates)$"}},
        {"LOWER": "president"},
    ]
    result = lowercase_spacy_rules(pattern)
    assert result == [
        {"LOWER": {"REGEX": "^[Uu](\\.?|nited)$"}},
        {"LOWER": {"REGEX": "^[Ss](\\.?|tates)$"}},
        {"LOWER": "president"},
    ]


def test_move_milestone_before(doc):
    """Test move_milestone with start='before'."""
    spans = [doc[2:4]]  # "a test"
    new_spans = move_milestone(doc, spans, start="before")
    assert len(new_spans) == 1
    assert new_spans[0].start == 1  # "is"
    assert new_spans[0].end == 2
    assert doc[1]._.milestone_iob == "B"
    assert doc[1]._.milestone_label == ""


def test_move_milestone_after(doc):
    """Test move_milestone with start='after'."""
    spans = [doc[2:4]]  # "a test"
    new_spans = move_milestone(doc, spans, start="after")
    assert len(new_spans) == 1
    assert new_spans[0].start == 4  # "document"
    assert new_spans[0].end == 5
    assert doc[4]._.milestone_iob == "B"
    assert doc[4]._.milestone_label == ""


def test_move_milestone_before_at_start(doc):
    """Test move_milestone with start='before' at the start of the doc."""
    spans = [doc[0:2]]  # "This is"
    new_spans = move_milestone(doc, spans, start="before")
    assert len(new_spans) == 0  # No new spans should be created


def test_move_milestone_after_at_end(doc):
    """Test move_milestone with start='after' at the end of the doc."""
    spans = [doc[-2:]]  # "test document"
    new_spans = move_milestone(doc, spans, start="after")
    assert len(new_spans) == 0  # No new spans should be created


def test_move_milestone_empty_spans(doc):
    """Test move_milestone with empty spans."""
    new_spans = move_milestone(doc, [], start="before")
    assert len(new_spans) == 0  # No new spans should be created


if __name__ == "__main__":
    pytest.main()
