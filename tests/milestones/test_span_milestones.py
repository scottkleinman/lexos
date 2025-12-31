"""test_span_milestones.py.

Coverage: 87%. Missing: 67, 119-125, 166, 168, 227, 229, 261-263, 284, 322, 324, 360
Last Update: 12/27/2024
"""

import pytest
import spacy

from lexos.milestones.span_milestones import (
    CustomMilestones,
    LineMilestones,
    SentenceMilestones,
    SpanMilestones,
)


@pytest.fixture
def nlp():
    """Return a spaCy NLP model."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def nlp_no_sents():
    """Return a spaCy NLP model with sentence boundary detection disabled."""
    return spacy.load("en_core_web_sm", disable=["parser", "sentencizer"])


@pytest.fixture
def doc(nlp):
    """Doc fixture for SpanMilestones tests."""
    text = "This is a test document. Here is another sentence."
    return nlp(text)


@pytest.fixture
def doc_no_sents(nlp_no_sents):
    """Doc fixture without sentence boundaries."""
    text = "This is a test document. Here is another sentence."
    return nlp_no_sents(text)


@pytest.fixture
def doc_line_breaks(nlp):
    """Doc fixture with line breaks for LineMilestones tests."""
    text = "This is a test document.\nHere is another sentence."
    return nlp(text)


@pytest.fixture
def spans(doc):
    """Spans fixture for milestones tests."""
    return [doc[0:4], doc[5:9]]


@pytest.fixture
def span_milestones(doc, spans):
    """SpanMilestones fixture."""
    return SpanMilestones(doc=doc, spans=spans)


@pytest.fixture
def sentence_milestones(doc, spans):
    """SentenceMilestones fixture."""
    return SentenceMilestones(doc=doc, spans=spans)


@pytest.fixture
def line_milestones(doc_line_breaks, spans):
    """LineMilestones fixture."""
    return LineMilestones(doc=doc_line_breaks, spans=spans)


@pytest.fixture
def custom_milestones(doc, spans):
    """CustomMilestones fixture."""
    return CustomMilestones(doc=doc, spans=spans)


# SpanMilestones tests


def test_span_milestones_init(span_milestones, doc):
    """Test SpanMilestones initialization."""
    assert span_milestones.doc == doc
    assert span_milestones.spans == []
    assert span_milestones.doc[0]._.milestone_iob == "O"
    assert span_milestones.doc[0]._.milestone_label == ""


def test_span_milestones_spans(span_milestones, spans):
    """Test SpanMilestones spans property."""
    span_milestones.doc.spans["milestones"] = spans
    assert span_milestones.spans == spans


def test_span_milestones_iter(span_milestones, spans):
    """Test SpanMilestones iter method."""
    span_milestones.doc.spans["milestones"] = spans
    assert list(span_milestones) == spans


def test_span_milestones__assign_token_attributes_basic(span_milestones, doc, spans):
    """Test basic token attribute assignment."""
    spans.append(doc[5:9])
    span_milestones._assign_token_attributes(spans)
    assert span_milestones.doc[0]._.milestone_iob == "B"
    assert span_milestones.doc[1]._.milestone_iob == "I"
    assert span_milestones.doc[2]._.milestone_iob == "I"
    assert span_milestones.doc[3]._.milestone_iob == "I"
    assert span_milestones.doc[4]._.milestone_iob == "O"
    assert span_milestones.doc[5]._.milestone_iob == "B"


def test_span_milestones__assign_token_attributes_empty_spans(span_milestones):
    """Test with empty spans list."""
    span_milestones._assign_token_attributes([])
    for token in span_milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_span_milestones__assign_token_attributes_long_label(span_milestones, spans):
    """Test truncation of long labels."""
    span_milestones._assign_token_attributes(spans, max_label_length=3)
    assert span_milestones.doc[0]._.milestone_label == "Thi..."


def test_span_milestones__get_list_basic(span_milestones):
    """Test _get_list method with default strip_punct=True."""
    milestone_dicts = span_milestones._get_list()
    assert isinstance(milestone_dicts, list)
    assert len(milestone_dicts) == len(span_milestones.spans)
    for milestone in milestone_dicts:
        assert isinstance(milestone, dict)
        assert "text" in milestone
        assert "characters" in milestone
        assert "start_token" in milestone
        assert "end_token" in milestone
        assert "start_char" in milestone
        assert "end_char" in milestone


# WARNING: I can't reproduce the behaviour that required the strip_punct variation.
# def test_span_milestones__get_list_strip_punct_true(span_milestones):
#     """Test _get_list method with strip_punct=True."""
#     span_milestones.doc.text += "."
#     milestone_dicts = span_milestones._get_list(strip_punct=True)
#     for milestone in milestone_dicts:
#         assert milestone["characters"][-1] not in punctuation

# def test_span_milestones__get_list_strip_punct_false(span_milestones):
#     """Test _get_list method with strip_punct=False."""
#     span_milestones.doc.text += "."
#     milestone_dicts = span_milestones._get_list(strip_punct=False)
#     for milestone in milestone_dicts:
#         assert milestone["characters"][-1] in punctuation


def test_span_milestones__get_list_empty_spans(span_milestones):
    """Test _get_list method with no spans."""
    span_milestones.doc.spans["milestones"] = []
    milestone_dicts = span_milestones._get_list()
    assert milestone_dicts == []


def test_span_milestones__reset(span_milestones):
    """Test _reset method."""
    span_milestones._reset()
    assert list(span_milestones.doc.spans["milestones"]) == []
    for token in span_milestones.doc:
        assert token._.milestone_iob == "O"
        assert token._.milestone_label == ""


def test_sentence_milestones_init(sentence_milestones, doc):
    """Test SentenceMilestones initialization."""
    sentence_milestones.set()
    assert sentence_milestones.doc == doc
    assert sentence_milestones.type == "sentences"
    assert len(list(sentence_milestones.doc.sents)) == 2
    assert len(sentence_milestones.doc.spans["milestones"]) == 2
    assert sentence_milestones.doc[0]._.milestone_iob == "B"
    assert len(sentence_milestones.doc[0]._.milestone_label) == 23


def test_sentence_milestones_invalid_model(doc_no_sents):
    """Test SentenceMilestones initialization."""
    with pytest.raises(
        ValueError,
        match="Either the document's model does not parse sentence boundaries",
    ):
        _ = SentenceMilestones(doc=doc_no_sents)


def test_sentence_milestones_reset(sentence_milestones, spans):
    """Test SentenceMilestones reset."""
    sentence_milestones.doc[0]._.milestone_iob = "B"
    sentence_milestones.doc[0]._.milestone_label = "Test"
    sentence_milestones.doc.spans["milestones"] = spans
    sentence_milestones.reset()
    sentence_milestones.doc.spans["milestones"] = []
    assert sentence_milestones.doc[0]._.milestone_iob == "O"
    assert sentence_milestones.doc[0]._.milestone_label == ""


def test_sentence_milestones_set_basic(sentence_milestones, doc):
    """Test SentenceMilestones set."""
    sentence_milestones.set()
    assert sentence_milestones.doc == doc
    assert sentence_milestones.type == "sentences"
    assert len(list(sentence_milestones.doc.sents)) == 2
    assert len(sentence_milestones.doc.spans["milestones"]) == 2
    assert sentence_milestones.doc[0]._.milestone_iob == "B"
    assert len(sentence_milestones.doc[0]._.milestone_label) == 23


def test_sentence_milestones_set_with_step(sentence_milestones, doc):
    """Test SentenceMilestones set with step."""
    sentence_milestones.set(step=2)
    assert sentence_milestones.doc == doc
    assert sentence_milestones.type == "sentences"
    assert len(list(sentence_milestones.doc.sents)) == 2
    assert len(sentence_milestones.doc.spans["milestones"]) == 1
    assert sentence_milestones.doc[0]._.milestone_iob == "B"


def test_sentence_milestones_set_with_max_label_length(sentence_milestones, doc):
    """Test SentenceMilestones set with max_label_length."""
    sentence_milestones.set(max_label_length=10)
    assert sentence_milestones.doc == doc
    assert sentence_milestones.type == "sentences"
    assert len(list(sentence_milestones.doc.sents)) == 2
    assert len(sentence_milestones.doc.spans["milestones"]) == 2
    assert len(sentence_milestones.doc[0]._.milestone_label) == 13


def test_sentence_milestones_to_list_basic(sentence_milestones):
    """Test SentenceMilestones to_list."""
    assert isinstance(sentence_milestones.to_list(), list)


def test_sentence_milestones_to_list_strip_punct_true(sentence_milestones):
    """Test SentenceMilestones to_list with strip_punct=True."""
    assert isinstance(sentence_milestones.to_list(strip_punct=True), list)


# LineMilestones tests


def test_line_milestones_init(line_milestones, doc_line_breaks):
    """Test LineMilestones initialization."""
    assert line_milestones.doc == doc_line_breaks
    assert line_milestones.type == "lines"
    assert len(line_milestones.doc.spans["milestones"]) == 0
    assert line_milestones.doc[0]._.milestone_iob == "O"
    assert line_milestones.doc[0]._.milestone_label == ""


def test_line_milestones_reset(line_milestones, doc_line_breaks):
    """Test LineMilestones reset."""
    spans = [doc_line_breaks[0:4], doc_line_breaks[5:9]]
    line_milestones.doc[0]._.milestone_iob = "B"
    line_milestones.doc[0]._.milestone_label = "Test"
    line_milestones.doc.spans["milestones"] = spans
    line_milestones.reset()
    line_milestones.doc.spans["milestones"] = []
    assert line_milestones.doc[0]._.milestone_iob == "O"
    assert line_milestones.doc[0]._.milestone_label == ""


def test_line_milestones_set_basic(line_milestones, doc_line_breaks):
    """Test LineMilestones set."""
    spans = doc_line_breaks.text.split("\n")
    line_milestones.set()
    assert line_milestones.doc == doc_line_breaks
    assert line_milestones.type == "lines"
    assert len(line_milestones.doc.spans["milestones"]) == 2
    assert line_milestones.spans[0].text == spans[0]
    assert line_milestones.spans[1].text == spans[1]
    assert line_milestones.doc[0]._.milestone_iob == "B"
    assert len(line_milestones.doc[0]._.milestone_label) == 23


def test_line_milestones_set_with_step(line_milestones, doc_line_breaks):
    """Test LineMilestones set with step."""
    line_milestones.set(step=2)
    assert line_milestones.doc == doc_line_breaks
    assert line_milestones.type == "lines"
    assert line_milestones.spans[0].text == doc_line_breaks.text
    assert line_milestones.doc[0]._.milestone_iob == "B"
    assert len(line_milestones.doc[0]._.milestone_label) == 23


def test_line_milestones_set_with_max_label_length(line_milestones):
    """Test LineMilestones set with max_label_length."""
    line_milestones.set(max_label_length=10)
    assert len(line_milestones.doc.spans["milestones"]) == 2
    assert len(line_milestones.doc[0]._.milestone_label) == 13


def test_line_milestones_set_with_pattern(nlp):
    """Test LineMilestones set with pattern."""
    text = "This is a test document.\nHere is another sentence.\n\nHere is a sentence after two linebreaks."
    doc = nlp(text)
    line_milestones = LineMilestones(doc=doc)
    spans = line_milestones.doc.text.split("\n\n")
    line_milestones.set(pattern="\n\n")
    assert line_milestones.spans[0].text == spans[0]


def test_line_milestones_to_list_basic(line_milestones):
    """Test LineMilestones to_list."""
    assert isinstance(line_milestones.to_list(), list)


def test_line_milestones_to_list_strip_punct_true(line_milestones):
    """Test LineMilestones to_list with strip_punct=True."""
    assert isinstance(line_milestones.to_list(strip_punct=True), list)


# CustomMilestones tests


def test_custom_milestones_init(custom_milestones, doc):
    """Test CustomMilestones initialization."""
    assert custom_milestones.doc == doc
    assert custom_milestones.type == "custom"
    assert len(custom_milestones.doc.spans["milestones"]) == 0
    assert custom_milestones.doc[0]._.milestone_iob == "O"
    assert custom_milestones.doc[0]._.milestone_label == ""


def test_custom_milestones_milestones_reset(custom_milestones, spans):
    """Test CustomMilestones reset."""
    custom_milestones.doc[0]._.milestone_iob = "B"
    custom_milestones.doc[0]._.milestone_label = "Test"
    custom_milestones.doc.spans["milestones"] = spans
    custom_milestones.reset()
    custom_milestones.doc.spans["milestones"] = []
    assert custom_milestones.doc[0]._.milestone_iob == "O"
    assert custom_milestones.doc[0]._.milestone_label == ""


def test_custom_milestones_set_basic(custom_milestones, doc, spans):
    """Test CustomMilestones set."""
    custom_milestones.set(spans=spans)
    assert custom_milestones.doc == doc
    assert custom_milestones.type == "custom"
    assert len(custom_milestones.doc.spans["milestones"]) == 2
    assert custom_milestones.spans[0].text == spans[0].text
    assert custom_milestones.spans[1].text == spans[1].text
    assert custom_milestones.doc[0]._.milestone_iob == "B"
    assert len(custom_milestones.doc[0]._.milestone_label) == 14


def test_custom_milestones_set_with_step(custom_milestones, doc):
    """Test CustomMilestones set with step."""
    spans = list(doc.sents)
    custom_milestones.set(spans, step=2)
    assert custom_milestones.doc == doc
    assert custom_milestones.type == "custom"
    assert custom_milestones.spans[0].text == doc.text
    assert custom_milestones.doc[0]._.milestone_iob == "B"
    assert len(custom_milestones.doc[0]._.milestone_label) == 23


def test_custom_milestones_set_with_max_label_length(custom_milestones, spans):
    """Test CustomMilestones set with max_label_length."""
    custom_milestones.set(spans, max_label_length=10)
    assert len(custom_milestones.doc.spans["milestones"]) == 2
    assert len(custom_milestones.doc[0]._.milestone_label) == 13


def test_custom_milestones_to_list_basic(custom_milestones):
    """Test CustomMilestones to_list."""
    assert isinstance(custom_milestones.to_list(), list)


def test_custom_milestones_to_list_strip_punct_true(custom_milestones):
    """Test CustomMilestones to_list with strip_punct=True."""
    assert isinstance(custom_milestones.to_list(strip_punct=True), list)


if __name__ == "__main__":
    pytest.main()
