"""test_token_cutter.py.

Coverage: 100%
Last updated: 23 December, 2025
"""

import numpy as np
import pytest
import spacy
from spacy.tokens import Doc, Token

from lexos.cutter.token_cutter import TokenCutter
from lexos.exceptions import LexosException

# Fixtures


@pytest.fixture
def nlp():
    """Return a blank English spaCy NLP pipeline."""
    return spacy.blank("en")


@pytest.fixture
def doc(nlp):
    """Return a basic test document with a simple sentence."""
    text = "The quick brown fox jumps over the lazy dog."
    return nlp(text)


@pytest.fixture
def cutter():
    """Return a new TokenCutter instance with default settings."""
    return TokenCutter()


@pytest.fixture
def doc2(nlp):
    """Return a short test document."""
    return nlp("Test document")


@pytest.fixture
def sample_chunks_for_merge_threshold(nlp):
    """Return two chunks of varying lengths to test merge threshold logic."""
    chunk1 = nlp("This is a long chunk with many tokens.")
    chunk2 = nlp("Short chunk.")
    return [chunk1, chunk2]


@pytest.fixture
def chunk_doc(nlp):
    """Return a document with multiple tokens for chunking tests."""
    return nlp("This is a test document with multiple tokens for testing chunking.")


@pytest.fixture
def chunk_doc_with_ents(nlp):
    """Return a document with manually assigned named entities."""
    doc = nlp("Barack Obama was born in Hawaii.")
    ents = [(0, 12, "PERSON"), (25, 31, "GPE")]
    doc.ents = [doc.char_span(start, end, label=label) for start, end, label in ents]
    return doc


@pytest.fixture
def sample_chunks_apply_overlap(nlp):
    """Return a list of three short docs for testing overlap logic."""
    return [nlp("one two three"), nlp("four five six"), nlp("seven eight nine")]


@pytest.fixture
def doc_split_doc(nlp):
    """Return a document to test general chunking behavior."""
    return nlp(
        "This is a test document with multiple tokens for testing chunking mechanisms"
    )


@pytest.fixture
def doc_with_lines(nlp):
    """Return a document with newline-separated lines for line splitting tests."""
    return nlp("Line1\nLine2\nLine3\nLine4\nLine5\n")


@pytest.fixture
def doc_with_sentences():
    """Return a document with multiple sentence boundaries using en_core_web_sm."""
    nlp_attr = spacy.load("en_core_web_sm")
    return nlp_attr(
        "First sentence. Second one. Third here. Fourth now. Last sentence."
    )


@pytest.fixture
def doc_for_write_chunk(nlp):
    """Return a short document for write_chunk file output tests."""
    return nlp("Test content")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Return a temporary directory for chunk output files."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_chunks_to_merge(nlp):
    """Return sample chunks for testing Doc merging."""
    return [nlp("First chunk."), nlp("Second chunk."), nlp("Third chunk.")]


@pytest.fixture
def cutter_save(nlp):
    """Return a TokenCutter instance with preloaded chunks for save_text tests."""
    cutter = TokenCutter()
    cutter.chunks = [
        [nlp("First chunk."), nlp("Second chunk.")],
        [nlp("Third chunk."), nlp("Fourth chunk.")],
    ]
    return cutter


@pytest.fixture
def output_dir(tmp_path):
    """Return a temporary output directory for saving files."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def doc_for_split(nlp):
    """Return a document with multiple sentences for split tests."""
    return nlp(
        "This is a test document with multiple sentences. Here is another one. And a third."
    )


@pytest.fixture
def milestone_doc(nlp):
    """Return a short document for milestone span-based splitting tests."""
    return nlp("quick jumps")


@pytest.fixture
def cutter_with_chunks(nlp):
    """Return a TokenCutter instance with preloaded chunks for merging or saving tests."""
    cutter = TokenCutter()
    cutter.chunks = [
        [nlp("First chunk"), nlp("Second chunk")],
        [nlp("Third chunk"), nlp("Fourth chunk")],
    ]
    return cutter


@pytest.fixture
def doc_with_sentences2(nlp):
    """Return a document with sentence boundaries using a sentencizer."""
    nlp.add_pipe("sentencizer")
    return nlp(
        "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    )


# Tests


def test_cutter_init_defaults():
    """Test TokenCutter initialization with default values."""
    cutter = TokenCutter()
    assert cutter.chunks == []
    assert cutter.chunksize == 1000
    assert cutter.n is None
    assert cutter.names == []
    assert cutter.newline is False
    assert cutter.merge_threshold == 0.5
    assert cutter.output_dir is None
    assert cutter.delimiter == "_"
    assert cutter.pad == 3
    assert cutter.strip_chunks is True


def test_cutter_init_custom():
    """Test TokenCutter initialization with custom values."""
    cutter = TokenCutter(
        chunksize=500,
        n=5,
        names=["doc1", "doc2"],
        newline=True,
        merge_threshold=0.5,
        output_dir="test_dir",
        delimiter="-",
        pad=4,
        strip_chunks=False,
    )
    assert cutter.chunksize == 500
    assert cutter.n == 5
    assert cutter.names == ["doc1", "doc2"]
    assert cutter.newline is True
    assert cutter.merge_threshold == 0.5
    assert cutter.output_dir == "test_dir"
    assert cutter.delimiter == "-"
    assert cutter.pad == 4
    assert cutter.strip_chunks is False


def test_cutter_iteration(cutter, doc):
    """Test iteration over TokenCutter chunks."""
    cutter.chunks = [[doc], [doc]]
    chunks = list(cutter)
    assert len(chunks) == 2
    assert all(isinstance(chunk, list) for chunk in chunks)


def test_cutter_len(cutter, doc):
    """Test TokenCutter length calculation."""
    cutter.docs = [doc, doc, doc]
    assert len(cutter) == 3


def test_cutter_empty():
    """Test TokenCutter with empty chunks."""
    cutter = TokenCutter()
    assert len(cutter) == 0
    assert list(cutter) == []


def test_cutter_invalid_pad():
    """Test TokenCutter with invalid pad value."""
    with pytest.raises(ValueError):
        TokenCutter(pad=-1)


def test_cutter_invalid_chunksize():
    """Test TokenCutter with invalid chunksize."""
    with pytest.raises(ValueError):
        TokenCutter(chunksize=0)


def test_merge_below_threshold(cutter, sample_chunks_for_merge_threshold):
    """Test merging when last chunk is below threshold."""
    cutter.chunksize = len(sample_chunks_for_merge_threshold[0])
    merged = cutter._apply_merge_threshold(sample_chunks_for_merge_threshold)
    assert len(merged) == 1
    # NB. Whitespace is supplied between merged chunks.
    assert merged[0].text == "This is a long chunk with many tokens. Short chunk."


def test_no_merge_above_threshold(cutter, nlp):
    """Test no merging when last chunk is above threshold."""
    chunks = [
        nlp("First chunk of text."),
        nlp("Second chunk that is long enough to exceed threshold."),
    ]
    cutter.chunksize = len(chunks[0])
    merged = cutter._apply_merge_threshold(chunks)
    assert len(merged) == 2


def test_merge_with_n(cutter, sample_chunks_for_merge_threshold):
    """Test merging when last chunk is below threshold."""
    cutter.n = 2
    merged = cutter._apply_merge_threshold(sample_chunks_for_merge_threshold)
    assert len(merged) == 1
    # NB. Whitespace is supplied between merged chunks.
    assert merged[0].text == "This is a long chunk with many tokens. Short chunk."


def test_no_merge_with_n(cutter, sample_chunks_for_merge_threshold):
    """Test merging when last chunk is below threshold."""
    cutter.n = 2
    cutter.merge_threshold = 0.0
    merged = cutter._apply_merge_threshold(sample_chunks_for_merge_threshold)
    assert len(merged) == 2
    # NB. Whitespace is supplied between merged chunks.
    assert merged[0].text == "This is a long chunk with many tokens."


def test_force_merge(cutter, sample_chunks_for_merge_threshold):
    """Test forced merge regardless of threshold."""
    merged = cutter._apply_merge_threshold(
        sample_chunks_for_merge_threshold, force=True
    )
    assert len(merged) == 1


def test_single_chunk(cutter, nlp):
    """Test handling single chunk."""
    chunks = [nlp("Single chunk.")]
    merged = cutter._apply_merge_threshold(chunks)
    assert len(merged) == 1
    assert merged[0].text == "Single chunk."


def test_different_threshold(nlp):
    """Test with different merge threshold."""
    cutter = TokenCutter(chunksize=5, merge_threshold=0.8)
    chunks = [nlp("First chunk."), nlp("Very tiny.")]
    merged = cutter._apply_merge_threshold(chunks)
    assert len(merged) == 1


def test_empty_chunks_merge_threshold(cutter):
    """Test handling empty chunks list."""
    with pytest.raises(IndexError):
        cutter._apply_merge_threshold([])


def test_merge_preserve_properties(cutter):
    """Test merging and preserviing properties."""
    nlp_attrs = spacy.load("en_core_web_sm")
    Token.set_extension("custom_attr", default="value", force=True)
    chunks = [
        nlp_attrs("Scotland is a small country with many haggises."),
        nlp_attrs("Short chunk."),
    ]
    cutter.n = 2
    merged = cutter._apply_merge_threshold(chunks)
    # NB. Whitespace is supplied between merged chunks.
    assert (
        merged[0].text == "Scotland is a small country with many haggises. Short chunk."
    )
    assert merged[0][0].ent_iob_ == chunks[0][0].ent_iob_
    assert merged[0][0]._.custom_attr == chunks[0][0]._.custom_attr
    assert merged[0].has_annotation("SENT_START")


def test_apply_overlap(cutter, sample_chunks_apply_overlap):
    """Test first chunk overlap."""
    cutter.overlap = 1
    overlapped = cutter._apply_overlap(sample_chunks_apply_overlap)
    # Assuming overlap = 1
    assert overlapped[0].text == "one two three four "
    assert overlapped[1].text == "four five six seven "
    assert overlapped[2].text == "seven eight nine"


def test_apply_overlap_single_chunk(cutter, nlp):
    """Test single chunk handling."""
    chunks = [nlp("one two three")]
    overlapped = cutter._apply_overlap(chunks)
    assert len(overlapped) == 1
    assert overlapped[0].text == "one two three"


def test_apply_overlap_empty_chunks(cutter):
    """Test empty chunks list."""
    overlapped = cutter._apply_overlap([])
    assert overlapped == []


def test_apply_overlap_different_size(nlp):
    """Test different overlap sizes."""
    cutter = TokenCutter(overlap=3)
    chunks = [nlp("one two three four"), nlp("five six seven eight")]
    overlapped = cutter._apply_overlap(chunks)
    assert len(overlapped[0].text.split()) == 7


def test_apply_overlap_properties(cutter, sample_chunks_apply_overlap):
    """Test preservation of doc properties."""
    nlp_attrs = spacy.load("en_core_web_sm")
    Token.set_extension("custom_attr", default="value", force=True)
    chunks = [
        nlp_attrs("This is a sentence about Scotland."),
        nlp_attrs("This is another sentence."),
    ]
    chunks[0][5]._.set("custom_attr", "new_value")
    cutter.overlap = 1
    overlapped = cutter._apply_overlap(chunks)
    assert overlapped[0][5].ent_iob_ == chunks[0][5].ent_iob_
    assert overlapped[0][5]._.custom_attr == "new_value"


def test_chunk_doc_basic(cutter, chunk_doc):
    """Test basic chunking with default attributes."""
    chunks = cutter._chunk_doc(chunk_doc)
    assert len(chunks) == 1
    assert all(isinstance(chunk, Doc) for chunk in chunks)


def test_chunk_doc_single_chunk_with_n(cutter, nlp):
    """Test _chunk_doc with n=1 that returns the original doc as one chunk."""
    doc = nlp("Short text.")
    cutter.n = 1
    chunks = cutter._chunk_doc(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunk_doc_entity_array_filled_forced_chunking():
    """Ensure ent_array is filled when doc.ents exists and chunking is forced."""
    import spacy

    from lexos.cutter.token_cutter import TokenCutter

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Barack Obama was born in Hawaii.")
    assert len(doc.ents) > 0
    cutter = TokenCutter(chunksize=1)
    chunks = cutter._chunk_doc(doc)
    assert len(chunks) > 1
    assert any(chunk.ents for chunk in chunks)


def test_chunk_doc_custom_attrs(cutter, chunk_doc):
    """Test chunking with custom attributes."""
    chunks = cutter._chunk_doc(chunk_doc, attrs=["ORTH", "LEMMA"])
    assert len(chunks) == 1
    assert all(isinstance(chunk, Doc) for chunk in chunks)


def test_chunk_doc_with_ents(cutter, chunk_doc_with_ents):
    """Test chunking with NER attributes."""
    chunks = cutter._chunk_doc(chunk_doc_with_ents)
    assert len(chunks) == 1
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert chunks[0].ents


def test_chunk_doc_with_extensions(cutter, chunk_doc):
    """Test chunking with custom extensions."""
    if not Token.has_extension("custom_attr"):
        Token.set_extension("custom_attr", default="None", force=True)
    chunk_doc[0]._.set("custom_attr", "value")

    chunks = cutter._chunk_doc(chunk_doc)
    assert len(chunks) == 1
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert chunks[0][0]._.get("custom_attr") == "value"


def test_chunk_doc_empty(cutter, nlp):
    """Test chunking with empty document."""
    doc = nlp("")
    with pytest.raises(LexosException, match="Document is empty"):
        cutter._chunk_doc(doc)


def test_chunk_doc_smaller_than_chunk(cutter, nlp):
    """Test document smaller than chunk size."""
    doc = nlp("Small doc")
    cutter.chunksize = 10
    chunks = cutter._chunk_doc(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Small doc"


def test_chunk_doc_exact_size(cutter, nlp):
    """Test document exactly chunk size."""
    doc = nlp("One two three four five")
    cutter.chunksize = 5
    chunks = cutter._chunk_doc(doc)
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_chunk_doc_multiple_chunks(cutter, nlp):
    """Test document with multiple chunks."""
    # import numpy as np
    doc = nlp("One two three four five six seven eight nine ten")
    cutter.chunksize = 3
    chunks = cutter._chunk_doc(doc)
    assert len(chunks) == 4
    assert all(isinstance(chunk, Doc) for chunk in chunks)


def test_keep_milestones_bool_basic(cutter, doc):
    """Tests basic splitting without keeping spans."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    chunks = cutter._keep_milestones_bool(doc, milestones)
    assert len(chunks) == 3
    assert chunks[0].text == "The"
    assert chunks[1].text == "brown fox"
    assert chunks[2].text == "over the lazy dog."


def test_keep_milestones_bool_with_spans(cutter, doc):
    """Tests splitting while keeping spans."""
    milestones = [doc[1:2]]  # "quick"
    chunks = cutter._keep_milestones_bool(doc, milestones, keep_spans=True)
    assert len(chunks) == 3
    assert chunks[0].text == "The"
    assert chunks[1].text == "quick"
    assert chunks[2].text == "brown fox jumps over the lazy dog."


def test_keep_milestones_bool_empty_doc(cutter, nlp):
    """Tests handling of empty document."""
    empty_doc = nlp("")
    chunks = cutter._keep_milestones_bool(empty_doc, [])
    assert chunks == []


def test_keep_milestones_bool_no_milestones(cutter, doc):
    """Tests handling with no milestones."""
    chunks = cutter._keep_milestones_bool(doc, [])
    assert len(chunks) == 1
    assert chunks[0].text == doc.text


def test_keep_milestones_bool_single_milestone(cutter, doc):
    """Tests handling of single milestone."""
    milestones = [doc[4:5]]  # "jumps"
    chunks = cutter._keep_milestones_bool(doc, milestones)
    assert len(chunks) == 2
    assert chunks[0].text == "The quick brown fox"
    assert chunks[1].text == "over the lazy dog."


def test_keep_milestones_bool_adjacent(cutter, doc):
    """Tests handling of adjacent milestones."""
    milestones = [doc[1:2], doc[2:3]]  # "quick", "brown"
    chunks = cutter._keep_milestones_bool(doc, milestones)
    assert len(chunks) == 2
    assert chunks[0].text == "The"
    assert chunks[1].text == "fox jumps over the lazy dog."


def test_keep_milestones_bool_doc_boundaries(cutter, doc):
    """Tests milestones at document boundaries."""
    milestones = [doc[0:1], doc[9:]]  # "The", "."
    chunks = cutter._keep_milestones_bool(doc, milestones, keep_spans=True)
    assert len(chunks) == 3
    assert chunks[0].text == "The"
    assert chunks[1].text == "quick brown fox jumps over the lazy dog"
    assert chunks[2].text == "."


def test_keep_milestones_following_basic(cutter, doc):
    """Test basic milestone following functionality."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    chunks = cutter._keep_milestones_following(doc, milestones)
    assert len(chunks) == 3
    assert chunks[0].text == "The"
    assert chunks[1].text == "quick brown fox"
    assert chunks[2].text == "jumps over the lazy dog."


def test_keep_milestones_following_empty(cutter, nlp):
    """Test empty document handling."""
    doc = nlp("")
    chunks = cutter._keep_milestones_following(doc, [])
    assert chunks == []


def test_keep_milestones_following_no_milestones(cutter, doc):
    """Test handling with no milestones."""
    chunks = cutter._keep_milestones_following(doc, [])
    assert len(chunks) == 0


def test_keep_milestones_following_single(cutter, doc):
    """Test single milestone handling."""
    milestones = [doc[1:2]]  # "quick"
    chunks = cutter._keep_milestones_following(doc, milestones)
    assert len(chunks) == 2
    assert chunks[0].text == "The"
    assert chunks[1].text == "quick brown fox jumps over the lazy dog."


def test_keep_milestones_following_adjacent(cutter, doc):
    """Test adjacent milestones handling."""
    milestones = [doc[1:2], doc[2:3]]  # "quick", "brown"
    chunks = cutter._keep_milestones_following(doc, milestones)
    assert len(chunks) == 3
    assert chunks[0].text == "The"
    assert chunks[1].text == "quick"
    assert chunks[2].text == "brown fox jumps over the lazy dog."


def test_keep_milestones_following_boundaries(cutter, doc):
    """Test milestones at document boundaries."""
    milestones = [doc[0:1], doc[-1:]]  # "The", "dog"
    chunks = cutter._keep_milestones_following(doc, milestones)
    assert len(chunks) == 2
    assert chunks[0].text == "The quick brown fox jumps over the lazy dog"
    assert chunks[1].text == "."


def test_keep_milestones_preceding_basic(cutter, doc):
    """Test basic milestone preceding functionality."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    chunks = cutter._keep_milestones_preceding(doc, milestones)
    assert len(chunks) == 3
    assert chunks[0].text == "The quick"
    assert chunks[1].text == "brown fox jumps"
    assert chunks[2].text == "over the lazy dog."


def test_keep_milestones_preceding_empty(cutter, nlp):
    """Test empty document handling."""
    empty_doc = nlp("")
    with pytest.raises(LexosException, match="Document is empty"):
        cutter._keep_milestones_preceding(empty_doc, [])


def test_keep_milestones_preceding_no_milestones(cutter, doc):
    """Test no milestones handling."""
    chunks = cutter._keep_milestones_preceding(doc, [])
    assert len(chunks) == 1


def test_keep_milestones_preceding_single(cutter, doc):
    """Test single milestone handling."""
    milestones = [doc[1:2]]  # "quick"
    chunks = cutter._keep_milestones_preceding(doc, milestones)

    assert len(chunks) == 2
    assert chunks[0].text == "The quick"


def test_keep_milestones_preceding_adjacent(cutter, doc):
    """Test adjacent milestones handling."""
    milestones = [doc[1:2], doc[2:3]]  # "quick", "brown"
    chunks = cutter._keep_milestones_preceding(doc, milestones)

    assert len(chunks) == 3
    assert chunks[0].text == "The quick"
    assert chunks[1].text == "brown"
    assert chunks[2].text == "fox jumps over the lazy dog."


def test_keep_milestones_preceding_boundaries(cutter, doc):
    """Test milestones at document boundaries."""
    milestones = [doc[0:1], doc[-1:]]  # "The", "."
    chunks = cutter._keep_milestones_preceding(doc, milestones)

    assert len(chunks) == 1
    assert chunks[0].text == "The quick brown fox jumps over the lazy dog."


def test_set_attributes_single(cutter):
    """Test setting single attribute."""
    cutter._set_attributes(chunksize=500)
    assert cutter.chunksize == 500


def test_set_attributes_multiple(cutter):
    """Test setting multiple attributes."""
    cutter._set_attributes(chunksize=500, n=5, newline=True)
    assert cutter.chunksize == 500
    assert cutter.n == 5
    assert cutter.newline is True


def test_set_attributes_existing(cutter):
    """Test overwriting existing attributes."""
    original_pad = cutter.pad
    cutter._set_attributes(pad=5)
    assert cutter.pad == 5
    assert cutter.pad != original_pad


def test_set_attributes_invalid(cutter):
    """Test setting invalid attribute."""
    with pytest.raises(ValueError):
        cutter._set_attributes(invalid_attr=123)


def test_set_attributes_empty(cutter):
    """Test setting no attributes."""
    original_state = vars(cutter).copy()
    cutter._set_attributes()
    assert vars(cutter) == original_state


def test_set_attributes_type_validation(cutter):
    """Test attribute type validation."""
    with pytest.raises(ValueError):
        cutter._set_attributes(pad="invalid")


def test_split_doc_by_tokens(cutter, doc_split_doc):
    """Test splitting by token count."""
    cutter.chunksize = 5
    chunks = cutter._split_doc(doc_split_doc)
    # The default merge_threshold merges the last two chunks
    assert len(chunks) == 2
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert len(chunks[0]) == 5


def test_split_doc_by_n(cutter, doc_split_doc):
    """Test splitting into n chunks."""
    cutter.n = 3
    chunks = cutter._split_doc(doc_split_doc)
    assert len(chunks) == 3
    assert all(isinstance(chunk, Doc) for chunk in chunks)


def test_split_doc_merge_final(cutter, doc_split_doc):
    """Test merging final chunk."""
    cutter.chunksize = 5
    cutter.merge_threshold = 0.5
    chunks = cutter._split_doc(doc_split_doc, merge_final=True)
    assert len(chunks[-1]) > 5


def test_split_doc_with_overlap(cutter, doc_split_doc):
    """Test splitting with overlap."""
    cutter.chunksize = 5
    cutter.overlap = 2
    chunks = cutter._split_doc(doc_split_doc)
    assert len(chunks[0]) > 5


def test_split_doc_strip_chunks(cutter, nlp):
    """Test stripping whitespace from chunks."""
    doc = nlp("  Text with spaces  ")
    cutter.strip_chunks = True
    chunks = cutter._split_doc(doc)
    # Strip needed to remove trailing whitespace
    assert chunks[0].text.strip() == "Text with spaces"


def test_split_doc_empty(cutter, nlp):
    """Test handling empty document."""
    doc = nlp("")
    with pytest.raises(LexosException, match="Document is empty"):
        cutter._split_doc(doc)


def test_split_doc_single_chunk(cutter, nlp):
    """Test document smaller than chunk size."""
    doc = nlp("Small doc")
    cutter.chunksize = 10
    chunks = cutter._split_doc(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Small doc"


def test_split_doc_properties(cutter):
    """Test preservation of doc properties."""
    nlp_attrs = spacy.load("en_core_web_sm")
    Token.set_extension("custom_attr", default="value", force=True)
    doc = nlp_attrs(
        "This is a test document with multiple tokens for testing chunking mechanisms."
    )
    cutter.chunksize = 5
    chunks = cutter._split_doc(doc)
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert chunks[0][0]._.custom_attr == "value"
    assert all(chunk.has_annotation("SENT_START") for chunk in chunks)


def test_split_doc_returns_span_as_doc(cutter, nlp):
    """Test fallback in _split_doc that converts Span to Doc when strip_chunks is False."""
    doc = nlp("A sentence with words.")
    span = doc[0:3]
    cutter.strip_chunks = False
    cutter._chunk_doc = lambda *_: [span]
    cutter._apply_merge_threshold = lambda chunks, force=False: chunks
    cutter._apply_overlap = lambda chunks: chunks
    chunks = cutter._split_doc(doc)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Doc)
    assert chunks[0].text.strip() == "A sentence with"


def test_split_doc_by_lines_basic(cutter, doc_with_lines):
    """Test basic line splitting."""
    # Parameters to yield the values tested below
    cutter.n = 2
    cutter.merge_threshold = 0.0
    cutter.strip_chunks = False
    # Get chunks
    chunks = cutter._split_doc_by_lines(doc_with_lines)
    # Test chunks
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == "Line1\nLine2\n"
    assert chunks[1].text_with_ws == "Line3\nLine4\n"
    assert chunks[2].text_with_ws == "Line5\n"


def test_split_doc_by_lines_empty(cutter, nlp):
    """Test empty document handling."""
    doc = nlp("")
    with pytest.raises(LexosException, match="Document is empty"):
        cutter._split_doc_by_lines(doc)


def test_split_doc_by_lines_no_newlines(cutter, nlp):
    """Test document with no newlines."""
    doc = nlp("Text without newlines")
    chunks = cutter._split_doc_by_lines(doc)
    assert chunks[0].text == doc.text


def test_split_doc_by_lines_merge_final(cutter, doc_with_lines):
    """Test merging final chunk."""
    cutter.n = 2
    cutter.merge_threshold = 0.5
    cutter.strip_chunks = False
    chunks = cutter._split_doc_by_lines(doc_with_lines, merge_final=True)
    assert len(chunks) == 2


def test_split_doc_by_lines_with_overlap(cutter, doc_with_lines):
    """Test line splitting with overlap."""
    cutter.n = 2
    cutter.merge_threshold = 0.0
    cutter.strip_chunks = False
    cutter.overlap = 1
    chunks = cutter._split_doc_by_lines(doc_with_lines)
    assert "Line3" in chunks[0].text
    assert "Line3" in chunks[1].text
    assert "Line4" in chunks[1].text
    assert "Line5" in chunks[1].text
    assert "Line5" in chunks[2].text


def test_split_doc_by_lines_strip(cutter, nlp):
    """Test stripping whitespace from chunks."""
    doc = nlp("  Line1  \n  Line2  \nLine3  \n")
    cutter.n = 2
    cutter.merge_threshold = 0.0
    cutter.strip_chunks = True
    chunks = cutter._split_doc_by_lines(doc)
    assert chunks[0].text.strip() == "Line1  \n  Line2"


def test_split_doc_by_lines_partial(cutter, nlp):
    """Test handling partial final chunk."""
    doc = nlp("Line1\nLine2\nLine3\n")
    cutter.n = 2
    chunks = cutter._split_doc_by_lines(doc)
    assert len(chunks) == 2
    assert chunks[1].text == "Line3"


def test_split_doc_by_sentences_basic(cutter, doc_with_sentences):
    """Test basic sentence splitting functionality."""
    cutter.n = 2
    chunks = cutter._split_doc_by_sentences(doc_with_sentences)
    assert len(chunks) == 3
    assert len(list(chunks[0].sents)) == 2
    assert len(list(chunks[1].sents)) == 2
    assert len(list(chunks[2].sents)) == 1


def test_split_doc_by_sentences_empty(cutter, nlp):
    """Test empty document handling."""
    doc = nlp("")
    with pytest.raises(LexosException, match="Document is empty."):
        cutter._split_doc_by_sentences(doc)


# def test_split_doc_by_sentences_no_sents(cutter, nlp):
#     """Test handling document with no sentences."""
#     # NOTE: This is hard to test because of an open bug in spaCy: https://github.com/explosion/spaCy/issues/13591
#     pass
#     # nlp.disable_pipe("senter")
#     # doc = nlp("This is a sentence. This is not a third sentence.")
#     # with pytest.raises(LexosException, match="The document has no assigned sentences."):
#     #     cutter._split_doc_by_sentences(doc)


def test_split_doc_by_sentences_single(cutter, doc_with_sentences):
    """Test single sentence document."""
    sents = list(doc_with_sentences.sents)
    doc = sents[0].as_doc()
    cutter.n = 2
    chunks = cutter._split_doc_by_sentences(doc)
    assert len(chunks) == 1
    assert len(list(chunks[0].sents)) == 1


def test_split_doc_by_sentences_merge_final(cutter, doc_with_sentences):
    """Test merging final chunk."""
    cutter.n = 2
    cutter.merge_threshold = 0.5
    chunks = cutter._split_doc_by_sentences(doc_with_sentences, merge_final=True)
    assert len(chunks) == 2


def test_split_doc_by_sentences_with_overlap(cutter, doc_with_sentences):
    """Test sentence splitting with overlap."""
    cutter.n = 2
    cutter.overlap = 1
    chunks = cutter._split_doc_by_sentences(doc_with_sentences)
    assert len(chunks) == 3
    # Check overlap in middle chunk
    assert len(list(chunks[1].sents)) > 2


def test_split_doc_by_sentences_strip(cutter, nlp):
    """Test stripping whitespace from chunks."""
    nlp.add_pipe("sentencizer")
    doc = nlp("  First sentence.  Second sentence.  ")
    cutter.n = 2
    cutter.strip_chunks = True
    chunks = cutter._split_doc_by_sentences(doc)
    assert chunks[0].text.strip() == "First sentence.  Second sentence."


def test_split_doc_by_sentences_custom_n(doc_with_sentences):
    """Test custom number of sentences per chunk."""
    cutter = TokenCutter(n=3)
    chunks = cutter._split_doc_by_sentences(doc_with_sentences)
    assert len(chunks) == 2
    assert len(list(chunks[0].sents)) == 3


def test_split_doc_on_milestones_following(cutter, doc):
    """Test splitting with milestones in following chunks."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    cutter.merge_threshold = 0.0
    chunks = cutter._split_doc_on_milestones(doc, milestones, keep_spans="following")
    assert len(chunks) == 3
    assert chunks[0].text == "The "
    assert chunks[1].text == "quick brown fox "
    assert chunks[2].text == "jumps over the lazy dog."


def test_split_doc_on_milestones_preceding(cutter, doc):
    """Test splitting with milestones in preceding chunks."""
    milestones = [doc[1:2], doc[4:5]]
    cutter.merge_threshold = 0.0
    chunks = cutter._split_doc_on_milestones(doc, milestones, keep_spans="preceding")
    assert len(chunks) == 3
    assert chunks[0].text == "The quick "
    assert chunks[1].text == "brown fox jumps "
    assert chunks[2].text == "over the lazy dog."


def test_split_doc_on_milestones_bool(cutter, doc):
    """Test splitting with boolean keep_spans."""
    milestones = [doc[1:2]]
    cutter.merge_threshold = 0.0
    chunks = cutter._split_doc_on_milestones(doc, milestones, keep_spans=True)
    assert len(chunks) == 3
    assert chunks[1].text == "quick "


def test_split_doc_on_milestones_strip(cutter, nlp):
    """Test stripping whitespace from chunks."""
    doc = nlp("  The  quick  brown  ")
    milestones = [doc[3:4]]  # "quick"
    cutter.merge_threshold = 0.0
    cutter.strip_chunks = True
    chunks = cutter._split_doc_on_milestones(doc, milestones)
    assert chunks[0].text == "The "


def test_split_doc_on_milestones_empty(cutter, nlp):
    """Test empty document handling."""
    doc = nlp("")
    milestones = [doc[3:4]]  # "quick"
    with pytest.raises(LexosException, match="Document is empty."):
        cutter._split_doc_on_milestones(doc, milestones)


def test_split_doc_on_milestones_single_span(cutter, doc):
    """Test splitting with single Span."""
    milestone = doc[1:2]  # "quick"
    cutter.merge_threshold = 0.0
    chunks = cutter._split_doc_on_milestones(doc, milestone)
    assert len(chunks) == 2


def test_split_doc_on_milestones_multiple(cutter, doc):
    """Test splitting with multiple milestones."""
    milestones = [doc[1:2], doc[4:5], doc[7:8]]
    cutter.merge_threshold = 0.0
    chunks = cutter._split_doc_on_milestones(doc, milestones)
    assert len(chunks) == 4


def test_split_doc_on_milestones_no_strip(cutter, nlp):
    """Test _split_doc_on_milestones without stripping chunks (hits final return)."""
    doc = nlp("The quick brown fox jumps over the lazy dog.")
    milestone = doc[2:3]
    cutter.strip_chunks = False
    chunks = cutter._split_doc_on_milestones(doc, milestone)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert "brown" not in chunks[0].text


def test_write_chunk_basic(cutter, doc_for_write_chunk, temp_output_dir):
    """Test basic chunk writing."""
    cutter._write_chunk("test", 1, doc_for_write_chunk, temp_output_dir)
    output_file = temp_output_dir / "test_001.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Test content"


def test_write_chunk_custom_delimiter(cutter, doc_for_write_chunk, temp_output_dir):
    """Test chunk writing with custom delimiter."""
    cutter.delimiter = "-"
    cutter._write_chunk("test", 1, doc_for_write_chunk, temp_output_dir)
    output_file = temp_output_dir / "test-001.txt"
    assert output_file.exists()


def test_write_chunk_custom_padding(cutter, doc_for_write_chunk, temp_output_dir):
    """Test chunk writing with custom padding."""
    cutter.pad = 5
    cutter._write_chunk("test", 1, doc_for_write_chunk, temp_output_dir)
    output_file = temp_output_dir / "test_00001.txt"
    assert output_file.exists()


def test_write_chunk_multiple(cutter, doc_for_write_chunk, temp_output_dir):
    """Test writing multiple chunks."""
    for i in range(3):
        cutter._write_chunk("test", i, doc_for_write_chunk, temp_output_dir)

    files = list(temp_output_dir.glob("*.txt"))
    assert len(files) == 3
    assert all(f.exists() for f in files)


def test_write_chunk_nonexistent_dir(cutter, doc_for_write_chunk, tmp_path):
    """Test writing to nonexistent directory."""
    output_dir = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        cutter._write_chunk("test", 1, doc_for_write_chunk, output_dir)


def test_write_chunk_invalid_path(cutter, doc_for_write_chunk, temp_output_dir):
    """Test writing with invalid path characters."""
    with pytest.raises(OSError):
        cutter._write_chunk("test/invalid", 1, doc_for_write_chunk, temp_output_dir)


def test_write_chunk_empty_content(cutter, nlp, temp_output_dir):
    """Test writing empty chunk."""
    empty_doc = nlp("")
    cutter._write_chunk("test", 1, empty_doc, temp_output_dir)
    output_file = temp_output_dir / "test_001.txt"
    assert output_file.exists()
    assert output_file.read_text() == ""


def test_merge_basic(cutter, sample_chunks_to_merge):
    """Test basic chunk merging."""
    merged = cutter.merge(sample_chunks_to_merge)
    assert isinstance(merged, Doc)
    assert merged.text == "First chunk. Second chunk. Third chunk."


def test_merge_single_chunk(cutter, nlp):
    """Test merging single chunk."""
    chunks = [nlp("Single chunk.")]
    merged = cutter.merge(chunks)
    assert merged.text == "Single chunk."


def test_merge_empty_chunks(cutter):
    """Test merging empty chunks list."""
    with pytest.raises(LexosException, match="No chunks to merge."):
        cutter.merge([])


def test_merge_with_attributes(cutter, nlp):
    """Test merging chunks with different attributes.

    Note: SpaCy will raise a W102 warning that the user data dict
    will be skipped. See https://github.com/explosion/spaCy/discussions/9106.
    """
    doc1 = nlp("Text one")
    doc2 = nlp("Text two")
    doc1.user_data["key"] = "value"
    merged = cutter.merge([doc1, doc2])
    assert isinstance(merged, Doc)
    assert merged.text == "Text one Text two"


def test_merge_maintains_properties(cutter, nlp):
    """Test merged doc maintains required properties."""
    chunks = [nlp("First."), nlp("Second.")]
    merged = cutter.merge(chunks)
    assert merged.has_annotation("SENT_START")
    assert isinstance(merged.vocab, spacy.vocab.Vocab)


def test_save_text_basic(cutter_save, output_dir):
    """Test basic save functionality."""
    cutter_save.save(output_dir, as_text=True)
    files = list(output_dir.glob("*.txt"))
    assert len(files) == 4
    assert (output_dir / "doc001_001.txt").exists()


def test_save_text_custom_names(cutter_save, output_dir):
    """Test save with custom doc names."""
    names = ["test1", "test2"]
    cutter_save.save(output_dir, names=names, as_text=True)
    assert (output_dir / "test1_001.txt").exists()
    assert (output_dir / "test2_001.txt").exists()


def test_save_text_custom_format(cutter_save, output_dir):
    """Test save with custom delimiter and padding."""
    cutter_save.save(output_dir, delimiter="-", pad=4, as_text=True)
    assert (output_dir / "doc0001-0001.txt").exists()


def test_save_text_strip_whitespace(cutter_save, output_dir, nlp):
    """Test whitespace stripping."""
    cutter_save.chunks = [[nlp("  Text with spaces  ")]]
    cutter_save.save(output_dir, strip_chunks=True, as_text=True)
    content = (output_dir / "doc001_001.txt").read_text()
    assert content == "Text with spaces "


def test_save_text_no_chunks(cutter_save, output_dir):
    """Test error when no chunks exist."""
    cutter_save.chunks = []
    with pytest.raises(LexosException, match="No chunks to save."):
        cutter_save.save(output_dir, as_text=True)


def test_save_text_no_output_dir(nlp):
    """Test error when no output directory provided."""
    cutter = TokenCutter()
    cutter.chunks = [[nlp("first")], [nlp("second")]]
    with pytest.raises(ValueError):
        cutter.save(None, as_text=True)


def test_save_text_mismatched_names(cutter_save, nlp, output_dir):
    """Test error when names length doesn't match chunks."""
    cutter_save.names = ["single_name"]
    cutter_save.chunks = [[nlp("first")], [nlp("second")]]
    with pytest.raises(LexosException, match="must equal"):
        cutter_save.save(output_dir, names=["single_name"], as_text=True)


def test_save_text_invalid_path(cutter_save):
    """Test error with invalid output path."""
    with pytest.raises(Exception):
        cutter_save.save("/invalid/path/here", as_text=True)


def test_split_single_doc_chunksize(cutter, doc_for_split):
    """Test splitting single doc by chunk size."""
    cutter.split(doc_for_split, chunksize=3)
    assert len(cutter.chunks) == 1  # One list of chunks
    assert len(cutter.chunks[0]) == 6  # Number of chunks


def test_split_multiple_docs(cutter, nlp):
    """Test splitting multiple docs."""
    docs = [nlp("Doc one."), nlp("Doc two.")]
    cutter.split(docs, chunksize=2)
    assert len(cutter.chunks) == 2


def test_split_by_n(cutter, doc_for_split):
    """Test splitting into n chunks."""
    cutter.split(doc_for_split, n=3)
    assert len(cutter.chunks[0]) == 3


def test_split_with_overlap(cutter, doc_for_split):
    """Test splitting with overlap."""
    cutter.split(doc_for_split, chunksize=5, overlap=2)
    assert len(cutter.chunks[0][0]) > 5


def test_split_with_merge_threshold(cutter, doc_for_split):
    """Test splitting with merge threshold."""
    cutter.split(doc_for_split, chunksize=3, merge_threshold=0.5)
    assert len(cutter.chunks[0][-1]) >= 3


def test_split_strip_chunks(cutter, nlp):
    """Test stripping whitespace from chunks."""
    doc = nlp("  Text with spaces  ")
    cutter.split(doc, chunksize=2, strip_chunks=True)
    chunks = cutter.chunks
    assert chunks[0][0].text.strip() == "Text"


def test_split_custom_names(cutter, doc_for_split):
    """Test splitting with custom doc names."""
    names = ["test1"]
    _ = cutter.split(doc_for_split, chunksize=5, names=names)
    assert cutter.names == names


def test_split_by_newline_basic(cutter, doc_with_lines):
    """Test basic line splitting with n=2 using newline=True."""
    cutter.split(doc_with_lines, n=2, newline=True)
    assert len(cutter.chunks) == 1  # One document's chunks
    assert len(cutter.chunks[0]) == 3  # Three chunks of two lines each


def test_split_by_newline_invalid_n_runtime(cutter, nlp):
    """Test LexosException is raised when n <= 0 at runtime with newline=True."""
    doc = nlp("Line 1\nLine 2\nLine 3")
    with pytest.raises(LexosException, match="n must be greater than 0."):
        cutter.split(doc, n=0, newline=True)


def test_split_by_newline_multiple_docs(cutter, nlp):
    """Test splitting multiple docs with newline=True."""
    docs = [nlp("Line1\nLine2\n"), nlp("Line3\nLine4\n")]
    cutter.split(docs, n=1, newline=True)
    assert len(cutter.chunks) == 2
    assert len(cutter.chunks[0]) == 2


def test_split_by_newline_merge_threshold(cutter, doc_with_lines):
    """Test merge threshold functionality with newline=True."""
    cutter.split(doc_with_lines, n=2, newline=True, merge_threshold=0.75)
    assert len(cutter.chunks[0][0]) == 3  # Has internal line breaks
    assert (
        len(cutter.chunks[0][1]) == 5
    )  # Merged last two chunks, has internal line breaks


def test_split_by_newline_overlap(cutter, doc_with_lines):
    """Test overlap functionality with newline=True."""
    cutter.split(doc_with_lines, n=2, newline=True, overlap=1)
    assert cutter.chunks[0][0][-1].text.strip() == cutter.chunks[0][1][0].text.strip()


def test_split_by_newline_strip(cutter, nlp):
    """Test stripping whitespace with newline=True."""
    doc = nlp("  Line1  \n  Line2  \n")
    cutter.split(doc, n=1, newline=True, strip_chunks=True)
    assert cutter.chunks[0][0].text.strip() == "Line1"


def test_split_by_newline_custom_names(cutter, doc_with_lines):
    """Test custom doc names with newline=True."""
    cutter.split(doc_with_lines, n=2, newline=True, names=["test1"])
    assert cutter.names == ["test1"]


def test_split_by_newline_no_newlines(cutter, nlp):
    """Test document with no line breaks using newline=True."""
    doc = nlp("No line breaks here")
    cutter.split(doc, n=1, newline=True)
    assert len(cutter.chunks[0]) == 1


def test_split_by_newline_merge_final(cutter, doc_with_lines):
    """Test merge_final parameter with newline=True."""
    cutter.split(doc_with_lines, n=2, newline=True, merge_final=True)
    assert len(cutter.chunks) == 1
    # When merge_final=True, last chunks should be merged
    assert len(cutter.chunks[0]) <= 3


def test_split_on_milestones_basic(cutter, doc):
    """Test basic milestone splitting."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    cutter.split_on_milestones(milestones, doc)
    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) > 1


def test_split_on_milestones_multiple_docs(cutter, nlp):
    """Test splitting multiple docs."""
    docs = [nlp("First text"), nlp("Second text")]
    milestone = nlp("text")[0:1]
    cutter.split_on_milestones(milestone, docs)
    assert len(cutter.chunks) == 2


def test_split_on_milestones_keep_spans(cutter, doc, milestone_doc):
    """Test different keep_spans options."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    cutter.chunksize = 3
    cutter.merge_threshold = 0.0
    cutter.split_on_milestones(milestones, doc, keep_spans=True)
    # The, quick, brown fox, jumps, over the lazy dog.
    assert cutter.chunks[0][1].text.strip() == "quick"
    assert cutter.chunks[0][3].text.strip() == "jumps"
    cutter.chunks = []  # Reset chunks
    cutter.split_on_milestones(milestones, doc, keep_spans=False)
    assert cutter.chunks[0][1].text.strip() == "brown fox"
    assert cutter.chunks[0][2].text.strip() == "over the lazy dog."
    cutter.chunks = []  # Reset chunks
    cutter.split_on_milestones(milestones, doc, keep_spans="following")
    assert cutter.chunks[0][0].text.strip() == "The"
    assert cutter.chunks[0][1].text.strip() == "quick brown fox"
    assert cutter.chunks[0][2].text.strip() == "jumps over the lazy dog."
    cutter.chunks = []  # Reset chunks
    cutter.split_on_milestones(milestones, doc, keep_spans="preceding")
    # The quick, brown fox jumps, over the lazy dog.
    assert cutter.chunks[0][0].text.strip() == "The quick"
    assert cutter.chunks[0][1].text.strip() == "brown fox jumps"
    assert cutter.chunks[0][2].text.strip() == "over the lazy dog."


def test_split_on_milestones_merge_threshold(cutter, doc, milestone_doc):
    """Test merge threshold functionality."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    cutter.chunksize = 3
    # Cannot find a threshold that works, so need to test with merge_final
    cutter.split_on_milestones(milestones, doc, merge_final=True)
    assert cutter.chunks[0][0].text.strip() == "The"
    assert cutter.chunks[0][1].text.strip() == "brown fox over the lazy dog."


def test_split_on_milestones_overlap(cutter, doc, milestone_doc):
    """Test overlap functionality."""
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"
    cutter.chunksize = 3
    # Cannot find a threshold that works, so need to test with merge_final
    cutter.split_on_milestones(milestones, doc, merge_final=True, overlap=1)
    assert cutter.chunks[0][0].text.strip() == "The brown"
    assert cutter.chunks[0][1].text.strip() == "brown fox over the lazy dog."


def test_split_on_milestones_strip(cutter, nlp):
    """Test strip chunks functionality."""
    doc = nlp("  Text with spaces  ")
    milestone = nlp("with")[0:1]
    cutter.split_on_milestones(milestone, doc, strip_chunks=True)
    assert cutter.chunks[0][0].text.strip() == "Text with spaces"


def test_split_on_milestones_custom_names(cutter, doc, nlp):
    """Test custom doc names."""
    milestone = nlp("with")[0:1]
    names = ["test1"]
    cutter.split_on_milestones(milestone, doc, names=names)
    assert cutter.names == names


def test_split_on_sentences_basic(cutter, doc_with_sentences2):
    """Test basic sentence splitting."""
    cutter.n = 2
    cutter.split_on_sentences(doc_with_sentences2, n=2)
    assert len(cutter.chunks) == 1  # One document
    assert len(cutter.chunks[0]) == 3  # Three chunks of two sentences each


def test_split_on_sentences_no_sentence_boundaries():
    """Test LexosException when doc.sents is not available."""
    from spacy.lang.en import English

    nlp = English()  # No sentencizer or parser
    doc = nlp("This is one sentence. This is another.")
    cutter = TokenCutter()
    with pytest.raises(
        LexosException, match="does not have sentence boundary detection"
    ):
        cutter.split_on_sentences(doc, n=1)


def test_split_doc_by_sentences_no_sents(cutter):
    """Test _split_doc_by_sentences raises when no sentence boundaries are assigned."""
    from spacy.lang.en import English

    nlp = English()
    doc = nlp("This is one sentence. This is another.")
    cutter.n = 1
    with pytest.raises(LexosException, match="The document has no assigned sentences."):
        cutter._split_doc_by_sentences(doc)


def test_split_doc_by_sentences_final_return(cutter, nlp):
    """Test _split_doc_by_sentences hits final return (strip_chunks=False)."""
    nlp.add_pipe("sentencizer")
    doc = nlp("First sentence. Second sentence. Third sentence.")
    cutter.n = 2
    cutter.strip_chunks = False
    chunks = cutter._split_doc_by_sentences(doc)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Doc) for chunk in chunks)
    assert len(chunks) >= 1


def test_split_on_sentences_multiple_docs(cutter, nlp):
    """Test splitting multiple docs."""
    nlp.add_pipe("sentencizer")
    docs = [
        nlp("First sentence. Second sentence."),
        nlp("Third sentence. Fourth sentence."),
    ]
    cutter.n = 2
    cutter.split_on_sentences(docs, n=1)
    assert len(cutter.chunks) == 2
    assert all(len(doc_chunks) == 2 for doc_chunks in cutter.chunks)


def test_split_on_sentences_merge_final(cutter, doc_with_sentences2):
    """Test merge final functionality."""
    cutter.split_on_sentences(doc_with_sentences2, n=2, merge_final=True)
    assert len(cutter.chunks[0]) == 2


def test_split_on_sentences_overlap(cutter, doc_with_sentences2):
    """Test overlap functionality."""
    cutter.split_on_sentences(doc_with_sentences2, n=2, overlap=1)
    assert cutter.chunks[0][0].text.strip() == "First sentence. Second sentence. Third"
    assert cutter.chunks[0][1].text.strip() == "Third sentence. Fourth sentence. Fifth"
    assert cutter.chunks[0][2].text.strip() == "Fifth sentence."


def test_split_on_sentences_strip(cutter, nlp):
    """Test stripping whitespace."""
    nlp.add_pipe("sentencizer")
    doc = nlp("  First sentence.  Second sentence.  ")
    cutter.split_on_sentences(doc, n=1, strip_chunks=True)
    assert cutter.chunks[0][0].text.strip() == "First sentence."


def test_split_on_sentences_custom_names(cutter, doc_with_sentences2):
    """Test custom doc names."""
    cutter.split_on_sentences(doc_with_sentences2, n=2, names=["test1"])
    assert cutter.names == ["test1"]


def test_to_dict_default_names(cutter_with_chunks):
    """Test dictionary conversion with default names."""
    result = cutter_with_chunks.to_dict()
    assert len(result) == 2
    assert "doc001" in result
    assert "doc002" in result
    assert len(result["doc001"]) == 2


def test_to_dict_custom_names(cutter_with_chunks):
    """Test dictionary conversion with custom names."""
    names = ["doc1", "doc2"]
    result = cutter_with_chunks.to_dict(names)
    assert "doc1" in result
    assert "doc2" in result


def test_to_dict_empty_chunks(cutter_with_chunks):
    """Test dictionary conversion with empty chunks."""
    cutter_with_chunks.chunks = []
    result = cutter_with_chunks.to_dict()
    assert result == {}


def test_to_dict_padding(cutter_with_chunks):
    """Test name padding in default names."""
    cutter_with_chunks.pad = 4
    result = cutter_with_chunks.to_dict()
    assert "doc0001" in result


def test_list_start_end_indexes_basic():
    """Test basic functionality with multiple arrays."""
    arrays = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])]
    result = TokenCutter.list_start_end_indexes(arrays)
    assert result == [(0, 3), (3, 5), (5, 9)]


def test_list_start_end_indexes_single_array():
    """Test functionality with a single array."""
    arrays = [np.array([1, 2, 3])]
    result = TokenCutter.list_start_end_indexes(arrays)
    assert result == [(0, 3)]


def test_list_start_end_indexes_empty():
    """Test functionality with an empty list."""
    arrays = []
    result = TokenCutter.list_start_end_indexes(arrays)
    assert result == []


def test_list_start_end_indexes_different_lengths():
    """Test functionality with arrays of different lengths."""
    arrays = [np.array([1]), np.array([2, 3, 4]), np.array([5, 6])]
    result = TokenCutter.list_start_end_indexes(arrays)
    assert result == [(0, 1), (1, 4), (4, 6)]


def test_list_start_end_indexes_validation():
    """Validate start and end indexes."""
    arrays = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
    result = TokenCutter.list_start_end_indexes(arrays)
    assert result == [(0, 2), (2, 5), (5, 6)]


def test_split_from_file(cutter, nlp, tmp_path):
    """Test loading and splitting spaCy Doc from file using Doc.from_disk()."""
    # Create a doc and save it to disk
    doc = nlp("This is a test document with multiple tokens for testing.")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    # Load and split using file=True with model (using blank:en for blank English model)
    cutter.split(doc_path, chunksize=5, file=True, model="blank:en")

    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) >= 2
    assert all(isinstance(chunk, Doc) for chunk in cutter.chunks[0])


def test_split_from_multiple_files(cutter, nlp, tmp_path):
    """Test loading and splitting multiple spaCy Docs from files."""
    # Create and save multiple docs
    doc1 = nlp("First document with some text.")
    doc2 = nlp("Second document with more text.")

    doc1_path = tmp_path / "doc1"
    doc2_path = tmp_path / "doc2"

    doc1.to_disk(doc1_path)
    doc2.to_disk(doc2_path)

    # Load and split using file=True with model (using blank:en for blank English model)
    cutter.split([doc1_path, doc2_path], chunksize=3, file=True, model="blank:en")

    assert len(cutter.chunks) == 2
    assert all(isinstance(chunk, Doc) for chunks in cutter.chunks for chunk in chunks)


def test_split_file_without_model(cutter, nlp, tmp_path):
    """Test that split() raises exception when file=True but model is not provided."""
    doc = nlp("Test document")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    with pytest.raises(
        LexosException, match="model parameter is required when file=True"
    ):
        cutter.split(doc_path, file=True)


def test_split_on_milestones_from_file(cutter, nlp, tmp_path):
    """Test split_on_milestones with file loading."""
    doc = nlp("The quick brown fox jumps over the lazy dog")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    # Create milestones using the same doc
    milestones = [doc[1:2], doc[4:5]]  # "quick", "jumps"

    cutter.split_on_milestones(milestones, doc_path, file=True, model="blank:en")
    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) > 0


def test_split_on_milestones_file_without_model(cutter, nlp, tmp_path):
    """Test that split_on_milestones raises exception when file=True but model is not provided."""
    doc = nlp("Test document")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    milestones = [doc[0:1]]
    with pytest.raises(
        LexosException, match="model parameter is required when file=True"
    ):
        cutter.split_on_milestones(milestones, doc_path, file=True)


def test_split_on_sentences_from_file(cutter, nlp, tmp_path):
    """Test split_on_sentences with file loading."""
    nlp.add_pipe("sentencizer")
    doc = nlp("First sentence. Second sentence. Third sentence.")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    cutter.split_on_sentences(doc_path, n=2, file=True, model="blank:en")
    assert len(cutter.chunks) == 1


def test_split_on_sentences_file_without_model(cutter, nlp, tmp_path):
    """Test that split_on_sentences raises exception when file=True but model is not provided."""
    nlp.add_pipe("sentencizer")
    doc = nlp("First sentence. Second sentence.")
    doc_path = tmp_path / "test_doc"
    doc.to_disk(doc_path)

    with pytest.raises(
        LexosException, match="model parameter is required when file=True"
    ):
        cutter.split_on_sentences(doc_path, n=1, file=True)


def test_split_on_sentences_n_defaults_to_chunksize(cutter, nlp):
    """Test that n defaults to chunksize when not provided in split_on_sentences."""
    nlp.add_pipe("sentencizer")
    doc = nlp(
        "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    )

    cutter.chunksize = 2
    cutter.split_on_sentences(doc)  # n not provided

    # Should use chunksize=2 as n
    assert len(cutter.chunks) == 1
    assert len(cutter.chunks[0]) >= 2


def test_split_on_sentences_invalid_n(cutter, nlp):
    """Test that split_on_sentences raises exception when n is invalid after defaulting."""
    nlp.add_pipe("sentencizer")
    doc = nlp("First sentence. Second sentence.")

    cutter.chunksize = None
    with pytest.raises(LexosException, match="n must be greater than 0"):
        cutter.split_on_sentences(doc)


def test_save_as_binary(cutter_save, output_dir):
    """Test saving chunks as binary spaCy Doc objects (as_text=False)."""
    cutter_save.save(output_dir, as_text=False)

    # Check that files were created (should be .spacy files or similar)
    files = list(output_dir.glob("*"))
    assert len(files) == 4

    # Verify files exist
    for f in files:
        assert f.exists()


def test_split_no_docs_after_init():
    """Test that split() raises exception when cutter has no docs after initialization."""
    cutter = TokenCutter()
    # cutter.docs is None by default
    with pytest.raises(LexosException, match="No documents provided for splitting"):
        cutter.split()


def test_split_from_invalid_file(cutter, nlp, tmp_path):
    """Test that split() raises exception when loading invalid doc file."""
    # Create a text file (not a valid spaCy serialized doc)
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("This is just text, not a spaCy doc")

    with pytest.raises(LexosException, match="Error loading doc from disk"):
        cutter.split(invalid_file, file=True, model="blank:en")


def test_split_on_milestones_no_docs_after_init(nlp):
    """Test that split_on_milestones() raises exception when cutter has no docs after initialization."""
    cutter = TokenCutter()
    doc = nlp("Test document")
    milestones = [doc[0:1]]
    # cutter.docs is None by default
    with pytest.raises(LexosException, match="No documents provided for splitting"):
        cutter.split_on_milestones(milestones)
