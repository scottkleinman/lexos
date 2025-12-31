"""test_whitespace_counter.py.

Coverage: 100%
Last Updated: December 20, 2025
"""

import pytest
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.tokenizer.whitespace_counter import WhitespaceCounter


@pytest.fixture
def whitespace_counter() -> WhitespaceCounter:
    """Return a WhitespaceCounter instance."""
    return WhitespaceCounter()


@pytest.fixture
def sample_text() -> str:
    """Return a sample text with varied whitespace."""
    return """  to be
                                      gone a
                     constant desire
        embarrassed for the
 giant leaning in for love
 we had enough
of
 the dance number but                 the whirling begins"""


@pytest.fixture
def simple_text() -> str:
    """Return a simple text for basic testing."""
    return "hello  world\nnew line"


class TestGetTokenWidths:
    """Test the _get_token_widths method."""

    def test_single_word(self, whitespace_counter):
        """Test tokenization of a single word."""
        tokens, widths = whitespace_counter._get_token_widths("hello")
        assert tokens == ["hello"]
        assert widths == [5]

    def test_two_words_single_space(self, whitespace_counter):
        """Test tokenization of two words separated by a single space."""
        tokens, widths = whitespace_counter._get_token_widths("hello world")
        assert tokens == ["hello", " ", "world"]
        assert widths == [5, 1, 5]

    def test_two_words_multiple_spaces(self, whitespace_counter):
        """Test tokenization of two words separated by multiple spaces."""
        tokens, widths = whitespace_counter._get_token_widths("hello   world")
        assert tokens == ["hello", " ", "world"]
        assert widths == [5, 3, 5]

    def test_line_break(self, whitespace_counter):
        """Test tokenization with a line break."""
        tokens, widths = whitespace_counter._get_token_widths("hello\nworld")
        assert tokens == ["hello", "\n", "world"]
        assert widths == [5, 1, 5]

    def test_leading_spaces(self, whitespace_counter):
        """Test tokenization with leading spaces."""
        tokens, widths = whitespace_counter._get_token_widths("  hello")
        assert tokens == [" ", "hello"]
        assert widths == [2, 5]

    def test_trailing_spaces(self, whitespace_counter):
        """Test tokenization with trailing spaces."""
        tokens, widths = whitespace_counter._get_token_widths("hello  ")
        assert tokens == ["hello", " "]
        assert widths == [5, 2]

    def test_complex_whitespace(self, whitespace_counter):
        """Test tokenization with complex whitespace patterns."""
        text = "  word1\n    word2     word3"
        tokens, widths = whitespace_counter._get_token_widths(text)
        assert tokens == [" ", "word1", "\n", " ", "word2", " ", "word3"]
        assert widths == [2, 5, 1, 4, 5, 5, 5]

    def test_empty_string(self, whitespace_counter):
        """Test tokenization of an empty string."""
        tokens, widths = whitespace_counter._get_token_widths("")
        assert tokens == []
        assert widths == []

    def test_only_spaces(self, whitespace_counter):
        """Test tokenization of only spaces."""
        tokens, widths = whitespace_counter._get_token_widths("     ")
        assert tokens == [" "]
        assert widths == [5]

    def test_only_newlines(self, whitespace_counter):
        """Test tokenization of only newlines."""
        tokens, widths = whitespace_counter._get_token_widths("\n\n\n")
        assert tokens == ["\n", "\n", "\n"]
        assert widths == [1, 1, 1]

    def test_mixed_whitespace_no_words(self, whitespace_counter):
        """Test tokenization with mixed whitespace but no words."""
        tokens, widths = whitespace_counter._get_token_widths("  \n  \n")
        assert tokens == [" ", "\n", " ", "\n"]
        assert widths == [2, 1, 2, 1]


class TestMakeDoc:
    """Test the make_doc method."""

    def test_make_doc_basic(self, whitespace_counter, simple_text):
        """Test basic doc creation."""
        doc = whitespace_counter.make_doc(simple_text)
        assert isinstance(doc, Doc)

    def test_make_doc_width_extension(self, whitespace_counter, simple_text):
        """Test that width extension is added to tokens."""
        doc = whitespace_counter.make_doc(simple_text)
        for token in doc:
            assert hasattr(token._, "width")
            assert isinstance(token._.width, int)

    def test_make_doc_token_widths(self, whitespace_counter):
        """Test that token widths are correctly assigned."""
        text = "hello  world"
        doc = whitespace_counter.make_doc(text)
        expected_widths = [5, 2, 5]
        actual_widths = [token._.width for token in doc]
        assert actual_widths == expected_widths

    def test_make_doc_preserves_line_breaks(self, whitespace_counter):
        """Test that line breaks are preserved as tokens."""
        text = "line1\nline2"
        doc = whitespace_counter.make_doc(text)
        tokens_text = [token.text for token in doc]
        assert "\n" in tokens_text

    def test_make_doc_newline_width(self, whitespace_counter):
        """Test that newline tokens have width of 1."""
        text = "line1\nline2"
        doc = whitespace_counter.make_doc(text)
        for token in doc:
            if token.text == "\n":
                assert token._.width == 1

    def test_make_doc_with_disable(self, whitespace_counter):
        """Test make_doc with disabled pipeline components."""
        text = "hello world"
        # Use 'senter' which is an actual component in xx_sent_ud_sm
        doc = whitespace_counter.make_doc(text, disable=["senter"])
        assert isinstance(doc, Doc)

    def test_make_doc_with_max_length(self, whitespace_counter):
        """Test make_doc with max_length parameter."""
        text = "short text"
        doc = whitespace_counter.make_doc(text, max_length=1000000)
        assert isinstance(doc, Doc)

    def test_make_doc_sample_text(self, whitespace_counter, sample_text):
        """Test make_doc with complex sample text."""
        doc = whitespace_counter.make_doc(sample_text)
        assert isinstance(doc, Doc)
        # Check that we have both words and whitespace tokens
        has_words = any(token.text not in [" ", "\n"] for token in doc)
        has_spaces = any(token.text == " " for token in doc)
        has_newlines = any(token.text == "\n" for token in doc)
        assert has_words
        assert has_spaces
        assert has_newlines


class TestMakeDocs:
    """Test the make_docs method."""

    def test_make_docs_basic(self, whitespace_counter):
        """Test basic docs creation from multiple texts."""
        texts = ["hello world", "foo bar"]
        docs = list(whitespace_counter.make_docs(texts))
        assert len(docs) == 2
        assert all(isinstance(doc, Doc) for doc in docs)

    def test_make_docs_width_extension(self, whitespace_counter):
        """Test that width extension is added to all docs."""
        texts = ["hello  world", "foo\nbar"]
        docs = list(whitespace_counter.make_docs(texts))
        for doc in docs:
            for token in doc:
                assert hasattr(token._, "width")

    def test_make_docs_preserves_whitespace(self, whitespace_counter):
        """Test that whitespace is preserved across multiple docs."""
        texts = ["hello  world", "foo\nbar"]
        docs = list(whitespace_counter.make_docs(texts))
        # First doc should have double space
        widths_0 = [token._.width for token in docs[0]]
        assert 2 in widths_0  # Double space
        # Second doc should have newline
        tokens_1 = [token.text for token in docs[1]]
        assert "\n" in tokens_1

    def test_make_docs_chunking(self, whitespace_counter):
        """Test that chunking works with small chunk size."""
        texts = [f"text {i}" for i in range(10)]
        docs = list(whitespace_counter.make_docs(texts, chunk_size=3))
        assert len(docs) == 10
        assert all(isinstance(doc, Doc) for doc in docs)

    def test_make_docs_large_chunk(self, whitespace_counter):
        """Test processing with chunk size larger than input."""
        texts = ["hello world", "foo bar"]
        docs = list(whitespace_counter.make_docs(texts, chunk_size=100))
        assert len(docs) == 2

    def test_make_docs_single_text(self, whitespace_counter):
        """Test make_docs with a single text."""
        texts = ["hello world"]
        docs = list(whitespace_counter.make_docs(texts))
        assert len(docs) == 1
        assert isinstance(docs[0], Doc)

    def test_make_docs_empty_list(self, whitespace_counter):
        """Test make_docs with an empty list."""
        texts = []
        docs = list(whitespace_counter.make_docs(texts))
        assert docs == []

    def test_make_docs_with_disable(self, whitespace_counter):
        """Test make_docs with disabled pipeline components."""
        texts = ["hello world", "foo bar"]
        docs = list(whitespace_counter.make_docs(texts, disable=["tagger"]))
        assert len(docs) == 2
        assert all(isinstance(doc, Doc) for doc in docs)

    def test_make_docs_with_max_length(self, whitespace_counter):
        """Test make_docs with max_length parameter."""
        texts = ["short text", "another short text"]
        docs = list(whitespace_counter.make_docs(texts, max_length=1000000))
        assert len(docs) == 2

    def test_make_docs_generator(self, whitespace_counter):
        """Test that make_docs returns a generator."""
        texts = ["hello world", "foo bar"]
        result = whitespace_counter.make_docs(texts)
        # Check that it's iterable without converting to list first
        first_doc = next(result)
        assert isinstance(first_doc, Doc)

    def test_make_docs_chunk_boundary(self, whitespace_counter):
        """Test that chunk boundaries don't affect output."""
        texts = [f"text {i}  spaced" for i in range(5)]
        # Process with different chunk sizes
        docs_chunk_2 = list(whitespace_counter.make_docs(texts, chunk_size=2))
        docs_chunk_5 = list(whitespace_counter.make_docs(texts, chunk_size=5))
        # Results should be the same
        assert len(docs_chunk_2) == len(docs_chunk_5) == 5
        for doc2, doc5 in zip(docs_chunk_2, docs_chunk_5):
            widths_2 = [t._.width for t in doc2]
            widths_5 = [t._.width for t in doc5]
            assert widths_2 == widths_5

    def test_make_docs_exact_chunk_size(self, whitespace_counter):
        """Test chunking when number of texts exactly matches chunk size."""
        # This tests line 115 (yield chunk) in the chunker when len(chunk) == size
        texts = [f"text {i}" for i in range(6)]
        docs = list(whitespace_counter.make_docs(texts, chunk_size=3))
        assert len(docs) == 6
        assert all(isinstance(doc, Doc) for doc in docs)

    def test_make_docs_chunk_size_one(self, whitespace_counter):
        """Test chunking with chunk_size=1 to exercise yield in chunker."""
        # This ensures line 115 is hit for every single text
        texts = ["hello world", "foo bar", "baz qux"]
        docs = list(whitespace_counter.make_docs(texts, chunk_size=1))
        assert len(docs) == 3
        assert all(isinstance(doc, Doc) for doc in docs)

    def test_make_docs_registers_width_extension(self):
        """Test that make_docs registers the width extension if not already registered."""
        from spacy.tokens import Token

        # Remove the extension if it exists
        if Token.has_extension("width"):
            Token.remove_extension("width")

        # Create a fresh instance and call make_docs
        # This will execute lines 114-115 to register the extension
        counter = WhitespaceCounter()
        texts = ["hello world"]
        docs = list(counter.make_docs(texts))

        # Verify the extension was registered and works
        assert Token.has_extension("width")
        assert len(docs) == 1
        assert all(hasattr(token._, "width") for token in docs[0])


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_long_spaces(self, whitespace_counter):
        """Test handling of very long runs of spaces."""
        text = "word1" + " " * 100 + "word2"
        tokens, widths = whitespace_counter._get_token_widths(text)
        assert tokens == ["word1", " ", "word2"]
        assert widths == [5, 100, 5]

    def test_consecutive_newlines(self, whitespace_counter):
        """Test handling of consecutive newlines."""
        text = "line1\n\n\nline2"
        tokens, widths = whitespace_counter._get_token_widths(text)
        assert tokens.count("\n") == 3
        assert all(widths[i] == 1 for i, t in enumerate(tokens) if t == "\n")

    def test_unicode_characters(self, whitespace_counter):
        """Test handling of Unicode characters."""
        text = "café  naïve"
        doc = whitespace_counter.make_doc(text)
        assert isinstance(doc, Doc)
        tokens_text = [token.text for token in doc]
        assert "café" in tokens_text
        assert "naïve" in tokens_text

    def test_punctuation(self, whitespace_counter):
        """Test handling of punctuation."""
        text = "hello,  world!"
        tokens, widths = whitespace_counter._get_token_widths(text)
        # Punctuation is treated as part of words
        assert "hello," in tokens
        assert "world!" in tokens

    def test_tabs_treated_as_whitespace(self, whitespace_counter):
        """Test that tabs are treated as whitespace."""
        text = "hello\tworld"
        tokens, widths = whitespace_counter._get_token_widths(text)
        # Tab is whitespace, so it splits the tokens
        assert len(tokens) == 2
        assert "hello" in tokens
        assert "world" in tokens

    def test_extension_already_registered(self, whitespace_counter):
        """Test that extension registration is idempotent."""
        text = "hello world"
        doc1 = whitespace_counter.make_doc(text)
        doc2 = whitespace_counter.make_doc(text)
        assert hasattr(doc1[0]._, "width")
        assert hasattr(doc2[0]._, "width")


class TestModelImportError:
    """Test handling of model import errors."""

    def test_import_error_raises_lexos_exception(self):
        """Test that ImportError when loading default model raises LexosException."""
        # We need to test the module-level import error that occurs
        # when spacy.load fails. This requires reloading the module
        # with a mocked spacy.load that raises ImportError.
        import sys
        from unittest.mock import patch

        # Remove the module from cache to force reimport
        if "lexos.tokenizer.whitespace_counter" in sys.modules:
            del sys.modules["lexos.tokenizer.whitespace_counter"]

        # Mock spacy.load to raise ImportError
        with patch("spacy.load", side_effect=ImportError("Model not found")):
            with pytest.raises(
                LexosException,
                match="The default model is not available. Please run `python -m spacy download xx_sent_ud_sm`",
            ):
                # This will trigger the module-level code that calls spacy.load
                import lexos.tokenizer.whitespace_counter  # noqa: F401

        # Clean up: reload the module properly for other tests
        if "lexos.tokenizer.whitespace_counter" in sys.modules:
            del sys.modules["lexos.tokenizer.whitespace_counter"]
        import lexos.tokenizer.whitespace_counter  # noqa: F401
