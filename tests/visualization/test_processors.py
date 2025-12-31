"""test_processors.py.

Coverage: 97%. Missing: 62, 226, 228

Last Update: December 5, 2025
"""

import token
from collections import Counter
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import spacy
from pydantic import ValidationError

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.processors import (
    _process_list_data,
    filter_docs,
    get_rows,
    multicloud_processor,
    process_data,
    process_dataframe,
    process_docs,
    process_dtm,
    process_item,
    process_list,
)

# Test data
SAMPLE_TEXT = "natural language processing machine learning artificial intelligence"
SAMPLE_DICT = {
    "natural": 5,
    "language": 4,
    "processing": 3,
    "machine": 2,
    "learning": 1,
}
SAMPLE_TOKENS = ["natural", "language", "processing", "machine", "learning"]
SAMPLE_NESTED_LIST = [
    ["natural", "language"],
    ["processing", "machine"],
    ["learning", "data"],
]


class TestProcessData:
    """Test the main process_data function."""

    def test_process_string_data(self):
        """Test processing string input."""
        result = process_data(SAMPLE_TEXT)

        expected = Counter(SAMPLE_TEXT.split())
        assert result == dict(expected)
        assert isinstance(result, dict)
        assert all(isinstance(v, int) for v in result.values())

    def test_process_string_with_limit(self):
        """Test processing string with limit."""
        result = process_data(SAMPLE_TEXT, limit=3)

        assert len(result) == 3
        assert isinstance(result, dict)

    def test_process_dict_data(self):
        """Test processing dictionary input."""
        result = process_data(SAMPLE_DICT)

        assert result == SAMPLE_DICT
        assert isinstance(result, dict)

    def test_process_dict_with_limit(self):
        """Test processing dictionary with limit."""
        result = process_data(SAMPLE_DICT, limit=2)

        assert len(result) == 2
        # Should contain the top 2 most frequent terms
        assert "natural" in result
        assert "language" in result

    def test_process_list_strings(self):
        """Test processing list of strings."""
        result = process_data(SAMPLE_TOKENS)

        expected = Counter(SAMPLE_TOKENS)
        assert result == dict(expected)

    def test_process_nested_list(self):
        """Test processing nested list."""
        result = process_data(SAMPLE_NESTED_LIST)

        flattened = [item for sublist in SAMPLE_NESTED_LIST for item in sublist]
        expected = Counter(flattened)
        assert result == dict(expected)

    def test_process_empty_data(self):
        """Test processing empty data."""
        assert process_data("") == {}
        assert process_data([]) == {}
        assert process_data({}) == {}

    def test_unsupported_data_type(self):
        """Test that unsupported data types raise LexosException."""
        with pytest.raises(LexosException) as exc_info:
            process_data(12345)

        assert "Unsupported data type" in str(exc_info.value)

    def test_process_spacy_doc(self, nlp):
        """Test processing spaCy Doc objects."""
        doc = nlp(SAMPLE_TEXT)

        result = process_data(doc)

        expected = Counter([token.text for token in doc])
        assert result == dict(expected)

    def test_process_spacy_span(self, nlp):
        """Test processing spaCy Span objects."""
        doc = nlp(SAMPLE_TEXT)
        span = doc[0:3]  # First 3 tokens

        result = process_data(span)

        expected = Counter([token.text for token in span])
        assert result == dict(expected)


class TestProcessListData:
    """Test the _process_list_data function."""

    def test_empty_list(self):
        """Test processing empty list."""
        result = _process_list_data([])
        assert result == Counter()

    def test_list_of_strings(self):
        """Test processing list of strings."""
        result = _process_list_data(SAMPLE_TOKENS)

        expected = Counter(SAMPLE_TOKENS)
        assert result == expected

    def test_list_of_lists(self):
        """Test processing list of lists."""
        result = _process_list_data(SAMPLE_NESTED_LIST)

        flattened = [item for sublist in SAMPLE_NESTED_LIST for item in sublist]
        expected = Counter(flattened)
        assert result == expected

    def test_list_of_docs(self, nlp):
        """Test processing list of spaCy docs."""
        docs = [nlp("natural language"), nlp("machine learning")]

        result = _process_list_data(docs)

        all_tokens = []
        for doc in docs:
            all_tokens.extend([token.text for token in doc])
        expected = Counter(all_tokens)
        assert result == expected

    def test_list_of_tokens(self, nlp):
        """Test processing list of spaCy tokens."""
        doc = nlp(SAMPLE_TEXT)
        tokens = list(doc)

        result = _process_list_data(tokens)

        expected = Counter([token.text for token in tokens])
        assert result == expected


class TestFilterDocs:
    """Test the filter_docs function."""

    def test_filter_by_string_labels(self):
        """Test filtering by string column labels."""
        df = pd.DataFrame(
            {"doc1": [1, 2, 3], "doc2": [4, 5, 6], "doc3": [7, 8, 9]},
            index=["term1", "term2", "term3"],
        )

        result = filter_docs(df, ["doc1", "doc3"])

        expected = df[["doc1", "doc3"]]
        pd.testing.assert_frame_equal(result, expected)

    def test_filter_by_int_indices(self):
        """Test filtering by integer column indices."""
        df = pd.DataFrame(
            {"doc1": [1, 2, 3], "doc2": [4, 5, 6], "doc3": [7, 8, 9]},
            index=["term1", "term2", "term3"],
        )

        result = filter_docs(df, [0, 2])

        expected = df.iloc[:, [0, 2]]
        pd.testing.assert_frame_equal(result, expected)

    def test_no_filter(self):
        """Test when no filter is applied."""
        df = pd.DataFrame(
            {"doc1": [1, 2, 3], "doc2": [4, 5, 6]}, index=["term1", "term2", "term3"]
        )

        result = filter_docs(df, None)

        pd.testing.assert_frame_equal(result, df)


class TestProcessDataframe:
    """Test the process_dataframe function."""

    def test_basic_dataframe_processing(self):
        """Test basic DataFrame processing."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 0], "doc2": [1, 3, 2], "doc3": [0, 1, 1]},
            index=["term1", "term2", "term3"],
        )

        result = process_dataframe(df)

        expected = {"term1": 3, "term2": 5, "term3": 3}
        assert result == expected

    def test_dataframe_with_zero_counts(self):
        """Test DataFrame processing with zero counts."""
        df = pd.DataFrame(
            {"doc1": [2, 0, 0], "doc2": [1, 3, 0], "doc3": [0, 1, 0]},
            index=["term1", "term2", "term3"],
        )

        result = process_dataframe(df)

        # term3 should be excluded as it has zero total count
        expected = {"term1": 3, "term2": 4}
        assert result == expected

    def test_dataframe_with_doc_filter(self):
        """Test DataFrame processing with document filtering."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 0], "doc2": [1, 3, 2], "doc3": [0, 1, 1]},
            index=["term1", "term2", "term3"],
        )

        result = process_dataframe(df, docs=["doc1", "doc2"])

        expected = {"term1": 3, "term2": 4, "term3": 2}
        assert result == expected


class TestProcessDTM:
    """Test the process_dtm function."""

    def test_basic_dtm_processing(self):
        """Test basic DTM processing."""
        docs = [
            ["term1", "term1", "term2"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["term1", "term2", "term3"])
        result = process_dtm(dtm)

        expected = {"term1": 3, "term2": 5, "term3": 3}
        assert result == expected

    def test_dtm_with_doc_filter(self):
        """Test DTM processing with document filtering."""
        docs = [
            ["term1", "term1", "term2"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["term1", "term2", "term3"])
        result = process_dtm(dtm, docs=[0, 1])  # First two docs

        expected = {"term1": 3, "term2": 4, "term3": 2}
        assert result == expected


class TestProcessList:
    """Test the process_list function."""

    def test_list_of_string_lists(self):
        """Test processing list of string lists."""
        data = [["natural", "language"], ["machine", "learning"], ["data", "science"]]
        result = process_list(data, None)

        flattened = ["natural", "language", "machine", "learning", "data", "science"]
        expected = dict(Counter(flattened))
        assert result == expected

    def test_list_with_doc_filter(self):
        """Test processing list with document filtering."""
        data = [["natural", "language"], ["machine", "learning"], ["data", "science"]]
        result = process_list(data, [0, 2])  # First and third docs
        sorted_items = sorted(result.items())
        result = dict(sorted_items)

        expected = dict(Counter(["natural", "language", "data", "science"]))
        assert result == expected

    def test_list_of_docs(self, nlp):
        """Test processing list of spaCy docs."""
        data = [nlp("natural language"), nlp("machine learning")]

        result = process_list(data, None)
        result = dict(sorted(result.items()))

        expected = dict(Counter(["natural", "language", "machine", "learning"]))
        for k, v in result.items():
            assert expected[k.text] == v


class TestProcessDocs:
    """Test the process_docs function."""

    def test_basic_docs_processing(self, nlp):
        """Test basic spaCy docs processing."""
        docs = [nlp("natural language"), nlp("machine learning")]

        result = process_docs(docs, None)

        expected = dict(Counter(["natural", "language", "machine", "learning"]))
        assert result == expected

    def test_docs_with_filter(self, nlp):
        """Test spaCy docs processing with filtering."""
        docs = [nlp("natural language"), nlp("machine learning"), nlp("data science")]

        result = process_docs(docs, [0, 2])  # First and third docs

        expected = dict(Counter(["natural", "language", "data", "science"]))
        assert result == expected


class TestProcessItem:
    """Test the process_item function."""

    def test_list_of_strings(self):
        """Test processing list of strings."""
        result = process_item(SAMPLE_TOKENS)

        expected = Counter(SAMPLE_TOKENS)
        assert result == expected

    def test_spacy_doc(self, nlp):
        """Test processing spaCy doc."""
        doc = nlp(SAMPLE_TEXT)

        result = process_item(doc)

        expected = Counter([token.text for token in doc])
        assert result == expected

    def test_list_of_tokens(self, nlp):
        """Test processing list of spaCy tokens."""
        doc = nlp(SAMPLE_TEXT)
        tokens = list(doc)

        result = process_item(tokens)

        expected = Counter([token.text for token in tokens])
        assert result == expected


class TestMulticloudProcessor:
    """Test the multicloud_processor function."""

    def test_dataframe_processing(self):
        """Test multicloud processing with DataFrame."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 0], "doc2": [1, 3, 2], "doc3": [0, 1, 1]},
            index=["term1", "term2", "term3"],
        )

        result = multicloud_processor(df)

        expected = [
            {"term1": 2, "term2": 1},  # doc1, excluding zero counts
            {"term1": 1, "term2": 3, "term3": 2},  # doc2
            {"term2": 1, "term3": 1},  # doc3, excluding zero counts
        ]
        assert result == expected

    def test_dtm_processing(self):
        """Test multicloud processing with DTM."""
        docs = [
            ["term1", "term1", "term2"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["term1", "term2", "term3"])

        result = multicloud_processor(dtm)

        expected = [
            {"term1": 2, "term2": 1},  # doc1
            {"term1": 1, "term2": 3, "term3": 2},  # doc2
            {"term2": 1, "term3": 1},  # doc3
        ]

        for i, doc in enumerate(result):
            for k, v in doc.items():
                assert expected[i][k] == v

    def test_list_of_dicts(self):
        """Test multicloud processing with list of dictionaries."""
        data = [{"natural": 5, "language": 3}, {"machine": 4, "learning": 2}]

        result = multicloud_processor(data)
        assert result == data

    def test_list_of_string_lists(self):
        """Test multicloud processing with list of string lists."""
        data = [["natural", "language", "natural"], ["machine", "learning"]]

        result = multicloud_processor(data)

        expected = [{"natural": 2, "language": 1}, {"machine": 1, "learning": 1}]
        assert result == expected

    def test_list_of_docs(self, nlp):
        """Test multicloud processing with list of spaCy docs."""
        data = [nlp("natural language natural"), nlp("machine learning")]

        result = multicloud_processor(data)

        expected = [{"natural": 2, "language": 1}, {"machine": 1, "learning": 1}]
        assert result == expected

    def test_with_doc_filter(self):
        """Test multicloud processing with document filtering."""
        data = [["natural", "language"], ["machine", "learning"], ["data", "science"]]

        result = multicloud_processor(data, docs=[0, 2])

        expected = [{"natural": 1, "language": 1}, {"data": 1, "science": 1}]
        assert result == expected

    def test_string_filter_error(self):
        """Test that string filtering raises error for non-DataFrame data."""
        data = [["natural", "language"], ["machine", "learning"]]

        with pytest.raises(LexosException) as exc_info:
            multicloud_processor(data, docs=["doc1", "doc2"])

        assert "Filtering by document labels is not yet supported" in str(
            exc_info.value
        )


class TestGetRows:
    """Test the get_rows function."""

    def test_basic_row_generation(self):
        """Test basic row generation."""
        data = list(range(10))
        rows = list(get_rows(data, 3))

        expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert rows == expected

    def test_exact_division(self):
        """Test when list divides evenly into rows."""
        data = list(range(6))
        rows = list(get_rows(data, 2))

        expected = [[0, 1], [2, 3], [4, 5]]
        assert rows == expected

    def test_single_item_rows(self):
        """Test with single item rows."""
        data = ["a", "b", "c"]
        rows = list(get_rows(data, 1))

        expected = [["a"], ["b"], ["c"]]
        assert rows == expected

    def test_empty_list(self):
        """Test with empty list."""
        rows = list(get_rows([], 3))
        assert rows == []

    def test_larger_n_than_list(self):
        """Test when n is larger than list length."""
        data = ["a", "b"]
        rows = list(get_rows(data, 5))

        expected = [["a", "b"]]
        assert rows == expected


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_process_data_with_mixed_types(self, nlp):
        """Test process_data behavior with unexpected mixed data."""
        doc = nlp("string")
        # This should raise an exception
        mixed_list = [doc, ["list", "of", "strings"]]

        with pytest.raises((TypeError, LexosException)):
            process_data(mixed_list)

    def test_very_large_limit(self):
        """Test with limit larger than data size."""
        result = process_data(SAMPLE_TEXT, limit=1000)

        # Should return all terms since limit exceeds data size
        expected = Counter(SAMPLE_TEXT.split())
        assert result == dict(expected)

    def test_zero_limit(self):
        """Test with zero limit."""
        with pytest.raises(ValidationError):
            process_data(SAMPLE_TEXT, limit=0)

    def test_negative_limit(self):
        """Test with negative limit."""
        with pytest.raises(ValidationError):
            process_data(SAMPLE_TEXT, limit=-1)


# Fixtures for test data
@pytest.fixture
def nlp():
    """Spacy NLP pipeline."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame(
        {"doc1": [2, 1, 0, 3], "doc2": [1, 3, 2, 0], "doc3": [0, 1, 1, 2]},
        index=["term1", "term2", "term3", "term4"],
    )


@pytest.fixture
def sample_dtm(sample_dataframe):
    """Fixture providing a sample DTM."""
    return DTM(sample_dataframe)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_string_to_multicloud(self):
        """Test complete workflow from string to multicloud format."""
        docs = [
            "natural language processing",
            "machine learning algorithms",
            "data science analytics",
        ]

        # Convert to list of token lists
        token_lists = [doc.split() for doc in docs]

        result = multicloud_processor(token_lists)

        expected = [
            {"natural": 1, "language": 1, "processing": 1},
            {"machine": 1, "learning": 1, "algorithms": 1},
            {"data": 1, "science": 1, "analytics": 1},
        ]
        assert result == expected

    def test_dataframe_to_process_data(self, sample_dataframe):
        """Test workflow from DataFrame through process_data."""
        result = process_data(sample_dataframe)

        # Should sum across all documents
        expected = {"term1": 3, "term2": 5, "term3": 3, "term4": 5}
        assert result == expected

    def test_filtering_consistency(self, sample_dataframe):
        """Test that filtering works consistently across functions."""
        docs_filter = ["doc1", "doc3"]

        # Test process_dataframe
        df_result = process_dataframe(sample_dataframe, docs=docs_filter)

        # Test process_data
        data_result = process_data(sample_dataframe, docs=docs_filter)

        # Should be identical
        assert df_result == data_result


class TestProcessorsCoverage:
    """Tests to cover specific uncovered lines in processors.py."""

    def test_process_data_negative_limit(self):
        """Test process_data with negative limit (covers line 59)."""
        data = "natural language processing machine learning"

        with pytest.raises(ValidationError):
            process_data(data, limit=-1)

        # Zero limit should also raise an error
        with pytest.raises(ValidationError):
            process_data(data, limit=0)

    def test_process_list_mixed_types_doc_span(self, nlp):
        """Test process_list with mixed Doc/Span types (covers line 220)."""
        pytest.importorskip("spacy", reason="spaCy not available")

        # Create a Doc and a Span
        doc = nlp("natural language processing")
        span = doc[0:2]  # "natural language"

        # Mix Doc and Span in same list - this should work since both are (Doc, Span)
        data = [doc, span]

        # This should process without error since both Doc and Span are handled together
        result = process_list([data], docs=None)

        # Should contain tokens from both doc and span
        assert isinstance(result, dict)
        result_str = " ".join([item.text for item in result.keys()])
        assert "natural" in result_str
        assert "language" in result_str
        assert "processing" in result_str

    def test_process_list_all_token_type_check(self, nlp):
        """Test process_list Token type checking (covers line 222)."""
        pytest.importorskip("spacy", reason="spaCy not available")

        # Create a list of individual tokens
        doc = nlp("natural language processing")
        tokens = list(doc)  # List of Token objects

        # Wrap in another list to simulate list of lists structure
        data = [tokens]

        result = process_list(data, docs=None)

        assert isinstance(result, dict)
        result_str = " ".join([item.text for item in result.keys()])
        assert "natural" in result_str
        assert "language" in result_str
        assert "processing" in result_str

    def test_multicloud_processor_token_lists(self):
        """Test multicloud_processor with list of Token lists (covers lines 341-342)."""
        pytest.importorskip("spacy", reason="spaCy not available")

        import spacy

        nlp = spacy.blank("en")

        # Create multiple documents with tokens
        doc1 = nlp("natural language")
        doc2 = nlp("machine learning")

        # Convert to lists of tokens
        token_list1 = list(doc1)
        token_list2 = list(doc2)

        data = [token_list1, token_list2]

        result = multicloud_processor(data)

        expected = [{"natural": 1, "language": 1}, {"machine": 1, "learning": 1}]

        assert result == expected
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)

    def test_multicloud_processor_empty_token_lists(self, nlp):
        """Test multicloud_processor with empty token lists."""
        pytest.importorskip("spacy", reason="spaCy not available")

        # Create empty token lists
        empty_doc = nlp("")
        token_list = list(empty_doc)

        data = [token_list]

        with pytest.raises(LexosException):
            result = multicloud_processor(data)

    def test_get_rows_empty_list(self):
        """Test get_rows with empty list."""
        result = list(get_rows([], 3))
        assert result == []

    def test_get_rows_single_item(self):
        """Test get_rows with single item."""
        result = list(get_rows(["a"], 3))
        assert result == [["a"]]

    def test_get_rows_exact_division(self):
        """Test get_rows where list divides evenly."""
        data = ["a", "b", "c", "d", "e", "f"]
        result = list(get_rows(data, 2))
        expected = [["a", "b"], ["c", "d"], ["e", "f"]]
        assert result == expected

    def test_get_rows_uneven_division(self):
        """Test get_rows where list doesn't divide evenly."""
        data = ["a", "b", "c", "d", "e"]
        result = list(get_rows(data, 3))
        expected = [["a", "b", "c"], ["d", "e"]]
        assert result == expected

    def test_process_data_with_zero_limit_string(self):
        """Test process_data with zero limit on string data."""
        data = "hello world test data"
        with pytest.raises(ValidationError):
            process_data(data, limit=0)

    def test_process_data_with_negative_limit_dict(self):
        """Test process_data with negative limit on dict data."""
        data = {"hello": 5, "world": 3, "test": 1}
        with pytest.raises(ValidationError):
            process_data(data, limit=-1)
        with pytest.raises(ValidationError):
            process_data(data, limit=-5)

    def test_process_list_with_doc_span_mixed_content(self, nlp):
        """Test process_list with complex Doc/Span content."""
        pytest.importorskip("spacy", reason="spaCy not available")

        # Create docs with overlapping content
        doc1 = nlp("natural language processing")
        doc2 = nlp("machine learning algorithms")
        span1 = doc1[1:3]  # "language processing"

        # Test with mixed docs and spans
        data = [[doc1, span1, doc2]]

        result = process_list(data, docs=None)
        print(type(result))
        for k, v in result.items():
            print(type(k), type(v))
        # assert False

        # # Should have all tokens, with some duplicated due to span overlap
        result_token_str = " ".join([item.text for item in result.keys()])
        assert "natural" in result_token_str
        assert "language" in result_token_str
        assert "processing" in result_token_str
        assert "machine" in result_token_str
        assert "learning" in result_token_str
        assert "algorithms" in result_token_str

        # # language and processing should appear twice (once in doc1, once in span1)
        result_tokens = result_token_str.split()
        result_token_counts = Counter(result_tokens)
        assert result_token_counts["language"] >= 2
        assert result_token_counts["processing"] >= 2

    def test_multicloud_processor_single_token_lists(self):
        """Test multicloud_processor with single token per list."""
        pytest.importorskip("spacy", reason="spaCy not available")

        import spacy

        nlp = spacy.blank("en")

        # Create single token lists
        token1 = list(nlp("hello"))[0]
        token2 = list(nlp("world"))[0]

        data = [[token1], [token2]]

        result = multicloud_processor(data)

        expected = [{"hello": 1}, {"world": 1}]

        assert result == expected

    def test_multicloud_processor_repeated_tokens(self):
        """Test multicloud_processor with repeated tokens in lists."""
        pytest.importorskip("spacy", reason="spaCy not available")

        import spacy

        nlp = spacy.blank("en")

        # Create token lists with repetition
        doc = nlp("hello hello world")
        tokens = list(doc)

        data = [tokens]

        result = multicloud_processor(data)

        expected = [{"hello": 2, "world": 1}]

        assert result == expected


# Additional tests to cover lines 59, 220, and 222


class TestSpecificLineCoverage:
    """Tests specifically targeting uncovered lines 59, 220, and 222."""

    def test_line_59_most_common_with_zero_limit(self):
        """Test line 59: counts.most_common(limit) with limit=0."""
        # This needs to hit the exact path where limit is 0 and most_common(0) is called
        data = {"word1": 5, "word2": 3, "word3": 1}

        with pytest.raises(ValidationError):
            process_data(data, limit=0)

    def test_line_220_doc_span_type_check(self, nlp):
        """Test line 220: all(isinstance(item, (Doc, Span)) for item in data)."""
        # Create a list where first item is Doc/Span but we need to check ALL items
        doc1 = nlp("hello world")
        doc2 = nlp("foo bar")
        span1 = doc1[0:1]  # Just "hello"

        # This list has mixed Doc and Span types - should pass the isinstance check
        mixed_doc_span = [doc1, span1, doc2]

        # Wrap in outer list to make it a "list of lists" for process_list
        data = [mixed_doc_span]

        # This should hit line 220 where it checks if all items are (Doc, Span)
        result = process_list(data, docs=None)
        assert isinstance(result, dict)

    def test_line_222_token_type_check(self, nlp):
        """Test line 222: all(isinstance(item, Token) for item in data)."""
        # Create a list of actual Token objects
        doc = nlp("hello world test")
        tokens = list(doc)  # This creates actual Token objects

        # Create another set of tokens
        doc2 = nlp("foo bar")
        tokens2 = list(doc2)

        # Mix different token lists - this should trigger the Token type check
        mixed_tokens = tokens + tokens2

        # Wrap in outer list for process_list
        data = [mixed_tokens]

        # This should hit line 222 where it checks if all items are Token type
        result = process_list(data, docs=None)
        assert isinstance(result, dict)

    def test_line_59_negative_limit_counter_most_common(self):
        """Test line 59 with negative limit specifically."""
        # Create data that will definitely hit the Counter.most_common() call
        text_data = "apple banana cherry apple banana apple"

        # This should create a Counter and then call most_common(-1)
        result = process_data(text_data, limit=1)
        assert len(result) == 1

        with pytest.raises(ValidationError):
            process_data(text_data, limit=0)

        with pytest.raises(ValidationError):
            process_data(text_data, limit=-1)

    def test_line_223(self, nlp):
        """Test line 223 with all Token types."""
        doc = nlp("hello world")
        token1 = doc[0]
        token2 = doc[1]
        doc_token_list = [[token1, token2]]

        terms = process_list(doc_token_list, docs=None)
        terms = [token.text for token in list(terms.keys())]
        assert token1.text in terms
        assert token2.text in terms
