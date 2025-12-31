"""test_dtm.py.

Coverage: 98%. Missing: 245-246
Last Update: June 15, 2025

Unit tests for the DTM (Document-Term Matrix) and related functionality in the lexos package.
Covers construction, sorting, statistics, conversion to DataFrame, and error handling.
"""

from typing import Any, Dict, Generator, List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import spacy
from natsort import ns
from scipy.sparse import csr_matrix
from textacy.representations.vectorizers import Vectorizer as TextacyVectorizer

from lexos.dtm import DTM, Vectorizer
from lexos.exceptions import LexosException

# Fixtures


@pytest.fixture
def nlp() -> spacy.language.Language:
    """Fixture for a blank English spaCy model."""
    return spacy.blank("en")


@pytest.fixture
def sample_docs(nlp: spacy.language.Language) -> List[Any]:
    """Fixture for sample spaCy docs."""
    texts = ["hello world", "test document"]
    return [nlp(text) for text in texts]


@pytest.fixture
def dtm() -> DTM:
    """Fixture for a DTM instance with TextacyVectorizer."""
    return DTM(vectorizer=TextacyVectorizer)


@pytest.fixture
def dtm_with_terms(nlp: spacy.language.Language) -> DTM:
    """Fixture for a DTM instance with sample terms and labels."""
    dtm = DTM()
    dtm(
        docs=[nlp("term2"), nlp("term1"), nlp("Term3"), nlp("10term"), nlp("2term")],
        labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return dtm


@pytest.fixture
def mock_dtm() -> DTM:
    """Create DTM with mocked vocabulary terms."""
    dtm_instance = DTM()

    class MockVectorizer:
        # These are the terms in the order we expect them from TextacyVectorizer
        terms_list = ["10term", "2term", "Term3", "term1", "term2"]

    dtm_instance.vectorizer = MockVectorizer()

    # Create a csr_matrix that, when summed along axis=0,
    # yields the desired total counts for each term.
    # example sums: {"10term": 2, "2term": 4, "Term3": 7, "term1": 3, "term2": 5}
    # The matrix below has 1 row (doc) and 5 coumns (terms), where each value is the total count
    dtm_instance.doc_term_matrix = csr_matrix(np.array([[2, 4, 7, 3, 5]]))
    dtm_instance.labels = ["doc1"]  # a dummy label as it's a mock

    return dtm_instance


@pytest.fixture
def mock_df_dtm() -> DTM:
    """Create DTM with sample data and mock vectorizer."""
    dtm = DTM()

    # Create sample data
    data = np.array([[1, 2], [3, 4], [5, 6]])
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3"]

    return dtm


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture for a mock DataFrame with term counts."""
    return pd.DataFrame(
        {"doc1": [10, 20, 30], "doc2": [5, 15, 25]}, index=["term1", "term2", "term3"]
    )


# @pytest.fixture
# def sample_df() -> pd.DataFrame:
#     """Fixture for a sample DataFrame with term counts."""
#     return pd.DataFrame(
#         {"doc1": [10, 20, 30], "doc2": [5, 15, 25]}, index=["term1", "term2", "term3"]
#     )


@pytest.fixture
def sample_corpus(nlp):
    """Fixture that provides a sample document corpus.

    Returns:
        pandas.DataFrame: A DataFrame with sample documents.
    """
    texts = [
        "This is a sample text for testing.",
        "Another example of a text to analyze.",
        "This text is different from the others.",
        "Yet another sample text for comparison.",
        "This text is similar to the first one.",
        "A completely different text for analysis.",
    ]
    return [nlp(text) for text in texts]


# Tests


def test_valid_docs_and_labels(dtm: DTM, sample_docs: List[Any]) -> None:
    """Test DTM with valid docs and labels."""
    labels = ["doc1", "doc2"]
    dtm(docs=sample_docs, labels=labels)
    assert len(dtm.docs) == len(labels)
    assert dtm.labels == labels
    assert dtm.doc_term_matrix is not None


def test_docs_with_token_lists(dtm: DTM) -> None:
    """Test DTM with docs as lists of tokens."""
    docs = [["hello", "world"], ["test", "document"]]
    labels = ["doc1", "doc2"]
    dtm(docs=docs, labels=labels)
    assert len(dtm.docs) == 2
    assert dtm.docs == docs


def test_docs_with_spacy_docs(dtm: DTM, sample_docs: List[Any]) -> None:
    """Test DTM with spaCy Doc objects as input."""
    labels = ["doc1", "doc2"]
    dtm(docs=sample_docs, labels=labels)
    assert len(dtm.docs) == 2
    assert all(isinstance(doc, list) for doc in dtm.docs)


def test_mismatched_docs_labels(dtm: DTM, sample_docs: List[Any]) -> None:
    """Test error when number of docs and labels do not match."""
    labels = ["doc1"]
    with pytest.raises(
        LexosException, match="The number of docs must match the number of labels."
    ):
        dtm(docs=sample_docs, labels=labels)


def test_default_labels(dtm: DTM, sample_docs: List[Any]) -> None:
    """Test default label assignment when labels are None."""
    dtm(docs=sample_docs, labels=None)
    assert dtm.labels == ["Doc1", "Doc2"]


def test_invalid_sorting_algorithm(dtm: DTM, sample_docs: List[Any]) -> None:
    """Test error when invalid sorting algorithm is set."""
    dtm.alg = "invalid_algorithm"
    with pytest.raises(LexosException):
        dtm(docs=sample_docs, labels=None)


def test_vectorizer_none(dtm: DTM) -> None:
    """Test whether the default vectorizer is used if it is set to None."""
    dtm.vectorizer = None
    # This should not raise an error
    dtm(docs=[["test"]], labels=["doc1"])


def test_sorted_terms_list_basic(dtm_with_terms: DTM) -> None:
    """Test basic natural sorting of terms."""
    expected = ["2term", "10term", "term1", "term2", "Term3"]
    assert dtm_with_terms.sorted_terms_list == expected


def test_sorted_terms_list_empty() -> None:
    """Test sorting with empty terms list."""
    dtm = DTM()
    with pytest.raises(LexosException):
        dtm(docs=[], labels=[])


def test_sorted_terms_list_with_different_alg(nlp: spacy.language.Language) -> None:
    """Test sorting with different algorithm."""
    dtm = DTM()
    dtm(
        docs=[nlp("Term2"), nlp("term1"), nlp("term3"), nlp("10term"), nlp("2term")],
        labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    dtm.alg = ns.IGNORECASE
    expected = ["2term", "10term", "term1", "Term2", "term3"]
    assert dtm.sorted_terms_list == expected


def test_sorted_terms_list_no_vectorizer(dtm_with_terms: DTM) -> None:
    """Test behavior when vectorizer is not initialized."""
    dtm_with_terms.vectorizer = None
    with pytest.raises(
        LexosException,
        match="Vectorizer or its 'terms_list' attribute is not available to get sorted terms.",
    ):
        _ = dtm_with_terms.sorted_terms_list


def test_sorted_term_counts_basic(mock_dtm: DTM) -> None:
    """Test basic natural sorting of term counts with correctly summed values."""
    # This 'expected dict now reflects the actual sums of the terms, sorted naturally
    expected = {"10term": 2, "2term": 4, "Term3": 7, "term1": 3, "term2": 5}
    assert mock_dtm.sorted_term_counts == expected


def test_sorted_term_counts_empty() -> None:
    """Test sorted_term_counts returns empty dict for empty DTM or uninitialized matrix."""
    # Test DTM without building (doc_term_matrix is None)
    dtm_unitialized = DTM()
    assert dtm_unitialized.sorted_term_counts == {}

    # Test DTM with an empty doc_term_matrix
    dtm_empty_matrix = DTM()

    class EmptyMockVectorizer:
        terms_list = []

    dtm_empty_matrix.vectorizer = EmptyMockVectorizer()
    dtm_empty_matrix.doc_term_matrix = csr_matrix(
        (0, 0)
    )  # Explicitly an empty sparse matrix
    assert dtm_empty_matrix.sorted_term_counts == {}


def test_sorted_term_counts_different_alg(mock_dtm: DTM) -> None:
    """Test sorting with different algorithm."""
    mock_dtm.alg = ns.IGNORECASE
    expected = {"2term": 4, "10term": 2, "term1": 3, "term2": 5, "Term3": 7}
    assert mock_dtm.sorted_term_counts == expected


def test_sorted_term_counts_no_vectorizer() -> None:
    """Test behavior when vectorizer is not initialized."""
    dtm = DTM()
    assert dtm.sorted_term_counts == {}


def test_basic_percentages(mock_df: pd.DataFrame) -> None:
    """Test calculation of term percentages."""
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, as_str=False)
    expected = pd.DataFrame(
        {"doc1": [9.524, 19.048, 28.571], "doc2": [4.762, 14.286, 23.810]},
        index=["term1", "term2", "term3"],
    ).round(3)
    pd.testing.assert_frame_equal(result.round(3), expected)


def test_zero_sum_case() -> None:
    """Test percentage calculation when all values are zero."""
    zero_df = pd.DataFrame({"doc1": [0, 0], "doc2": [0, 0]})
    dtm = DTM()
    result = dtm._get_term_percentages(zero_df, as_str=False)
    assert (result == 0).all().all()


def test_rounding(mock_df: pd.DataFrame) -> None:
    """Test rounding of percentage values."""
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, rounding=1, as_str=False)
    assert all(len(str(x).split(".")[-1]) <= 1 for x in result.values.flatten())


def test_with_statistics(mock_df: pd.DataFrame) -> None:
    """Test calculation of sum, mean, and median statistics."""
    dtm = DTM()
    result = dtm._get_term_percentages(
        mock_df, as_str=False, sum=True, mean=True, median=True
    )
    assert "Total" in result.columns
    assert "Mean" in result.columns
    assert "Median" in result.columns


def test_string_output(mock_df: pd.DataFrame) -> None:
    """Test output of percentages as strings with percent sign."""
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, as_str="string")
    assert all(isinstance(x, str) and x.endswith("%") for x in result.values.flatten())


def test_different_rounding_values(mock_df: pd.DataFrame) -> None:
    """Test different rounding values for percentages."""
    dtm = DTM()
    for rounding in [1, 2, 4]:
        result = dtm._get_term_percentages(mock_df, rounding=1, as_str=False)
        assert all(
            len(str(x).split(".")[-1]) <= rounding for x in result.values.flatten()
        )


def test_basic_conversion(mock_df_dtm: DTM) -> None:
    """Test conversion of DTM to DataFrame."""
    df = mock_df_dtm.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["doc1", "doc2", "doc3"]
    assert list(df.index) == ["term1", "term2"]


def test_dtm_shape_property(mock_df_dtm: DTM) -> None:
    """Test that the DTM.shape property returns the correct dimensions, matching the underlying doc_term_matrix shape."""
    expected_shape = (3, 2)
    assert mock_df_dtm.shape == expected_shape

    with pytest.raises(
        LexosException, match="DTM must be built before accessing its shape"
    ):
        dtm = DTM()
        _ = dtm.shape


def test_sorting(mock_df_dtm: DTM) -> None:
    """Test sorting of DataFrame by column values."""
    df = mock_df_dtm.to_df(by=["doc1"], ascending=False)
    assert df.index.tolist() == ["term2", "term1"]


def test_percentage_conversion(dtm_with_terms: DTM) -> None:
    """Test conversion of DTM to DataFrame with and without percentages."""
    df = dtm_with_terms.to_df(as_percent=False)
    print(df)
    df = dtm_with_terms.to_df(as_percent=True)
    assert all(isinstance(x, float) for x in df.values.flatten())


def test_percentage_conversion_string(mock_df_dtm: DTM) -> None:
    """Test conversion of DTM to DataFrame with percentages as strings."""
    df = mock_df_dtm.to_df(as_percent="string")
    assert all(isinstance(x, str) and x.endswith("%") for x in df.values.flatten())


def test_rounding_in_get_term_percentages_method(mock_df_dtm: DTM) -> None:
    """Test rounding in get_term_percentages method."""
    rounding = 2
    df = mock_df_dtm.to_df(as_percent=True, rounding=rounding)
    values = [str(x).split(".")[1] for x in df.values.flatten()]
    assert all(len(x) <= rounding for x in values)


def test_transpose(mock_df_dtm: DTM) -> None:
    """Test transposing the DataFrame output."""
    df = mock_df_dtm.to_df(transpose=True)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["term1", "term2"]
    assert list(df.index) == ["doc1", "doc2", "doc3"]


def test_statistics(mock_df_dtm: DTM) -> None:
    """Test inclusion of statistics columns in DataFrame."""
    df = mock_df_dtm.to_df(sum=True, mean=True, median=True)
    assert "Total" in df.columns
    assert "Mean" in df.columns
    assert "Median" in df.columns


def test_to_df_with_statistics_no_percentages(mock_df_dtm: DTM) -> None:
    """Test to_df method with sum, mean, and median when not converting to percentages."""
    df = mock_df_dtm.to_df(sum=True, mean=True, median=True, as_percent=False)
    assert "Total" in df.columns
    assert "Mean" in df.columns
    assert "Median" in df.columns
    assert df.shape == (
        2,
        3 + 3,
    )  # (terms, docs + stats) assuming 2 terms and 3 docs in mock_df_dtm


def test_vectorizer_instantiation_and_call() -> None:
    """Test direct instantiation and call of the Vectorizer class."""
    vectorizer_wrapper_instance = Vectorizer()
    textacy_vectorizer = vectorizer_wrapper_instance(tf_type="log", max_df=0.5)
    assert isinstance(textacy_vectorizer, TextacyVectorizer)
    assert textacy_vectorizer.tf_type == "log"
    assert textacy_vectorizer.max_df == 0.5


def test_to_df_handles_attribute_error_from_sparse(mock_df_dtm: DTM) -> None:
    """Test to_df handles AttributeError when using sparse.from_spmatrix."""
    mock_df_dtm.doc_term_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    with patch("pandas.DataFrame.sparse.from_spmatrix") as mock_from_spmatrix:
        mock_from_spmatrix.side_effect = AttributeError(
            "Simulate sparse conversion error"
        )
        df = mock_df_dtm.to_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["doc1", "doc2", "doc3"]
        assert list(df.index) == ["term1", "term2"]
        expected_df_data = np.array([[1, 3, 5], [2, 4, 6]])
        expected_df = pd.DataFrame(
            expected_df_data, index=["term1", "term2"], columns=["doc1", "doc2", "doc3"]
        )
        pd.testing.assert_frame_equal(df, expected_df)


def test_to_df_handles_general_exception(mock_df_dtm: DTM) -> None:
    """Test to_df handles general Exception during DataFrame conversion."""
    with patch("pandas.DataFrame.sparse.from_spmatrix") as mock_from_spmatrix:
        mock_from_spmatrix.side_effect = TypeError("Simulated general conversion error")
        with pytest.raises(LexosException, match="Error converting DTM to DataFrame"):
            mock_df_dtm.to_df()


def test_combined_options(mock_df_dtm: DTM) -> None:
    """Test to_df with combined options for sorting, percentages, rounding, transpose, and statistics."""
    rounding = 2
    df = mock_df_dtm.to_df(
        by=["doc1"],
        ascending=False,
        as_percent="string",
        rounding=rounding,
        transpose=True,
        sum=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert "Total" in df.index
    assert all(isinstance(x, str) and x.endswith("%") for x in df.values.flatten())
    values = [x.split(".")[1] for x in df.values.flatten()]
    assert all(len(x) <= rounding + 1 for x in values)


def test_valid_sorting_algorithm(dtm: DTM) -> None:
    """Test validation with valid sorting algorithm."""
    dtm.alg = ns.IGNORECASE
    assert dtm._validate_sorting_algorithm() is True


def test_invalid_sorting_algorithm_in_validation_method(dtm: DTM) -> None:
    """Test validation with invalid sorting algorithm."""
    dtm.alg = "invalid_alg"
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    assert "Invalid sorting algorithm" in str(excinfo.value)
    assert "Valid algorithms for `alg` are:" in str(excinfo.value)


def test_none_sorting_algorithm(dtm: DTM) -> None:
    """Test validation with None as sorting algorithm."""
    dtm.alg = None
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    assert "Invalid sorting algorithm" in str(excinfo.value)


def test_error_message_formatting(dtm: DTM) -> None:
    """Test error message contains all valid algorithms."""
    dtm.alg = "invalid_alg"
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    for locale in ns:
        assert f"ns.{locale.name}" in str(excinfo.value)


##### Additional Tests for Vectorizer Parameters and Properties #####


def test_vectorizer_args_are_passed_on_init(sample_corpus):
    """Test that vectorizer receives and uses custom parameters.

    Args:
        sample_corpus: Pytest fixture with sample documents.
    """
    # Create DTM with specific Vectorizer parameters
    labels = [f"doc{i}" for i in range(len(sample_corpus))]
    dtm1 = DTM(docs=sample_corpus, labels=labels)
    dtm2 = DTM(docs=sample_corpus, labels=labels, min_df=2, max_df=0.9)

    # Verify that the vectorizer has our custom parameters
    assert dtm2.vectorizer.min_df == 2
    assert dtm2.vectorizer.max_df == 0.9

    # Validate that the parameters actually affect the result
    dtm1(docs=sample_corpus, labels=labels)
    dtm2(docs=sample_corpus, labels=labels)
    assert dtm2.doc_term_matrix.shape != dtm1.doc_term_matrix.shape


def test_vectorizer_args_are_passed_after_init(sample_corpus):
    """Test that vectorizer receives and uses custom parameters.

    Args:
        sample_corpus: Pytest fixture with sample documents.
    """
    # Create DTM with specific Vectorizer parameters
    labels = [f"doc{i}" for i in range(len(sample_corpus))]
    dtm1 = DTM(docs=sample_corpus, labels=labels)
    dtm2 = DTM(docs=sample_corpus, labels=labels)
    dtm2.vectorizer.min_df = 2
    dtm2.vectorizer.max_df = 0.9

    # Verify that the vectorizer has our custom parameters
    assert dtm2.vectorizer.min_df == 2
    assert dtm2.vectorizer.max_df == 0.9

    # Validate that the parameters actually affect the result
    dtm1(docs=sample_corpus, labels=labels)
    dtm2(docs=sample_corpus, labels=labels)
    assert dtm2.doc_term_matrix.shape != dtm1.doc_term_matrix.shape


def test_vectorizer_args_are_passed_on_call(sample_corpus):
    """Test that vectorizer receives and uses custom parameters.

    Args:
        sample_corpus: Pytest fixture with sample documents.
    """
    # Create DTM with specific Vectorizer parameters
    labels = [f"doc{i}" for i in range(len(sample_corpus))]
    dtm1 = DTM()
    dtm1(docs=sample_corpus, labels=labels)
    dtm2 = DTM()
    dtm2(docs=sample_corpus, labels=labels, min_df=2)

    # Verify that the vectorizer has our custom parameters
    assert dtm2.vectorizer.min_df == 2

    # Validate that the parameters actually affect the result
    assert dtm2.doc_term_matrix.shape != dtm1.doc_term_matrix.shape


def test_custom_vectorizer_args(sample_corpus):
    """Test that custom vectorizer instances receive and use provided parameters.

    Args:
        sample_corpus: Pytest fixture with sample documents.
    """

    class CustomVectorizer:
        """Custom vectorizer for testing."""

        def __init__(self, min_df: Optional[int] = 1, ngram: Optional[int] = 1):
            """Initialize the custom vectorizer with specific parameters."""
            self.min_df = min_df
            self.ngram = ngram

    # Pass the custom vectorizer to DTM
    dtm1 = DTM(docs=sample_corpus, vectorizer=CustomVectorizer)
    dtm2 = DTM(docs=sample_corpus, vectorizer=CustomVectorizer)
    dtm2.vectorizer.min_df = 2
    dtm2.vectorizer.ngram = 2

    # Verify that DTM uses the custom vectorizer with its parameters
    assert dtm1.vectorizer.min_df == 1
    assert dtm1.vectorizer.ngram == 1
    assert dtm2.vectorizer.min_df == 2
    assert dtm2.vectorizer.ngram == 2


def test_sorted_term_counts_property_no_terms_list():
    """Test that the DTM.sorted_term_counts property raises a no terms_list error."""

    class CustomVectorizer:
        """Custom vectorizer for testing."""

        def __init__(self):
            self.no_terms_list = []

    with pytest.raises(
        LexosException,
        match="Vectorizer must have 'terms_list' attribute to get sorted term counts.",
    ):
        dtm = DTM()
        # Assign a dummy doc_term_matrix to bypass the first check
        dtm.doc_term_matrix = csr_matrix(np.array([[1, 2], [3, 4]]))
        # Assign the custom vectorizer that does not have a terms_list
        dtm.vectorizer = CustomVectorizer()
        # Attempt to access sorted_term_counts should raise an error
        _ = dtm.sorted_term_counts


def test_sorted_term_counts_property_mismatch(sample_corpus):
    """Test that the DTM.sorted_term_counts property raises a mismatch error."""
    dtm1 = DTM()
    dtm2 = DTM()
    docs1 = sample_corpus[0:2]
    docs2 = sample_corpus[0:3]
    dtm1(docs=docs1, labels=["doc1", "doc2"])
    dtm2(docs=docs2, labels=["doc1", "doc2", "doc3"])
    dtm1.doc_term_matrix = dtm2.doc_term_matrix
    with pytest.raises(LexosException, match="Mismatch between"):
        _ = dtm1.sorted_term_counts
