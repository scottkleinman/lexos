"""test_kwic.py.

Test suite for the Kwic class.

Coverage: 97%. Missing: 270-276
The missing lines are for invalid sorting locales, which should not be possible to submit because of Pydantic validation.

Last Update: July 28, 2025
"""

import pandas as pd
import pytest
import spacy
from natsort import ns
from pydantic import ValidationError

from lexos.exceptions import LexosException
from lexos.kwic import Kwic


class TestKwic:
    """Test suite for the Kwic class."""

    @pytest.fixture
    def sample_texts(self):
        """Create sample text documents for testing."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "A lazy cat sleeps under the warm sun.",
            "The dog runs quickly through the green park.",
            "Brown foxes are very quick animals in nature.",
        ]

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for documents."""
        return ["Document 1", "Document 2", "Document 3", "Document 4"]

    @pytest.fixture
    def kwic_instance(self):
        """Create a Kwic instance for testing."""
        # Use a small model that's likely to be available
        try:
            return Kwic(nlp="en_core_web_sm")
        except OSError:
            # Fallback to multilingual model if English model not available
            return Kwic(nlp="xx_sent_ud_sm")

    @pytest.fixture
    def spacy_docs(self, kwic_instance, sample_texts):
        """Create spaCy Doc objects from sample texts."""
        return [kwic_instance.nlp(text) for text in sample_texts]

    def test_kwic_initialization_default(self):
        """Test Kwic initialization with default parameters."""
        kwic = Kwic()
        assert kwic.nlp is not None
        assert kwic.alg == ns.LOCALE

    def test_kwic_initialization_custom_model(self):
        """Test Kwic initialization with custom spaCy model."""
        try:
            kwic = Kwic(nlp="en_core_web_sm")
            lang = kwic.nlp.meta["lang"]
            name = kwic.nlp.meta["name"]
            assert f"{lang}_{name}" == "en_core_web_sm"
        except OSError:
            # Skip if model not available
            pytest.skip("en_core_web_sm model not available")

    def test_kwic_initialization_invalid_model(self):
        """Test Kwic initialization with invalid spaCy model."""
        with pytest.raises(OSError):
            Kwic(nlp="invalid_model_name")

    def test_kwic_initialization_custom_sorting(self):
        """Test Kwic initialization with custom sorting algorithm."""
        kwic = Kwic(alg=ns.IGNORECASE)
        assert kwic.alg == ns.IGNORECASE

    def test_kwic_initialization_invalid_sorting(self):
        """Test Kwic initialization with invalid sorting algorithm."""
        with pytest.raises(ValidationError):
            Kwic(alg="invalid_algorithm")

    def test_characters_matcher_basic(self, kwic_instance, sample_texts, sample_labels):
        """Test basic character-based matching."""
        results = kwic_instance(
            docs=sample_texts,
            labels=sample_labels,
            patterns=["quick"],
            window=10,
            matcher="characters",
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3  # "quick" appears in 3 documents
        assert all(
            col in results.columns
            for col in ["doc", "context_before", "keyword", "context_after"]
        )
        assert all("quick" in str(keyword) for keyword in results["keyword"])

    def test_characters_matcher_case_sensitive(self, kwic_instance, sample_texts):
        """Test character-based matching with case sensitivity."""
        # Test case-insensitive (default)
        results_insensitive = kwic_instance(
            docs=sample_texts,
            patterns=["QUICK"],
            matcher="characters",
            case_sensitive=False,
            window=10,
            as_df=True,
        )

        # Test case-sensitive
        results_sensitive = kwic_instance(
            docs=sample_texts,
            patterns=["QUICK"],
            matcher="characters",
            case_sensitive=True,
            window=10,
            as_df=True,
        )

        assert len(results_insensitive) > 0  # Should find "quick"
        assert len(results_sensitive) == 0  # Should not find "QUICK"

    def test_characters_matcher_regex(self, kwic_instance, sample_texts):
        """Test character-based matching with regex patterns."""
        # Note: regex patterns work with character matcher
        results = kwic_instance(
            docs=sample_texts,
            patterns=[r"\b\w*ox\w*\b"],  # Words containing "ox"
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1  # Should find "fox" and "foxes"

    def test_characters_matcher_window_size(self, kwic_instance, sample_texts):
        """Test character-based matching with different window sizes."""
        # Small window
        results_small = kwic_instance(
            docs=sample_texts,
            patterns=["quick"],
            window=5,
            matcher="characters",
            as_df=True,
        )

        # Large window
        results_large = kwic_instance(
            docs=sample_texts,
            patterns=["quick"],
            window=20,
            matcher="characters",
            as_df=True,
        )

        assert len(results_small) == len(results_large)  # Same matches
        # Context should be different lengths
        for i in range(len(results_small)):
            assert len(results_large.iloc[i]["context_before"]) >= len(
                results_small.iloc[i]["context_before"]
            )
            assert len(results_large.iloc[i]["context_after"]) >= len(
                results_small.iloc[i]["context_after"]
            )

    def test_tokens_matcher_with_spacy_docs(
        self, kwic_instance, spacy_docs, sample_labels
    ):
        """Test token-based matching with spaCy documents."""
        results = kwic_instance(
            docs=spacy_docs,
            labels=sample_labels,
            patterns=["quick"],
            matcher="tokens",
            window=3,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1
        assert all("quick" in str(keyword).lower() for keyword in results["keyword"])

    def test_tokens_matcher_with_strings_raises_error(
        self, kwic_instance, sample_texts
    ):
        """Test that token matcher raises error with string documents."""
        with pytest.raises(LexosException, match="Docs must be spaCy Doc objects"):
            kwic_instance(
                docs=sample_texts, patterns=["quick"], matcher="tokens", window=10
            )

    def test_tokens_matcher_case_sensitive(self, kwic_instance, spacy_docs):
        """Test token-based matching with case sensitivity."""
        # Case-insensitive
        results_insensitive = kwic_instance(
            docs=spacy_docs,
            patterns=["QUICK"],
            matcher="tokens",
            case_sensitive=False,
            window=3,
            as_df=True,
        )

        # Case-sensitive
        results_sensitive = kwic_instance(
            docs=spacy_docs,
            patterns=["QUICK"],
            matcher="tokens",
            case_sensitive=True,
            window=3,
            as_df=True,
        )

        assert len(results_insensitive) > 0
        assert len(results_sensitive) == 0

    def test_tokens_matcher_with_regex(self, kwic_instance, spacy_docs):
        """Test token-based matching with regex patterns."""
        results = kwic_instance(
            docs=spacy_docs,
            patterns=[r"qu.*k"],  # Matches "quick"
            matcher="tokens",
            use_regex=True,
            window=3,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1

    def test_phrase_matcher_with_spacy_docs(
        self, kwic_instance, spacy_docs, sample_labels
    ):
        """Test phrase-based matching with spaCy documents."""
        results = kwic_instance(
            docs=spacy_docs,
            labels=sample_labels,
            patterns=["brown fox", "lazy dog"],
            matcher="phrase",
            window=2,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1
        # Check that matched phrases are found
        keywords = [str(keyword).lower() for keyword in results["keyword"]]
        assert any("brown fox" in keyword for keyword in keywords) or any(
            "lazy dog" in keyword for keyword in keywords
        )

    def test_phrase_matcher_case_sensitivity(self, kwic_instance, spacy_docs):
        """Test phrase matcher with case sensitivity."""
        # Case-insensitive
        results_insensitive = kwic_instance(
            docs=spacy_docs,
            patterns=["BROWN FOX"],
            matcher="phrase",
            case_sensitive=False,
            window=2,
            as_df=True,
        )

        # Case-sensitive
        results_sensitive = kwic_instance(
            docs=spacy_docs,
            patterns=["BROWN FOX"],
            matcher="phrase",
            case_sensitive=True,
            window=2,
            as_df=True,
        )

        assert len(results_insensitive) > 0
        assert len(results_sensitive) == 0

    def test_multiple_patterns(self, kwic_instance, sample_texts):
        """Test matching multiple patterns."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["quick", "lazy", "dog"],
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 3  # Should find all three patterns
        keywords = [str(keyword).lower() for keyword in results["keyword"]]
        assert any("quick" in keyword for keyword in keywords)
        assert any("lazy" in keyword for keyword in keywords)
        assert any("dog" in keyword for keyword in keywords)

    def test_auto_generated_labels(self, kwic_instance, sample_texts):
        """Test automatic label generation when labels not provided."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["the"],
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        # Check that auto-generated labels follow pattern "Doc 1", "Doc 2", etc.
        unique_labels = results["doc"].unique()
        expected_labels = [f"Doc {i + 1}" for i in range(len(sample_texts))]
        for label in unique_labels:
            assert label in expected_labels

    def test_labels_length_mismatch_error(self, kwic_instance, sample_texts):
        """Test error when labels length doesn't match docs length."""
        with pytest.raises(
            LexosException, match="The number of documents and labels must match"
        ):
            kwic_instance(
                docs=sample_texts,
                labels=["Doc 1", "Doc 2"],  # Only 2 labels for 4 docs
                patterns=["test"],
                window=10,
            )

    def test_return_as_list(self, kwic_instance, sample_texts):
        """Test returning results as list instead of DataFrame."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["quick"],
            matcher="characters",
            window=10,
            as_df=False,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        # Each result should be a tuple with 4 elements
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 4
            assert isinstance(result[0], str)  # doc label
            assert isinstance(result[1], str)  # context_before
            assert isinstance(result[2], str)  # keyword
            assert isinstance(result[3], str)  # context_after

    def test_sorting_by_keyword(self, kwic_instance, sample_texts):
        """Test sorting results by keyword."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["the", "a", "quick"],
            matcher="characters",
            sort_by="keyword",
            ascending=True,
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        if len(results) > 1:
            # Check that keywords are sorted
            keywords = results["keyword"].tolist()
            sorted_keywords = sorted(keywords, key=str.lower)
            assert keywords == sorted_keywords

    def test_sorting_by_document(self, kwic_instance, sample_texts, sample_labels):
        """Test sorting results by document."""
        results = kwic_instance(
            docs=sample_texts,
            labels=sample_labels,
            patterns=["the"],
            matcher="characters",
            sort_by="doc",
            ascending=True,
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        if len(results) > 1:
            # Check that documents are sorted
            docs = results["doc"].tolist()
            sorted_docs = sorted(docs)
            assert docs == sorted_docs

    def test_sorting_descending(self, kwic_instance, sample_texts):
        """Test descending sort order."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["the", "a"],
            matcher="characters",
            sort_by="keyword",
            ascending=False,
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        if len(results) > 1:
            keywords = results["keyword"].tolist()
            sorted_keywords = sorted(keywords, key=str.lower, reverse=True)
            assert keywords == sorted_keywords

    def test_empty_patterns(self, kwic_instance, sample_texts):
        """Test behavior with empty patterns list."""
        results = kwic_instance(
            docs=sample_texts, patterns=[], matcher="characters", window=10, as_df=True
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_empty_docs(self, kwic_instance):
        """Test behavior with empty documents list."""
        results = kwic_instance(
            docs=[], patterns=["test"], matcher="characters", window=10, as_df=True
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_no_matches_found(self, kwic_instance, sample_texts):
        """Test behavior when no matches are found."""
        results = kwic_instance(
            docs=sample_texts,
            patterns=["xyzzyx"],  # Non-existent word
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_window_edge_cases(self, kwic_instance):
        """Test window behavior at document edges."""
        short_text = ["Quick brown fox."]

        results = kwic_instance(
            docs=short_text,
            patterns=["brown"],
            window=100,  # Window larger than text
            matcher="characters",
            as_df=True,
        )

        assert len(results) == 1
        # Context should not exceed document boundaries
        row = results.iloc[0]
        full_context = row["context_before"] + row["keyword"] + row["context_after"]
        assert full_context == short_text[0]

    def test_overlapping_matches(self, kwic_instance):
        """Test behavior with overlapping matches."""
        text_with_overlap = ["The the the quick brown fox."]

        results = kwic_instance(
            docs=text_with_overlap,
            patterns=["the"],
            matcher="characters",
            case_sensitive=False,
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3  # Should find all three "the" instances

    def test_special_characters_in_patterns(self, kwic_instance):
        """Test patterns containing special characters."""
        text_with_special = ["Hello, world! How are you? Fine."]

        results = kwic_instance(
            docs=text_with_special,
            patterns=[r"\w+[!?]"],  # Words followed by ! or ?
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1

    def test_unicode_text(self, kwic_instance):
        """Test with Unicode text."""
        unicode_texts = ["Café, naïve résumé", "Montréal façade"]

        results = kwic_instance(
            docs=unicode_texts,
            patterns=["é"],
            matcher="characters",
            window=10,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1

    def test_convert_patterns_to_spacy_no_regex_case_sensitive(self, kwic_instance):
        """Test _convert_patterns_to_spacy method with no regex, case sensitive."""
        patterns = ["test", "example"]
        result = kwic_instance._convert_patterns_to_spacy(
            patterns, case_sensitive=True, use_regex=False
        )

        expected = [[{"TEXT": "test"}], [{"TEXT": "example"}]]
        assert result == expected

    def test_convert_patterns_to_spacy_no_regex_case_insensitive(self, kwic_instance):
        """Test _convert_patterns_to_spacy method with no regex, case insensitive."""
        patterns = ["TEST", "EXAMPLE"]
        result = kwic_instance._convert_patterns_to_spacy(
            patterns, case_sensitive=False, use_regex=False
        )

        expected = [[{"LOWER": "TEST"}], [{"LOWER": "EXAMPLE"}]]
        assert result == expected

    def test_convert_patterns_to_spacy_with_regex_case_sensitive(self, kwic_instance):
        """Test _convert_patterns_to_spacy method with regex, case sensitive."""
        patterns = [r"te.*t", r"ex.*e"]
        result = kwic_instance._convert_patterns_to_spacy(
            patterns, case_sensitive=True, use_regex=True
        )

        expected = [[{"TEXT": {"REGEX": r"te.*t"}}], [{"TEXT": {"REGEX": r"ex.*e"}}]]
        assert result == expected

    def test_convert_patterns_to_spacy_with_regex_case_insensitive(self, kwic_instance):
        """Test _convert_patterns_to_spacy method with regex, case insensitive."""
        patterns = [r"TE.*T", r"EX.*E"]
        result = kwic_instance._convert_patterns_to_spacy(
            patterns, case_sensitive=False, use_regex=True
        )

        expected = [[{"LOWER": {"REGEX": r"te.*t"}}], [{"LOWER": {"REGEX": r"ex.*e"}}]]
        assert result == expected

    def test_rule_matcher_with_spacy_docs(
        self, kwic_instance, spacy_docs, sample_labels
    ):
        """Test rule-based matching with spaCy documents."""
        # Rule matcher requires spaCy token patterns, not simple strings
        token_patterns = [
            [{"LOWER": "quick"}],  # Match "quick" (case insensitive)
            [{"LOWER": "brown"}, {"LOWER": "fox"}],  # Match "brown fox" phrase
        ]

        results = kwic_instance(
            docs=spacy_docs,
            labels=sample_labels,
            patterns=token_patterns,
            matcher="rule",  # This will trigger lines 130-132
            window=3,
            as_df=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1
        # Check that matches were found
        keywords = [str(keyword).lower() for keyword in results["keyword"]]
        assert any("quick" in keyword for keyword in keywords) or any(
            "brown" in keyword for keyword in keywords
        )

    def test_invalid_sorting_algorithm_raises_error(self):
        """Test that invalid sorting algorithm raises ValidationError with proper error message.

        Note: It should not be possible to submit an invalid sorting algorithm, because Pydantic should prevent it. The commented out
        portion of this test is kept in case it is ever needed.
        """
        with pytest.raises(ValidationError):
            # Pass an invalid sorting algorithm
            Kwic(alg="INVALID_ALGORITHM")

        # Check that the error message contains the expected content
        # error_message = str(exc_info.value)
        # assert "Invalid sorting algorithm" in error_message
        # assert "Valid algorithms for `alg` are:" in error_message
        # assert "ns." in error_message  # This ensures line 270 was executed
        # assert "natsort.readthedocs.io" in error_message
