"""Tests for compare.py module.

Coverage: 99%. Missing: 200

Last Update: November 14, 2025
"""

import pandas as pd
import pytest
import spacy
from pydantic import ValidationError
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.topwords.compare import Compare
from lexos.topwords.ztest import ZTest

# ---------------- Fixtures ----------------


@pytest.fixture
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "The cat sat on the mat with a hat.",
        "The dog barked loudly at the cat.",
        "A quick brown fox jumps over the lazy dog.",
        "The bird sang sweetly in the morning.",
        "The fish swam quickly through the water.",
    ]


@pytest.fixture
def sample_docs(nlp, sample_texts):
    """Create sample spaCy Doc objects."""
    return [nlp(text) for text in sample_texts]


@pytest.fixture
def calculator():
    """Create a basic ZTest calculator for testing."""
    return ZTest(target_docs=[], comparison_docs=[], topn=5)


# ---------------- Basic Initialization Tests ----------------


class TestCompareInitialization:
    """Test initialization of Compare class."""

    def test_compare_init_minimal(self, calculator):
        """Test Compare initialization with just a calculator."""
        compare = Compare(calculator=calculator)

        assert compare is not None
        assert compare.calculator == calculator
        assert compare.data == []
        assert compare.results == {}

    def test_compare_init_validates_calculator_type(self):
        """Test that Compare requires a TopWords calculator."""
        with pytest.raises(ValidationError):
            Compare(calculator="not_a_calculator")

    def test_calculator_configuration_preserved(self):
        """Test that calculator configuration is preserved."""
        calc = ZTest(target_docs=[], comparison_docs=[], topn=20)
        compare = Compare(calculator=calc)

        assert compare.calculator.topn == 20


# ---------------- document_to_corpus Tests ----------------


class TestDocumentToCorpus:
    """Test document_to_corpus method."""

    def test_document_to_corpus_with_strings(self, calculator, nlp):
        """Test comparing each document to corpus with strings."""
        texts = [
            "The cat sat on the mat.",
            "The dog barked loudly.",
            "A bird sang sweetly.",
        ]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        result = compare.document_to_corpus(docs)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert "Doc 1" in result
        assert "Doc 2" in result
        assert "Doc 3" in result
        assert all(isinstance(v, list) for v in result.values())

    def test_document_to_corpus_with_custom_labels(self, calculator, nlp):
        """Test document_to_corpus with custom doc labels."""
        texts = ["Text one.", "Text two.", "Text three."]
        docs = [nlp(text) for text in texts]
        labels = ["Article A", "Article B", "Article C"]
        compare = Compare(calculator=calculator)

        result = compare.document_to_corpus(docs, doc_labels=labels)

        assert "Article A" in result
        assert "Article B" in result
        assert "Article C" in result

    def test_document_to_corpus_dataframe_output(self, calculator, nlp):
        """Test document_to_corpus with dataframe output."""
        texts = ["Cat meow.", "Dog bark.", "Bird chirp."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        result = compare.document_to_corpus(docs, output_format="dataframe")

        assert isinstance(result, pd.DataFrame)
        assert "doc_label" in result.index.name or "doc_label" in result.columns

    def test_document_to_corpus_list_of_dicts_output(self, calculator, nlp):
        """Test document_to_corpus with list_of_dicts output."""
        texts = ["Cat meow.", "Dog bark.", "Bird chirp."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        result = compare.document_to_corpus(docs, output_format="list_of_dicts")

        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
        if result:
            assert "doc_label" in result[0]
            assert "term" in result[0]
            assert "score" in result[0]

    def test_document_to_corpus_requires_min_two_docs(self, calculator, nlp):
        """Test that document_to_corpus requires at least 2 documents."""
        docs = [nlp("Only one document.")]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="at least two documents"):
            compare.document_to_corpus(docs)

    def test_document_to_corpus_caches_results(self, calculator, nlp):
        """Test that document_to_corpus caches results."""
        texts = ["Cat meow.", "Dog bark."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        compare.document_to_corpus(docs)

        assert compare.results != {}
        assert len(compare.results) == 2

    def test_document_to_corpus_stores_data(self, calculator, nlp):
        """Test that document_to_corpus stores data attribute."""
        texts = ["Cat meow.", "Dog bark."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        compare.document_to_corpus(docs)

        assert compare.data != []
        assert len(compare.data) == 2
        assert all("doc_label" in item for item in compare.data)
        assert all("doc" in item for item in compare.data)


# ---------------- documents_to_classes Tests ----------------


class TestDocumentsToClasses:
    """Test documents_to_classes method."""

    def test_documents_to_classes_basic(self, calculator, nlp):
        """Test comparing documents to classes."""
        texts = [
            "Shakespeare wrote plays.",
            "Shakespeare wrote sonnets.",
            "Marlowe wrote plays.",
            "Marlowe wrote poetry.",
        ]
        docs = [nlp(text) for text in texts]
        class_labels = ["Shakespeare", "Shakespeare", "Marlowe", "Marlowe"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(docs, class_labels=class_labels)

        assert isinstance(result, dict)
        assert len(result) == 4  # One result per document

    def test_documents_to_classes_with_custom_labels(self, calculator, nlp):
        """Test documents_to_classes with custom doc labels."""
        texts = ["Text A.", "Text B.", "Text C.", "Text D."]
        docs = [nlp(text) for text in texts]
        doc_labels = ["Doc1", "Doc2", "Doc3", "Doc4"]
        class_labels = ["ClassA", "ClassA", "ClassB", "ClassB"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(
            docs, doc_labels=doc_labels, class_labels=class_labels
        )

        assert "Doc1" in result
        assert "Doc2" in result
        assert "Doc3" in result
        assert "Doc4" in result

    def test_documents_to_classes_dataframe_output(self, calculator, nlp):
        """Test documents_to_classes with dataframe output."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(
            docs, class_labels=class_labels, output_format="dataframe"
        )

        assert isinstance(result, pd.DataFrame)
        assert "doc_label" in result.columns
        assert "comparison_class" in result.columns
        assert "term" in result.columns
        assert "score" in result.columns

    def test_documents_to_classes_list_of_dicts_output(self, calculator, nlp):
        """Test documents_to_classes with list_of_dicts output."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(
            docs, class_labels=class_labels, output_format="list_of_dicts"
        )

        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
        if result:
            assert "doc_label" in result[0]
            assert "comparison_class" in result[0]
            assert "term" in result[0]
            assert "score" in result[0]

    def test_documents_to_classes_requires_min_two_docs(self, calculator, nlp):
        """Test that documents_to_classes requires at least 2 documents."""
        docs = [nlp("Only one.")]
        class_labels = ["ClassA"]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="At least two documents"):
            compare.documents_to_classes(docs, class_labels=class_labels)

    def test_documents_to_classes_requires_min_two_classes(self, calculator, nlp):
        """Test that documents_to_classes requires at least 2 different classes."""
        texts = ["A.", "B.", "C."]
        docs = [nlp(text) for text in texts]
        class_labels = ["ClassA", "ClassA", "ClassA"]  # All same class
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="At least two different classes"):
            compare.documents_to_classes(docs, class_labels=class_labels)

    def test_documents_to_classes_requires_class_labels(self, calculator, nlp):
        """Test that documents_to_classes requires class_labels parameter."""
        texts = ["A.", "B."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="must be provided"):
            compare.documents_to_classes(docs)

    def test_documents_to_classes_result_structure(self, calculator, nlp):
        """Test the structure of documents_to_classes results."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(docs, class_labels=class_labels)

        # Check that results have comparison_class and topwords
        for doc_result in result.values():
            assert "comparison_class" in doc_result
            assert "topwords" in doc_result
            assert isinstance(doc_result["topwords"], list)


# ---------------- classes_to_classes Tests ----------------


class TestClassesToClasses:
    """Test classes_to_classes method."""

    def test_classes_to_classes_basic(self, calculator, nlp):
        """Test comparing classes to classes."""
        texts = [
            "Shakespeare wrote plays.",
            "Shakespeare wrote sonnets.",
            "Marlowe wrote plays.",
            "Marlowe wrote poetry.",
        ]
        docs = [nlp(text) for text in texts]
        class_labels = ["Shakespeare", "Shakespeare", "Marlowe", "Marlowe"]
        compare = Compare(calculator=calculator)

        result = compare.classes_to_classes(docs, class_labels=class_labels)

        assert isinstance(result, dict)
        assert len(result) == 2  # One result per class
        assert "Shakespeare" in result
        assert "Marlowe" in result

    def test_classes_to_classes_dataframe_output(self, calculator, nlp):
        """Test classes_to_classes with dataframe output."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.classes_to_classes(
            docs, class_labels=class_labels, output_format="dataframe"
        )

        assert isinstance(result, pd.DataFrame)
        assert "class_label" in result.columns
        assert "comparison_class" in result.columns
        assert "term" in result.columns
        assert "score" in result.columns

    def test_classes_to_classes_list_of_dicts_output(self, calculator, nlp):
        """Test classes_to_classes with list_of_dicts output."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.classes_to_classes(
            docs, class_labels=class_labels, output_format="list_of_dicts"
        )

        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
        if result:
            assert "class_label" in result[0]
            assert "comparison_class" in result[0]
            assert "term" in result[0]
            assert "score" in result[0]

    def test_classes_to_classes_requires_min_two_docs(self, calculator, nlp):
        """Test that classes_to_classes requires at least 2 documents."""
        docs = [nlp("Only one.")]
        class_labels = ["ClassA"]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="At least two documents"):
            compare.classes_to_classes(docs, class_labels=class_labels)

    def test_classes_to_classes_requires_min_two_classes(self, calculator, nlp):
        """Test that classes_to_classes requires at least 2 different classes."""
        texts = ["A.", "B.", "C."]
        docs = [nlp(text) for text in texts]
        class_labels = ["ClassA", "ClassA", "ClassA"]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="At least two different classes"):
            compare.classes_to_classes(docs, class_labels=class_labels)

    def test_classes_to_classes_result_structure(self, calculator, nlp):
        """Test the structure of classes_to_classes results."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        result = compare.classes_to_classes(docs, class_labels=class_labels)

        # Check that results have comparison_class and topwords
        for class_result in result.values():
            assert "comparison_class" in class_result
            assert "topwords" in class_result
            assert isinstance(class_result["topwords"], list)


# ---------------- convert_output Tests ----------------


class TestConvertOutput:
    """Test convert_output method."""

    def test_convert_output_after_document_to_corpus(self, calculator, nlp):
        """Test converting output format after document_to_corpus."""
        texts = ["Cat.", "Dog.", "Bird."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        # Run comparison first
        compare.document_to_corpus(docs)

        # Convert to dataframe
        df_result = compare.convert_output("dataframe")
        assert isinstance(df_result, pd.DataFrame)

        # Convert to list_of_dicts
        list_result = compare.convert_output("list_of_dicts")
        assert isinstance(list_result, list)

    def test_convert_output_after_documents_to_classes(self, calculator, nlp):
        """Test converting output format after documents_to_classes."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        # Run comparison first
        compare.documents_to_classes(docs, class_labels=class_labels)

        # Convert to dataframe
        df_result = compare.convert_output("dataframe")
        assert isinstance(df_result, pd.DataFrame)
        assert "doc_label" in df_result.columns

    def test_convert_output_after_classes_to_classes(self, calculator, nlp):
        """Test converting output format after classes_to_classes."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        # Run comparison first
        compare.classes_to_classes(docs, class_labels=class_labels)

        # Convert to dataframe
        df_result = compare.convert_output("dataframe")
        assert isinstance(df_result, pd.DataFrame)
        assert "class_label" in df_result.columns

    def test_convert_output_requires_cached_results(self, calculator):
        """Test that convert_output requires cached results."""
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="No results cached"):
            compare.convert_output("dataframe")

    def test_convert_output_invalid_format(self, calculator, nlp):
        """Test that convert_output validates output format."""
        texts = ["A.", "B."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        compare.document_to_corpus(docs)

        with pytest.raises(LexosException, match="Unsupported output_format"):
            compare.convert_output("invalid_format")


# ---------------- Helper Method Tests ----------------


class TestHelperMethods:
    """Test helper methods."""

    def test_get_class(self, calculator, nlp):
        """Test get_class method."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        doc_labels = ["Doc1", "Doc2", "Doc3", "Doc4"]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        compare.documents_to_classes(
            docs, doc_labels=doc_labels, class_labels=class_labels
        )

        assert compare.get_class("Doc1") == "X"
        assert compare.get_class("Doc2") == "X"
        assert compare.get_class("Doc3") == "Y"
        assert compare.get_class("Doc4") == "Y"

    def test_get_class_not_found(self, calculator, nlp):
        """Test get_class with non-existent label."""
        texts = ["A.", "B."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "Y"]
        compare = Compare(calculator=calculator)

        compare.documents_to_classes(docs, class_labels=class_labels)

        with pytest.raises(LexosException, match="not found"):
            compare.get_class("NonExistent")

    def test_get_docs_by_class(self, calculator, nlp):
        """Test get_docs_by_class method."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        doc_labels = ["Doc1", "Doc2", "Doc3", "Doc4"]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        compare.documents_to_classes(
            docs, doc_labels=doc_labels, class_labels=class_labels
        )

        result = compare.get_docs_by_class()

        assert "X" in result
        assert "Y" in result
        assert len(result["X"]) == 2
        assert len(result["Y"]) == 2
        assert "Doc1" in result["X"]
        assert "Doc2" in result["X"]
        assert "Doc3" in result["Y"]
        assert "Doc4" in result["Y"]

    def test_get_docs_by_class_specific_class(self, calculator, nlp):
        """Test get_docs_by_class for a specific class."""
        texts = ["A.", "B.", "C.", "D."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "X", "Y", "Y"]
        compare = Compare(calculator=calculator)

        compare.documents_to_classes(docs, class_labels=class_labels)

        result = compare.get_docs_by_class("X")

        assert "X" in result
        assert "Y" not in result
        assert len(result["X"]) == 2


# ---------------- Edge Cases and Error Handling ----------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_doc_input_converted_to_list(self, calculator, nlp):
        """Test that single document is converted to list."""
        doc = nlp("Single document.")
        compare = Compare(calculator=calculator)

        # Should raise error because we need at least 2 docs
        with pytest.raises(LexosException):
            compare.document_to_corpus(doc)

    def test_empty_results_handling(self, calculator, nlp):
        """Test handling of empty/minimal results."""
        texts = ["A.", "B."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        result = compare.document_to_corpus(docs)

        # Should still return a dict structure even if topwords are empty
        assert isinstance(result, dict)

    def test_invalid_output_format(self, calculator, nlp):
        """Test invalid output format raises error."""
        texts = ["A.", "B."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="Unsupported output_format"):
            compare.document_to_corpus(docs, output_format="invalid")

    def test_docs_dict_with_missing_required_keys(self, calculator, nlp):
        """Test that _validate_docs_dict raises error for missing required keys."""
        # Create docs dict missing 'class_label'
        docs = [
            {"doc": nlp("Text one.")},
            {"doc": nlp("Text two.")},
        ]
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="must contain the keys"):
            compare.documents_to_classes(docs)

    def test_docs_dict_auto_generates_doc_label(self, calculator, nlp):
        """Test that _validate_docs_dict auto-generates doc_label when missing."""
        docs = [
            {"doc": nlp("Text one."), "class_label": "A"},
            {"doc": nlp("Text two."), "class_label": "B"},
        ]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(docs)

        # Check that auto-generated labels were used
        assert "Doc 1" in result
        assert "Doc 2" in result

    def test_docs_dict_preserves_existing_doc_label(self, calculator, nlp):
        """Test that _validate_docs_dict preserves existing doc_label."""
        docs = [
            {"doc": nlp("Text one."), "class_label": "A", "doc_label": "Custom1"},
            {"doc": nlp("Text two."), "class_label": "B", "doc_label": "Custom2"},
        ]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(docs)

        # Check that custom labels were preserved
        assert "Custom1" in result
        assert "Custom2" in result

    def test_mismatched_doc_classes_length(self, calculator, nlp):
        """Test error when doc_classes length doesn't match docs length."""
        texts = ["A.", "B.", "C."]
        docs = [nlp(text) for text in texts]
        class_labels = ["X", "Y"]  # Only 2 labels for 3 docs
        compare = Compare(calculator=calculator)

        with pytest.raises(LexosException, match="must be the same length"):
            compare.documents_to_classes(docs, class_labels=class_labels)

    def test_convert_output_label_key_auto_detection_document_to_corpus(
        self, calculator, nlp
    ):
        """Test that convert_output auto-detects doc_label for document_to_corpus."""
        texts = ["A.", "B.", "C."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        # Run document_to_corpus (no class_label in data)
        compare.document_to_corpus(docs)

        # Convert without specifying label_key - should auto-detect doc_label
        result = compare.convert_output("dataframe", label_key=None)

        assert isinstance(result, pd.DataFrame)
        # Should use doc_label as index or column
        assert "doc_label" in result.index.name or "doc_label" in result.columns

    def test_docs_with_spacy_extension_for_class_labels(self, calculator, nlp):
        """Test using spaCy Doc extension attributes for class labels."""
        # Set up custom extension
        if not Doc.has_extension("author"):
            Doc.set_extension("author", default=None, force=True)

        # Create docs with extension attributes
        doc1 = nlp("Shakespeare wrote plays.")
        doc1._.author = "Shakespeare"
        doc2 = nlp("Marlowe wrote plays.")
        doc2._.author = "Marlowe"
        doc3 = nlp("Shakespeare wrote sonnets.")
        doc3._.author = "Shakespeare"
        doc4 = nlp("Marlowe wrote poetry.")
        doc4._.author = "Marlowe"

        docs = [doc1, doc2, doc3, doc4]
        # Use extension name as class_labels parameter
        class_labels = ["author", "author", "author", "author"]
        compare = Compare(calculator=calculator)

        result = compare.documents_to_classes(docs, class_labels=class_labels)

        # Should successfully create results using extension values
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_create_data_dict_with_invalid_type(self, calculator, nlp):
        """Test _create_data_dict with non-list, non-dict type to cover line 234."""
        compare = Compare(calculator=calculator)

        # Create a non-list, non-indexable object (like a generator or custom object)
        # We need to bypass ensure_list to test this edge case
        class NotAList:
            """A class that is neither a list nor has dict-like first element."""

            def __getitem__(self, index):
                # Return something that's not a dict to trigger the isinstance check
                return "not a dict"

        invalid_docs = NotAList()

        with pytest.raises(
            LexosException, match="must be a list of dicts, strings, or Doc objects"
        ):
            compare._create_data_dict(invalid_docs)

    def test_convert_output_with_simple_results_format(self, calculator, nlp):
        """Test convert_output with results that don't have 'topwords' key to cover line 200."""
        texts = ["A.", "B.", "C."]
        docs = [nlp(text) for text in texts]
        compare = Compare(calculator=calculator)

        # Run document_to_corpus which produces simple list results (not dict with 'topwords')
        compare.document_to_corpus(docs)

        # Now the results should be simple lists, not dicts with 'topwords'
        # Manually modify results to ensure we test the code path
        # The auto-detection should handle this case
        result = compare.convert_output("dataframe", label_key=None)

        assert isinstance(result, pd.DataFrame)
