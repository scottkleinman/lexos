"""test_communication_architecture.py.

Comprehensive tests for corpus communication architecture methods.
Targets 100% coverage of import_analysis_results, get_analysis_results,
export_statistical_fingerprint, and validation methods.

Coverage: 100%
Last Update: 2025-06-20.
"""

import tempfile
from pathlib import Path

import pytest

from lexos.corpus.corpus import Corpus


class TestCorpusCommunicationArchitecture:
    """Test communication architecture methods comprehensively."""

    @pytest.fixture
    def temp_corpus_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def populated_corpus(self, temp_corpus_dir):
        """Create a corpus with some test data."""
        corpus = Corpus(corpus_dir=temp_corpus_dir, name="TestCorpus")
        corpus.add("Hello world document", name="doc1")
        corpus.add("Another test document", name="doc2")
        return corpus

    def test_import_analysis_results_basic(self, populated_corpus):
        """Test basic import_analysis_results functionality."""
        test_results = {
            "clusters": [0, 1],
            "silhouette_score": 0.75,
            "analysis_type": "kmeans",
        }

        # Test basic import
        populated_corpus.import_analysis_results(
            "kmeans", test_results, version="1.0.0"
        )

        # Verify results were stored
        assert "kmeans" in populated_corpus.analysis_results
        stored = populated_corpus.analysis_results["kmeans"]
        assert stored["version"] == "1.0.0"
        assert stored["results"] == test_results
        assert "timestamp" in stored
        assert "corpus_state" in stored

        print("✓ Basic import_analysis_results works")

    def test_import_analysis_results_overwrite_protection(self, populated_corpus):
        """Test overwrite protection in import_analysis_results."""
        test_results = {"test": "data"}

        # Import initial results
        populated_corpus.import_analysis_results("test_module", test_results)

        # Try to import again without overwrite - should raise ValueError (lines 652-656)
        with pytest.raises(
            ValueError, match="Results for module 'test_module' already exist"
        ):
            populated_corpus.import_analysis_results("test_module", {"new": "data"})

        # Import with overwrite=True should work
        populated_corpus.import_analysis_results(
            "test_module", {"new": "data"}, overwrite=True
        )
        assert populated_corpus.analysis_results["test_module"]["results"] == {
            "new": "data"
        }

        print("✓ Overwrite protection works correctly")

    def test_get_analysis_results_specific_module(self, populated_corpus):
        """Test get_analysis_results with specific module name."""
        test_results = {"specific": "data"}
        populated_corpus.import_analysis_results("specific_module", test_results)

        # Test retrieving specific module (lines 682-685)
        results = populated_corpus.get_analysis_results("specific_module")
        assert results["results"] == test_results
        assert results["version"] == "1.0.0"

        print("✓ Get specific module results works")

    def test_get_analysis_results_module_not_found(self, populated_corpus):
        """Test get_analysis_results with non-existent module."""
        # Test error when module doesn't exist (lines 683-684)
        with pytest.raises(
            ValueError, match="No results found for module 'nonexistent'"
        ):
            populated_corpus.get_analysis_results("nonexistent")

        print("✓ Module not found error handling works")

    def test_get_analysis_results_all_modules(self, populated_corpus):
        """Test get_analysis_results without module name."""
        # Add multiple modules
        populated_corpus.import_analysis_results("module1", {"data1": "test"})
        populated_corpus.import_analysis_results("module2", {"data2": "test"})

        # Test retrieving all results (line 687)
        all_results = populated_corpus.get_analysis_results()
        assert "module1" in all_results
        assert "module2" in all_results
        assert len(all_results) == 2

        print("✓ Get all module results works")

    def test_export_statistical_fingerprint_success(self, populated_corpus):
        """Test successful export_statistical_fingerprint."""
        # Test successful export (lines 702-732)
        fingerprint = populated_corpus.export_statistical_fingerprint()

        # Verify structure
        assert "corpus_metadata" in fingerprint
        assert "distribution_stats" in fingerprint
        assert "percentiles" in fingerprint
        assert "text_diversity" in fingerprint
        assert "basic_stats" in fingerprint
        assert "document_features" in fingerprint
        assert "term_frequencies" in fingerprint

        # Verify corpus metadata
        metadata = fingerprint["corpus_metadata"]
        assert metadata["name"] == "TestCorpus"
        assert metadata["num_docs"] == 2
        assert "corpus_fingerprint" in metadata

        print("✓ Successful statistical fingerprint export works")

    def test_export_statistical_fingerprint_error_fallback(self, temp_corpus_dir):
        """Test export_statistical_fingerprint error fallback."""
        # Create corpus that will cause CorpusStats to fail
        corpus = Corpus(corpus_dir=temp_corpus_dir, name="ErrorCorpus")
        # Don't add any documents, which might cause CorpusStats issues

        # Test fallback when CorpusStats fails (lines 730-746)
        fingerprint = corpus.export_statistical_fingerprint()

        # Should have fallback structure
        assert "corpus_metadata" in fingerprint
        assert "error" in fingerprint
        assert "basic_features" in fingerprint

        # Verify fallback metadata
        metadata = fingerprint["corpus_metadata"]
        assert metadata["name"] == "ErrorCorpus"
        assert metadata["num_docs"] == 0

        print("✓ Statistical fingerprint error fallback works")

    def test_generate_corpus_fingerprint(self, populated_corpus):
        """Test _generate_corpus_fingerprint method."""
        # Test fingerprint generation (lines 754-765)
        fingerprint = populated_corpus._generate_corpus_fingerprint()

        # Should be a 16-character hash
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16

        # Same corpus should generate same fingerprint
        fingerprint2 = populated_corpus._generate_corpus_fingerprint()
        assert fingerprint == fingerprint2

        # Different corpus state should generate different fingerprint
        populated_corpus.add("New document changes fingerprint", name="doc3")
        fingerprint3 = populated_corpus._generate_corpus_fingerprint()
        assert fingerprint != fingerprint3

        print("✓ Corpus fingerprint generation works")

    def test_validate_analysis_compatibility_compatible(self, populated_corpus):
        """Test validate_analysis_compatibility with compatible results."""
        # Import results first
        test_results = {"test": "data"}
        populated_corpus.import_analysis_results("compatible_module", test_results)

        # Test compatibility validation (lines 777-814)
        compatibility = populated_corpus.validate_analysis_compatibility(
            "compatible_module"
        )

        # Should be compatible since corpus hasn't changed
        assert compatibility["compatible"] is True
        assert "current_fingerprint" in compatibility
        assert "stored_fingerprint" in compatibility
        assert (
            compatibility["current_fingerprint"] == compatibility["stored_fingerprint"]
        )

        print("✓ Compatible analysis validation works")

    def test_validate_analysis_compatibility_incompatible(self, populated_corpus):
        """Test validate_analysis_compatibility with incompatible results."""
        # Import results first
        test_results = {"test": "data"}
        populated_corpus.import_analysis_results("incompatible_module", test_results)

        # Change corpus state
        populated_corpus.add("This changes the corpus fingerprint", name="doc3")

        # Test compatibility validation
        compatibility = populated_corpus.validate_analysis_compatibility(
            "incompatible_module"
        )

        # Should be incompatible since corpus changed
        assert compatibility["compatible"] is False
        assert "reason" in compatibility
        assert "recommendation" in compatibility
        assert "state_changes" in compatibility
        assert (
            compatibility["current_fingerprint"] != compatibility["stored_fingerprint"]
        )

        # Check state changes details
        state_changes = compatibility["state_changes"]
        assert "num_docs" in state_changes
        assert "num_active_docs" in state_changes
        assert state_changes["num_docs"]["changed"] is True
        assert state_changes["num_docs"]["current"] == 3
        assert state_changes["num_docs"]["stored"] == 2

        print("✓ Incompatible analysis validation works")

    def test_validate_analysis_compatibility_no_results(self, populated_corpus):
        """Test validate_analysis_compatibility with no existing results."""
        # Test validation for non-existent module
        compatibility = populated_corpus.validate_analysis_compatibility(
            "nonexistent_module"
        )

        assert compatibility["compatible"] is False
        assert "No analysis results found" in compatibility["reason"]

        print("✓ No results validation works")

    def test_corpus_state_tracking_in_results(self, populated_corpus):
        """Test that corpus state is properly tracked in analysis results."""
        # Import results and check corpus state tracking
        test_results = {"tracking": "test"}
        populated_corpus.import_analysis_results("tracking_module", test_results)

        stored_results = populated_corpus.analysis_results["tracking_module"]
        corpus_state = stored_results["corpus_state"]

        # Verify corpus state was captured correctly
        assert corpus_state["num_docs"] == populated_corpus.num_docs
        assert corpus_state["num_active_docs"] == populated_corpus.num_active_docs
        assert len(corpus_state["corpus_fingerprint"]) == 16

        print("✓ Corpus state tracking in results works")

    def test_analysis_results_versioning(self, populated_corpus):
        """Test versioning in analysis results."""
        test_results = {"version": "test"}

        # Test default version
        populated_corpus.import_analysis_results("version_module", test_results)
        assert populated_corpus.analysis_results["version_module"]["version"] == "1.0.0"

        # Test custom version
        populated_corpus.import_analysis_results(
            "version_module2", test_results, version="2.1.0"
        )
        assert (
            populated_corpus.analysis_results["version_module2"]["version"] == "2.1.0"
        )

        print("✓ Analysis results versioning works")
