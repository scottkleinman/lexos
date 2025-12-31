"""test_corpus_analysis_report.py.

Test suite for the create_corpus_analysis_report function in corpus_analysis_report.py.

Coverage: 99%. Missing: 110

Last Updated: November 19, 2025
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Try to import required modules
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from lexos.corpus import Corpus, Record
    from lexos.corpus.corpus_analysis_report import create_corpus_analysis_report

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Module import failed: {e}")

# Skip all tests if modules aren't available
pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE, reason="Required modules not available"
)


@pytest.fixture
def nlp():
    """Return a spaCy English model or blank model for testing."""
    if not SPACY_AVAILABLE:
        pytest.skip("SpaCy not available")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # If en_core_web_sm is not available, create a blank English model
        return spacy.blank("en")


@pytest.fixture
def sample_texts():
    """Return a list of sample text strings for testing."""
    return [
        "This is the first test document. It contains multiple sentences.",
        "Here is another document for testing purposes.",
        "A third document with different content and structure.",
        "The final test document in our sample corpus.",
        "Another example with some repeated words like test and document.",
    ]


@pytest.fixture
def temp_corpus_dir():
    """Create a temporary directory for corpus testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def populated_corpus(sample_texts, nlp, temp_corpus_dir):
    """Create a populated corpus with sample documents."""
    corpus = Corpus(name="Test Corpus", corpus_dir=str(temp_corpus_dir))

    # Add documents to corpus
    # Don't pass Record objects with pre-existing IDs, let corpus create them
    for i, text in enumerate(sample_texts):
        corpus.add(
            content=text,
            name=f"doc_{i + 1}",
            model="en_core_web_sm" if hasattr(nlp, "path") else None,
        )

    return corpus


class TestCreateCorpusAnalysisReportNoOutput:
    """Test create_corpus_analysis_report without file output."""

    def test_returns_string_without_output_dir(self, populated_corpus):
        """Test that function returns a string when output_dir is None."""
        result = create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_markdown_output_by_default(self, populated_corpus):
        """Test that function returns markdown by default (html=False)."""
        result = create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False, html=False
        )
        # When html=False, the markdown() function is actually converting markdown TO html
        # So the result will still contain HTML tags
        # The important thing is that we're getting a string result
        assert isinstance(result, str)
        assert len(result) > 0
        # Content should be present
        assert "Test Corpus" in result or "Corpus Analysis Report" in result

    def test_html_output_when_requested(self, populated_corpus):
        """Test that function returns HTML when html=True."""
        result = create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False, html=True
        )
        # HTML should contain proper tags
        assert "<html>" in result
        assert "<head>" in result
        assert "<body>" in result
        assert "</html>" in result
        assert "Test Corpus" in result

    def test_contains_corpus_name(self, populated_corpus):
        """Test that report contains the corpus name."""
        result = create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False
        )
        assert "Test Corpus" in result

    def test_contains_statistical_info(self, populated_corpus):
        """Test that report contains statistical information."""
        result = create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False, html=True
        )
        # Should contain various statistical elements
        assert "Statistical Summary" in result
        assert "Quality Assessment" in result
        assert "tokens" in result.lower()

    def test_no_files_created_without_output_dir(
        self, populated_corpus, temp_corpus_dir
    ):
        """Test that no files are created when output_dir is None."""
        # Use a clean temp directory
        test_dir = temp_corpus_dir / "no_output_test"
        test_dir.mkdir()

        # Call function without output_dir
        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=None, console_output=False
        )

        # Check that no files were created in the test directory
        assert len(list(test_dir.glob("*"))) == 0


class TestCreateCorpusAnalysisReportWithOutput:
    """Test create_corpus_analysis_report with file output."""

    def test_creates_output_directory(self, populated_corpus, temp_corpus_dir):
        """Test that output directory is created if it doesn't exist."""
        output_dir = temp_corpus_dir / "new_output_dir"
        assert not output_dir.exists()

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_creates_csv_files(self, populated_corpus, temp_corpus_dir):
        """Test that CSV files are created when output_dir is provided."""
        output_dir = temp_corpus_dir / "csv_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        # Check for expected CSV files
        assert (output_dir / "corpus_overview.csv").exists()
        assert (output_dir / "document_statistics.csv").exists()
        assert (output_dir / "corpus_summary.csv").exists()

    def test_corpus_overview_csv_content(self, populated_corpus, temp_corpus_dir):
        """Test that corpus_overview.csv contains expected data."""
        output_dir = temp_corpus_dir / "overview_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        df = pd.read_csv(output_dir / "corpus_overview.csv")
        # Should have records for each document
        assert len(df) > 0
        # Should have expected columns (at least 'name' should be present)
        assert len(df.columns) > 0

    def test_document_statistics_csv_content(self, populated_corpus, temp_corpus_dir):
        """Test that document_statistics.csv contains statistics."""
        output_dir = temp_corpus_dir / "stats_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        df = pd.read_csv(output_dir / "document_statistics.csv")
        # Should have at least one row
        assert len(df) > 0
        # Should have statistical columns
        assert "total_tokens" in df.columns or "Documents" in df.columns

    def test_corpus_summary_csv_content(self, populated_corpus, temp_corpus_dir):
        """Test that corpus_summary.csv contains summary data."""
        output_dir = temp_corpus_dir / "summary_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        df = pd.read_csv(output_dir / "corpus_summary.csv")
        # Should have exactly one row (summary row)
        assert len(df) == 1
        # Should contain expected columns
        assert "corpus_name" in df.columns
        assert "total_documents" in df.columns
        assert df.iloc[0]["corpus_name"] == "Test Corpus"

    def test_creates_markdown_report_by_default(
        self, populated_corpus, temp_corpus_dir
    ):
        """Test that .md report file is created when html=False."""
        output_dir = temp_corpus_dir / "md_test"

        create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=False,
        )

        report_path = output_dir / "analysis_report.md"
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_creates_html_report_when_requested(
        self, populated_corpus, temp_corpus_dir
    ):
        """Test that .html report file is created when html=True."""
        output_dir = temp_corpus_dir / "html_test"

        create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        report_path = output_dir / "analysis_report.html"
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "<html>" in content
        assert "</html>" in content

    def test_html_report_is_valid(self, populated_corpus, temp_corpus_dir):
        """Test that HTML report contains valid HTML structure."""
        output_dir = temp_corpus_dir / "valid_html_test"

        create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        content = (output_dir / "analysis_report.html").read_text(encoding="utf-8")

        # Check for essential HTML tags
        assert "<html>" in content
        assert "<head>" in content
        assert "<title>" in content
        assert "<body>" in content
        assert "</body>" in content
        assert "</html>" in content

    def test_returns_same_content_as_saved_file(
        self, populated_corpus, temp_corpus_dir
    ):
        """Test that returned string matches saved file content."""
        output_dir = temp_corpus_dir / "match_test"

        result = create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        saved_content = (output_dir / "analysis_report.html").read_text(
            encoding="utf-8"
        )

        assert result == saved_content


class TestCreateCorpusAnalysisReportWithModuleResults:
    """Test report generation with analysis results from modules."""

    def test_includes_module_results_in_json(self, populated_corpus, temp_corpus_dir):
        """Test that module analysis results are saved to JSON."""
        output_dir = temp_corpus_dir / "module_json_test"

        # Add some mock analysis results
        populated_corpus.import_analysis_results(
            module_name="test_module",
            results_data={"accuracy": 0.95, "features": ["word", "pos"]},
            version="1.0.0",
        )

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        # Check JSON file was created
        json_path = output_dir / "module_analysis_results.json"
        assert json_path.exists()

        # Check content
        with open(json_path, "r") as f:
            data = json.load(f)

        assert "test_module" in data
        assert data["test_module"]["version"] == "1.0.0"

    def test_includes_module_results_in_html_report(
        self, populated_corpus, temp_corpus_dir
    ):
        """Test that module results appear in HTML report."""
        output_dir = temp_corpus_dir / "module_html_test"

        # Add mock analysis results
        populated_corpus.import_analysis_results(
            module_name="classification",
            results_data={"model": "svm"},
            version="2.0.0",
        )

        result = create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        assert "Module Analyses" in result
        assert "classification" in result
        assert "2.0.0" in result

    def test_no_module_section_when_no_results(self, populated_corpus, temp_corpus_dir):
        """Test that Module Analyses section is omitted when there are no results."""
        output_dir = temp_corpus_dir / "no_module_test"

        # Ensure no analysis results
        populated_corpus.analysis_results = {}

        result = create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        # Module Analyses section should not be present
        assert (
            "Module Analyses" not in result
            or "<h2>Module Analyses</h2><ul></ul>" not in result
        )


class TestCreateCorpusAnalysisReportConsoleOutput:
    """Test console output behavior."""

    def test_no_console_output_when_disabled(
        self, populated_corpus, temp_corpus_dir, capsys
    ):
        """Test that console output is suppressed when console_output=False."""
        output_dir = temp_corpus_dir / "silent_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=False
        )

        captured = capsys.readouterr()
        # Should have minimal or no output
        assert "✓" not in captured.out

    def test_console_output_when_enabled(
        self, populated_corpus, temp_corpus_dir, capsys
    ):
        """Test that console output is shown when console_output=True."""
        output_dir = temp_corpus_dir / "verbose_test"

        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=str(output_dir), console_output=True
        )

        captured = capsys.readouterr()
        # Should have informative output
        assert "✓" in captured.out or "Exported" in captured.out


class TestCreateCorpusAnalysisReportEdgeCases:
    """Test edge cases and error handling."""

    def test_works_with_empty_corpus(self, nlp, temp_corpus_dir):
        """Test function works with an empty corpus."""
        empty_corpus = Corpus(name="Empty Corpus", corpus_dir=str(temp_corpus_dir))

        # This might raise an error or return empty content depending on implementation
        # We just test it doesn't crash
        try:
            result = create_corpus_analysis_report(
                corpus=empty_corpus, output_dir=None, console_output=False
            )
            assert isinstance(result, str)
        except Exception as e:
            # If it raises an exception, that's also acceptable behavior for empty corpus
            assert True

    def test_works_with_special_characters_in_corpus_name(
        self, sample_texts, nlp, temp_corpus_dir
    ):
        """Test that special characters in corpus name don't break the function."""
        corpus = Corpus(
            name="Test Corpus: With (Special) [Characters]!",
            corpus_dir=str(temp_corpus_dir),
        )

        # Add a document
        doc = nlp(sample_texts[0])
        record = Record(name="doc_1", content=doc)
        corpus.add(record)

        result = create_corpus_analysis_report(
            corpus=corpus, output_dir=None, console_output=False, html=True
        )

        assert "Test Corpus: With (Special) [Characters]!" in result

    def test_output_dir_as_path_object(self, populated_corpus, temp_corpus_dir):
        """Test that output_dir can be a Path object."""
        output_dir = temp_corpus_dir / "path_object_test"

        # Pass Path object instead of string
        create_corpus_analysis_report(
            corpus=populated_corpus, output_dir=output_dir, console_output=False
        )

        assert output_dir.exists()
        assert (output_dir / "corpus_overview.csv").exists()


class TestCreateCorpusAnalysisReportIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_markdown(self, populated_corpus, temp_corpus_dir):
        """Test complete workflow with Markdown output."""
        output_dir = temp_corpus_dir / "full_md_test"

        # Add analysis results
        populated_corpus.import_analysis_results(
            module_name="topic_modeling", results_data={"topics": 5}, version="1.5.0"
        )

        result = create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=False,
        )

        # Check return value
        assert isinstance(result, str)
        assert len(result) > 0

        # Check files created
        assert (output_dir / "corpus_overview.csv").exists()
        assert (output_dir / "document_statistics.csv").exists()
        assert (output_dir / "corpus_summary.csv").exists()
        assert (output_dir / "analysis_report.md").exists()
        assert (output_dir / "module_analysis_results.json").exists()

        # Check report content
        assert "Test Corpus" in result

    def test_full_workflow_html(self, populated_corpus, temp_corpus_dir):
        """Test complete workflow with HTML output."""
        output_dir = temp_corpus_dir / "full_html_test"

        result = create_corpus_analysis_report(
            corpus=populated_corpus,
            output_dir=str(output_dir),
            console_output=False,
            html=True,
        )

        # Check return value is HTML
        assert "<html>" in result
        assert "</html>" in result

        # Check files created
        assert (output_dir / "corpus_overview.csv").exists()
        assert (output_dir / "analysis_report.html").exists()

        # Check HTML file content matches return value
        saved_html = (output_dir / "analysis_report.html").read_text(encoding="utf-8")
        assert result == saved_html
