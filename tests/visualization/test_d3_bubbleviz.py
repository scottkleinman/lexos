"""test_d3_bubbleviz.py.

Coverage: 98%. Missing: 150
Line 150 is skipped because tests were run in WSL. At some point, this should be run in an environment that can find the pytest temporary directory.

Last Update: August 12, 2025
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import spacy

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.d3_bubbleviz import D3BubbleChart


# Load spaCy model once for all tests
@pytest.fixture(scope="session")
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog. The fox is quick."


@pytest.fixture
def sample_counts():
    """Sample word counts dictionary."""
    return {
        "the": 3,
        "fox": 2,
        "quick": 2,
        "brown": 1,
        "jumps": 1,
        "over": 1,
        "lazy": 1,
        "dog": 1,
        "is": 1,
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {"doc1": [2, 1, 3, 0], "doc2": [1, 3, 0, 2], "doc3": [0, 2, 1, 1]},
        index=["term1", "term2", "term3", "term4"],
    )


@pytest.fixture
def mock_template_content():
    """Mock HTML template content."""
    return """
    <html>
    <head><title>{{ title }}</title></head>
    <body>
        <div id="chart"></div>
        <script>
            var data = {{ term_counts | tojson }};
            var width = {{ width }};
            var height = {{ height }};
            var margin = {{ margin | tojson }};
            var color = "{{ color }}";
        </script>
    </body>
    </html>
    """


class TestD3BubbleChartInitialization:
    """Test D3BubbleChart initialization with different data types."""

    def test_init_with_string_data(self, sample_text, mock_template_content):
        """Test initialization with string data."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_text, auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0
                assert chart.html is not None

    def test_init_with_dict_data(self, sample_counts, mock_template_content):
        """Test initialization with dictionary data."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)
                assert chart.counts == sample_counts
                assert chart.html is not None

    def test_init_with_spacy_doc(self, nlp, sample_text, mock_template_content):
        """Test initialization with spaCy Doc."""
        doc = nlp(sample_text)
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=doc, auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0

    def test_init_with_spacy_span(self, nlp, sample_text, mock_template_content):
        """Test initialization with spaCy Span."""
        doc = nlp(sample_text)
        span = doc[:5]  # First 5 tokens
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=span, auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0

    def test_init_with_token_list(self, nlp, sample_text, mock_template_content):
        """Test initialization with list of spaCy tokens."""
        doc = nlp(sample_text)
        tokens = [token for token in doc if not token.is_punct]
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=tokens, auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0

    def test_init_with_string_list(self, mock_template_content):
        """Test initialization with list of strings."""
        data = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=data, auto_open=False)
                assert chart.counts["apple"] == 3
                assert chart.counts["banana"] == 2
                assert chart.counts["cherry"] == 1

    def test_init_with_dataframe(self, sample_dataframe, mock_template_content):
        """Test initialization with DataFrame."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_dataframe, auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0


class TestD3BubbleChartParameters:
    """Test D3BubbleChart with different parameter configurations."""

    def test_custom_title(self, sample_counts, mock_template_content):
        """Test chart with custom title."""
        title = "My Custom Bubble Chart"
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, title=title, auto_open=False)
                assert chart.title == title
                assert title in chart.html

    def test_custom_dimensions(self, sample_counts, mock_template_content):
        """Test chart with custom width and height."""
        width, height = 800, 500
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(
                    data=sample_counts, width=width, height=height, auto_open=False
                )
                assert chart.width == width
                assert chart.height == height
                assert str(width) in chart.html
                assert str(height) in chart.html

    def test_custom_margin(self, sample_counts, mock_template_content):
        """Test chart with custom margins."""
        margin = {"top": 30, "right": 30, "bottom": 30, "left": 30}
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(
                    data=sample_counts, margin=margin, auto_open=False
                )
                assert chart.margin == margin

    def test_custom_color_scheme(self, sample_counts, mock_template_content):
        """Test chart with custom color scheme."""
        color = "schemeSet3"
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, color=color, auto_open=False)
                assert chart.color == color
                assert color in chart.html

    def test_custom_color_list(self, sample_counts, mock_template_content):
        """Test chart with custom color list."""
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, color=colors, auto_open=False)
                assert chart.color == colors

    def test_limit_parameter(self, sample_counts, mock_template_content):
        """Test chart with limit parameter."""
        limit = 3
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, limit=limit, auto_open=False)
                assert len(chart.counts) <= limit

    def test_include_d3js_parameter(self, sample_counts, mock_template_content):
        """Test chart with include_d3js parameter."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(
                    data=sample_counts, include_d3js=True, auto_open=False
                )
                assert chart.include_d3js is True


class TestD3BubbleChartMethods:
    """Test D3BubbleChart methods."""

    @pytest.mark.skip(
        reason="Tested in WSL which has trouble finding the temporary directory."
    )
    def test_save_method(self, sample_counts, mock_template_content, tmp_path):
        """Test saving chart to file."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)

                output_path = tmp_path / "test_chart.html"
                chart.save(output_path)

                assert output_path.exists()

    def test_get_asset_path(self, sample_counts, mock_template_content):
        """Test asset path resolution."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)

                asset_path = chart._get_asset_path("test_file.html")
                expected_path = (
                    Path(__file__).parent.parent
                    / "src"
                    / "lexos"
                    / "visualization"
                    / "d3_cloud_assets"
                    / "test_file.html"
                )

                assert isinstance(asset_path, Path)
                assert asset_path.name == "test_file.html"

    def test_open_method(self, sample_counts, mock_template_content):
        """Test opening chart in browser."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open") as mock_browser:
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = (
                        "/tmp/test.html"
                    )

                    chart = D3BubbleChart(data=sample_counts, auto_open=False)
                    chart._open()

                    mock_browser.assert_called_once()

    def test_auto_open_true(self, sample_counts, mock_template_content):
        """Test automatic opening when auto_open is True."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open") as mock_browser:
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = (
                        "/tmp/test.html"
                    )

                    D3BubbleChart(data=sample_counts, auto_open=True)
                    mock_browser.assert_called_once()

    def test_auto_open_false(self, sample_counts, mock_template_content):
        """Test no automatic opening when auto_open is False."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open") as mock_browser:
                D3BubbleChart(data=sample_counts, auto_open=False)
                mock_browser.assert_not_called()


class TestD3BubbleChartErrorHandling:
    """Test error handling in D3BubbleChart."""

    def test_template_not_found(self, sample_counts):
        """Test handling of missing template file."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(LexosException, match="Template file not found"):
                D3BubbleChart(data=sample_counts, auto_open=False)

    def test_invalid_save_path(self, sample_counts, mock_template_content):
        """Test error handling for invalid save path."""
        # First patch is only for chart initialization
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)

        # Now test the save method without the open patch
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            chart.save("/nonexistent/directory/file.html")


class TestD3BubbleChartWithDocs:
    """Test D3BubbleChart with docs parameter for DTM data."""

    def test_with_docs_parameter(self, mock_template_content):
        """Test chart creation with docs parameter."""
        # Create sample DTM-like data
        dtm_data = pd.DataFrame(
            {"doc1": [3, 2, 1, 0], "doc2": [1, 1, 2, 3], "doc3": [2, 0, 1, 1]},
            index=["apple", "banana", "cherry", "date"],
        )

        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=dtm_data, docs=["doc1"], auto_open=False)
                assert isinstance(chart.counts, dict)
                assert len(chart.counts) > 0


class TestD3BubbleChartHtmlOutput:
    """Test HTML output generation."""

    def test_html_contains_required_elements(
        self, sample_counts, mock_template_content
    ):
        """Test that generated HTML contains required elements."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)

                # Check that HTML contains the data
                assert "the" in chart.html  # Word from sample_counts
                assert str(chart.width) in chart.html
                assert str(chart.height) in chart.html
                assert chart.title in chart.html

    def test_html_structure(self, sample_counts, mock_template_content):
        """Test basic HTML structure."""
        with patch("builtins.open", mock_open(read_data=mock_template_content)):
            with patch("webbrowser.open"):
                chart = D3BubbleChart(data=sample_counts, auto_open=False)

                assert "<html>" in chart.html
                assert "<head>" in chart.html
                assert "<body>" in chart.html
                assert "</html>" in chart.html
