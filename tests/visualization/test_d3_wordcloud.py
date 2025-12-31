"""test_d3_wordcloud.py.

Test suite for D3WordCloud and D3MultiCloud classes.

Coverage: 95%. Missing: 151-153, 216-217, 352, 360, 432-433, 461-466

Last Updated: December 05, 2025
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import spacy
from spacy.tokens import Doc

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.d3_wordcloud import D3MultiCloud, D3WordCloud


# Load spaCy model once for all tests
@pytest.fixture(scope="session")
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog. The fox is quick and smart. The dog is lazy but friendly."


@pytest.fixture
def sample_counts():
    """Sample word counts dictionary."""
    return {
        "the": 4,
        "fox": 2,
        "quick": 2,
        "brown": 1,
        "jumps": 1,
        "over": 1,
        "lazy": 2,
        "dog": 2,
        "is": 2,
        "and": 1,
        "smart": 1,
        "but": 1,
        "friendly": 1,
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
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <script id="d3"></script>
        <script id="d3cloud"></script>
    </head>
    <body>
        <div id="wordcloud"></div>
        <script>
            var termCounts = {{ termCounts }};
            var width = {{ width }};
            var height = {{ height }};
            var backgroundColor = "{{ backgroundColor }}";
            var colorscale = "{{ colorscale }}";
            var font = "{{ font }}";
            var spiral = "{{ spiral }}";
            var scale = "{{ scale }}";
            var angleCount = {{ angleCount }};
            var angleFrom = {{ angleFrom }};
            var angleTo = {{ angleTo }};
        </script>
    </body>
    </html>
    """


@pytest.fixture
def mock_multicloud_template():
    """Mock multi-cloud HTML template content."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title or "Multi Word Cloud" }}</title>
        <script id="d3"></script>
        <script id="d3cloud"></script>
    </head>
    <body>
        <svg width="{{ total_width }}" height="{{ total_height }}">
        </svg>
        <script>
            var cloudData = {{ cloud_data }};
            var font = "{{ font }}";
            var spiral = "{{ spiral }}";
            var scale = "{{ scale }}";
            var angleCount = {{ angleCount }};
            var angleFrom = {{ angleFrom }};
            var angleTo = {{ angleTo }};
            var backgroundColor = "{{ backgroundColor }}";
            var colorscale = "{{ colorscale }}";
        </script>
    </body>
    </html>
    """


@pytest.fixture
def mock_d3_script():
    """Mock D3.js script content."""
    return "// Mock D3.js library content"


@pytest.fixture
def mock_d3_cloud_script():
    """Mock D3 cloud script content."""
    return "// Mock D3 cloud library content"


class TestD3WordCloudInitialization:
    """Test D3WordCloud initialization with different data types."""

    def test_init_with_string_data(
        self, sample_text, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test initialization with string data."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_text, auto_open=False)
                assert isinstance(cloud.counts, dict)
                assert len(cloud.counts) > 0
                assert cloud.html != ""
                assert "The" in cloud.counts or "the" in cloud.counts

    def test_init_with_dict_data(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test initialization with dictionary data."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, auto_open=False)
                assert cloud.counts == sample_counts
                assert cloud.html != ""

    def test_init_with_spacy_doc(
        self,
        nlp,
        sample_text,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
    ):
        """Test initialization with spaCy Doc."""
        doc = nlp(sample_text)

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=doc, auto_open=False)
                assert isinstance(cloud.counts, dict)
                assert len(cloud.counts) > 0

    def test_init_with_spacy_span(
        self,
        nlp,
        sample_text,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
    ):
        """Test initialization with spaCy Span."""
        doc = nlp(sample_text)
        span = doc[:10]  # First 10 tokens

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=span, auto_open=False)
                assert isinstance(cloud.counts, dict)
                assert len(cloud.counts) > 0

    def test_init_with_token_list(
        self,
        nlp,
        sample_text,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
    ):
        """Test initialization with list of spaCy tokens."""
        doc = nlp(sample_text)
        tokens = [token for token in doc if not token.is_punct and not token.is_space]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=tokens, auto_open=False)
                assert isinstance(cloud.counts, dict)
                assert len(cloud.counts) > 0

    def test_init_with_string_list(
        self, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test initialization with list of strings."""
        data = ["apple", "banana", "apple", "cherry", "banana", "apple", "date"]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=data, auto_open=False)
                assert cloud.counts["apple"] == 3
                assert cloud.counts["banana"] == 2
                assert cloud.counts["cherry"] == 1

    def test_init_with_dataframe(
        self,
        sample_dataframe,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
    ):
        """Test initialization with DataFrame."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_dataframe, auto_open=False)
                assert isinstance(cloud.counts, dict)
                assert len(cloud.counts) > 0


class TestD3WordCloudParameters:
    """Test D3WordCloud with different parameter configurations."""

    def test_custom_dimensions(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with custom dimensions."""
        width, height = 800, 500

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts, width=width, height=height, auto_open=False
                )
                assert cloud.width == width
                assert cloud.height == height
                assert str(width) in cloud.html
                assert str(height) in cloud.html

    def test_custom_title(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with custom title."""
        title = "My Custom Word Cloud"

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, title=title, auto_open=False)
                assert cloud.title == title
                assert title in cloud.html

    def test_limit_parameter(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with limit parameter."""
        limit = 3

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, limit=limit, auto_open=False)
                assert len(cloud.counts) <= limit

    def test_font_parameter(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with custom font."""
        font = "Arial"

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, font=font, auto_open=False)
                assert cloud.font == font
                assert font in cloud.html

    def test_spiral_parameter(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with different spiral types."""
        for spiral in ["archimedean", "rectangular"]:

            def mock_open_handler(filename, *args, **kwargs):
                if "d3_cloud_template" in str(filename):
                    return mock_open(read_data=mock_template_content).return_value
                elif "d3.min.js" in str(filename):
                    return mock_open(read_data=mock_d3_script).return_value
                elif "d3cloud_bundle.min.js" in str(filename):
                    return mock_open(read_data=mock_d3_cloud_script).return_value
                else:
                    return mock_open().return_value

            with patch("builtins.open", side_effect=mock_open_handler):
                with patch("webbrowser.open"):
                    cloud = D3WordCloud(
                        data=sample_counts, spiral=spiral, auto_open=False
                    )
                    assert cloud.spiral == spiral

    def test_scale_parameter(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with different scale types."""
        for scale in ["log", "sqrt", "linear"]:

            def mock_open_handler(filename, *args, **kwargs):
                if "d3_cloud_template" in str(filename):
                    return mock_open(read_data=mock_template_content).return_value
                elif "d3.min.js" in str(filename):
                    return mock_open(read_data=mock_d3_script).return_value
                elif "d3cloud_bundle.min.js" in str(filename):
                    return mock_open(read_data=mock_d3_cloud_script).return_value
                else:
                    return mock_open().return_value

            with patch("builtins.open", side_effect=mock_open_handler):
                with patch("webbrowser.open"):
                    cloud = D3WordCloud(
                        data=sample_counts, scale=scale, auto_open=False
                    )
                    assert cloud.scale == scale

    def test_angle_parameters(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test word cloud with custom angle parameters."""
        angle_from, angle_to, angle_count = -90, 90, 7

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    angle_from=angle_from,
                    angle_to=angle_to,
                    angle_count=angle_count,
                    auto_open=False,
                )
                assert cloud.angle_from == angle_from
                assert cloud.angle_to == angle_to
                assert cloud.angle_count == angle_count


class TestD3WordCloudValidation:
    """Test D3WordCloud parameter validation."""

    def test_invalid_spiral(self, sample_counts):
        """Test error handling for invalid spiral parameter."""
        with pytest.raises((LexosException, ValueError)):
            D3WordCloud(data=sample_counts, spiral="invalid", auto_open=False)

    def test_invalid_scale(self, sample_counts):
        """Test error handling for invalid scale parameter."""
        with pytest.raises((LexosException, ValueError)):
            D3WordCloud(data=sample_counts, scale="invalid", auto_open=False)

    def test_invalid_angles(self, sample_counts):
        """Test error handling for invalid angle parameters."""
        with pytest.raises((LexosException, ValueError)):
            D3WordCloud(
                data=sample_counts, angle_from=60, angle_to=-60, auto_open=False
            )

    def test_invalid_dimensions(self, sample_counts):
        """Test error handling for invalid dimensions."""
        with pytest.raises((LexosException, ValueError)):
            D3WordCloud(data=sample_counts, width=30, auto_open=False)  # Below minimum

        with pytest.raises((LexosException, ValueError)):
            D3WordCloud(data=sample_counts, height=30, auto_open=False)  # Below minimum


class TestD3WordCloudMethods:
    """Test D3WordCloud methods."""

    def test_save_method(
        self,
        sample_counts,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
        tmp_path,
    ):
        """Test saving word cloud to file."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, auto_open=False)

        # Test saving without minification
        output_path = tmp_path / "test_cloud.html"
        cloud.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0
        assert cloud.html == content

    def test_save_with_minify(
        self,
        sample_counts,
        mock_template_content,
        mock_d3_script,
        mock_d3_cloud_script,
        tmp_path,
    ):
        """Test saving word cloud with minification."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, auto_open=False)

        output_path = tmp_path / "test_cloud_minified.html"
        cloud.save(output_path, minify=True)

        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0
        # Minified content should be shorter
        assert len(content) <= len(cloud.html)

    def test_auto_open_behavior(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test auto_open behavior."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        # Test auto_open=True
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open") as mock_browser:
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = (
                        "/tmp/test.html"
                    )
                    D3WordCloud(data=sample_counts, auto_open=True)
                    mock_browser.assert_called_once()

        # Test auto_open=False
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open") as mock_browser:
                D3WordCloud(data=sample_counts, auto_open=False)
                mock_browser.assert_not_called()


class TestD3WordCloudErrorHandling:
    """Test error handling in D3WordCloud."""

    def test_template_not_found(self, sample_counts):
        """Test handling of missing template file."""
        with pytest.raises(FileNotFoundError):
            D3WordCloud(
                data=sample_counts,
                template="nonexistent_template.html",
                auto_open=False,
            )


class TestD3MultiCloud:
    """Test D3MultiCloud functionality."""

    def test_multicloud_initialization(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud initialization."""
        data_sources = [
            {"apple": 10, "banana": 5},
            {"cherry": 8, "date": 3},
            {"elderberry": 6, "fig": 4},
        ]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(data_sources=data_sources, auto_open=False)
                assert len(multicloud.word_clouds) == 3
                assert len(multicloud.labels) == 3
                assert multicloud.html != ""

    def test_multicloud_with_custom_labels(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with custom labels."""
        data_sources = [{"apple": 10, "banana": 5}, {"cherry": 8, "date": 3}]
        labels = ["Fruits A", "Fruits B"]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=data_sources, labels=labels, auto_open=False
                )
                assert multicloud.labels == labels
                assert multicloud.word_clouds[0].title == "Fruits A"
                assert multicloud.word_clouds[1].title == "Fruits B"

    def test_multicloud_save_method(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script, tmp_path
    ):
        """Test D3MultiCloud save method."""
        data_sources = [{"apple": 10}, {"banana": 5}]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(data_sources=data_sources, auto_open=False)

        output_path = tmp_path / "test_multicloud.html"
        multicloud.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0

    def test_multicloud_get_cloud_counts(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test getting word counts from specific clouds."""
        data_sources = [{"apple": 10, "banana": 5}, {"cherry": 8, "date": 3}]

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(data_sources=data_sources, auto_open=False)

        counts_0 = multicloud.get_cloud_counts(0)
        counts_1 = multicloud.get_cloud_counts(1)

        assert "apple" in counts_0
        assert "cherry" in counts_1

        # Test invalid index
        with pytest.raises(IndexError):
            multicloud.get_cloud_counts(5)

    def test_multicloud_validation_errors(self):
        """Test D3MultiCloud validation errors."""
        # Test mismatched labels
        data_sources = [{"apple": 10}, {"banana": 5}]
        labels = ["Only One Label"]  # Should have 2 labels

        with pytest.raises(LexosException, match="Number of labels must match"):
            D3MultiCloud(data_sources=data_sources, labels=labels, auto_open=False)

    @pytest.mark.skip(reason="Couldn't test temporary files on local WSL system.")
    def test_load_template_method_coverage(self, sample_counts, tmp_path):
        """Test to ensure _load_template method lines are covered."""
        # Create a real template file
        template_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Template</title></head>
        <body><div id="wordcloud"></div></body>
        </html>
        """
        template_file = tmp_path / "real_template.html"
        template_file.write_text(template_content)

        # Mock only the D3 library files, let template loading happen naturally
        def selective_mock_open_handler(filename, *args, **kwargs):
            filename_str = str(filename)
            if "d3.min.js" in filename_str:
                return mock_open(read_data="// Mock D3.js").return_value
            elif "d3cloud_bundle.min.js" in filename_str:
                return mock_open(read_data="// Mock D3 cloud").return_value
            else:
                # Use real file system for everything else (including template)
                return open(filename, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_mock_open_handler):
            with patch("webbrowser.open"):
                # This should execute the actual _load_template method
                cloud = D3WordCloud(
                    data=sample_counts, template=str(template_file), auto_open=False
                )

                # Verify it worked
                assert template_content.strip() in cloud.html

                # Call _load_template directly to ensure coverage
                loaded = cloud._load_template()
                assert loaded == template_content

    def test_d3_library_directory_inclusion(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3 library inclusion with 'directory' option."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        # Test directory inclusion - this should execute line 209
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3js="directory",  # This triggers line 209
                    auto_open=False,
                )
                # Check that it creates a script tag with src attribute (not embedded)
                assert 'src="' in cloud.html
                assert "d3.min.js" in cloud.html

        # Test case-insensitive "DIRECTORY"
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3js="DIRECTORY",  # Test case insensitivity
                    auto_open=False,
                )
                assert 'src="' in cloud.html
                assert "d3.min.js" in cloud.html

    def test_d3_cloud_custom_path_inclusion(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3 cloud library inclusion with custom JavaScript file path (line 229)."""

        def mock_open_handler(filename, *args, **kwargs):
            filename_str = str(filename)
            if "d3_cloud_template" in filename_str:
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in filename_str:
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in filename_str:
                return mock_open(read_data=mock_d3_cloud_script).return_value
            elif filename_str.endswith(
                ".js"
            ):  # Handle any .js file (including custom paths)
                return mock_open(
                    read_data="// Custom D3 cloud library content"
                ).return_value
            else:
                return mock_open().return_value

        # Test custom D3 cloud JS file path - this should execute line 229
        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3_cloud="custom_d3_cloud.js",  # This triggers line 229
                    auto_open=False,
                )
                # Check that the custom D3 cloud content is embedded
                assert "Custom D3 cloud library content" in cloud.html

        # Test with absolute path
        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3_cloud="/path/to/custom_d3_cloud.js",
                    auto_open=False,
                )
                assert "Custom D3 cloud library content" in cloud.html

        # Test with relative path
        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3_cloud="./assets/my_d3_cloud.js",
                    auto_open=False,
                )
                assert "Custom D3 cloud library content" in cloud.html

    def test_d3_library_custom_path_not_found(
        self, sample_counts, mock_template_content
    ):
        """Test error handling when custom D3 file is not found."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "nonexistent.js" in str(filename):
                raise FileNotFoundError("Custom D3 file not found")
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                with pytest.raises(LexosException, match="Script file not found"):
                    D3WordCloud(
                        data=sample_counts,
                        include_d3js="nonexistent.js",
                        auto_open=False,
                    )


class TestD3WordCloudHtmlOutput:
    """Test HTML output generation."""

    def test_html_contains_data(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test that generated HTML contains the data."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(data=sample_counts, auto_open=False)

        # Check that HTML contains some of the key words from our sample data
        # This is more reliable than checking exact JSON format
        for word in ["the", "fox", "quick"]:
            assert word in cloud.html.lower()

        # Check basic HTML structure
        assert "<!DOCTYPE html>" in cloud.html
        assert "<head>" in cloud.html
        assert "<body>" in cloud.html
        assert "</html>" in cloud.html

        # Check that the data is actually being rendered (not just template variables)
        assert "fox" in cloud.html

    def test_d3_library_inclusion(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3 library inclusion options."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        # Test CDN inclusion
        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts, include_d3js="cdn", auto_open=False
                )
                # Check for CDN URL pattern instead of exact d3js.org
                assert "https://" in cloud.html or "cdn" in cloud.html.lower()

        # Test local inclusion
        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts, include_d3js=True, auto_open=False
                )
                # Check that the mock D3 script content is included
                assert "Mock D3.js library content" in cloud.html


class TestD3WordMultiCloudValidation:
    """Test D3WordCloud parameter validation."""


class TestD3WordMultiCloudValidation:
    """Test D3MultiCloud parameter validation."""

    def test_invalid_spiral_multicloud(self):
        """Test error handling for invalid spiral parameter in D3MultiCloud."""
        data_sources = [{"apple": 10}, {"banana": 5}]

        with pytest.raises(
            LexosException, match='spiral must be "archimedean" or "rectangular"'
        ):
            D3MultiCloud(
                data_sources=data_sources,
                spiral="invalid_spiral",  # This triggers the D3MultiCloud validator
                auto_open=False,
            )

    def test_invalid_scale_multicloud(self):
        """Test error handling for invalid scale parameter in D3MultiCloud."""
        data_sources = [{"apple": 10}, {"banana": 5}]

        with pytest.raises(
            LexosException, match='scale must be "log", "sqrt", or "linear"'
        ):
            D3MultiCloud(
                data_sources=data_sources,
                scale="invalid_scale",  # This triggers the D3MultiCloud validator
                auto_open=False,
            )

    def test_invalid_angles_multicloud(self):
        """Test error handling for invalid angle parameters in D3MultiCloud."""
        data_sources = [{"apple": 10}, {"banana": 5}]

        with pytest.raises(LexosException):
            D3MultiCloud(
                data_sources=data_sources,
                angle_from=60,
                angle_to=-60,  # This triggers the D3MultiCloud validator
                auto_open=False,
            )

    def test_mismatched_labels_multicloud(self):
        """Test error handling for mismatched labels in D3MultiCloud."""
        data_sources = [{"apple": 10}, {"banana": 5}]
        labels = ["Only One Label"]  # Should have 2 labels

        with pytest.raises(LexosException, match="Number of labels must match"):
            D3MultiCloud(data_sources=data_sources, labels=labels, auto_open=False)


class TestD3WordCloudUncoveredPaths:
    """Test uncovered code paths in D3WordCloud and D3MultiCloud."""

    def test_d3_include_false(
        self, sample_counts, mock_template_content, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3WordCloud with include_d3js=False (line 216-217)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts, include_d3js=False, auto_open=False
                )
                # Should contain empty script tag (not replaced with d3 content)
                assert '<script id="d3"></script>' in cloud.html
                # Should not contain d3 library content
                assert "Mock D3.js library" not in cloud.html

    def test_d3_include_custom_js_file(
        self, sample_counts, mock_template_content, mock_d3_cloud_script
    ):
        """Test D3WordCloud with custom .js file path (line 229)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "custom_d3.js" in str(filename):
                return mock_open(read_data="// Custom D3 script").return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3js="custom_d3.js",
                    auto_open=False,
                )
                assert "Custom D3 script" in cloud.html

    def test_multicloud_auto_generated_labels(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud auto-generates labels when not provided (line 352)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}, {"banana": 5}, {"cherry": 3}],
                    auto_open=False,
                )
                assert multicloud.labels == ["Doc 1", "Doc 2", "Doc 3"]

    def test_multicloud_get_cloud_counts(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud.get_cloud_counts() method (line 552)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10, "banana": 5}, {"cherry": 8}],
                    auto_open=False,
                )
                counts_0 = multicloud.get_cloud_counts(0)
                assert counts_0 == {"apple": 10, "banana": 5}

    def test_multicloud_save_with_minify(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script, tmp_path
    ):
        """Test D3MultiCloud.save() with minification (lines 565-569)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}, {"banana": 5}], auto_open=False
                )

        output_path = tmp_path / "test_multicloud_minified.html"
        multicloud.save(output_path, minify=True)

        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0

    def test_multicloud_include_d3_false(
        self, mock_multicloud_template, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with include_d3js=False (line 447)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}], include_d3js=False, auto_open=False
                )
                # D3 script should not be included
                assert '<script id="d3">' not in multicloud.html

    def test_multicloud_include_d3_custom_path(
        self, mock_multicloud_template, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with custom D3.js path (line 438)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "custom_d3.js" in str(filename):
                return mock_open(read_data="// Custom D3").return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}],
                    include_d3js="custom_d3.js",
                    auto_open=False,
                )
                assert "Custom D3" in multicloud.html

    def test_multicloud_include_d3_cloud_cdn_fallback(
        self, mock_multicloud_template, mock_d3_script
    ):
        """Test D3MultiCloud fallback to CDN for d3-cloud (lines 461-466)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                raise FileNotFoundError("d3cloud not found")
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}],
                    include_d3_cloud=True,
                    auto_open=False,
                )
                # Should have CDN fallback
                assert "cdn.jsdelivr.net" in multicloud.html

    def test_multicloud_render_calculates_dimensions(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud._render() calculates grid dimensions (lines 491-498)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                # Test with 2 columns and 5 clouds (should be 3 rows)
                multicloud = D3MultiCloud(
                    data_sources=[
                        {"a": 1},
                        {"b": 2},
                        {"c": 3},
                        {"d": 4},
                        {"e": 5},
                    ],
                    columns=2,
                    cloud_width=300,
                    cloud_height=300,
                    cloud_spacing=20,
                    auto_open=False,
                )
                # Verify the HTML contains calculated dimensions
                assert multicloud.html is not None
                assert len(multicloud.word_clouds) == 5

    def test_multicloud_auto_open(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with auto_open=True (line 450)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open") as mock_browser:
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = (
                        "/tmp/test.html"
                    )
                    D3MultiCloud(
                        data_sources=[{"apple": 10}],
                        auto_open=True,
                    )
                    mock_browser.assert_called_once()

    def test_wordcloud_include_d3_directory(
        self, sample_counts, mock_template_content, mock_d3_cloud_script
    ):
        """Test D3WordCloud with include_d3js='directory' (line 217)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_cloud_template" in str(filename):
                return mock_open(read_data=mock_template_content).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                cloud = D3WordCloud(
                    data=sample_counts,
                    include_d3js="directory",
                    auto_open=False,
                )
                assert 'src="' in cloud.html
                assert "d3.min.js" in cloud.html

    def test_multicloud_include_d3_directory(
        self, mock_multicloud_template, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with include_d3js='directory' (line 445)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}],
                    include_d3js="directory",
                    auto_open=False,
                )
                assert 'src="' in multicloud.html
                assert "d3.min.js" in multicloud.html

    def test_multicloud_include_d3_cdn(
        self, mock_multicloud_template, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with include_d3js='cdn' (line 432)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}],
                    include_d3js="cdn",
                    auto_open=False,
                )
                assert "d3js.org" in multicloud.html

    def test_multicloud_include_d3_none(
        self, mock_multicloud_template, mock_d3_script, mock_d3_cloud_script
    ):
        """Test D3MultiCloud with include_d3js=None (line 453)."""

        def mock_open_handler(filename, *args, **kwargs):
            if "d3_multicloud_template" in str(filename):
                return mock_open(read_data=mock_multicloud_template).return_value
            elif "d3_cloud_template" in str(filename):
                return mock_open(read_data="<html></html>").return_value
            elif "d3.min.js" in str(filename):
                return mock_open(read_data=mock_d3_script).return_value
            elif "d3cloud_bundle.min.js" in str(filename):
                return mock_open(read_data=mock_d3_cloud_script).return_value
            else:
                return mock_open().return_value

        with patch(
            "lexos.visualization.d3_wordcloud.open", side_effect=mock_open_handler
        ):
            with patch("webbrowser.open"):
                multicloud = D3MultiCloud(
                    data_sources=[{"apple": 10}],
                    include_d3js=None,
                    auto_open=False,
                )
                # Should include D3 (None defaults to True behavior)
                assert '<script id="d3">' in multicloud.html
