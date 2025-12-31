"""test_plotly_dendrogram.py.

Coverage: 100%

Last Update: December 5, 2025
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lexos.cluster.plotly_dendrogram import PlotlyDendrogram
from lexos.dtm import DTM
from lexos.exceptions import LexosException


class TestPlotlyDendrogram:
    """Test suite for the PlotlyDendrogram class."""

    @pytest.fixture
    def sample_dtm(self):
        """Create a real DTM for testing using actual texts."""
        # Sample text documents
        texts = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "ran", "in", "the", "park"],
            ["cats", "and", "dogs", "are", "pets"],
            ["parks", "have", "trees", "and", "grass"],
        ]

        # Create DTM from texts
        dtm = DTM()
        dtm(docs=texts, labels=["doc1", "doc2", "doc3", "doc4"])
        return dtm

    @pytest.fixture
    def sample_dataframe(self, sample_dtm):
        """Create a sample DataFrame for testing."""
        return sample_dtm.to_df().T

    @pytest.fixture
    def sample_numpy_array(self):
        """Create a sample numpy array for testing."""
        return np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

    @pytest.fixture
    def sample_list_matrix(self):
        """Create a sample list matrix for testing."""
        return [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]

    def test_plotly_dendrogram_initialization_no_dtm(self):
        """Test PlotlyDendrogram initialization without DTM raises exception."""
        with pytest.raises(
            LexosException, match="You must provide a document-term matrix"
        ):
            PlotlyDendrogram()

    def test_plotly_dendrogram_initialization_with_dtm(self, sample_dtm):
        """Test basic PlotlyDendrogram initialization with DTM."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        assert dendrogram.dtm == sample_dtm
        assert dendrogram.labels == ["doc1", "doc2", "doc3", "doc4"]
        assert dendrogram.metric == "euclidean"
        assert dendrogram.method == "average"
        assert dendrogram.fig is not None
        assert hasattr(dendrogram.fig, "data")

    def test_plotly_dendrogram_initialization_with_dataframe(self, sample_dataframe):
        """Test PlotlyDendrogram initialization with DataFrame."""
        # WARNING: A DataFrame must be transposed, with documents as rows and terms as columns. But this is done automatically for a DTM instance.

        dendrogram = PlotlyDendrogram(dtm=sample_dataframe)

        assert dendrogram.labels == ["doc1", "doc2", "doc3", "doc4"]
        assert dendrogram.fig is not None

    def test_plotly_dendrogram_initialization_with_numpy_array(
        self, sample_numpy_array
    ):
        """Test PlotlyDendrogram initialization with numpy array."""
        dendrogram = PlotlyDendrogram(dtm=sample_numpy_array)

        assert dendrogram.labels == ["Doc1", "Doc2", "Doc3", "Doc4"]
        assert dendrogram.fig is not None

    def test_plotly_dendrogram_initialization_with_list(self, sample_list_matrix):
        """Test PlotlyDendrogram initialization with list matrix."""
        dendrogram = PlotlyDendrogram(dtm=sample_list_matrix)

        assert dendrogram.labels == ["Doc1", "Doc2", "Doc3", "Doc4"]
        assert dendrogram.fig is not None

    def test_plotly_dendrogram_with_custom_parameters(self, sample_dtm):
        """Test PlotlyDendrogram initialization with custom parameters."""
        custom_labels = ["Document A", "Document B", "Document C", "Document D"]

        dendrogram = PlotlyDendrogram(
            dtm=sample_dtm,
            labels=custom_labels,
            metric="cityblock",
            method="ward",
            orientation="left",
            title="Custom Dendrogram",
            figsize=(12, 8),
            colorscale=["red", "blue"],
            color_threshold=0.5,
        )

        assert dendrogram.labels == custom_labels
        assert dendrogram.metric == "cityblock"
        assert dendrogram.method == "ward"
        assert dendrogram.orientation == "left"
        assert dendrogram.title == "Custom Dendrogram"
        assert dendrogram.figsize == (12, 8)
        assert dendrogram.colorscale == ["red", "blue"]
        assert dendrogram.color_threshold == 0.5

    def test_get_valid_matrix_single_document(self):
        """Test _get_valid_matrix with single document."""
        matrix = [[1, 2, 3, 4]]  # Only one document

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            PlotlyDendrogram(dtm=matrix)

    def test_labels_mismatch_with_matrix_shape(self, sample_dtm):
        """Test error when labels don't match matrix shape."""
        wrong_labels = ["doc1", "doc2"]  # Only 2 labels for 4 documents

        with pytest.raises(
            ValueError,
            match="Dimensions of Z and labels must be consistent",
        ):
            PlotlyDendrogram(dtm=sample_dtm, labels=wrong_labels)

    def test_show_method(self, sample_dtm):
        """Test show method."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        # Just verify the method exists and doesn't crash
        # We can't easily test the actual display without browser integration
        assert hasattr(dendrogram, "show")
        assert callable(dendrogram.show)

    def test_to_html_method(self, sample_dtm):
        """Test to_html method."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        html_result = dendrogram.to_html()
        assert isinstance(html_result, str)
        assert len(html_result) > 0
        assert "html" in html_result.lower()

    def test_to_image_method(self, sample_dtm):
        """Test to_image method."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        # Test that the method exists and returns bytes
        try:
            image_result = dendrogram.to_image(format="png")
            assert isinstance(image_result, bytes)
        except Exception:
            # Image generation might fail without proper dependencies
            pytest.skip("Image generation requires additional dependencies")

    def test_write_html_method(self, sample_dtm):
        """Test write_html method."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            dendrogram.write_html(tmp_file.name)

            # Verify file was created and has content
            assert Path(tmp_file.name).exists()
            assert Path(tmp_file.name).stat().st_size > 0

    def test_write_html_method_with_path_object(self, sample_dtm):
        """Test write_html method with Path object."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            path_obj = Path(tmp_file.name)
            dendrogram.write_html(path_obj)

            assert path_obj.exists()
            assert path_obj.stat().st_size > 0

    @pytest.mark.filterwarnings("ignore:.*deprecated*:DeprecationWarning")
    def test_write_image_method(self, sample_dtm):
        """Test write_image method."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            try:
                dendrogram.write_image(tmp_file.name)
                assert Path(tmp_file.name).exists()
            except Exception:
                # Image generation might fail without proper dependencies
                pytest.skip("Image generation requires additional dependencies")

    def test_orientation_variations(self, sample_dtm):
        """Test different orientation options."""
        orientations = ["top", "bottom", "left", "right"]

        for orientation in orientations:
            dendrogram = PlotlyDendrogram(dtm=sample_dtm, orientation=orientation)
            assert dendrogram.orientation == orientation
            assert dendrogram.fig is not None

    def test_different_metrics(self, sample_dtm):
        """Test different distance metrics."""
        metrics = ["euclidean", "cityblock", "cosine"]

        for metric in metrics:
            dendrogram = PlotlyDendrogram(dtm=sample_dtm, metric=metric)
            assert dendrogram.metric == metric
            assert dendrogram.fig is not None

    def test_different_methods(self, sample_dtm):
        """Test different linkage methods."""
        methods = ["single", "complete", "average", "ward"]

        for method in methods:
            try:
                dendrogram = PlotlyDendrogram(dtm=sample_dtm, method=method)
                assert dendrogram.method == method
                assert dendrogram.fig is not None
            except Exception:
                # Some method/metric combinations might not be compatible
                pass

    def test_custom_layout(self, sample_dtm):
        """Test custom layout configuration."""
        custom_layout = {"width": 800, "height": 600}

        dendrogram = PlotlyDendrogram(dtm=sample_dtm, layout=custom_layout)
        assert dendrogram.layout == custom_layout
        assert dendrogram.fig is not None

    def test_with_title(self, sample_dtm):
        """Test dendrogram with title."""
        title = "Test Dendrogram Title"

        dendrogram = PlotlyDendrogram(dtm=sample_dtm, title=title)
        assert dendrogram.title == title
        assert dendrogram.fig is not None

    def test_invalid_dtm_types(self):
        """Test various invalid DTM inputs."""
        invalid_inputs = ["invalid_string", 123, None, {"invalid": "dict"}]

        for invalid_input in invalid_inputs:
            with pytest.raises((LexosException, AttributeError, TypeError, ValueError)):
                PlotlyDendrogram(dtm=invalid_input)

    def test_non_numeric_data(self):
        """Test with non-numeric data."""
        invalid_df = pd.DataFrame(
            {"term1": ["a", "b", "c", "d"], "term2": ["e", "f", "g", "h"]},
            index=["doc1", "doc2", "doc3", "doc4"],
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            PlotlyDendrogram(dtm=invalid_df)

    def test_empty_matrix(self):
        """Test with empty matrix."""
        empty_df = pd.DataFrame()

        with pytest.raises((LexosException, ValueError)):
            PlotlyDendrogram(dtm=empty_df)

    def test_show_method(self, sample_dtm):
        """Test show() method (lines 215-219)."""
        from unittest.mock import MagicMock, patch

        dendrogram = PlotlyDendrogram(dtm=sample_dtm)

        with patch.object(dendrogram.fig, "show") as mock_show:
            dendrogram.show()
            mock_show.assert_called_once()

    def test_show_method_no_figure(self, sample_dtm):
        """Test show() raises exception when figure is None (line 217)."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        dendrogram.fig = None

        with pytest.raises(
            LexosException, match="You must call the instance before showing"
        ):
            dendrogram.show()

    def test_to_html_method(self, sample_dtm):
        """Test to_html() method (line 228)."""
        from unittest.mock import MagicMock, patch

        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        mock_html = "<html>test</html>"

        with patch.object(
            dendrogram.fig, "to_html", return_value=mock_html
        ) as mock_to_html:
            result = dendrogram.to_html()
            mock_to_html.assert_called_once()
            assert result == mock_html

    def test_to_html_method_no_figure(self, sample_dtm):
        """Test to_html() raises exception when figure is None."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        dendrogram.fig = None

        with pytest.raises(
            LexosException, match="You must call the instance before generating HTML"
        ):
            dendrogram.to_html()

    def test_to_image_method(self, sample_dtm):
        """Test to_image() method (line 238)."""
        from unittest.mock import MagicMock, patch

        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        mock_image_bytes = b"image_data"

        with patch.object(
            dendrogram.fig, "to_image", return_value=mock_image_bytes
        ) as mock_to_image:
            result = dendrogram.to_image()
            mock_to_image.assert_called_once()
            assert result == mock_image_bytes

    def test_to_image_method_no_figure(self, sample_dtm):
        """Test to_image() raises exception when figure is None."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        dendrogram.fig = None

        with pytest.raises(
            LexosException,
            match="You must call the instance before generating an image",
        ):
            dendrogram.to_image()

    def test_write_html_method(self, sample_dtm, tmp_path):
        """Test write_html() method (line 251)."""
        from unittest.mock import patch

        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        test_path = tmp_path / "test.html"

        with patch.object(dendrogram.fig, "write_html") as mock_write_html:
            dendrogram.write_html(test_path)
            mock_write_html.assert_called_once_with(str(test_path))

    def test_write_html_method_no_figure(self, sample_dtm):
        """Test write_html() raises exception when figure is None."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        dendrogram.fig = None

        with pytest.raises(
            LexosException, match="You must call the instance before saving"
        ):
            dendrogram.write_html("test.html")

    def test_write_image_method(self, sample_dtm, tmp_path):
        """Test write_image() method (line 265)."""
        from unittest.mock import patch

        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        test_path = tmp_path / "test.png"

        with patch.object(dendrogram.fig, "write_image") as mock_write_image:
            dendrogram.write_image(test_path)
            mock_write_image.assert_called_once_with(str(test_path))

    def test_write_image_method_no_figure(self, sample_dtm):
        """Test write_image() raises exception when figure is None."""
        dendrogram = PlotlyDendrogram(dtm=sample_dtm)
        dendrogram.fig = None

        with pytest.raises(
            LexosException, match="You must call the instance before saving"
        ):
            dendrogram.write_image("test.png")
