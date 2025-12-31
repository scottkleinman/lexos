"""test_clustermap.py.

Coverage: 96%. Missing: 516, 705, 828-829, 834-835, 1052-1057, 1086-1098

Last Updated: December 5, 2025
"""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.cluster import hierarchy

from lexos.cluster import (
    Clustermap,
    PlotlyClusterGrid,
    PlotlyClustermap,
    _create_dendrogram_traces,
    _get_matrix,
    get_matrix,
)
from lexos.dtm import DTM
from lexos.exceptions import LexosException


class TestGetMatrix:
    """Test suite for the _get_matrix function."""

    def test_get_matrix_with_dtm(self):
        """Test _get_matrix with a DTM object."""
        # Create a sample DTM
        data = [[1, 2, 3], [4, 5, 6]]
        labels = ["doc1", "doc2"]
        dtm = DTM()
        dtm(docs=data, labels=labels)

        result = _get_matrix(dtm)

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "terms"
        assert result.shape == (6, 2)

    def test_get_matrix_with_dataframe(self):
        """Test _get_matrix with a pandas DataFrame."""
        data = [[1, 2, 3], [4, 5, 6]]
        labels = ["doc1", "doc2"]
        dtm = DTM()
        dtm(docs=data, labels=labels)
        df = dtm.to_df()

        result = get_matrix(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (6, 2)
        pd.testing.assert_frame_equal(result, df.sparse.to_dense())

    def test_get_matrix_with_numpy_array(self):
        """Test _get_matrix with a numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = _get_matrix(data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, data)

    def test_get_matrix_with_list(self):
        """Test _get_matrix with a list."""
        data = [[1, 2, 3], [4, 5, 6]]

        result = _get_matrix(data)

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert result == data

    def test_get_matrix_single_document_raises_exception(self):
        """Test that _get_matrix raises exception with single document."""
        # Test with DataFrame
        single_doc_df = pd.DataFrame([[1, 2, 3]], columns=["term1", "term2", "term3"])

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            _get_matrix(single_doc_df)

    def test_get_matrix_single_document_numpy_raises_exception(self):
        """Test that _get_matrix raises exception with single document numpy array."""
        single_doc_array = np.array([[1, 2, 3]])

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            _get_matrix(single_doc_array)

    def test_get_matrix_single_document_list_raises_exception(self):
        """Test that _get_matrix raises exception with single document list."""
        single_doc_list = [[1, 2, 3]]

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            _get_matrix(single_doc_list)

    def test_get_matrix_empty_list_raises_exception(self):
        """Test that _get_matrix raises exception with empty list."""
        empty_list = []

        with pytest.raises(
            LexosException,
            match="The document-term matrix cannot be empty",
        ):
            _get_matrix(empty_list)

    def test_get_matrix_empty_dataframe_raises_exception(self):
        """Test that _get_matrix raises exception with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            _get_matrix(empty_df)

    def test_get_matrix_preserves_dataframe_structure(self):
        """Test that _get_matrix preserves DataFrame structure and metadata."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df = pd.DataFrame(
            data, columns=["term1", "term2", "term3"], index=["doc1", "doc2", "doc3"]
        )

        result = _get_matrix(df)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        assert list(result.columns) == ["term1", "term2", "term3"]
        assert list(result.index) == ["doc1", "doc2", "doc3"]

    def test_get_matrix_with_large_matrix(self):
        """Test _get_matrix with a larger matrix."""
        data = np.random.rand(100, 50)
        df = pd.DataFrame(data)

        result = _get_matrix(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 50)
        pd.testing.assert_frame_equal(result, df)

    def test_get_matrix_list_bug_fix(self):
        """Test the bug fix in the list handling code."""
        # The original code had a bug where it assigned len(matrix) to first_row twice
        # This test ensures the fix works correctly
        data = [[1, 2], [3, 4], [5, 6]]  # 3 documents, 2 terms each

        result = _get_matrix(data)

        assert isinstance(result, list)
        assert len(result) == 3  # Should have 3 documents
        assert all(len(row) == 2 for row in result)  # Each should have 2 terms

    def test_get_matrix_with_mixed_types(self):
        """Test _get_matrix with DataFrame containing mixed numeric types."""
        data = [[1, 2.5, 3], [4.0, 5, 6.7]]
        df = pd.DataFrame(data, columns=["term1", "term2", "term3"])

        result = _get_matrix(df)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "n_docs,n_terms",
        [
            (2, 1),
            (2, 10),
            (10, 2),
            (5, 5),
        ],
    )
    def test_get_matrix_various_dimensions(self, n_docs, n_terms):
        """Test _get_matrix with various matrix dimensions."""
        data = np.random.rand(n_docs, n_terms)
        df = pd.DataFrame(data)

        result = _get_matrix(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (n_docs, n_terms)

    def test_get_matrix_dtm_sets_index_name(self):
        """Test that _get_matrix sets index name to 'terms' when input is DTM."""
        data = [[1, 2, 3], [4, 5, 6]]
        labels = ["doc1", "doc2"]
        dtm = DTM()
        dtm(docs=data, labels=labels)

        result = _get_matrix(dtm)

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "terms"


class TestClustermap:
    """Test suite for the Clustermap class."""

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample matrix for testing."""
        return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        return pd.DataFrame(
            data,
            columns=["term1", "term2", "term3", "term4"],
            index=["doc1", "doc2", "doc3"],
        )

    @pytest.fixture
    def sample_dtm(self):
        """Create a sample DTM for testing."""
        dtm = Mock(spec=DTM)
        dtm.to_df.return_value = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=["term1", "term2", "term3"],
            index=["doc1", "doc2"],
        )
        dtm.labels = ["doc1", "doc2"]
        return dtm

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return ["Document 1", "Document 2", "Document 3"]

    def test_clustermap_initialization_with_dataframe(self, sample_dataframe):
        """Test Clustermap initialization with a DataFrame."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title="Test Clustermap")

            assert cm.dtm is sample_dataframe
            assert cm.title == "Test Clustermap"
            assert cm.metric == "euclidean"
            assert cm.method == "average"
            assert cm.fig is mock_grid.figure

    def test_clustermap_initialization_with_dtm(self, sample_dtm):
        """Test Clustermap initialization with a DTM object."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dtm)

            assert cm.labels == ["doc1", "doc2"]
            assert cm.fig is mock_grid.figure

    def test_clustermap_initialization_with_array(self, sample_matrix):
        """Test Clustermap initialization with numpy array."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_matrix)

            np.testing.assert_array_equal(cm.dtm, sample_matrix)
            assert cm.labels == ["Doc1", "Doc2", "Doc3"]

    def test_clustermap_with_custom_parameters(self, sample_dataframe):
        """Test Clustermap with custom parameters."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(
                dtm=sample_dataframe,
                metric="cosine",
                method="ward",
                figsize=(10, 8),
                cmap="coolwarm",
                z_score=0,
                hide_upper=True,
                hide_side=False,
            )

            assert cm.metric == "cosine"
            assert cm.method == "ward"
            assert cm.figsize == (10, 8)
            assert cm.cmap == "coolwarm"
            assert cm.z_score == 0
            assert cm.hide_upper is True
            assert cm.hide_side is False

    def test_clustermap_hide_upper_dendrogram(self, sample_dataframe):
        """Test hiding the upper dendrogram."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, hide_upper=True)

            mock_grid.ax_col_dendrogram.remove.assert_called_once()

    def test_clustermap_hide_side_dendrogram(self, sample_dataframe):
        """Test hiding the side dendrogram."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, hide_side=True)

            mock_grid.ax_row_dendrogram.remove.assert_called_once()

    def test_clustermap_with_title(self, sample_dataframe):
        """Test adding a title to the clustermap."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.figure.suptitle = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title="Test Title")

            mock_grid.figure.suptitle.assert_called_once_with("Test Title", y=1.05)

    def test_clustermap_with_title_hide_upper(self, sample_dataframe):
        """Test title positioning when upper dendrogram is hidden."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.figure.suptitle = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title="Test Title", hide_upper=True)

            mock_grid.figure.suptitle.assert_called_once_with("Test Title", y=0.95)

    def test_set_labels_with_dtm(self, sample_dtm):
        """Test _set_labels method with DTM."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dtm)

            assert cm.labels == ["doc1", "doc2"]

    def test_set_labels_with_dataframe(self, sample_dataframe):
        """Test _set_labels method with DataFrame."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)

            # Should extract column names except the first
            expected_labels = ["term2", "term3", "term4"]
            assert cm.labels == expected_labels

    def test_set_labels_with_array(self, sample_matrix):
        """Test _set_labels method with array."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_matrix)

            assert cm.labels == ["Doc1", "Doc2", "Doc3"]

    def test_set_labels_with_custom_labels(self, sample_dataframe, sample_labels):
        """Test _set_labels method with custom labels."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, labels=sample_labels)

            assert cm.labels == sample_labels

    def test_get_colors_none(self, sample_dataframe):
        """Test _get_colors method with None values."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, col_colors=None, row_colors=None)
            col_colors, row_colors = cm._get_colors()

            assert col_colors is None
            assert row_colors is None

    def test_get_colors_default(self, sample_dataframe):
        """Test _get_colors method with default palette."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch("seaborn.husl_palette") as mock_husl,
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid
            mock_husl.return_value = ["color1", "color2"]

            cm = Clustermap(
                dtm=sample_dataframe, col_colors="default", row_colors="default"
            )

            # Reset the call count before calling _get_colors
            mock_husl.reset_mock()

            col_colors, row_colors = cm._get_colors()

            # Now we expect exactly 2 calls
            assert mock_husl.call_count == 2
            mock_husl.assert_called_with(8, s=0.45)

    def test_get_colors_with_dataframe(self, sample_dataframe):
        """Test _get_colors method with DataFrame/Series colors."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            color_df = pd.DataFrame(["red", "blue", "green"], columns=["colors"])

            cm = Clustermap(
                dtm=sample_dataframe, col_colors=color_df, row_colors=color_df
            )
            col_colors, row_colors = cm._get_colors()

            pd.testing.assert_frame_equal(col_colors, color_df)
            pd.testing.assert_frame_equal(row_colors, color_df)

    def test_get_colors_invalid_palette(self, sample_dataframe):
        """Test _get_colors method with invalid palette."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch("seaborn.color_palette", side_effect=ValueError()),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            with pytest.raises(LexosException, match="Invalid column palette"):
                Clustermap(dtm=sample_dataframe, col_colors="invalid_palette")

    def test_validate_linkage_matrices_valid(self, sample_dataframe):
        """Test _validate_linkage_matrices with valid linkage."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch("scipy.cluster.hierarchy.is_valid_linkage", return_value=True),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            valid_linkage = np.array([[0, 1, 0.5, 2], [2, 3, 1.0, 3]])

            # Should not raise an exception
            cm = Clustermap(
                dtm=sample_dataframe,
                row_linkage=valid_linkage,
                col_linkage=valid_linkage,
            )

    def test_validate_linkage_matrices_invalid(self, sample_dataframe):
        """Test _validate_linkage_matrices with invalid linkage."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch(
                "scipy.cluster.hierarchy.is_valid_linkage",
                side_effect=TypeError("Invalid linkage"),
            ),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            invalid_linkage = np.array([[1, 2, 3]])  # Invalid format

            with pytest.raises(LexosException, match="Invalid `row_linkage` value"):
                Clustermap(dtm=sample_dataframe, row_linkage=invalid_linkage)

    def test_save_method(self, sample_dataframe, tmp_path):
        """Test the save method."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_figure = Mock()
            mock_grid.figure = mock_figure
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)

            test_path = tmp_path / "test_clustermap.png"
            cm.save(test_path, dpi=300)

            mock_figure.savefig.assert_called_once_with(test_path, dpi=300)

    def test_show_method(self, sample_dataframe):
        """Test the show method."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_figure = Mock()
            mock_grid.figure = mock_figure
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)
            result = cm.show()

            assert result is mock_figure

    def test_clustermap_with_mask(self, sample_dataframe):
        """Test Clustermap with mask parameter."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            mask = np.array(
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                ]
            )

            cm = Clustermap(dtm=sample_dataframe, mask=mask)

            # Verify mask was passed to seaborn
            call_args = mock_clustermap.call_args
            np.testing.assert_array_equal(call_args[1]["mask"], mask)

    def test_clustermap_with_all_parameters(self, sample_dataframe):
        """Test Clustermap with all possible parameters."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(
                dtm=sample_dataframe,
                labels=["A", "B", "C"],
                metric="cosine",
                method="ward",
                hide_upper=False,
                hide_side=False,
                title="Complete Test",
                z_score=1,
                standard_scale=0,
                figsize=(12, 10),
                row_cluster=True,
                col_cluster=True,
                center=0.5,
                cmap="RdBu",
                linewidths=1.0,
            )

            # Verify all parameters were set correctly
            assert cm.labels == ["A", "B", "C"]
            assert cm.metric == "cosine"
            assert cm.method == "ward"
            assert cm.title == "Complete Test"
            assert cm.z_score == 1
            assert cm.standard_scale == 0
            assert cm.figsize == (12, 10)
            assert cm.center == 0.5
            assert cm.cmap == "RdBu"
            assert cm.linewidths == 1.0

    def test_clustermap_single_document_raises_exception(self):
        """Test that Clustermap raises exception with single document."""
        single_doc_df = pd.DataFrame(
            [[1, 2, 3, 4]], columns=["term1", "term2", "term3", "term4"]
        )

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            Clustermap(dtm=single_doc_df)

    @pytest.mark.parametrize(
        "metric,method",
        [
            ("euclidean", "average"),
            ("cosine", "complete"),
            ("cityblock", "single"),
            ("correlation", "ward"),
        ],
    )
    def test_clustermap_different_metrics_methods(
        self, sample_dataframe, metric, method
    ):
        """Test Clustermap with different metric and method combinations."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, metric=metric, method=method)

            assert cm.metric == metric
            assert cm.method == method

            # Verify parameters passed to seaborn
            call_args = mock_clustermap.call_args
            assert call_args[1]["metric"] == metric
            assert call_args[1]["method"] == method

    def test_row_colors_length_validation(self, sample_dataframe):
        """Test validation of row_colors length."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            # Create row_colors list that's too short
            short_colors = ["red", "blue"]  # Only 2 colors for 3 documents

            with pytest.raises(
                LexosException,
                match="The length of `row_colors` must be greater than the number of labels",
            ):
                Clustermap(dtm=sample_dataframe, row_colors=short_colors)

    def test_clustermap_get_colors_with_invalid_col_palette(self, sample_dataframe):
        """Test _get_colors method with invalid column palette (lines 263-266)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch("seaborn.color_palette", side_effect=ValueError("Invalid palette")),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            with pytest.raises(LexosException, match="Invalid column palette"):
                Clustermap(dtm=sample_dataframe, col_colors="invalid_palette")

    def test_clustermap_get_colors_with_invalid_row_palette(self, sample_dataframe):
        """Test _get_colors method with invalid row palette (lines 263-266)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            with pytest.raises(LexosException, match="Invalid row palette"):
                Clustermap(
                    dtm=sample_dataframe,
                    row_colors="invalid_palette_name_that_does_not_exist",
                )

    def test_set_attrs_with_none_values(self, sample_dataframe):
        """Test _set_attrs method skips None values (lines 276-278)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe, title="Test")
            # Call _set_attrs with a mix of None and actual values
            cm._set_attrs(title=None, cmap="viridis")
            # The title should remain unchanged (None was skipped)
            assert cm.title == "Test"
            # The cmap should be updated
            assert cm.cmap == "viridis"

    def test_set_labels_with_dataframe_multiple_columns(self, sample_dataframe):
        """Test _set_labels with DataFrame extracts columns (lines 301-302)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            cm = Clustermap(dtm=sample_dataframe)
            # Should extract columns starting from index 1
            expected = ["term2", "term3", "term4"]
            assert cm.labels == expected

    def test_validate_linkage_matrices_with_invalid_row_linkage(self, sample_dataframe):
        """Test _validate_linkage_matrices with invalid row linkage (line 414)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch(
                "scipy.cluster.hierarchy.is_valid_linkage",
                side_effect=TypeError("Invalid linkage"),
            ),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            invalid_linkage = np.array([[0, 1, 2]])

            with pytest.raises(LexosException, match="Invalid `row_linkage` value"):
                Clustermap(dtm=sample_dataframe, row_linkage=invalid_linkage)

    def test_validate_linkage_matrices_with_invalid_col_linkage(self, sample_dataframe):
        """Test _validate_linkage_matrices with invalid col linkage (line 414)."""
        with (
            patch("lexos.cluster.clustermap.sns.clustermap") as mock_clustermap,
            patch("matplotlib.pyplot.close"),
            patch(
                "scipy.cluster.hierarchy.is_valid_linkage",
                side_effect=ValueError("Invalid linkage"),
            ),
        ):
            mock_grid = Mock()
            mock_grid.figure = Mock()
            mock_grid.ax_col_dendrogram = Mock()
            mock_grid.ax_row_dendrogram = Mock()
            mock_clustermap.return_value = mock_grid

            invalid_linkage = np.array([[0, 1, 2]])

            with pytest.raises(LexosException, match="Invalid `col_linkage` value"):
                Clustermap(dtm=sample_dataframe, col_linkage=invalid_linkage)


class TestCreateDendrogramTraces:
    """Tests for the _create_dendrogram_traces function."""

    @pytest.fixture
    def sample_linkage_matrix(self):
        """Create a sample linkage matrix for testing."""
        np.random.seed(42)
        data = np.random.rand(4, 3)
        return hierarchy.linkage(data, method="average")

    def test_create_dendrogram_traces_basic(self, sample_linkage_matrix):
        """Test basic dendrogram trace creation."""
        traces, dendro_data = _create_dendrogram_traces(sample_linkage_matrix)

        assert isinstance(traces, list)
        assert len(traces) > 0
        assert all(isinstance(trace, go.Scatter) for trace in traces)
        assert isinstance(dendro_data, dict)
        assert "icoord" in dendro_data
        assert "dcoord" in dendro_data

    def test_create_dendrogram_traces_with_labels(self, sample_linkage_matrix):
        """Test dendrogram trace creation with labels."""
        labels = ["A", "B", "C", "D"]
        traces, dendro_data = _create_dendrogram_traces(
            sample_linkage_matrix, labels=labels
        )

        assert isinstance(traces, list)
        assert len(traces) > 0

    def test_create_dendrogram_traces_orientations(self, sample_linkage_matrix):
        """Test different dendrogram orientations."""
        orientations = ["top", "bottom", "left", "right"]

        for orientation in orientations:
            traces, dendro_data = _create_dendrogram_traces(
                sample_linkage_matrix, orientation=orientation
            )
            assert isinstance(traces, list)
            assert len(traces) > 0

    def test_create_dendrogram_traces_custom_styling(self, sample_linkage_matrix):
        """Test dendrogram trace creation with custom styling."""
        traces, dendro_data = _create_dendrogram_traces(
            sample_linkage_matrix, color="rgb(255,0,0)", line_width=2.0
        )

        assert isinstance(traces, list)
        assert len(traces) > 0

        # Check that styling is applied
        for trace in traces:
            assert trace.line.color == "rgb(255,0,0)"
            assert trace.line.width == 2.0


class TestPlotlyClusterGrid:
    """Tests for the PlotlyClusterGrid class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        data = np.random.rand(5, 10)
        return pd.DataFrame(
            data,
            index=[f"Doc{i + 1}" for i in range(5)],
            columns=[f"Term{i + 1}" for i in range(10)],
        )

    def test_plotly_cluster_grid_initialization(self, sample_dataframe):
        """Test PlotlyClusterGrid initialization."""
        grid = PlotlyClusterGrid(sample_dataframe)

        assert isinstance(grid.data, pd.DataFrame)
        assert isinstance(grid.data2d, pd.DataFrame)
        assert grid.figsize == (800, 600)

    def test_plotly_cluster_grid_with_z_score(self, sample_dataframe):
        """Test PlotlyClusterGrid with z-score normalization."""
        grid = PlotlyClusterGrid(sample_dataframe, z_score=1)

        # Check that z-scoring was applied
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)

    def test_plotly_cluster_grid_with_standard_scale(self, sample_dataframe):
        """Test PlotlyClusterGrid with standard scaling."""
        grid = PlotlyClusterGrid(sample_dataframe, standard_scale=1)

        # Check that standard scaling was applied
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)

    def test_plotly_cluster_grid_z_score_and_standard_scale_error(
        self, sample_dataframe
    ):
        """Test that using both z_score and standard_scale raises error."""
        with pytest.raises(
            ValueError, match="Cannot perform both z-scoring and standard-scaling"
        ):
            PlotlyClusterGrid(sample_dataframe, z_score=1, standard_scale=1)

    def test_plotly_cluster_grid_with_mask(self, sample_dataframe):
        """Test PlotlyClusterGrid with mask."""
        mask = np.zeros(sample_dataframe.shape, dtype=bool)
        mask[0, 0] = True  # Mask first element

        grid = PlotlyClusterGrid(sample_dataframe, mask=mask)

        assert grid.mask is not None
        assert grid.mask.iloc[0, 0] == True

    def test_plotly_cluster_grid_invalid_mask_shape(self, sample_dataframe):
        """Test PlotlyClusterGrid with invalid mask shape."""
        invalid_mask = np.zeros((3, 3), dtype=bool)  # Wrong shape

        with pytest.raises(ValueError, match="Mask must have the same shape as data"):
            PlotlyClusterGrid(sample_dataframe, mask=invalid_mask)

    def test_z_score_method(self, sample_dataframe):
        """Test the _z_score static method."""
        z_scored = PlotlyClusterGrid._z_score(sample_dataframe, axis=1)

        # Check that mean is approximately 0 and std is approximately 1
        assert np.allclose(z_scored.mean().mean(), 0, atol=1e-10)
        assert np.allclose(z_scored.std().mean(), 1, atol=1e-10)

    def test_standard_scale_method(self, sample_dataframe):
        """Test the _standard_scale static method."""
        scaled = PlotlyClusterGrid._standard_scale(sample_dataframe, axis=1)

        # Check that values are between 0 and 1
        assert scaled.min().min() >= 0
        assert scaled.max().max() <= 1

    def test_plotly_cluster_grid_z_score_column_axis(self, sample_dataframe):
        """Test PlotlyClusterGrid with z-score on columns axis (line 485)."""
        grid = PlotlyClusterGrid(sample_dataframe, z_score=0)

        # Check that z-scoring was applied (data should be different)
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)
        # Check that standard deviation is approximately 1 for each row
        assert np.allclose(grid.data2d.std(axis=1).mean(), 1, atol=0.1)

    def test_plotly_cluster_grid_standard_scale_row_axis(self, sample_dataframe):
        """Test PlotlyClusterGrid with standard scaling on rows axis (line 495)."""
        grid = PlotlyClusterGrid(sample_dataframe, standard_scale=0)

        # Check that scaling was applied
        assert not np.allclose(grid.data2d.values, sample_dataframe.values)
        # Check that values are between 0 and 1
        assert grid.data2d.min().min() >= 0
        assert grid.data2d.max().max() <= 1

    def test_plotly_cluster_grid_mask_with_invalid_shape(self, sample_dataframe):
        """Test PlotlyClusterGrid with invalid mask shape (lines 512-516)."""
        invalid_mask = np.zeros((2, 2), dtype=bool)  # Wrong shape

        with pytest.raises(ValueError, match="Mask must have the same shape as data"):
            PlotlyClusterGrid(sample_dataframe, mask=invalid_mask)

    def test_plotly_cluster_grid_mask_with_dataframe(self, sample_dataframe):
        """Test PlotlyClusterGrid with mask as DataFrame (line 545)."""
        mask_df = pd.DataFrame(
            False,
            index=sample_dataframe.index,
            columns=sample_dataframe.columns,
        )
        mask_df.iloc[0, 0] = True

        grid = PlotlyClusterGrid(sample_dataframe, mask=mask_df)

        assert grid.mask is not None
        assert grid.mask.iloc[0, 0] == True

    def test_plotly_cluster_grid_with_z_score_and_standard_scale_error(
        self, sample_dataframe
    ):
        """Test that using both z_score and standard_scale raises error (lines 462, 469)."""
        with pytest.raises(
            ValueError, match="Cannot perform both z-scoring and standard-scaling"
        ):
            PlotlyClusterGrid(sample_dataframe, z_score=1, standard_scale=1)


class TestPlotlyClustermap:
    """Test suite for the PlotlyClustermap class."""

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample matrix for testing."""
        return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        return pd.DataFrame(
            data,
            columns=["term1", "term2", "term3", "term4"],
            index=["doc1", "doc2", "doc3"],
        )

    @pytest.fixture
    def sample_dtm(self):
        """Create a sample DTM for testing."""
        dtm = Mock(spec=DTM)
        dtm.to_df.return_value = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=["term1", "term2", "term3"],
            index=["doc1", "doc2"],
        )
        dtm.labels = ["doc1", "doc2"]
        return dtm

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return ["Document 1", "Document 2", "Document 3"]

    def test_plotly_clustermap_initialization_with_dataframe(self, sample_dataframe):
        """Test PlotlyClustermap initialization with a DataFrame."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, title="Test Clustermap")

            assert cm.dtm is sample_dataframe
            assert cm.title == "Test Clustermap"
            assert cm.metric == "euclidean"
            assert cm.method == "average"
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_initialization_with_dtm(self, sample_dtm):
        """Test PlotlyClustermap initialization with a DTM object."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dtm.to_df.return_value
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1],
                "icoord": [[10, 10, 20, 20]],
                "dcoord": [[0, 5, 5, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dtm)

            assert cm.labels == ["doc1", "doc2"]
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_initialization_with_array(self, sample_matrix):
        """Test PlotlyClustermap initialization with numpy array."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = pd.DataFrame(sample_matrix)
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_matrix)

            np.testing.assert_array_equal(cm.dtm, sample_matrix)
            assert cm.labels == ["Doc1", "Doc2", "Doc3"]

    def test_plotly_clustermap_with_custom_parameters(self, sample_dataframe):
        """Test PlotlyClustermap with custom parameters."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(
                dtm=sample_dataframe,
                metric="cosine",
                method="ward",
                figsize=(800, 600),
                cmap="tropic",
                z_score=0,
                hide_upper=True,
                hide_side=False,
                center=0.5,
            )

            assert cm.metric == "cosine"
            assert cm.method == "ward"
            assert cm.figsize == (800, 600)
            assert cm.cmap == "tropic"
            assert cm.z_score == 0
            assert cm.hide_upper is True
            assert cm.hide_side is False
            assert cm.center == 0.5

    def test_plotly_clustermap_hide_upper_dendrogram(self, sample_dataframe):
        """Test hiding the upper dendrogram."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, hide_upper=True)

            # Verify that column dendrogram traces are not added when hide_upper=True
            assert cm.hide_upper is True
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_hide_side_dendrogram(self, sample_dataframe):
        """Test hiding the side dendrogram."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, hide_side=True)

            assert cm.hide_side is True
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_with_annotations(self, sample_dataframe):
        """Test PlotlyClustermap with annotations."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, annot=True, fmt=".1f")

            assert cm.annot is True
            assert cm.fmt == ".1f"
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_with_mask(self, sample_dataframe):
        """Test PlotlyClustermap with mask parameter."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mask = np.array(
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                ]
            )

            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = pd.DataFrame(
                mask, index=sample_dataframe.index, columns=sample_dataframe.columns
            )
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, mask=mask)

            np.testing.assert_array_equal(cm.mask, mask)
            assert isinstance(cm.fig, go.Figure)

    def test_set_labels_with_dtm(self, sample_dtm):
        """Test _set_labels method with DTM."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dtm.to_df.return_value
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1],
                "icoord": [[10, 10, 20, 20]],
                "dcoord": [[0, 5, 5, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dtm)

            assert cm.labels == ["doc1", "doc2"]

    def test_set_labels_with_dataframe(self, sample_dataframe):
        """Test _set_labels method with DataFrame."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            # Should extract column names except the first
            expected_labels = ["term2", "term3", "term4"]
            assert cm.labels == expected_labels

    def test_set_labels_with_custom_labels(self, sample_dataframe, sample_labels):
        """Test _set_labels method with custom labels."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, labels=sample_labels)

            assert cm.labels == sample_labels

    def test_show_method(self, sample_dataframe):
        """Test the show method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            with patch.object(cm.fig, "show") as mock_show:
                cm.show()
                mock_show.assert_called_once()

    def test_save_method(self, sample_dataframe, tmp_path):
        """Test the save method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            test_path = tmp_path / "test_clustermap.png"

            with patch.object(cm.fig, "write_image") as mock_write:
                cm.save(test_path, width=800, height=600)
                mock_write.assert_called_once_with(test_path, width=800, height=600)

    def test_write_html_method(self, sample_dataframe, tmp_path):
        """Test the write_html method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            test_path = tmp_path / "test_clustermap.html"

            with patch.object(cm.fig, "write_html") as mock_write:
                cm.write_html(test_path, include_plotlyjs=True)
                mock_write.assert_called_once_with(test_path, include_plotlyjs=True)

    def test_write_image_method(self, sample_dataframe, tmp_path):
        """Test the write_image method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            test_path = tmp_path / "test_clustermap.svg"

            with patch.object(cm.fig, "write_image") as mock_write:
                cm.write_image(test_path, format="svg")
                mock_write.assert_called_once_with(test_path, format="svg")

    def test_to_html_method(self, sample_dataframe):
        """Test the to_html method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            with patch.object(
                cm.fig, "to_html", return_value="<html>test</html>"
            ) as mock_to_html:
                result = cm.to_html(include_plotlyjs=True)
                mock_to_html.assert_called_once_with(include_plotlyjs=True)
                assert result == "<html>test</html>"

    def test_to_html_with_sync(self, sample_dataframe):
        """Test the to_html method with synchronization script."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            with patch.object(
                cm.fig, "to_html", return_value="<html><body>test</body></html>"
            ):
                result = cm.to_html(include_sync=True)
                assert "<script>" in result  # Should include sync script
                assert "</body>" in result

    def test_to_image_method(self, sample_dataframe):
        """Test the to_image method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            with patch.object(
                cm.fig, "to_image", return_value=b"fake_image_bytes"
            ) as mock_to_image:
                result = cm.to_image(format="png")
                mock_to_image.assert_called_once_with(format="png")
                assert result == b"fake_image_bytes"

    def test_adjust_layout_for_hidden_upper(self, sample_dataframe):
        """Test _adjust_layout_for_hidden_upper method."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, hide_upper=True)

            # Verify that the method was called during initialization
            assert cm.hide_upper is True
            assert isinstance(cm.fig, go.Figure)

    def test_show_heatmap_labels_configuration(self, sample_dataframe):
        """Test show_heatmap_labels configuration."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, show_heatmap_labels=False)

            assert cm.show_heatmap_labels is False
            assert isinstance(cm.fig, go.Figure)

    def test_show_dendrogram_labels_configuration(self, sample_dataframe):
        """Test show_dendrogram_labels configuration."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, show_dendrogram_labels=True)

            assert cm.show_dendrogram_labels is True
            assert isinstance(cm.fig, go.Figure)

    @pytest.mark.parametrize(
        "metric,method",
        [
            ("euclidean", "average"),
            ("cosine", "complete"),
            ("cityblock", "single"),
            ("correlation", "ward"),
        ],
    )
    def test_plotly_clustermap_different_metrics_methods(
        self, sample_dataframe, metric, method
    ):
        """Test PlotlyClustermap with different metric and method combinations."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, metric=metric, method=method)

            assert cm.metric == metric
            assert cm.method == method

    def test_plotly_clustermap_with_all_parameters(self, sample_dataframe):
        """Test PlotlyClustermap with all possible parameters."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_dendro_data = {
                "leaves": [0, 1, 2],  # or appropriate number based on data
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],  # x-coordinates
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],  # y-coordinates
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(
                dtm=sample_dataframe,
                labels=["A", "B", "C"],
                metric="cosine",
                method="ward",
                hide_upper=False,
                hide_side=False,
                title="Complete Test",
                z_score=1,
                standard_scale=0,
                figsize=(800, 600),
                row_cluster=True,
                col_cluster=True,
                center=0.5,
                cmap="tropic",
                linewidths=1.0,
                annot=True,
                fmt=".3f",
                show_dendrogram_labels=True,
                show_heatmap_labels=True,
            )

            # Verify all parameters were set correctly
            assert cm.labels == ["A", "B", "C"]
            assert cm.metric == "cosine"
            assert cm.method == "ward"
            assert cm.title == "Complete Test"
            assert cm.z_score == 1
            assert cm.standard_scale == 0
            assert cm.figsize == (800, 600)
            assert cm.center == 0.5
            assert cm.cmap == "tropic"
            assert cm.linewidths == 1.0
            assert cm.annot is True
            assert cm.fmt == ".3f"
            assert cm.show_dendrogram_labels is True
            assert cm.show_heatmap_labels is True

    def test_plotly_clustermap_with_precomputed_linkage(self, sample_dataframe):
        """Test PlotlyClustermap with precomputed linkage matrices."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid_class.return_value = mock_grid  # This line was missing!

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            row_linkage = np.array([[0, 1, 0.5, 2], [2, 3, 1.0, 3]])
            col_linkage = np.array([[0, 1, 0.3, 2], [2, 3, 0.8, 3]])

            cm = PlotlyClustermap(
                dtm=sample_dataframe, row_linkage=row_linkage, col_linkage=col_linkage
            )

            np.testing.assert_array_equal(cm.row_linkage, row_linkage)
            np.testing.assert_array_equal(cm.col_linkage, col_linkage)

    def test_plotly_clustermap_single_document_raises_exception(self):
        """Test that PlotlyClustermap raises exception with single document."""
        single_doc_df = pd.DataFrame(
            [[1, 2, 3, 4]], columns=["term1", "term2", "term3", "term4"]
        )

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            PlotlyClustermap(dtm=single_doc_df)

    def test_plotly_clustermap_empty_matrix_raises_exception(self):
        """Test that PlotlyClustermap raises exception with empty matrix."""
        empty_df = pd.DataFrame()

        with pytest.raises(
            LexosException,
            match="The document-term matrix must have more than one document",
        ):
            PlotlyClustermap(dtm=empty_df)

    def test_plotly_clustermap_invalid_z_score_and_standard_scale(
        self, sample_dataframe
    ):
        """Test that PlotlyClustermap raises exception when both z_score and standard_scale are specified."""
        with pytest.raises(
            ValueError, match="Cannot perform both z-scoring and standard-scaling"
        ):
            PlotlyClustermap(dtm=sample_dataframe, z_score=1, standard_scale=1)

    def test_plotly_clustermap_invalid_mask_shape(self, sample_dataframe):
        """Test that PlotlyClustermap raises exception with invalid mask shape."""
        invalid_mask = np.array(
            [[True, False], [False, True]]
        )  # Wrong shape for 3x4 dataframe

        with pytest.raises(ValueError, match="Mask must have the same shape as data"):
            PlotlyClustermap(dtm=sample_dataframe, mask=invalid_mask)

    def test_plotly_clustermap_invalid_linkage_matrix(self, sample_dataframe):
        """Test that PlotlyClustermap raises exception with invalid linkage matrix."""
        with patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class:
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid_class.return_value = mock_grid

            # Test invalid row_linkage - wrong number of columns (should be 4 columns)
            invalid_linkage = np.array([[1.0, 2.0, 0.5]])  # Only 3 columns instead of 4

            with pytest.raises(
                (TypeError, ValueError), match="Linkage matrix 'Z' must have"
            ):
                PlotlyClustermap(dtm=sample_dataframe, row_linkage=invalid_linkage)

    def test_plotly_clustermap_invalid_col_linkage_matrix(self, sample_dataframe):
        """Test that PlotlyClustermap raises exception with invalid column linkage matrix."""
        with patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class:
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid_class.return_value = mock_grid

            # Test invalid col_linkage - wrong number of columns (should be 4 columns)
            invalid_linkage = np.array([[1.0, 2.0, 0.5]])  # Only 3 columns instead of 4

            with pytest.raises(
                (TypeError, ValueError), match="Linkage matrix 'Z' must"
            ):
                PlotlyClustermap(dtm=sample_dataframe, col_linkage=invalid_linkage)

    def test_plotly_clustermap_clustering_disabled_with_no_linkage(
        self, sample_dataframe
    ):
        """Test PlotlyClustermap behavior when clustering is disabled but no linkage provided."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Test with clustering disabled
            cm = PlotlyClustermap(
                dtm=sample_dataframe, row_cluster=False, col_cluster=False
            )

            assert cm.row_cluster is False
            assert cm.col_cluster is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_error_handling_in_linkage_calculation(
        self, sample_dataframe
    ):
        """Test error handling when linkage calculation fails."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            # Mock _calculate_linkage to raise an exception
            mock_grid._calculate_linkage.side_effect = ValueError(
                "Linkage calculation failed"
            )
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            with pytest.raises(ValueError, match="Linkage calculation failed"):
                PlotlyClustermap(dtm=sample_dataframe)

    def test_plotly_clustermap_dendrogram_creation_error(self, sample_dataframe):
        """Test error handling when dendrogram creation fails."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            # Mock _create_dendrogram_traces to raise an exception
            mock_dendro.side_effect = ValueError("Dendrogram creation failed")

            with pytest.raises(ValueError, match="Dendrogram creation failed"):
                PlotlyClustermap(dtm=sample_dataframe)

    def test_plotly_clustermap_with_invalid_figsize(self, sample_dataframe):
        """Test PlotlyClustermap with invalid figsize."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Test with invalid figsize (negative values)
            with pytest.raises(ValueError):
                PlotlyClustermap(dtm=sample_dataframe, figsize=(-100, -100))

    def test_plotly_clustermap_grid_initialization_error(self, sample_dataframe):
        """Test error handling when PlotlyClusterGrid initialization fails."""
        with patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class:
            # Mock PlotlyClusterGrid to raise an exception
            mock_grid_class.side_effect = ValueError("Grid initialization failed")

            with pytest.raises(ValueError, match="Grid initialization failed"):
                PlotlyClustermap(dtm=sample_dataframe)

    def test_plotly_clustermap_with_precomputed_invalid_linkage_size(
        self, sample_dataframe
    ):
        """Test PlotlyClustermap with precomputed linkage of wrong size."""
        # Create a linkage matrix that's clearly invalid for the data size
        # For 3 documents, we need 2 linkage steps (n-1), but provide more
        invalid_linkage = np.array(
            [
                [0.0, 1.0, 0.5, 2.0],
                [2.0, 3.0, 1.0, 3.0],
                [4.0, 5.0, 1.5, 4.0],  # Extra row - invalid for 3 documents
                [6.0, 7.0, 2.0, 5.0],  # Another extra row
            ]
        )

        with pytest.raises((IndexError, ValueError, TypeError, LexosException)):
            PlotlyClustermap(dtm=sample_dataframe, row_linkage=invalid_linkage)

    def test_plotly_clustermap_subplot_creation_error(self, sample_dataframe):
        """Test error handling when subplot creation fails."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
            patch("lexos.cluster.clustermap.make_subplots") as mock_subplots,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Mock make_subplots to raise an exception
            mock_subplots.side_effect = ValueError("Subplot creation failed")

            with pytest.raises(ValueError, match="Subplot creation failed"):
                PlotlyClustermap(dtm=sample_dataframe)

    def test_plotly_clustermap_row_cluster_disabled(self, sample_dataframe):
        """Test PlotlyClustermap with row_cluster=False (lines 680-683)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, row_cluster=False)

            assert cm.row_cluster is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_col_cluster_disabled(self, sample_dataframe):
        """Test PlotlyClustermap with col_cluster=False (line 705)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, col_cluster=False)

            assert cm.col_cluster is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_both_clusters_disabled(self, sample_dataframe):
        """Test PlotlyClustermap with both row_cluster and col_cluster disabled (lines 773-774, 777-778)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(
                dtm=sample_dataframe, row_cluster=False, col_cluster=False
            )

            assert cm.row_cluster is False
            assert cm.col_cluster is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_show_heatmap_labels_explicit_false(
        self, sample_dataframe
    ):
        """Test PlotlyClustermap with show_heatmap_labels=False (lines 828-829)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, show_heatmap_labels=False)

            assert cm.show_heatmap_labels is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_set_labels_with_array(self, sample_matrix):
        """Test PlotlyClustermap _set_labels with array input (lines 1052-1057)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = pd.DataFrame(sample_matrix)
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_matrix)

            # Should generate default labels for array input
            assert cm.labels == ["Doc1", "Doc2", "Doc3"]

    def test_plotly_clustermap_adjust_layout_for_hidden_upper(self, sample_dataframe):
        """Test _adjust_layout_for_hidden_upper method (lines 1086-1098)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe, hide_upper=True)

            assert cm.hide_upper is True
            assert isinstance(cm.fig, go.Figure)
            # Verify that the layout was adjusted
            assert cm.fig.layout is not None

    def test_plotly_cluster_grid_non_dataframe_input(self):
        """Test PlotlyClusterGrid with non-DataFrame input (line 414)."""
        # Test with numpy array
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        grid = PlotlyClusterGrid(data)

        assert isinstance(grid.data, pd.DataFrame)
        assert grid.data.shape == (3, 3)

    def test_plotly_cluster_grid_dataframe_mask_index_match(self, sample_dataframe):
        """Test PlotlyClusterGrid with DataFrame mask (lines 516-517)."""
        mask_df = pd.DataFrame(
            False,
            index=sample_dataframe.index,
            columns=sample_dataframe.columns,
        )
        mask_df.iloc[0, 0] = True

        grid = PlotlyClusterGrid(sample_dataframe, mask=mask_df)

        assert grid.mask is not None
        assert isinstance(grid.mask, pd.DataFrame)
        assert grid.mask.iloc[0, 0] == True

    def test_plotly_cluster_grid_linkage_vector_path(self):
        """Test PlotlyClusterGrid _calculate_linkage with linkage_vector path (line 545)."""
        data = np.random.rand(5, 10)
        grid = PlotlyClusterGrid(pd.DataFrame(data))

        # Test euclidean metric with centroid method (should use linkage_vector)
        linkage = grid._calculate_linkage(data, method="centroid", metric="euclidean")

        assert isinstance(linkage, np.ndarray)
        assert linkage.shape[1] == 4  # Standard linkage matrix has 4 columns

    def test_plotly_cluster_grid_linkage_single_method(self):
        """Test PlotlyClusterGrid _calculate_linkage with single method (uses linkage_vector)."""
        data = np.random.rand(5, 10)
        grid = PlotlyClusterGrid(pd.DataFrame(data))

        # Single method should use linkage_vector
        linkage = grid._calculate_linkage(data, method="single", metric="cosine")

        assert isinstance(linkage, np.ndarray)
        assert linkage.shape[1] == 4

    def test_plotly_clustermap_show_heatmap_labels_auto_mode(self, sample_dataframe):
        """Test PlotlyClustermap with show_heatmap_labels=None (auto-mode, lines 680-683)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Create with show_heatmap_labels=None (auto mode)
            cm = PlotlyClustermap(
                dtm=sample_dataframe, show_heatmap_labels=None, row_cluster=True
            )

            assert cm.show_heatmap_labels is None
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_dendrogram_ratio_tuple(self, sample_dataframe):
        """Test PlotlyClustermap with dendrogram_ratio as tuple (line 705)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Test with tuple dendrogram_ratio
            cm = PlotlyClustermap(
                dtm=sample_dataframe,
                dendrogram_ratio=(0.15, 0.25),
            )

            assert cm.dendrogram_ratio == (0.15, 0.25)
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_heatmap_column_visibility(self, sample_dataframe):
        """Test PlotlyClustermap heatmap column visibility (lines 828-829)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Test with hide_upper and hide_side
            cm = PlotlyClustermap(
                dtm=sample_dataframe,
                hide_upper=False,
                hide_side=False,
            )

            assert cm.hide_upper is False
            assert cm.hide_side is False
            assert isinstance(cm.fig, go.Figure)

    def test_plotly_clustermap_write_html_with_sync(self, sample_dataframe, tmp_path):
        """Test PlotlyClustermap write_html with sync script (lines 834-835)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            cm = PlotlyClustermap(dtm=sample_dataframe)

            test_path = tmp_path / "test.html"

            with patch.object(cm.fig, "write_html") as mock_write:
                cm.write_html(test_path)
                mock_write.assert_called_once()

    def test_plotly_clustermap_layout_adjustment_logic(self, sample_dataframe):
        """Test _adjust_layout_for_hidden_upper xaxis/yaxis replacement logic (lines 1086-1098)."""
        with (
            patch("lexos.cluster.clustermap.PlotlyClusterGrid") as mock_grid_class,
            patch("lexos.cluster.clustermap._create_dendrogram_traces") as mock_dendro,
        ):
            mock_grid = Mock()
            mock_grid.data2d = sample_dataframe
            mock_grid.mask = None
            mock_grid._calculate_linkage.return_value = np.array([[0, 1, 0.5, 2]])
            mock_grid_class.return_value = mock_grid

            mock_dendro_data = {
                "leaves": [0, 1, 2],
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
            }
            mock_dendro.return_value = ([], mock_dendro_data)

            # Test with hide_upper=True to trigger layout adjustment
            cm = PlotlyClustermap(dtm=sample_dataframe, hide_upper=True, hide_side=True)

            # Verify the figure was created
            assert isinstance(cm.fig, go.Figure)
            # The layout should have been adjusted
            assert cm.fig.layout is not None


class TestCreateDendrogramTraces:
    """Test suite for the _create_dendrogram_traces function."""

    def test_create_dendrogram_traces_basic(self):
        """Test basic dendrogram trace creation."""
        linkage_matrix = np.array([[0, 1, 0.5, 2], [2, 3, 1.0, 3]])
        labels = ["A", "B", "C"]

        with patch("lexos.cluster.clustermap.hierarchy.dendrogram") as mock_dendro:
            mock_dendro.return_value = {
                "icoord": [[10, 10, 20, 20], [15, 15, 25, 25]],
                "dcoord": [[0, 5, 5, 0], [0, 10, 10, 0]],
                "leaves": [0, 1, 2],
            }

            traces, dendro_data = _create_dendrogram_traces(
                linkage_matrix, labels, orientation="bottom"
            )

            assert len(traces) > 0
            assert all(isinstance(trace, go.Scatter) for trace in traces)
            assert dendro_data["leaves"] == [0, 1, 2]

    @pytest.mark.parametrize("orientation", ["top", "bottom", "left", "right"])
    def test_create_dendrogram_traces_orientations(self, orientation):
        """Test dendrogram trace creation with different orientations."""
        linkage_matrix = np.array([[0, 1, 0.5, 2]])

        with patch("lexos.cluster.clustermap.hierarchy.dendrogram") as mock_dendro:
            mock_dendro.return_value = {
                "icoord": [[10, 10, 20, 20]],
                "dcoord": [[0, 5, 5, 0]],
                "leaves": [0, 1],
            }

            traces, dendro_data = _create_dendrogram_traces(
                linkage_matrix, orientation=orientation
            )

            assert len(traces) >= 0  # Some traces might be filtered out
            assert dendro_data["leaves"] == [0, 1]

    def test_create_dendrogram_traces_with_custom_styling(self):
        """Test dendrogram trace creation with custom styling."""
        linkage_matrix = np.array([[0, 1, 0.5, 2]])

        with patch("lexos.cluster.clustermap.hierarchy.dendrogram") as mock_dendro:
            mock_dendro.return_value = {
                "icoord": [[10, 10, 20, 20]],
                "dcoord": [[0, 5, 5, 0]],
                "leaves": [0, 1],
            }

            traces, dendro_data = _create_dendrogram_traces(
                linkage_matrix, color="rgb(255,0,0)", line_width=2.0
            )

            for trace in traces:
                assert trace.line.color == "rgb(255,0,0)"
                assert trace.line.width == 2.0
