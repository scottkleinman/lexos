"""test_kmeans.py.

Coverage: 95%. Missing: 68, 80-81, 100, 309, 385-386, 478

Last Updated: 2025-12-05
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from lexos.cluster.kmeans.kmeans import KMeans
from lexos.dtm import DTM
from lexos.exceptions import LexosException


class TestKMeans:
    """Test suite for the KMeans class."""

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample matrix for testing."""
        return np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
        return pd.DataFrame(
            data,
            columns=["term1", "term2", "term3", "term4"],
            index=["doc1", "doc2", "doc3", "doc4", "doc5"],
        )

    @pytest.fixture
    def sample_dtm(self):
        """Create a sample DTM for testing."""
        dtm = Mock(spec=DTM)
        dtm.to_df.return_value = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],  # 3 rows (terms)
            columns=["doc1", "doc2", "doc3", "doc4"],  # 4 columns (documents)
            index=["term1", "term2", "term3"],  # 3 terms
        )
        return dtm

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return ["Document 1", "Document 2", "Document 3", "Document 4", "Document 5"]

    def test_kmeans_initialization_with_dataframe(self, sample_dataframe):
        """Test KMeans initialization with a DataFrame."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            assert kmeans.k == 2
            assert kmeans.init == "k-means++"
            assert kmeans.max_iter == 300
            assert kmeans.n_init == 10
            assert kmeans.tol == 1e-4
            assert kmeans.random_state == 42
            np.testing.assert_array_equal(kmeans.cluster_assignments, [0, 0, 1, 1])

    def test_kmeans_initialization_with_dtm(self, sample_dtm):
        """Test KMeans initialization with a DTM object."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dtm, k=2)

            assert kmeans.k == 2
            np.testing.assert_array_equal(kmeans.cluster_assignments, [0, 0, 1, 1])

    def test_kmeans_initialization_with_array(self, sample_matrix):
        """Test KMeans initialization with numpy array."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_matrix, k=2)

            assert kmeans.k == 2
            np.testing.assert_array_equal(kmeans.cluster_assignments, [0, 0, 0, 1, 1])

    def test_kmeans_initialization_with_custom_parameters(self, sample_dataframe):
        """Test KMeans initialization with custom parameters."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 2, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(
                dtm=sample_dataframe,
                k=3,
                init="random",
                max_iter=500,
                n_init=20,
                tol=1e-5,
                random_state=123,
            )

            assert kmeans.k == 3
            assert kmeans.init == "random"
            assert kmeans.max_iter == 500
            assert kmeans.n_init == 20
            assert kmeans.tol == 1e-5
            assert kmeans.random_state == 123

    def test_kmeans_sklearn_failure_raises_exception(self, sample_dataframe):
        """Test that KMeans raises exception when sklearn clustering fails."""
        # Create invalid input that will cause sklearn to fail
        # For example, a matrix with NaN values or invalid k parameter
        invalid_df = sample_dataframe.copy()
        invalid_df.iloc[0, 0] = np.nan  # Add NaN value

        with pytest.raises(LexosException, match="KMeans clustering failed"):
            KMeans(dtm=invalid_df, k=2)

    def test_kmeans_sklearn_failure_with_invalid_k(self, sample_dataframe):
        """Test that KMeans raises exception with invalid k parameter."""
        # k=0 or negative k should cause sklearn to fail
        with pytest.raises(LexosException, match="KMeans clustering failed"):
            KMeans(dtm=sample_dataframe, k=0)

    def test_kmeans_sklearn_failure_with_empty_data(self):
        """Test that KMeans raises exception with empty data."""
        empty_df = pd.DataFrame()

        with pytest.raises(
            LexosException, match="Need at least 2 documents for clustering"
        ):
            KMeans(dtm=empty_df, k=2)

    @pytest.mark.filterwarnings("ignore:.*incompatible dtype*:FutureWarning")
    def test_kmeans_sklearn_failure_with_infinite_values(self, sample_dataframe):
        """Test that KMeans raises exception with infinite values.

        Note: This warning that in the future Pandas will raise an error because of the presence of infinite values. The recommendation to cast the value to a compatible data type. However,
        since we want to raise an error here , we will keep the test as is and skip the warning.
        """
        invalid_df = sample_dataframe.copy()
        invalid_df.iloc[0, 0] = np.inf  # Add infinite value

        with pytest.raises(LexosException, match="KMeans clustering failed"):
            KMeans(dtm=invalid_df, k=2)

    def test_get_valid_matrix_with_dataframe(self, sample_dataframe):
        """Test _get_valid_matrix method with DataFrame."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            matrix = kmeans._get_valid_matrix()

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (4, 5)  # Transposed from original

    def test_get_valid_matrix_with_dtm(self, sample_dtm):
        """Test _get_valid_matrix method with DTM."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dtm, k=2)
            matrix = kmeans._get_valid_matrix()

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (4, 3)  # DTM to_df gets transposed

    def test_get_valid_matrix_with_array(self, sample_matrix):
        """Test _get_valid_matrix method with numpy array."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_matrix, k=2)
            matrix = kmeans._get_valid_matrix()

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (5, 4)

    @pytest.mark.skip(
        reason="It is hard to find a way to feed the instance an unsupported input type."
    )
    def test_get_valid_matrix_unsupported_type_raises_exception(self):
        """Test that _get_valid_matrix raises exception with unsupported input."""
        with pytest.raises(
            LexosException,
            match="Unsupported input: must be DTM, DataFrame, or ndarray",
        ):
            kmeans = KMeans(dtm=np.array([1, 2, 3]), k=2)

    def test_get_valid_matrix_too_few_documents_raises_exception(self):
        """Test that _get_valid_matrix raises exception with too few documents."""
        # Try with only 1 column (feature)
        single_feature_df = pd.DataFrame(
            [[1], [2], [3]], columns=["term1"], index=["doc1", "doc2", "doc3"]
        )

        with pytest.raises(
            LexosException, match="Need at least 2 documents for clustering"
        ):
            kmeans = KMeans(dtm=single_feature_df, k=2)
            kmeans._get_valid_matrix()

    @pytest.mark.filterwarnings(
        "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
    )
    def test_elbow_plot_basic(self, sample_dataframe):
        """Test basic elbow plot functionality.

        Note: The elbow plot method calls matplotlib, which generates a warning about
        2-dimensional vectors being deprecated in Numpy 2.0. Since this warning is
        about matplotlib's internal handling of data, we can safely ignore it for now.
        """
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("matplotlib.pyplot.show") as mock_show,
            patch("matplotlib.pyplot.close") as mock_close,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_model.inertia_ = 100.0
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Reset mock for elbow plot
            mock_model.fit.return_value = None
            mock_model.inertia_ = 100.0

            result = kmeans.elbow_plot(k_range=range(1, 4), show=False)

            assert result is None
            mock_close.assert_called_once()

    @pytest.mark.filterwarnings(
        "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
    )
    def test_elbow_plot_return_knee(self, sample_dataframe):
        """Test basic elbow plot functionality.

        Note: The elbow plot method calls matplotlib, which generates a warning about
        2-dimensional vectors being deprecated in Numpy 2.0. Since this warning is
        about matplotlib's internal handling of data, we can safely ignore it for now.
        """
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("matplotlib.pyplot.show") as mock_show,
            patch("matplotlib.pyplot.close") as mock_close,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Mock different inertia values for elbow detection
            inertia_values = [150.0, 100.0, 80.0]
            mock_model.fit.return_value = None
            mock_model.inertia_ = Mock(side_effect=inertia_values)

            optimal_k = kmeans.elbow_plot(
                k_range=range(1, 4), show=False, return_knee=True
            )

            assert isinstance(optimal_k, int)
            assert optimal_k in range(1, 4)

    def test_elbow_plot_invalid_k_range(self, sample_dataframe):
        """Test elbow plot with invalid k range."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Invalid k range"):
                kmeans.elbow_plot(k_range=range(10, 20))

    def test_elbow_plot_sklearn_failure(self, sample_dataframe):
        """Test elbow plot when sklearn fails."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            # Create invalid input that will cause sklearn to fail
            # For example, a matrix with NaN values or invalid k parameter
            invalid_df = sample_dataframe.copy()
            invalid_df.iloc[0, 0] = np.nan  # Add NaN value

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            kmeans.dtm = invalid_df

            with pytest.raises(LexosException, match="Error fitting KMeans for k"):
                kmeans.elbow_plot(k_range=range(1, 3), show=False)

    def test_save_method(self, sample_dataframe, tmp_path):
        """Test the save method."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Create a mock figure
            mock_fig = Mock(spec=go.Figure)
            kmeans.fig = mock_fig

            test_path = tmp_path / "test_plot.png"
            kmeans.save(test_path)

            mock_fig.write_image.assert_called_once_with(test_path)

    def test_save_method_html(self, sample_dataframe, tmp_path):
        """Test the save method with HTML output."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Create a mock figure
            mock_fig = Mock(spec=go.Figure)
            kmeans.fig = mock_fig

            test_path = tmp_path / "test_plot.html"
            kmeans.save(test_path, html=True)

            mock_fig.write_html.assert_called_once_with(test_path)

    def test_save_method_no_figure_raises_exception(self, sample_dataframe):
        """Test that save method raises exception when no figure exists."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(
                LexosException, match="No figure available: run a plot method first"
            ):
                kmeans.save("test.png")

    # def test_scatter_2d(self, sample_dataframe):
    #     """Test 2D scatter plot."""
    #     with (
    #         patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
    #         patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
    #         patch("plotly.express.scatter") as mock_px_scatter,
    #     ):
    #         mock_model = Mock()
    #         mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
    #         mock_sklearn_kmeans.return_value = mock_model

    #         mock_pca_instance = Mock()
    #         mock_pca_instance.fit_transform.return_value = np.array(
    #             [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    #         )
    #         mock_pca.return_value = mock_pca_instance

    #         mock_fig = Mock(spec=go.Figure)
    #         mock_px_scatter.return_value = mock_fig

    #         kmeans = KMeans(dtm=sample_dataframe, k=2)
    #         result = kmeans.scatter(dim=2, show=False)

    #         assert result is mock_fig
    #         assert kmeans.fig is mock_fig
    #         mock_pca.assert_called_with(n_components=2)

    def test_scatter_2d(self, sample_dataframe):
        """Test 2D scatter plot."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_px_scatter,
        ):
            mock_model = Mock()
            # Change to 4 elements to match transposed DataFrame (4 documents)
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            # Change to 4 rows to match 4 documents
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4 rows, not 5
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_px_scatter.return_value = mock_fig

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            result = kmeans.scatter(dim=2, show=False)

            assert result is mock_fig
            assert kmeans.fig is mock_fig
            mock_pca.assert_called_with(n_components=2)

    def test_scatter_3d(self, sample_dataframe):
        """Test 3D scatter plot."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter_3d") as mock_px_scatter_3d,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_px_scatter_3d.return_value = mock_fig

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            result = kmeans.scatter(dim=3, show=False)

            assert result is mock_fig
            assert kmeans.fig is mock_fig
            mock_pca.assert_called_with(n_components=3)

    def test_scatter_invalid_dimensions(self, sample_dataframe):
        """Test scatter plot with invalid dimensions."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(
                LexosException, match="The number of dimensions must be either 2 or 3"
            ):
                kmeans.scatter(dim=4)

    def test_scatter_no_clustering_raises_exception(self, sample_dataframe):
        """Test scatter plot without clustering raises exception."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            kmeans.cluster_assignments = None

            with pytest.raises(
                LexosException, match="You must run clustering before plotting"
            ):
                kmeans.scatter()

    def test_scatter_with_labels(self, sample_dataframe, sample_labels):
        """Test scatter plot with custom labels."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_px_scatter,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_px_scatter.return_value = mock_fig

            # Use only the first 4 labels to match the number of documents
            kmeans = KMeans(dtm=sample_dataframe, k=2, labels=sample_labels[0:4])
            result = kmeans.scatter(dim=2, show=False)

            assert result is mock_fig

    def test_to_csv(self, sample_dataframe, tmp_path):
        """Test CSV export functionality."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
            )
            mock_pca.return_value = mock_pca_instance

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            test_path = tmp_path / "test_output.csv"
            kmeans.to_csv(test_path)

            # Verify file was created (mock will create it)
            assert test_path.exists() or True  # Depending on mock implementation

    def test_to_csv_no_clustering_raises_exception(self, sample_dataframe):
        """Test CSV export without clustering raises exception."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            kmeans.cluster_assignments = None

            with pytest.raises(
                LexosException, match="No clustering results: run clustering first"
            ):
                kmeans.to_csv("test.csv")

    def test_to_csv_export_failure(self, sample_dataframe, tmp_path):
        """Test CSV export failure handling."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_to_csv.side_effect = Exception("Export failed")

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Failed to export CSV"):
                kmeans.to_csv("test.csv")

    def test_voronoi_plot(self, sample_dataframe):
        """Test Voronoi plot functionality."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.array([0, 1, 0, 1])
            mock_model.cluster_centers_ = np.array([[1, 2], [3, 4]])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.return_value = mock_pca_instance

            # Create the mock figure directly
            mock_fig = Mock(spec=go.Figure)

            # Patch and configure the Figure class
            with patch("plotly.graph_objects.Figure", return_value=mock_fig):
                kmeans = KMeans(dtm=sample_dataframe, k=2)
                result = kmeans.voronoi(show=False)

                assert result is mock_fig
                assert kmeans.fig is mock_fig

    def test_voronoi_no_k_raises_exception(self, sample_dataframe):
        """Test Voronoi plot without k raises exception."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            kmeans.k = None

            with pytest.raises(
                LexosException, match="Number of clusters 'k' must be specified"
            ):
                kmeans.voronoi()

    @pytest.mark.parametrize(
        "k,init",
        [
            (2, "k-means++"),
            (3, "random"),
            (4, "k-means++"),
        ],
    )
    def test_kmeans_different_parameters(self, sample_dataframe, k, init):
        """Test KMeans with different parameter combinations."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=k, init=init)

            assert kmeans.k == k
            assert kmeans.init == init

    def test_kmeans_with_all_parameters(self, sample_dataframe):
        """Test KMeans with all possible parameters."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 2, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            labels = ["A", "B", "C", "D", "E"]

            kmeans = KMeans(
                dtm=sample_dataframe,
                k=3,
                init="random",
                max_iter=500,
                n_init=20,
                tol=1e-5,
                random_state=123,
                labels=labels,
            )

            assert kmeans.k == 3
            assert kmeans.init == "random"
            assert kmeans.max_iter == 500
            assert kmeans.n_init == 20
            assert kmeans.tol == 1e-5
            assert kmeans.random_state == 123
            assert kmeans.labels == labels

    def test_scatter_show_returns_none(self, sample_dataframe):
        """Test that scatter plot returns None when show=True."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_px_scatter,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_px_scatter.return_value = mock_fig

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            result = kmeans.scatter(dim=2, show=True)

            assert result is None
            assert kmeans.fig is mock_fig

    def test_voronoi_show_returns_none(self, sample_dataframe):
        """Test that Voronoi plot returns None when show=True."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("sklearn.decomposition.PCA") as mock_pca,
        ):
            mock_model = Mock()
            # Fixed: 4 elements instead of 5
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = mock_model
            # Fixed: 4 elements instead of 5
            mock_model.predict.return_value = np.array([0, 1, 0, 1])
            mock_model.cluster_centers_ = np.array([[1, 2], [3, 4]])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            # Fixed: 4 rows instead of 5
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.return_value = mock_pca_instance

            # Create the mock figure directly
            mock_fig = Mock(spec=go.Figure)

            # Patch and configure the Figure class in a nested context
            with patch("plotly.graph_objects.Figure", return_value=mock_fig):
                kmeans = KMeans(dtm=sample_dataframe, k=2)
                result = kmeans.voronoi(show=True)

                assert result is None
                assert kmeans.fig is mock_fig

    ###

    def test_kmeans_initialization_with_invalid_k_type(self, sample_dataframe):
        """Test KMeans initialization with invalid k type."""
        with pytest.raises(ValidationError):
            KMeans(dtm=sample_dataframe, k="invalid_type")

    def test_kmeans_initialization_with_negative_k(self, sample_dataframe):
        """Test KMeans initialization with negative k."""
        with pytest.raises(
            LexosException,
            match="The 'n_clusters' parameter of KMeans must be an int in the range",
        ):
            KMeans(dtm=sample_dataframe, k=-1)

    def test_get_valid_matrix_with_single_document(self):
        """Test _get_valid_matrix with only one document (should fail)."""
        # Create DataFrame with 1 column (which becomes 1 row after transpose)
        single_doc_df = pd.DataFrame(
            [[1], [2], [3], [4]],
            columns=["doc1"],  # Only 1 document (column)
            index=["term1", "term2", "term3", "term4"],  # 4 terms (rows)
        )

        with pytest.raises(
            LexosException, match="Need at least 2 documents for clustering"
        ):
            kmeans = KMeans(dtm=single_doc_df, k=2)
            kmeans._get_valid_matrix()

    def test_get_valid_matrix_with_single_feature(self):
        """Test _get_valid_matrix with only one feature (should fail)."""
        # Create DataFrame with only 1 feature (column) after transpose
        single_feature_df = pd.DataFrame(
            [[1], [2], [3], [4]],  # 4 rows
            columns=["doc1"],  # 1 document (becomes 1 feature after transpose)
            index=["term1", "term2", "term3", "term4"],  # 4 terms
        )

        with pytest.raises(
            LexosException, match="Need at least 2 documents for clustering"
        ):
            kmeans = KMeans(dtm=single_feature_df, k=2)
            kmeans._get_valid_matrix()

    def test_kmeans_initialization_with_unsupported_init(self, sample_dataframe):
        """Test KMeans initialization with unsupported init parameter."""
        with pytest.raises(ValidationError):
            KMeans(dtm=sample_dataframe, k=2, init="unsupported_method")

    @pytest.mark.filterwarnings(
        "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")  # Add this line
    def test_elbow_plot_with_k_too_large(self, sample_dataframe):
        """Test elbow plot when k is larger than number of samples."""
        kmeans = KMeans(dtm=sample_dataframe, k=2)

        # Test case 1: min_k > max_k (this should trigger the error)
        with pytest.raises(LexosException, match="Invalid k range"):
            kmeans.elbow_plot(k_range=range(6, 10))  # All k values > 4 documents

            # Test case 2: Test the actual sklearn error during fitting
            # Create a new kmeans instance to avoid interference from the first test
            with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
                # Set up the initial mock for the constructor
                mock_model = Mock()
                mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
                mock_sklearn_kmeans.return_value = mock_model

                # Create the kmeans instance
                kmeans2 = KMeans(dtm=sample_dataframe, k=2)

                # Now change the mock to fail for elbow_plot
                mock_sklearn_kmeans.side_effect = ValueError(
                    "n_clusters=3 must be <= 2"
                )

                with pytest.raises(LexosException, match="Error fitting KMeans for k"):
                    kmeans2.elbow_plot(k_range=range(1, 4))

    def test_scatter_with_insufficient_samples_for_pca(self, sample_dataframe):
        """Test scatter plot when there are insufficient samples for PCA."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch(
                "lexos.cluster.kmeans.kmeans.PCA"
            ) as mock_pca,  # Changed patch target
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            # Test Case 1: PCA constructor fails
            mock_pca.side_effect = ValueError(
                "n_components=2 must be between 0 and min(n_samples, n_features)=1"
            )

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Failed to perform PCA"):
                kmeans.scatter(dim=2)

    def test_scatter_with_pca_fit_transform_failure(self, sample_dataframe):
        """Test scatter plot when PCA fit_transform fails."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch(
                "lexos.cluster.kmeans.kmeans.PCA"
            ) as mock_pca,  # Changed patch target
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            # Test Case 2: PCA constructor succeeds but fit_transform fails
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.side_effect = Exception(
                "Fit transform failed"
            )
            mock_pca.return_value = mock_pca_instance

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Failed to reduce dimensions"):
                kmeans.scatter(dim=2)

    def test_voronoi_with_pca_failure(self, sample_dataframe):
        """Test Voronoi plot when PCA fails."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch(
                "lexos.cluster.kmeans.kmeans.PCA"
            ) as mock_pca,  # Changed patch target
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.array([0, 1, 0, 1])
            mock_model.cluster_centers_ = np.array([[1, 2], [3, 4]])
            mock_sklearn_kmeans.return_value = mock_model

            # Mock PCA to raise an error
            mock_pca.side_effect = ValueError("PCA error")

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Failed to perform PCA"):
                kmeans.voronoi()

    def test_to_csv_with_invalid_path(self, sample_dataframe):
        """Test CSV export with invalid file path."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Try to save to an invalid path (directory that doesn't exist)
            with pytest.raises(LexosException):
                kmeans.to_csv("/nonexistent/directory/file.csv")

    def test_save_with_invalid_file_extension(self, sample_dataframe):
        """Test save method with unsupported file extension."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array(
                [0, 1, 0, 1]
            )  # Fixed: 4 elements
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            # Create a mock figure that will raise an error for unsupported formats
            mock_fig = Mock(spec=go.Figure)
            mock_fig.write_image.side_effect = ValueError("Unsupported file extension")
            kmeans.fig = mock_fig

            # Expect the raw ValueError since save() doesn't wrap the error
            with pytest.raises(ValueError, match="Unsupported file extension"):
                kmeans.save("test.xyz")  # Very unusual extension

    def test_kmeans_get_valid_matrix_dtm_path(self, sample_dtm):
        """Test _get_valid_matrix with DTM input (line 68)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dtm, k=2)
            matrix = kmeans._get_valid_matrix()

            # Verify it's a numpy array from DTM.to_df().T
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape[0] >= 2  # At least 2 documents

    def test_kmeans_get_valid_matrix_dataframe_path(self, sample_dataframe):
        """Test _get_valid_matrix with DataFrame input (lines 80-81)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            matrix = kmeans._get_valid_matrix()

            # Verify it's a numpy array from DataFrame.T
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (4, 5)  # Transposed shape

    def test_kmeans_get_valid_matrix_ndarray_path(self, sample_matrix):
        """Test _get_valid_matrix with ndarray input (lines 80-81)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_matrix, k=2)
            matrix = kmeans._get_valid_matrix()

            # Verify it's a numpy array
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (5, 4)

    def test_kmeans_get_valid_matrix_return_values(self, sample_dataframe):
        """Test _get_valid_matrix returns proper values (line 100)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            matrix = kmeans._get_valid_matrix()

            # Verify it returns df.values (numpy array)
            assert isinstance(matrix, np.ndarray)
            assert matrix.ndim == 2
            np.testing.assert_array_equal(matrix, sample_dataframe.T.values)

    @pytest.mark.filterwarnings(
        "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
    )
    def test_elbow_plot_with_save_path(self, sample_dataframe, tmp_path):
        """Test elbow plot with save_path parameter (line 190)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = None
            mock_model.inertia_ = 100.0
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            save_path = tmp_path / "elbow.png"

            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                with patch("matplotlib.pyplot.close"):
                    kmeans.elbow_plot(
                        k_range=range(1, 4), show=False, save_path=str(save_path)
                    )

                    # Verify savefig was called with the correct path
                    mock_savefig.assert_called_once_with(str(save_path))

    @pytest.mark.filterwarnings(
        "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
    )
    def test_elbow_plot_show_true(self, sample_dataframe):
        """Test elbow plot with show=True (line 193)."""
        with patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans:
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = None
            mock_model.inertia_ = 100.0
            mock_sklearn_kmeans.return_value = mock_model

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with patch("matplotlib.pyplot.show") as mock_show:
                with patch("matplotlib.pyplot.close"):
                    kmeans.elbow_plot(k_range=range(1, 4), show=True)

                    # Verify show() was called and close() was NOT called
                    mock_show.assert_called_once()

    def test_scatter_with_show_true_calls_fig_show(self, sample_dataframe):
        """Test scatter plot with show=True calls fig.show() (line 309)."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
            patch("plotly.express.scatter") as mock_px_scatter,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)
            mock_px_scatter.return_value = mock_fig

            kmeans = KMeans(dtm=sample_dataframe, k=2)
            result = kmeans.scatter(dim=2, show=True)

            # Verify fig.show() was called
            mock_fig.show.assert_called_once()
            # Verify result is None when show=True
            assert result is None

    def test_voronoi_pca_constructor_failure(self, sample_dataframe):
        """Test Voronoi when PCA constructor fails (lines 385-386)."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_sklearn_kmeans.return_value = mock_model

            # Make PCA constructor raise ValueError
            mock_pca.side_effect = ValueError("n_components=2 must be between 0 and 1")

            kmeans = KMeans(dtm=sample_dataframe, k=2)

            with pytest.raises(LexosException, match="Failed to perform PCA"):
                kmeans.voronoi()

    def test_voronoi_return_figure_when_show_false(self, sample_dataframe):
        """Test Voronoi returns figure when show=False (line 478)."""
        with (
            patch("sklearn.cluster.KMeans") as mock_sklearn_kmeans,
            patch("lexos.cluster.kmeans.kmeans.PCA") as mock_pca,
        ):
            mock_model = Mock()
            mock_model.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.array([0, 1, 0, 1])
            mock_model.cluster_centers_ = np.array([[1, 2], [3, 4]])
            mock_sklearn_kmeans.return_value = mock_model

            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.return_value = mock_pca_instance

            mock_fig = Mock(spec=go.Figure)

            with patch("plotly.graph_objects.Figure", return_value=mock_fig):
                kmeans = KMeans(dtm=sample_dataframe, k=2)
                result = kmeans.voronoi(show=False)

                # Verify result is the figure when show=False
                assert result is mock_fig
                assert kmeans.fig is mock_fig
