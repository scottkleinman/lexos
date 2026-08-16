"""test_seetrees.py.

Tests for the SeeTrees class including DTM initialization, distance computation,
visualization entry points, and error handling.

Coverage: 97%. Missing: 115, 119, 272, 284, 330

Last Update: August 16, 2026
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lexos.cluster.seetrees import SeeTrees
from lexos.dtm import DTM


def test_seetrees_initializes_from_dtm():
    """SeeTrees should accept a Lexos DTM and convert it to a frequency table."""
    dtm = DTM()
    dtm(docs=[["apple", "banana"], ["apple", "cherry"]], labels=["doc1", "doc2"])

    st = SeeTrees(dtm=dtm)

    assert not st.frequencies.empty
    assert set(st.frequencies.index) == {"doc1", "doc2"}
    assert "apple" in st.frequencies.columns


def test_seetrees_initializes_from_frequency_dataframe():
    """SeeTrees should preserve a supplied frequency DataFrame."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )

    st = SeeTrees(frequencies=frequencies)

    assert st.frequencies.equals(frequencies)


def test_compute_distances_delta_returns_matrix():
    """compute_distances should return a valid distance matrix for delta."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    distance_table = st.compute_distances(metric="delta")

    assert distance_table.shape == (2, 2)
    assert np.allclose(np.diag(distance_table), np.zeros(2))


def test_compute_distances_invalid_metric_raises_error():
    """An unknown metric should raise a ValueError."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Unknown metric"):
        st.compute_distances(metric="invalid_metric")


def test_view_scores_plots_without_error():
    """view_scores should execute without error for a valid document."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with patch("matplotlib.pyplot.show"):
        st.view_scores(target_text="doc1", top=2)


def test_view_tree_plots_without_error():
    """view_tree should execute without error after computing distances."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)
    st.compute_distances(metric="cosine")

    with patch("matplotlib.pyplot.show"):
        st.view_tree(k=2)


def test_seetrees_initializes_from_stylo_res():
    """SeeTrees should initialize from a stylo_res dictionary."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0], "word2": [0.5, 0.5]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["docA", "docB"], columns=["docA", "docB"]
    )
    stylo_res = {
        "frequencies": frequencies,
        "distance_table": distance_table,
        "features": ["word1", "word2"],
    }

    st = SeeTrees(stylo_res=stylo_res)

    assert st.features == ["word1", "word2"]
    assert st.labels == ["docA", "docB"]
    assert st.frequencies.equals(frequencies)


def test_init_labels_prefers_distance_table_index():
    """Labels should be derived from the distance table when available."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["d1", "d2"], columns=["d1", "d2"]
    )

    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    assert st.labels == ["d1", "d2"]


def test_compare_scores_difference_view():
    """The difference view should execute without error."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with patch("matplotlib.pyplot.show"):
        st.compare_scores("doc1", "doc2", top_diff=2, view_type="difference")


def test_compare_scores_missing_text_raises_error():
    """compare_scores should raise if a source or target label is missing."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Source text 'docX' not found"):
        st.compare_scores("docX", "doc2")


def test_compute_distances_empty_frequencies_raises_error():
    """compute_distances should raise when no frequency data is available."""
    st = SeeTrees()

    with pytest.raises(ValueError, match="Frequency table is required"):
        st.compute_distances(metric="delta")


@pytest.mark.parametrize(
    "metric",
    ["eder_delta", "cosine_delta", "manhattan", "cosine"],
)
def test_compute_distances_other_metrics(metric):
    """compute_distances should support additional metric values."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5, 1.0], "word2": [0.2, 0.8, 0.5]},
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)

    distance_table = st.compute_distances(metric=metric)

    assert distance_table.shape == (3, 3)
    assert np.allclose(np.diag(distance_table), np.zeros(3))


def test_view_distances_mds_plots_without_error():
    """view_distances should plot MDS when a metric is supplied."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with patch("matplotlib.pyplot.show"):
        st.view_distances(method="MDS", metric="delta")


def test_view_distances_pca_plots_without_error():
    """view_distances should plot PCA when requested."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with patch("matplotlib.pyplot.show"):
        st.view_distances(method="PCA")


def test_compare_scores_profile_view():
    """The profile view should execute without error."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with patch("matplotlib.pyplot.show"):
        st.compare_scores("doc1", "doc2", top_diff=2, view_type="profile")


def test_view_distances_invalid_method_raises_error():
    """view_distances should raise for unsupported methods."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Method must be either"):
        st.view_distances(method="INVALID")


def test_view_scores_missing_target_raises_error():
    """view_scores should raise if the target document is not found."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.view_scores(target_text="docX", top=2)


def test_view_tree_raises_without_distance_table():
    """view_tree should raise if no distance table has been computed."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Distance table is required"):
        st.view_tree(k=2)


def test_view_tree_skips_constant_features():
    """view_tree should skip constant features when computing eta-squared."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 1.0, 1.0], "word2": [1.0, 2.0, 3.0]},
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)
    st.compute_distances(metric="delta")

    with patch("matplotlib.pyplot.show"):
        st.view_tree(k=2)
