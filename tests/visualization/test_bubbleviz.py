"""test_bubbleviz.py.

Coverage: 100%

Last Tested: December 5, 2025
"""

import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.bubbleviz import DEFAULT_COLORS, BubbleChart

# Test data
SAMPLE_TEXT = (
    "natural language processing machine learning artificial intelligence data science"
)
SAMPLE_DICT = {
    "natural": 10,
    "language": 8,
    "processing": 6,
    "machine": 4,
    "learning": 2,
}
SAMPLE_TOKENS = ["natural", "language", "processing", "machine", "learning"]
SAMPLE_DOCS = [
    "natural language processing",
    "machine learning algorithms",
    "data science analytics",
    "artificial intelligence systems",
]


class TestBubbleChart:
    """Test the BubbleChart class."""

    def test_bubblechart_string_initialization(self):
        """Test BubbleChart initialization with string data."""
        bc = BubbleChart(data=SAMPLE_TEXT)

        assert isinstance(bc.counts, dict)
        assert bc.figsize == (10, 10)  # Default figsize becomes tuple
        assert bc.limit == 100
        assert bc.bubble_spacing == 0.1
        assert bc.colors == DEFAULT_COLORS
        assert bc.font_family == "DejaVu Sans"
        assert bc.showfig == True
        assert bc.fig is not None
        assert isinstance(bc.bubbles, np.ndarray)

    def test_bubblechart_dict_initialization(self):
        """Test BubbleChart initialization with dictionary data."""
        bc = BubbleChart(data=SAMPLE_DICT)

        assert bc.counts == SAMPLE_DICT
        assert bc.fig is not None
        assert len(bc.bubbles) == len(SAMPLE_DICT)

    def test_bubblechart_list_initialization(self):
        """Test BubbleChart initialization with list data."""
        bc = BubbleChart(data=SAMPLE_TOKENS)

        expected_counts = Counter(SAMPLE_TOKENS)
        assert bc.counts == dict(expected_counts)
        assert bc.fig is not None

    def test_bubblechart_custom_parameters(self):
        """Test BubbleChart with custom parameters."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        bc = BubbleChart(
            data=SAMPLE_TEXT,
            limit=5,
            title="Custom Bubble Chart",
            bubble_spacing=0.2,
            colors=custom_colors,
            figsize=8,
            font_family="Arial",
            showfig=False,
        )

        assert bc.limit == 5
        assert bc.title == "Custom Bubble Chart"
        assert bc.bubble_spacing == 0.2
        assert bc.colors == custom_colors
        assert bc.figsize == (8, 8)
        assert bc.font_family == "Arial"
        assert bc.showfig == False
        assert len(bc.counts) <= 5

    def test_bubblechart_with_spacy_doc(self):
        """Test BubbleChart with spaCy Doc object."""
        import spacy

        nlp = spacy.blank("en")
        doc = nlp(SAMPLE_TEXT)

        bc = BubbleChart(data=doc)

        expected_counts = Counter([token.text for token in doc])
        assert bc.counts == dict(expected_counts)
        assert bc.fig is not None

    def test_bubblechart_with_spacy_span(self):
        """Test BubbleChart with spaCy Span object."""
        import spacy

        nlp = spacy.blank("en")
        doc = nlp(SAMPLE_TEXT)
        span = doc[2:5]  # Select middle tokens

        bc = BubbleChart(data=span)

        expected_counts = Counter([token.text for token in span])
        assert bc.counts == dict(expected_counts)
        assert bc.fig is not None

    def test_bubblechart_with_token_list(self):
        """Test BubbleChart with list of spaCy Token objects."""
        import spacy

        nlp = spacy.blank("en")
        doc = nlp(SAMPLE_TEXT)
        tokens = list(doc)

        bc = BubbleChart(data=tokens)

        expected_counts = Counter([token.text for token in tokens])
        assert bc.counts == dict(expected_counts)
        assert bc.fig is not None

    def test_bubblechart_with_nested_lists(self):
        """Test BubbleChart with nested list structure."""
        nested_data = [
            ["natural", "language", "processing"],
            ["machine", "learning", "algorithms"],
            ["data", "science", "analytics"],
        ]

        bc = BubbleChart(data=nested_data)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_with_doc_list(self):
        """Test BubbleChart with list of spaCy Doc objects."""
        import spacy

        nlp = spacy.blank("en")
        docs = [nlp(text) for text in SAMPLE_DOCS]

        bc = BubbleChart(data=docs)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_with_dataframe(self):
        """Test BubbleChart with DataFrame data."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 3, 0], "doc2": [1, 3, 0, 2], "doc3": [0, 2, 1, 1]},
            index=["term1", "term2", "term3", "term4"],
        )

        bc = BubbleChart(data=df)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_with_dtm(self):
        """Test BubbleChart with DTM data."""
        docs = [
            ["term1", "term2", "term3"],
            ["term1", "term3", "term4"],
            ["term2", "term4", "term5"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])
        bc = BubbleChart(data=dtm)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_with_limit(self):
        """Test BubbleChart with term limit."""
        large_text = " ".join([f"word{i}" for i in range(50)])
        bc = BubbleChart(data=large_text, limit=10)

        assert len(bc.counts) <= 10
        assert len(bc.bubbles) <= 10

    def test_bubblechart_with_docs_parameter(self):
        """Test BubbleChart with docs parameter for document selection."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 0], "doc2": [1, 3, 2], "doc3": [0, 1, 1]},
            index=["term1", "term2", "term3"],
        )

        bc = BubbleChart(data=df, docs=[0, 2])  # Select first and third docs

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_empty_data_validation(self):
        """Test BubbleChart validation with empty data."""
        # Empty string
        with pytest.raises(LexosException) as exc_info:
            BubbleChart(data="")
        assert "Data is an empty list or string" in str(exc_info.value)

        # Empty list
        with pytest.raises(LexosException) as exc_info:
            BubbleChart(data=[])
        assert "Data is an empty list or string" in str(exc_info.value)

        # Empty dict
        with pytest.raises(LexosException) as exc_info:
            BubbleChart(data={})
        assert "Data is an empty list or string" in str(exc_info.value)

    def test_bubblechart_empty_dataframe_validation(self):
        """Test BubbleChart validation with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(LexosException) as exc_info:
            BubbleChart(data=empty_df)
        assert "Dataframe is empty" in str(exc_info.value)

    def test_bubblechart_center_distance(self):
        """Test BubbleChart _center_distance method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        bubble = np.array([0, 0, 1, 1])
        bubbles = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])

        distances = bc._center_distance(bubble, bubbles)
        expected = np.array([np.sqrt(2), np.sqrt(8)])
        np.testing.assert_array_almost_equal(distances, expected)

    def test_bubblechart_center_of_mass(self):
        """Test BubbleChart _center_of_mass method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        # Test with known bubble configuration
        bc.bubbles = np.array(
            [
                [0, 0, 1, 4],  # weight 4 at origin
                [2, 0, 1, 1],  # weight 1 at (2,0)
            ]
        )

        com = bc._center_of_mass()
        expected_x = (0 * 4 + 2 * 1) / (4 + 1)  # weighted average
        expected_y = (0 * 4 + 0 * 1) / (4 + 1)
        expected = np.array([expected_x, expected_y])

        np.testing.assert_array_almost_equal(com, expected)

    def test_bubblechart_outline_distance(self):
        """Test BubbleChart _outline_distance method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        bubble = np.array([0, 0, 1, 1])  # radius 1
        bubbles = np.array([[3, 0, 1, 1]])  # radius 1, distance 3

        outline_dist = bc._outline_distance(bubble, bubbles)
        # Distance = 3, radii = 1+1=2, spacing = 0.1, so outline = 3-2-0.1 = 0.9
        expected = np.array([3 - 1 - 1 - bc.bubble_spacing])

        np.testing.assert_array_almost_equal(outline_dist, expected)

    def test_bubblechart_check_collisions(self):
        """Test BubbleChart _check_collisions method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        # Two overlapping bubbles
        bubble = np.array([0, 0, 1, 1])
        bubbles = np.array([[0.5, 0, 1, 1]])  # Overlapping bubble

        collisions = bc._check_collisions(bubble, bubbles)
        assert collisions > 0  # Should detect collision

    def test_bubblechart_collides_with(self):
        """Test BubbleChart _collides_with method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        bubble = np.array([0, 0, 1, 1])
        bubbles = np.array(
            [
                [0.5, 0, 1, 1],  # Close bubble
                [10, 10, 1, 1],  # Far bubble
            ]
        )

        colliding_indices = bc._collides_with(bubble, bubbles)
        assert 0 in colliding_indices  # Should return index of closest bubble

    def test_bubblechart_collapse(self):
        """Test BubbleChart _collapse method."""
        bc = BubbleChart(data={"a": 5, "b": 3})

        initial_positions = bc.bubbles[:, :2].copy()
        bc._collapse(n_iterations=5)
        final_positions = bc.bubbles[:, :2]

        # Positions should have changed (bubbles moved toward center)
        assert not np.array_equal(initial_positions, final_positions)

    def test_bubblechart_plot_method(self):
        """Test BubbleChart _plot method."""
        bc = BubbleChart(data=SAMPLE_DICT)

        fig, ax = plt.subplots()
        labels = list(SAMPLE_DICT.keys())

        # Should not raise an exception
        bc._plot(ax, labels)

        # Check that patches were added
        assert len(ax.patches) == len(labels)
        plt.close(fig)

    def test_bubblechart_save_method(self):
        """Test BubbleChart save method."""
        bc = BubbleChart(data=SAMPLE_TEXT)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            bc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_bubblechart_save_empty_path(self):
        """Test BubbleChart save with empty path."""
        bc = BubbleChart(data=SAMPLE_TEXT)

        with pytest.raises(LexosException) as exc_info:
            bc.save("")
        assert "You must provide a valid path" in str(exc_info.value)

    def test_bubblechart_save_no_figure(self):
        """Test BubbleChart save when no figure exists."""
        bc = BubbleChart(data=SAMPLE_TEXT)
        bc.fig = None  # Manually remove figure

        with pytest.raises(LexosException) as exc_info:
            bc.save("test.png")
        assert "The figure has not yet been generated" in str(exc_info.value)

    def test_bubblechart_show_method(self):
        """Test BubbleChart show method."""
        bc = BubbleChart(data=SAMPLE_TEXT)

        fig = bc.show()
        assert fig is not None
        assert fig == bc.fig

    def test_bubblechart_with_title(self):
        """Test BubbleChart with title."""
        title = "Test Bubble Chart"
        bc = BubbleChart(data=SAMPLE_TEXT, title=title)

        assert bc.title == title
        assert bc.fig is not None

    def test_bubblechart_custom_colors(self):
        """Test BubbleChart with custom colors."""
        custom_colors = ["#FF5733", "#33FF57", "#3357FF"]
        bc = BubbleChart(data=SAMPLE_DICT, colors=custom_colors)

        assert bc.colors == custom_colors

    def test_bubblechart_different_bubble_spacing(self):
        """Test BubbleChart with different bubble spacing."""
        bc1 = BubbleChart(data=SAMPLE_DICT, bubble_spacing=0.05)
        bc2 = BubbleChart(data=SAMPLE_DICT, bubble_spacing=0.5)

        assert bc1.bubble_spacing == 0.05
        assert bc2.bubble_spacing == 0.5
        assert bc1.bubble_spacing != bc2.bubble_spacing

    def test_bubblechart_different_figsize(self):
        """Test BubbleChart with different figure sizes."""
        bc = BubbleChart(data=SAMPLE_TEXT, figsize=15)

        assert bc.figsize == (15, 15)

    def test_bubblechart_no_showfig(self):
        """Test BubbleChart with showfig=False."""
        bc = BubbleChart(data=SAMPLE_TEXT, showfig=False)

        assert bc.showfig == False
        assert bc.fig is not None  # Figure should still be created

    def test_bubblechart_large_dataset(self):
        """Test BubbleChart with large dataset."""
        # Create large text with many unique words
        large_text = " ".join([f"word{i}" for i in range(200)])
        bc = BubbleChart(data=large_text, limit=50)

        assert len(bc.counts) <= 50
        assert bc.fig is not None

    def test_bubblechart_single_word(self):
        """Test BubbleChart with single word."""
        bc = BubbleChart(data="word")

        assert bc.counts == {"word": 1}
        assert len(bc.bubbles) == 1

    def test_bubblechart_repeated_words(self):
        """Test BubbleChart with repeated words."""
        text = "apple apple banana apple banana banana banana"
        bc = BubbleChart(data=text)

        expected = {"apple": 3, "banana": 4}
        assert bc.counts == expected

    def test_bubblechart_unicode_text(self):
        """Test BubbleChart with Unicode characters."""
        unicode_text = "café résumé naïve 数据 科学"
        bc = BubbleChart(data=unicode_text)

        assert "café" in bc.counts
        assert "数据" in bc.counts
        assert bc.fig is not None

    def test_bubblechart_special_characters(self):
        """Test BubbleChart with special characters."""
        special_text = "hello@world.com test-case user#123"
        bc = BubbleChart(data=special_text)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_dataframe_with_docs_selection(self):
        """Test BubbleChart with DataFrame and document selection."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 0], "doc2": [1, 3, 2], "doc3": [0, 1, 1]},
            index=["term1", "term2", "term3"],
        )

        bc = BubbleChart(data=df, docs=[0, 2])

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_dtm_with_docs_selection(self):
        """Test BubbleChart with DTM and document selection."""
        docs = [["apple", "banana"], ["cherry", "date"], ["elderberry", "fig"]]

        dtm = DTM()
        dtm(docs=docs, labels=["fruits1", "fruits2", "fruits3"])
        bc = BubbleChart(data=dtm, docs=[0, 2])

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_font_family(self):
        """Test BubbleChart with different font families."""
        bc = BubbleChart(data=SAMPLE_TEXT, font_family="Times New Roman")

        assert bc.font_family == "Times New Roman"

    def test_bubblechart_zero_limit(self):
        """Test BubbleChart with zero limit."""
        with pytest.raises(ValidationError):
            bc = BubbleChart(data=SAMPLE_TEXT, limit=0)

    def test_bubblechart_none_limit(self):
        """Test BubbleChart with None limit."""
        bc = BubbleChart(data=SAMPLE_TEXT, limit=None)

        # Should include all words when limit is None
        expected_count = len(SAMPLE_TEXT.split())
        assert len(bc.counts) == len(set(SAMPLE_TEXT.split()))

    def test_bubblechart_invalid_data_type(self):
        """Test BubbleChart with invalid data type."""
        with pytest.raises(ValidationError):
            BubbleChart(data=12345)  # int is not in allowed types

    def test_bubblechart_nested_token_lists(self):
        """Test BubbleChart with nested lists of tokens."""
        import spacy

        nlp = spacy.blank("en")

        doc1 = nlp("hello world")
        doc2 = nlp("foo bar")

        nested_tokens = [list(doc1), list(doc2)]
        bc = BubbleChart(data=nested_tokens)

        assert isinstance(bc.counts, dict)
        assert bc.fig is not None

    def test_bubblechart_edge_case_single_bubble(self):
        """Test BubbleChart edge case with single bubble."""
        bc = BubbleChart(data={"single": 1})

        assert len(bc.bubbles) == 1
        assert bc.counts == {"single": 1}
        assert bc.fig is not None

    def test_bubblechart_very_small_counts(self):
        """Test BubbleChart with very small count values."""
        small_counts = {"a": 1, "b": 1, "c": 1}
        bc = BubbleChart(data=small_counts)

        assert bc.counts == small_counts
        assert len(bc.bubbles) == 3

    def test_bubblechart_very_large_counts(self):
        """Test BubbleChart with very large count values."""
        large_counts = {"word1": 1000, "word2": 500, "word3": 2000}
        bc = BubbleChart(data=large_counts)

        assert bc.counts == large_counts
        assert bc.fig is not None

    def test_bubblechart_matplotlib_cleanup(self):
        """Test that BubbleChart properly closes matplotlib figures."""
        initial_figs = len(plt.get_fignums())

        bc = BubbleChart(data=SAMPLE_TEXT)

        # Figure should be created but closed (due to plt.close() in __init__)
        current_figs = len(plt.get_fignums())
        assert current_figs == initial_figs  # Should be the same due to plt.close()

        # But bc.fig should still exist
        assert bc.fig is not None


class TestBubbleChartIntegration:
    """Integration tests for complete BubbleChart workflows."""

    def test_bubblechart_complete_workflow(self):
        """Test complete BubbleChart workflow."""
        # Create bubble chart
        bc = BubbleChart(
            data=SAMPLE_DICT,
            title="Integration Test",
            limit=5,
            bubble_spacing=0.15,
            figsize=12,
        )

        # Verify creation
        assert bc.fig is not None
        assert len(bc.counts) <= 5
        assert bc.title == "Integration Test"
        assert bc.bubble_spacing == 0.15
        assert bc.figsize == (12, 12)

        # Test save
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            bc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_bubblechart_multiple_data_types(self):
        """Test BubbleChart with multiple data types in sequence."""
        data_types = [SAMPLE_TEXT, SAMPLE_DICT, SAMPLE_TOKENS]

        bubble_charts = []
        for data in data_types:
            bc = BubbleChart(data=data, limit=10)
            bubble_charts.append(bc)
            assert bc.fig is not None
            assert isinstance(bc.counts, dict)

        # Verify all charts were created successfully
        assert len(bubble_charts) == 3


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame(
        {"doc1": [2, 1, 0, 3], "doc2": [1, 3, 2, 0], "doc3": [0, 1, 1, 2]},
        index=["term1", "term2", "term3", "term4"],
    )


@pytest.fixture
def sample_dtm(sample_dataframe):
    """Fixture providing a sample DTM."""
    docs = [
        ["term1", "term1", "term2"],
        ["term2", "term2", "term3"],
        ["term3", "term4", "term4"],
    ]
    dtm = DTM()
    dtm(docs=docs, labels=["doc1", "doc2", "doc3"])
    return dtm


@pytest.fixture
def cleanup_plots():
    """Fixture to cleanup matplotlib plots after tests."""
    yield
    plt.close("all")


# Use the cleanup fixture in tests that create plots
pytestmark = pytest.mark.usefixtures("cleanup_plots")
