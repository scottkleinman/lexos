"""test_cloud.py.

Coverage: 99%. Missing: 266, 281

Last Update: August 16, 2026
"""

import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.cloud import MultiCloud, WordCloud

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


# Fixtures
@pytest.fixture
def nlp():
    """Fixture spaCy pipeline."""
    return spacy.load("en_core_web_sm")


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
    return DTM(sample_dataframe.T)  # Transpose for DTM format


@pytest.fixture
def cleanup_plots():
    """Fixture to cleanup matplotlib plots after tests."""
    yield
    plt.close("all")


# Use the cleanup fixture in tests that create plots
pytestmark = pytest.mark.usefixtures("cleanup_plots")


class TestWordCloud:
    """Test the WordCloud class."""

    def test_wordcloud_string_initialization(self):
        """Test WordCloud initialization with string data."""
        wc = WordCloud(data=SAMPLE_TEXT)

        assert wc.data == SAMPLE_TEXT
        assert wc.height == 200
        assert wc.width == 200
        assert isinstance(wc.counts, dict)
        assert wc.cloud is not None
        assert len(wc.counts) == 9  # Number of unique words

    def test_wordcloud_dict_initialization(self):
        """Test WordCloud initialization with dictionary data."""
        wc = WordCloud(data=SAMPLE_DICT)

        assert wc.data == SAMPLE_DICT
        assert wc.counts == SAMPLE_DICT
        assert wc.cloud is not None

    def test_wordcloud_list_initialization(self):
        """Test WordCloud initialization with list data."""
        wc = WordCloud(data=SAMPLE_TOKENS)

        expected_counts = Counter(SAMPLE_TOKENS)
        assert wc.counts == dict(expected_counts)
        assert wc.cloud is not None

    def test_wordcloud_custom_dimensions(self):
        """Test WordCloud with custom dimensions."""
        wc = WordCloud(data=SAMPLE_TEXT, height=400, width=600)

        assert wc.height == 400
        assert wc.width == 600
        assert wc.opts["height"] == 400
        assert wc.opts["width"] == 600

    def test_wordcloud_with_limit(self):
        """Test WordCloud with term limit."""
        wc = WordCloud(data=SAMPLE_DICT, limit=3)

        assert len(wc.counts) <= 3
        # Should contain the top 3 most frequent terms
        top_terms = sorted(SAMPLE_DICT.items(), key=lambda x: x[1], reverse=True)[:3]
        for term, count in top_terms:
            assert term in wc.counts

    def test_wordcloud_with_title(self):
        """Test WordCloud with title."""
        title = "Test Word Cloud"
        wc = WordCloud(data=SAMPLE_TEXT, title=title)

        assert wc.title == title

    def test_wordcloud_with_custom_opts(self):
        """Test WordCloud with custom WordCloud options."""
        custom_opts = {
            "background_color": "black",
            "max_words": 100,
            "colormap": "viridis",
        }
        wc = WordCloud(data=SAMPLE_TEXT, opts=custom_opts)

        assert wc.opts["background_color"] == "black"
        assert wc.opts["max_words"] == 100
        assert wc.opts["colormap"] == "viridis"
        # Should still have height and width set
        assert "height" in wc.opts
        assert "width" in wc.opts

    def test_wordcloud_with_round_mask(self):
        """Test WordCloud with circular mask."""
        wc = WordCloud(data=SAMPLE_TEXT, round=100)

        assert wc.round == 100
        assert "mask" in wc.opts
        assert isinstance(wc.opts["mask"], np.ndarray)

    def test_wordcloud_with_figure_opts(self):
        """Test WordCloud with figure options."""
        figure_opts = {"figsize": (10, 8), "dpi": 150}
        wc = WordCloud(data=SAMPLE_TEXT, figure_opts=figure_opts)

        assert wc.figure_opts == figure_opts

    def test_wordcloud_empty_data(self):
        """Test WordCloud with empty data."""
        with pytest.raises(ValueError):
            wc = WordCloud(data="")

    def test_wordcloud_with_dataframe(self):
        """Test WordCloud with DataFrame data."""
        df = pd.DataFrame(
            {"doc1": [2, 1, 3], "doc2": [1, 3, 0], "doc3": [0, 2, 1]},
            index=["term1", "term2", "term3"],
        )

        wc = WordCloud(data=df)

        assert isinstance(wc.counts, dict)
        assert wc.cloud is not None

    def test_wordcloud_with_spacy_doc(self, nlp):
        """Test WordCloud with spaCy Doc object."""
        doc = nlp(SAMPLE_TEXT)

        wc = WordCloud(data=doc)

        expected_counts = Counter([token.text for token in doc])
        assert wc.counts == dict(expected_counts)
        assert wc.cloud is not None

    def test_wordcloud_save(self):
        """Test saving WordCloud to file."""
        wc = WordCloud(data=SAMPLE_TEXT)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            wc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_wordcloud_save_no_cloud(self):
        """Test saving when no cloud object exists."""
        wc = WordCloud(data=SAMPLE_TEXT)
        wc.cloud = None  # Manually set to None

        with pytest.raises(LexosException) as exc_info:
            wc.save("test.png")

        assert "No WordCloud object to save" in str(exc_info.value)

    def test_wordcloud_show(self):
        """Test showing WordCloud without title."""
        wc = WordCloud(data=SAMPLE_TEXT)

        with (
            patch("matplotlib.pyplot.imshow") as mock_imshow,
            patch("matplotlib.pyplot.axis") as mock_axis,
        ):
            wc.show()
            assert mock_imshow.call_count == 1
            args, kwargs = mock_imshow.call_args
            assert np.array_equal(args[0], wc.cloud.to_array())
            assert kwargs == {"interpolation": "bilinear"}
            mock_axis.assert_called_once_with("off")

    def test_wordcloud_show_with_title(self):
        """Test showing WordCloud with title."""
        title = "Test Title"
        wc = WordCloud(data=SAMPLE_TEXT, title=title)

        with (
            patch("matplotlib.pyplot.imshow") as mock_imshow,
            patch("matplotlib.pyplot.axis") as mock_axis,
            patch("matplotlib.pyplot.figure") as mock_figure,
        ):
            # Create a mock figure object
            mock_fig = Mock()
            mock_figure.return_value = mock_fig

            wc.show()

            # Assert basic matplotlib calls
            assert mock_imshow.call_count == 1
            args, kwargs = mock_imshow.call_args
            assert np.array_equal(args[0], wc.cloud.to_array())
            assert kwargs == {"interpolation": "bilinear"}
            mock_axis.assert_called_once_with("off")
            mock_figure.assert_called_once_with(**wc.figure_opts)

            # Assert that suptitle was called with the correct title
            mock_fig.suptitle.assert_called_once_with(title)


class TestMultiCloud:
    """Test the new MultiCloud class (topic_clouds approach)."""

    def test_multicloud_list_initialization(self):
        """Test MultiCloud initialization with list of documents."""
        mc = MultiCloud(data=SAMPLE_DOCS)

        assert mc.doc_data is not None
        assert len(mc.doc_data) == 4
        assert mc.wordcloud is not None
        assert mc.fig is not None

    def test_multicloud_with_limit(self):
        """Test MultiCloud with term limit."""
        mc = MultiCloud(data=SAMPLE_DOCS, limit=5)

        assert mc.limit == 5
        # Check that the wordcloud respects the max_words setting
        assert mc.opts["max_words"] == 5

    def test_multicloud_with_custom_figsize(self):
        """Test MultiCloud with custom figure size."""
        mc = MultiCloud(data=SAMPLE_DOCS, figsize=(12, 8))

        assert mc.figsize == (12, 8)
        assert mc.fig is not None

    def test_multicloud_with_title_and_labels(self):
        """Test MultiCloud with title and labels."""
        title = "Multiple Word Clouds"
        labels = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        mc = MultiCloud(data=SAMPLE_DOCS, title=title, labels=labels)

        assert mc.title == title
        assert mc.labels == labels

    def test_multicloud_with_custom_opts(self):
        """Test MultiCloud with custom WordCloud options."""
        custom_opts = {"background_color": "black", "max_words": 50}
        mc = MultiCloud(data=SAMPLE_DOCS, opts=custom_opts)

        assert mc.opts["background_color"] == "black"
        assert mc.opts["max_words"] == 50

    def test_multicloud_with_round_mask(self):
        """Test MultiCloud with circular mask."""
        mc = MultiCloud(data=SAMPLE_DOCS, round=150)

        assert mc.round == 150
        assert "mask" in mc.opts

    def test_multicloud_with_auto_layout(self):
        """Test MultiCloud with auto layout."""
        mc = MultiCloud(data=SAMPLE_DOCS, layout="auto")

        assert mc.layout == "auto"
        assert mc.fig is not None

    def test_multicloud_with_custom_layout(self):
        """Test MultiCloud with custom layout."""
        mc = MultiCloud(data=SAMPLE_DOCS, layout=(2, 2))

        assert mc.layout == (2, 2)
        assert mc.fig is not None

    def test_multicloud_dataframe_processing(self):
        """Test MultiCloud with DataFrame data."""
        df = pd.DataFrame(
            {"term1": [2, 1, 0, 3], "term2": [1, 3, 2, 0], "term3": [0, 1, 1, 2]},
            index=["doc1", "doc2", "doc3", "doc4"],
        )

        mc = MultiCloud(data=df)

        assert len(mc.doc_data) == 4
        assert mc.fig is not None

    def test_multicloud_dataframe_with_doc_selection(self):
        """Test MultiCloud with DataFrame and document selection."""
        df = pd.DataFrame(
            {"term1": [2, 1, 0, 3], "term2": [1, 3, 2, 0], "term3": [0, 1, 1, 2]},
            index=["doc1", "doc2", "doc3", "doc4"],
        )

        mc = MultiCloud(data=df, docs=[0, 2])

        assert len(mc.doc_data) == 2

    def test_multicloud_dtm_processing(self):
        """Test MultiCloud with DTM data."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])
        mc = MultiCloud(data=dtm)

        assert len(mc.doc_data) == 3
        assert mc.fig is not None

    def test_multicloud_dtm_with_doc_selection(self):
        """Test MultiCloud with DTM and document selection."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]

        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])
        mc = MultiCloud(data=dtm, docs=[0, 2])

        assert len(mc.doc_data) == 2

    def test_multicloud_empty_data(self):
        """Test MultiCloud with empty list."""
        with pytest.raises(LexosException):
            MultiCloud(data=[])

    def test_multicloud_empty_dtm(self):
        """Test MultiCloud with empty DTM."""
        with pytest.raises(LexosException):
            dtm = DTM()
            MultiCloud(data=dtm)

    def test_multicloud_empty_dataframe(self):
        """Test MultiCloud with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(LexosException) as exc_info:
            MultiCloud(data=empty_df)

        assert "Empty DataFrame provided" in str(exc_info.value)

    def test_multicloud_unsupported_data_type(self):
        """Test MultiCloud raises on unsupported data types."""
        from lexos.visualization.cloud import MultiCloud

        multicloud = MultiCloud.model_construct(data=object())
        with pytest.raises(
            LexosException, match="Unsupported data type for MultiCloud"
        ):
            multicloud._process_data()

    def test_multicloud_show_fallback(self):
        """Test MultiCloud.show fallback when IPython is unavailable."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "IPython.display":
                raise ImportError
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("lexos.visualization.cloud.plt.figure") as mock_figure:
                with patch("lexos.visualization.cloud.plt.show") as mock_show:
                    mc.show()
                    mock_figure.assert_called_once()
                    mock_show.assert_called_once()

    def test_multicloud_single_document(self):
        """Test MultiCloud with single document."""
        mc = MultiCloud(data=[SAMPLE_DOCS[0]])

        assert len(mc.doc_data) == 1
        assert mc.fig is not None

    def test_multicloud_spacy_docs(self, nlp):
        """Test MultiCloud with spaCy Doc objects."""
        docs = [nlp(text) for text in SAMPLE_DOCS]

        mc = MultiCloud(data=docs)

        assert len(mc.doc_data) == 4
        assert mc.fig is not None

    def test_multicloud_save(self):
        """Test saving MultiCloud to file."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            mc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multicloud_save_no_figure(self):
        """Test MultiCloud save when no figure exists."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])
        mc.fig = None

        with pytest.raises(LexosException) as exc_info:
            mc.save("test.png")

        assert "No figure to save" in str(exc_info.value)

    def test_multicloud_show(self):
        """Test MultiCloud show method."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        # The show() method should not raise an exception
        mc.show()

    def test_multicloud_show_no_figure(self):
        """Test MultiCloud show when no figure exists."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])
        mc.fig = None

        with pytest.raises(LexosException) as exc_info:
            mc.show()

        assert "No figure to show" in str(exc_info.value)

    def test_multicloud_dtm_with_zero_counts(self):
        """Test MultiCloud DTM processing with zero counts."""
        docs = [
            ["term1", "term1", "term2"],
            ["term2", "term3", "term3"],
            ["term1", "term3"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])

        mc = MultiCloud(data=dtm)

        assert len(mc.doc_data) == 3
        # Verify zero counts are filtered out
        assert "term3" not in mc.doc_data[0]
        assert "term1" not in mc.doc_data[1]
        assert "term2" not in mc.doc_data[2]

    def test_multicloud_dtm_single_doc_by_index(self):
        """Test MultiCloud with DTM selecting single document by index."""
        docs = [
            ["term1", "term2", "term3"],
            ["term4", "term5", "term6"],
            ["term7", "term8", "term9"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["doc1", "doc2", "doc3"])

        mc = MultiCloud(data=dtm, docs=1)

        assert len(mc.doc_data) == 1
        assert "term4" in mc.doc_data[0]

    def test_multicloud_dtm_single_doc_by_label(self):
        """Test MultiCloud with DTM selecting single document by label."""
        docs = [
            ["apple", "banana", "cherry"],
            ["dog", "elephant", "fox"],
            ["guitar", "harmonica", "instrument"],
        ]

        dtm = DTM()
        dtm(docs=docs, labels=["fruits", "animals", "music"])

        mc = MultiCloud(data=dtm, docs="animals")

        assert len(mc.doc_data) == 1
        assert "dog" in mc.doc_data[0]

    def test_multicloud_dataframe_single_doc_by_index(self):
        """Test MultiCloud with DataFrame selecting single document by index."""
        df = pd.DataFrame(
            {"term1": [2, 0, 1], "term2": [1, 3, 0], "term3": [0, 1, 2]},
            index=["doc1", "doc2", "doc3"],
        )

        mc = MultiCloud(data=df, docs=0)

        assert len(mc.doc_data) == 1

    def test_multicloud_dataframe_single_doc_by_label(self):
        """Test MultiCloud with DataFrame selecting single document by label."""
        df = pd.DataFrame(
            {"word1": [5, 0, 2], "word2": [1, 4, 0], "word3": [0, 2, 3]},
            index=["document_a", "document_b", "document_c"],
        )

        mc = MultiCloud(data=df, docs="document_b")

        assert len(mc.doc_data) == 1
        assert "word2" in mc.doc_data[0]

    def test_multicloud_invalid_layout(self):
        """Test MultiCloud with invalid layout specification."""
        with pytest.raises(LexosException) as exc_info:
            MultiCloud(data=SAMPLE_DOCS, layout="invalid")

        assert "Invalid layout specification" in str(exc_info.value)

    def test_multicloud_with_unicode_terms(self):
        """Test MultiCloud with Unicode characters in terms."""
        unicode_data = ["café résumé naïve", "数据 科学 机器学习", "café résumé データ"]

        mc = MultiCloud(data=unicode_data)

        assert len(mc.doc_data) == 3
        assert mc.fig is not None

    def test_multicloud_process_data_dtm_string_docs(self):
        """Test MultiCloud DTM processing with string document selection."""
        data = [
            ["term1", "term1"],
            ["term1", "term2", "term2", "term2", "term3", "term3"],
            ["term2", "term3"],
        ]
        dtm = DTM()
        dtm(docs=data, labels=["doc1", "doc2", "doc3"])

        mc = MultiCloud(data=dtm, docs=["doc1", "doc3"])

        assert len(mc.doc_data) == 2

    def test_multicloud_process_data_dataframe_string_docs(self):
        """Test MultiCloud DataFrame processing with string document selection."""
        df = pd.DataFrame(
            {"term1": [2, 1, 0], "term2": [1, 3, 2], "term3": [0, 1, 1]},
            index=["doc1", "doc2", "doc3"],
        )

        mc = MultiCloud(data=df, docs=["doc1", "doc3"])

        assert len(mc.doc_data) == 2

    def test_multicloud_save_different_formats(self):
        """Test MultiCloud save with different file formats."""
        mc = MultiCloud(data=SAMPLE_DOCS[:2])

        formats = [".png", ".jpg", ".pdf", ".svg"]

        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                mc.save(tmp_path)
                assert Path(tmp_path).exists()
            finally:
                Path(tmp_path).unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_wordcloud_invalid_dimensions(self):
        """Test WordCloud with invalid dimensions."""
        with pytest.raises(ValueError):
            WordCloud(data=SAMPLE_TEXT, height=10)  # Too small

        with pytest.raises(ValueError):
            WordCloud(data=SAMPLE_TEXT, width=10)  # Too small

    def test_wordcloud_very_large_round(self):
        """Test WordCloud with very large round value."""
        wc = WordCloud(data=SAMPLE_TEXT, round=1000)

        assert wc.round == 1000
        assert "mask" in wc.opts


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_wordcloud_complete_workflow(self):
        """Test complete WordCloud workflow."""
        # Create WordCloud
        wc = WordCloud(
            data=SAMPLE_DICT, title="Integration Test", height=300, width=400, limit=5
        )

        # Verify creation
        assert wc.cloud is not None
        assert len(wc.counts) <= 5
        assert wc.title == "Integration Test"

        # Test save
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            wc.save(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
