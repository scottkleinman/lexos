"""Test the public API for supplementary Lexos modules.

Coverage: 100%
Last Updated: 22 July, 2026
"""

import pytest

from lexos.cluster import BootstrapConsensus, Dendrogram, KMeans
from lexos.dtm import DTM, Vectorizer
from lexos.kwic import Kwic
from lexos.scrubber import Pipe, Scrubber, scrub
from lexos.tokenizer import Tokenizer


def test_dtm_exports():
    """Test that the DTM and Vectorizer classes are correctly exported from the lexos.dtm module."""
    assert DTM is not None
    assert Vectorizer is not None
    import lexos.dtm

    assert set(lexos.dtm.__all__) == {"DTM", "Vectorizer"}


def test_tokenizer_exports():
    """Test that the Tokenizer class is correctly exported from the lexos.tokenizer module."""
    assert Tokenizer is not None
    import lexos.tokenizer

    assert "Tokenizer" in lexos.tokenizer.__all__


def test_kwic_exports():
    """Test that the Kwic class is correctly exported from the lexos.kwic module."""
    assert Kwic is not None
    import lexos.kwic

    assert set(lexos.kwic.__all__) == {"Kwic"}


def test_scrubber_exports():
    """Test that the Scrubber, Pipe, and scrub functions are correctly exported from the lexos.scrubber module."""
    assert Scrubber is not None
    assert Pipe is not None
    assert scrub is not None
    import lexos.scrubber

    assert set(lexos.scrubber.__all__) == {"Scrubber", "Pipe", "scrub"}


def test_cluster_exports():
    """Test that the BootstrapConsensus, Dendrogram, and KMeans classes are correctly exported from the lexos.cluster module."""
    assert BootstrapConsensus is not None
    assert Dendrogram is not None
    assert KMeans is not None
    import lexos.cluster

    assert set(lexos.cluster.__all__) == {
        "BootstrapConsensus",
        "Clustermap",
        "Dendrogram",
        "PlotlyClusterGrid",
        "PlotlyClustermap",
        "KMeans",
    }
