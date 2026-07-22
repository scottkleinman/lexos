"""Test the public API for supplementary Lexos modules."""

import pytest

from lexos.cluster import BootstrapConsensus, Dendrogram, KMeans
from lexos.dtm import DTM, Vectorizer
from lexos.kwic import Kwic
from lexos.scrubber import Pipe, Scrubber, scrub
from lexos.tokenizer import Tokenizer


def test_dtm_exports():
    assert DTM is not None
    assert Vectorizer is not None
    import lexos.dtm

    assert set(lexos.dtm.__all__) == {"DTM", "Vectorizer"}


def test_tokenizer_exports():
    assert Tokenizer is not None
    import lexos.tokenizer

    assert "Tokenizer" in lexos.tokenizer.__all__


def test_kwic_exports():
    assert Kwic is not None
    import lexos.kwic

    assert set(lexos.kwic.__all__) == {"Kwic"}


def test_scrubber_exports():
    assert Scrubber is not None
    assert Pipe is not None
    assert scrub is not None
    import lexos.scrubber

    assert set(lexos.scrubber.__all__) == {"Scrubber", "Pipe", "scrub"}


def test_cluster_exports():
    assert BootstrapConsensus is not None
    assert Dendrogram is not None
    assert KMeans is not None
    import lexos.cluster

    assert set(lexos.cluster.__all__) == {"BootstrapConsensus", "Dendrogram", "KMeans"}
