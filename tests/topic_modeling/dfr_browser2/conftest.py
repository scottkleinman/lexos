"""Pytest fixtures for dfr_browser2 tests."""

import json
from pathlib import Path
from typing import Callable, List

import pytest


def _create_template(template_dir: Path) -> Path:
    template_dir.mkdir(parents=True, exist_ok=True)
    template_config = {
        "data_source": "data/docs.txt",
        "topic_keys_file": "data/topic-keys.txt",
        "doc_topic_file": "data/doc-topic.txt",
        "metadata_file": "data/metadata.csv",
    }
    (template_dir / "config.json").write_text(
        json.dumps(template_config), encoding="utf-8"
    )
    (template_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    return template_dir


def _create_mallet_dir(mallet_dir: Path, files: List[str]) -> Path:
    mallet_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        (mallet_dir / fname).write_text("content", encoding="utf-8")
    return mallet_dir


@pytest.fixture
def dist_template_dir(tmp_path: Path) -> Path:
    """Return a path to a minimal template (dist) folder with config.json and index.html."""
    return _create_template(tmp_path / "dist")


@pytest.fixture
def mallet_dir_factory(tmp_path: Path) -> Callable[[List[str]], Path]:
    """Return a factory that creates a mallet dir with the specified files.

    Usage:
        mallet_dir = mallet_dir_factory()
        # or custom
        mallet_dir = mallet_dir_factory(["doc-topics.txt", "metadata.csv", ...])
    """
    default_files = [
        "metadata.csv",
        "topic-keys.txt",
        "doc-topic.txt",
        "topic-state.gz",
        "topic_coords.csv",
    ]

    def _factory(files: List[str] | None = None) -> Path:
        if files is None:
            files = default_files
        mallet_dir = tmp_path / "mallet"
        return _create_mallet_dir(mallet_dir, files)

    return _factory


@pytest.fixture
def sample_tsv(tmp_path: Path) -> Path:
    """Return a path to a small TSV file used as docs/data input."""
    data_file = tmp_path / "data_src.tsv"
    data_file.write_text("1\tTitle1\n2\tTitle2\n", encoding="utf-8")
    return data_file
