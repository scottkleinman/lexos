"""test_base_loader.py.

Coverage: 100%
Last Update: 2025-06-29
"""

from typing import Generator, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.io.base_loader import BaseLoader


def test_texts_field_definition():
    """Test texts field definition and default value.

    This test ensures 100% coverage.
    """
    from lexos.io.data_loader import DataLoader

    # Create fresh instance to trigger field defaults
    loader = DataLoader()

    # Verify texts field exists and has correct default
    assert loader.texts == []

    # Test field info
    field_info = DataLoader.model_fields["texts"]
    assert field_info.default == []
    assert field_info.description == "The list of loaded texts."


class ConcreteLoader(BaseLoader):
    """Concrete implementation of BaseLoader for testing."""

    def load_dataset(self, dataset):
        """Load a dataset."""
        pass


# Fixtures


@pytest.fixture
def base_loader():
    """Fixture to create a concrete BaseLoader instance."""
    return ConcreteLoader()


@pytest.fixture
def sample_paths():
    """Fixture to create sample paths."""
    return ["test1.txt", "test2.txt", "test3.txt"]


@pytest.fixture
def sample_mime_types():
    """Fixture to create sample mime_types."""
    return ["text/plain", "text/plain", "text/plain"]


@pytest.fixture
def sample_names():
    """Fixture to create sample names."""
    return ["test1", "test2", "test3"]


@pytest.fixture
def sample_texts():
    """Fixture to create sample texts."""
    return ["test1", "test2", "test3"]


@pytest.fixture
def sample_errors():
    """Fixture to create sample errors."""
    return []


@pytest.fixture
def base_loader_with_data(
    sample_paths, sample_mime_types, sample_names, sample_texts, sample_errors
):
    """Fixture to create a baseloader with data."""
    loader = ConcreteLoader()
    loader.paths = sample_paths
    loader.mime_types = sample_mime_types
    loader.names = sample_names
    loader.texts = sample_texts
    loader.errors = sample_errors
    return loader


@pytest.fixture
def test_duplicate_rows_loader():
    """Fixture to create a concrete a baseloader with duplicate rows."""
    loader = ConcreteLoader()
    loader.paths = ["file1.txt", "file1.txt", "file2.txt"]
    loader.mime_types = ["text/plain", "text/plain", "text/plain"]
    loader.names = ["doc1", "doc1", "doc2"]
    loader.texts = ["content1", "content1", "content2"]
    return loader


@pytest.fixture
def test_duplicate_rows_loader_no_dupes():
    """Fixture to create a concrete a baseloader with no duplicates."""
    loader = ConcreteLoader()
    loader.paths = ["file1.txt", "file2.txt"]
    loader.mime_types = ["text/plain", "text/plain"]
    loader.names = ["doc1", "doc2"]
    loader.texts = ["content1", "content2"]
    return loader


#########################################################################################################
# Tests


def test_base_loader_abstract_methods():
    """Test that BaseLoader cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseLoader()


def test_base_loader_init(base_loader):
    """Test BaseLoader initialization."""
    assert isinstance(base_loader, BaseLoader)
    assert base_loader.paths == []
    assert base_loader.mime_types == []
    assert base_loader.names == []
    assert base_loader.texts == []
    assert base_loader.errors == []


def test_base_loader_data(
    base_loader_with_data,
    sample_paths,
    sample_mime_types,
    sample_names,
    sample_texts,
    sample_errors,
):
    """Test BaseLoader data property."""
    assert base_loader_with_data.data == {
        "paths": sample_paths,
        "mime_types": sample_mime_types,
        "names": sample_names,
        "texts": sample_texts,
        "errors": sample_errors,
    }


def test_base_loader_records(
    base_loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test BaseLoader records property."""
    assert base_loader_with_data.records == [
        {
            "name": sample_names[0],
            "path": sample_paths[0],
            "mime_type": sample_mime_types[0],
            "text": sample_texts[0],
        },
        {
            "name": sample_names[1],
            "path": sample_paths[1],
            "mime_type": sample_mime_types[1],
            "text": sample_texts[1],
        },
        {
            "name": sample_names[2],
            "path": sample_paths[2],
            "mime_type": sample_mime_types[2],
            "text": sample_texts[2],
        },
    ]


def test_base_loader_records_mismatched_lengths():
    """Test BaseLoader records mismatched lengths exception throws correctly."""
    loader = ConcreteLoader()
    loader.paths = ["file1.txt"]
    loader.mime_types = ["text/plain", "text/plain"]  # length mismatch
    loader.names = ["doc1"]
    loader.texts = ["text1"]

    with pytest.raises(LexosException, match="Mismatched lengths in file records data"):
        _ = loader.records


def test_base_loader_iter(
    base_loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test BaseLoader records property."""
    assert isinstance(base_loader_with_data.__iter__(), Generator)
    for item in base_loader_with_data:
        assert item in [
            {
                "name": sample_names[0],
                "path": sample_paths[0],
                "mime_type": sample_mime_types[0],
                "text": sample_texts[0],
            },
            {
                "name": sample_names[1],
                "path": sample_paths[1],
                "mime_type": sample_mime_types[1],
                "text": sample_texts[1],
            },
            {
                "name": sample_names[2],
                "path": sample_paths[2],
                "mime_type": sample_mime_types[2],
                "text": sample_texts[2],
            },
        ]


def test_base_loader_df(
    base_loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test BaseLoader df property."""
    assert isinstance(base_loader_with_data.df, pd.DataFrame)
    df = pd.DataFrame(
        [
            {
                "name": sample_names[0],
                "path": sample_paths[0],
                "mime_type": sample_mime_types[0],
                "text": sample_texts[0],
            },
            {
                "name": sample_names[1],
                "path": sample_paths[1],
                "mime_type": sample_mime_types[1],
                "text": sample_texts[1],
            },
            {
                "name": sample_names[2],
                "path": sample_paths[2],
                "mime_type": sample_mime_types[2],
                "text": sample_texts[2],
            },
        ]
    )
    assert base_loader_with_data.df.equals(df)


def test_base_loader_df_empty(base_loader):
    """Test BaseLoader df property."""
    assert base_loader.df.empty


def test_base_loader_reset(base_loader_with_data):
    """Test BaseLoader reset method."""
    base_loader_with_data.errors = ["error1", "error2"]
    assert base_loader_with_data.paths != []
    assert base_loader_with_data.mime_types != []
    assert base_loader_with_data.names != []
    assert base_loader_with_data.texts != []
    base_loader_with_data.reset()
    assert base_loader_with_data.paths == []
    assert base_loader_with_data.mime_types == []
    assert base_loader_with_data.names == []
    assert base_loader_with_data.texts == []
    assert base_loader_with_data.errors == []


def test_dedupe_basic(test_duplicate_rows_loader):
    """Test basic deduplication."""
    test_duplicate_rows_loader.dedupe()

    # Check attributes were updated
    assert len(test_duplicate_rows_loader.paths) == 2
    assert len(test_duplicate_rows_loader.mime_types) == 2
    assert len(test_duplicate_rows_loader.names) == 2
    assert len(test_duplicate_rows_loader.texts) == 2

    # Verify specific values
    assert test_duplicate_rows_loader.paths == ["file1.txt", "file2.txt"]
    assert test_duplicate_rows_loader.texts == ["content1", "content2"]


def test_dedupe_single_column(test_duplicate_rows_loader):
    """Test deduplication with single column."""
    test_duplicate_rows_loader.dedupe(["path"])

    # Check attributes were updated
    assert len(test_duplicate_rows_loader.paths) == 2
    assert len(test_duplicate_rows_loader.mime_types) == 2
    assert len(test_duplicate_rows_loader.names) == 2
    assert len(test_duplicate_rows_loader.texts) == 2


def test_dedupe_no_duplicates(test_duplicate_rows_loader_no_dupes):
    """Test deduplication with no duplicates."""
    test_duplicate_rows_loader_no_dupes.dedupe()
    assert len(test_duplicate_rows_loader_no_dupes.paths) == 2
    assert len(test_duplicate_rows_loader_no_dupes.mime_types) == 2
    assert len(test_duplicate_rows_loader_no_dupes.names) == 2
    assert len(test_duplicate_rows_loader_no_dupes.texts) == 2


def test_dedupe_empty_df(base_loader):
    """Test deduplication with empty DataFrame."""
    base_loader.dedupe(["path"])
    assert base_loader.paths == []
    assert base_loader.mime_types == []
    assert base_loader.names == []
    assert base_loader.texts == []


def test_show_duplicates(test_duplicate_rows_loader):
    """Test show_duplicates method."""
    df = test_duplicate_rows_loader.show_duplicates()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_show_duplicates_empty_loader(base_loader):
    """Test show_duplicates method with empty loader."""
    assert base_loader.show_duplicates() is None


def test_to_csv_calls_df_to_csv(base_loader_with_data):
    """Test to_csv method."""
    with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        base_loader_with_data.to_csv("fake.csv")
        mock_to_csv.assert_called_once()


def test_to_json_calls_df_to_json(base_loader_with_data):
    """Test to_json method."""
    with patch.object(pd.DataFrame, "to_json") as mock_to_json:
        base_loader_with_data.to_json("fake.json")
        mock_to_json.assert_called_once()


def test_to_excel_calls_df_to_csv(base_loader_with_data):
    """Test to_excel method."""
    with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:  # Note: uses to_csv()
        base_loader_with_data.to_excel("fake.xlsx")
        mock_to_csv.assert_called_once()
