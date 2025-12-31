"""test_data_loader.py.

Coverage: 100%
Last Update: 2025-06-29
"""

from io import StringIO
from typing import Generator
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.io.data_loader import DataLoader

# Fixtures


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
def sample_dataframe():
    """Fixture to create a dataframe with sample data."""
    return pd.DataFrame(
        [
            {"name": "test1", "text": "test1"},
            {"name": "test2", "text": "test2"},
            {"name": "test3", "text": "test3"},
        ]
    )


@pytest.fixture
def sample_dataframe_invalid_data():
    """Fixture to create a dataframe with invalid sample data."""
    return pd.DataFrame(
        [
            {"title": "test1", "content": "test1"},
            {"title": "test2", "content": "test2"},
            {"title": "test3", "content": "test3"},
        ]
    )


@pytest.fixture
def dataframe():
    """Fixture to create a dataframe from sample data."""
    return []


@pytest.fixture
def data_loader():
    """Fixture to create an empty DataLoader instance."""
    return DataLoader()


@pytest.fixture
def data_loader_with_data(
    sample_paths, sample_mime_types, sample_names, sample_texts, sample_errors
):
    """Fixture to create a DataLoader with data."""
    loader = DataLoader()
    loader.paths = sample_paths
    loader.mime_types = sample_mime_types
    loader.names = sample_names
    loader.texts = sample_texts
    loader.errors = sample_errors
    return loader


#########################################################################################################
# Tests


def test_data_loader_init(data_loader):
    """Test BaseLoader initialization."""
    assert isinstance(data_loader, DataLoader)
    assert data_loader.paths == []
    assert data_loader.mime_types == []
    assert data_loader.names == []
    assert data_loader.texts == []
    assert data_loader.errors == []


def test_data_loader_data(
    data_loader_with_data,
    sample_paths,
    sample_mime_types,
    sample_names,
    sample_texts,
    sample_errors,
):
    """Test DataLoader data property."""
    assert data_loader_with_data.data == {
        "paths": sample_paths,
        "mime_types": sample_mime_types,
        "names": sample_names,
        "texts": sample_texts,
        "errors": sample_errors,
    }


def test_data_loader_records(
    data_loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test DataLoader records property."""
    assert data_loader_with_data.records == [
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


def test_data_loader_df(
    data_loader_with_data, sample_paths, sample_mime_types, sample_names, sample_texts
):
    """Test DataLoader df property."""
    assert isinstance(data_loader_with_data.df, pd.DataFrame)
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
    assert data_loader_with_data.df.equals(df)


def test_data_loader_df_empty(data_loader):
    """Test DataLoader df property."""
    assert data_loader.df.empty


def test_data_loader_reset(data_loader_with_data):
    """Test DataLoader reset method."""
    data_loader_with_data.errors == [1]
    assert data_loader_with_data.paths != []
    assert data_loader_with_data.mime_types != []
    assert data_loader_with_data.names != []
    assert data_loader_with_data.texts != []
    data_loader_with_data.reset()
    assert data_loader_with_data.paths == []
    assert data_loader_with_data.mime_types == []
    assert data_loader_with_data.names == []
    assert data_loader_with_data.texts == []
    assert data_loader_with_data.errors == []


def test_data_loader_load_csv(data_loader, sample_dataframe):
    """Test DataLoader load_csv method."""
    csv = StringIO(sample_dataframe.to_csv(index=False))
    data_loader.load_csv(csv)
    assert data_loader.paths == ["csv_string", "csv_string", "csv_string"]
    assert data_loader.mime_types == ["text/csv", "text/csv", "text/csv"]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_csv_with_tabs(data_loader, sample_dataframe):
    """Test DataLoader load_csv method with tsv format."""
    csv = StringIO(sample_dataframe.to_csv(index=False, sep="\t"))
    data_loader.load_csv(csv, sep="\t")
    assert data_loader.paths == ["csv_string", "csv_string", "csv_string"]
    assert data_loader.mime_types == [
        "text/tab-separated-values",
        "text/tab-separated-values",
        "text/tab-separated-values",
    ]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_csv_with_invalid_headers_error(
    data_loader, sample_dataframe_invalid_data
):
    """Test DataLoader load_csv method with invalid headers."""
    csv = StringIO(sample_dataframe_invalid_data.to_csv(index=False))
    with pytest.raises(
        LexosException,
        match="CSV and TSV files must contain headers named `name` and `text`.",
    ):
        data_loader.load_csv(csv)


def test_data_loader_load_csv_with_invalid_headers(
    data_loader, sample_dataframe_invalid_data
):
    """Test DataLoader load_csv method with invalid headers and replacements."""
    csv = StringIO(sample_dataframe_invalid_data.to_csv(index=False))
    data_loader.load_csv(csv, name_col="title", text_col="content")
    assert data_loader.paths == ["csv_string", "csv_string", "csv_string"]
    assert data_loader.mime_types == ["text/csv", "text/csv", "text/csv"]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_load_csv_raises_lexos_exception_on_unexpected_error(data_loader):
    """Test DataLoader load_csv method raises LexosException on unexpected error."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = RuntimeError("Unexpected error!")

        with pytest.raises(LexosException) as exc_info:
            data_loader.load_csv("dummy_path.csv")

    assert "Unexpected error!" in str(exc_info.value)


def test_data_loader_load_dataset(data_loader_with_data):
    """Test DataLoader load_dataset method."""
    data_loader2 = DataLoader()
    data_loader2.paths = ["test4.txt", "test5.txt", "test6.txt"]
    data_loader2.mime_types = ["text/plain", "text/plain", "text/plain"]
    data_loader2.names = ["test4", "test5", "test6"]
    data_loader2.texts = ["test4", "test5", "test6"]
    data_loader2.errors = []
    data_loader_with_data.load_dataset(data_loader2)
    assert data_loader_with_data.paths == [
        "test1.txt",
        "test2.txt",
        "test3.txt",
        "test4.txt",
        "test5.txt",
        "test6.txt",
    ]
    assert data_loader_with_data.mime_types == [
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
        "text/plain",
    ]
    assert data_loader_with_data.names == [
        "test1",
        "test2",
        "test3",
        "test4",
        "test5",
        "test6",
    ]
    assert data_loader_with_data.texts == [
        "test1",
        "test2",
        "test3",
        "test4",
        "test5",
        "test6",
    ]
    assert data_loader_with_data.errors == []


def test_data_loader_load_invalid_dataset(data_loader_with_data):
    """Test DataLoader load_dataset method with invalid data type."""
    invalid_data = ["test4.txt", "test5.txt", "test6.txt"]
    with pytest.raises(LexosException, match="Invalid dataset type."):
        data_loader_with_data.load_dataset(invalid_data)


def test_data_loader_load_excel():
    """Test DataLoader load_excel method.

    This test is not implemented since it is the same procedure as load_csv.
    """
    pass


def test_data_loader_load_json(data_loader, sample_dataframe):
    """Test DataLoader load_json method."""
    json_string_io = StringIO(sample_dataframe.to_json(orient="records"))
    data_loader.load_json(json_string_io, orient="records")
    assert data_loader.paths == ["json_string", "json_string", "json_string"]
    assert data_loader.mime_types == [
        "application/json",
        "application/json",
        "application/json",
    ]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_json_with_invalid_headers_error(
    data_loader, sample_dataframe_invalid_data
):
    """Test DataLoader load_json method with invalid headers."""
    json_string_io = StringIO(sample_dataframe_invalid_data.to_json(orient="records"))
    with pytest.raises(LexosException, match="JSON files must contain"):
        data_loader.load_json(json_string_io)


def test_data_loader_load_json_with_invalid_headers(
    data_loader, sample_dataframe_invalid_data
):
    """Test DataLoader load_json method with invalid headers and replacements."""
    json_string_io = StringIO(sample_dataframe_invalid_data.to_json(orient="records"))
    data_loader.load_json(json_string_io, name_field="title", text_field="content")
    assert data_loader.paths == ["json_string", "json_string", "json_string"]
    assert data_loader.mime_types == [
        "application/json",
        "application/json",
        "application/json",
    ]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_json_line_delimited(data_loader, sample_dataframe):
    """Test DataLoader load_json method with line-delimited json."""
    json_string_io = StringIO(sample_dataframe.to_json(orient="records", lines=True))
    data_loader.load_json(json_string_io, orient="records", lines=True)
    assert data_loader.paths == ["json_string", "json_string", "json_string"]
    assert data_loader.mime_types == [
        "application/json",
        "application/json",
        "application/json",
    ]
    assert data_loader.names == ["test1", "test2", "test3"]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_load_json_raises_lexos_exception_on_unexpected_error(data_loader):
    """Test DataLoader load_json method raises LexosException on unexpected error."""
    with patch("pandas.read_json") as mock_read_json:
        mock_read_json.side_effect = RuntimeError("Unexpected error!")

        with pytest.raises(LexosException) as exc_info:
            data_loader.load_json("dummy_path.json")

    assert "Unexpected error!" in str(exc_info.value)


def test_data_loader_load_lineated_text(data_loader):
    """Test DataLoader load_lineated_text method."""
    source = "test1\ntest2\ntest3"
    data_loader.load_lineated_text(source)
    assert data_loader.paths == ["text_string", "text_string", "text_string"]
    assert data_loader.mime_types == ["text/plain", "text/plain", "text/plain"]
    assert data_loader.names == [f"text{i + 1:03d}" for i in range(0, 3)]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_lineated_text_start(data_loader):
    """Test DataLoader load_lineated_text method with start argument."""
    source = "test1\ntest2\ntest3"
    start = 2
    data_loader.load_lineated_text(source, start=start)
    assert data_loader.paths == ["text_string", "text_string", "text_string"]
    assert data_loader.mime_types == ["text/plain", "text/plain", "text/plain"]
    assert data_loader.names == [f"text{i + start:03d}" for i in range(0, 3)]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_data_loader_load_lineated_text_zero_pad(data_loader):
    """Test DataLoader load_lineated_text method with zero_pad argument."""
    source = "test1\ntest2\ntest3"
    pad = "02"
    data_loader.load_lineated_text(source, zero_pad=pad)
    assert data_loader.paths == ["text_string", "text_string", "text_string"]
    assert data_loader.mime_types == ["text/plain", "text/plain", "text/plain"]
    assert data_loader.names == [f"text{i + 1:{pad}d}" for i in range(0, 3)]
    assert data_loader.texts == ["test1", "test2", "test3"]
    assert data_loader.errors == []


def test_load_lineated_text_reads_lines_successfully(data_loader):
    """Test DataLoader load_lineated_text method reads lines successfully."""
    mock_file_content = b"line1\nline2\nline3\n"
    m_open = mock_open(read_data=mock_file_content)

    with patch("lexos.io.data_loader.open", m_open):
        data_loader.load_lineated_text("dummy_path.txt")

    assert data_loader.texts == ["line1\n", "line2\n", "line3\n"]
    assert data_loader.paths == ["text_string"] * 3
    assert data_loader.mime_types == ["text/plain"] * 3
    # default names
    assert data_loader.names == ["text001", "text002", "text003"]


def test_load_lineated_text_raises_lexos_exception_on_unexpected_error(data_loader):
    """Test DataLoader load_lineated_text method raises LexosException on unexpected error."""
    # Patch 'open' where it is used inside data_loader module
    with patch("lexos.io.data_loader.open") as mock_open:
        mock_open.side_effect = RuntimeError("Unexpected error!")

        with pytest.raises(LexosException) as exc_info:
            data_loader.load_lineated_text("dummy_path.txt")

    assert "Unexpected error!" in str(exc_info.value)


## Additional tests for DataLoader.__iter__ method
class TestDataLoaderIter:
    """Test cases for DataLoader.__iter__() method."""

    def test_iter_empty_data_loader(self):
        """Test __iter__ with an empty DataLoader."""
        loader = DataLoader()

        # Convert iterator to list to check contents
        datasets = list(loader)

        assert len(datasets) == 0
        assert datasets == []

    def test_iter_single_item(self):
        """Test __iter__ with a single item in DataLoader."""
        loader = DataLoader()

        # Set individual attributes instead of data property
        loader.names = ["test_doc"]
        loader.paths = ["/path/to/test.txt"]
        loader.mime_types = ["text/plain"]
        loader.texts = ["This is test content"]
        loader.errors = []

        # Test iteration
        datasets = list(loader)

        assert len(datasets) == 1
        from lexos.io.data_loader import Dataset

        assert isinstance(datasets[0], Dataset)
        assert datasets[0].name == "test_doc"
        assert datasets[0].path == "/path/to/test.txt"
        assert datasets[0].mime_type == "text/plain"
        assert datasets[0].text == "This is test content"

    def test_iter_multiple_items(self):
        """Test __iter__ with multiple items in DataLoader."""
        loader = DataLoader()

        # Set individual attributes instead of data property
        loader.names = ["doc1", "doc2", "doc3"]
        loader.paths = ["/path/to/doc1.txt", "/path/to/doc2.txt", "/path/to/doc3.txt"]
        loader.mime_types = ["text/plain", "text/csv", "application/json"]
        loader.texts = ["Content 1", "Content 2", "Content 3"]
        loader.errors = []

        # Test iteration
        datasets = list(loader)

        assert len(datasets) == 3

        from lexos.io.data_loader import Dataset

        # Check first dataset
        assert isinstance(datasets[0], Dataset)
        assert datasets[0].name == "doc1"
        assert datasets[0].path == "/path/to/doc1.txt"
        assert datasets[0].mime_type == "text/plain"
        assert datasets[0].text == "Content 1"

        # Check second dataset
        assert isinstance(datasets[1], Dataset)
        assert datasets[1].name == "doc2"
        assert datasets[1].path == "/path/to/doc2.txt"
        assert datasets[1].mime_type == "text/csv"
        assert datasets[1].text == "Content 2"

        # Check third dataset
        assert isinstance(datasets[2], Dataset)
        assert datasets[2].name == "doc3"
        assert datasets[2].path == "/path/to/doc3.txt"
        assert datasets[2].mime_type == "application/json"
        assert datasets[2].text == "Content 3"

    def test_iter_is_generator(self):
        """Test that __iter__ returns a generator."""
        loader = DataLoader()

        # Set individual attributes
        loader.names = ["test1", "test2"]
        loader.paths = ["/path1", "/path2"]
        loader.mime_types = ["text/plain", "text/plain"]
        loader.texts = ["Text 1", "Text 2"]
        loader.errors = []

        # Test that iter returns a generator
        iterator = iter(loader)

        from lexos.io.data_loader import Dataset

        # Get items one by one
        first_item = next(iterator)
        assert isinstance(first_item, Dataset)
        assert first_item.name == "test1"

        second_item = next(iterator)
        assert isinstance(second_item, Dataset)
        assert second_item.name == "test2"

        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iter_multiple_iterations(self):
        """Test that __iter__ can be called multiple times."""
        loader = DataLoader()

        # Set individual attributes
        loader.names = ["doc_a", "doc_b"]
        loader.paths = ["/a.txt", "/b.txt"]
        loader.mime_types = ["text/plain", "text/plain"]
        loader.texts = ["Content A", "Content B"]
        loader.errors = []

        # First iteration
        first_iteration = list(loader)
        assert len(first_iteration) == 2
        assert first_iteration[0].name == "doc_a"
        assert first_iteration[1].name == "doc_b"

        # Second iteration should produce the same results
        second_iteration = list(loader)
        assert len(second_iteration) == 2
        assert second_iteration[0].name == "doc_a"
        assert second_iteration[1].name == "doc_b"

    def test_iter_with_for_loop(self):
        """Test __iter__ using a for loop."""
        loader = DataLoader()

        # Set individual attributes
        loader.names = ["item1", "item2", "item3"]
        loader.paths = ["/path1", "/path2", "/path3"]
        loader.mime_types = ["text/plain", "text/csv", "text/json"]
        loader.texts = ["Text 1", "Text 2", "Text 3"]
        loader.errors = []

        # Test using for loop
        collected_names = []
        collected_texts = []

        from lexos.io.data_loader import Dataset

        for dataset in loader:
            assert isinstance(dataset, Dataset)
            collected_names.append(dataset.name)
            collected_texts.append(dataset.text)

        assert collected_names == ["item1", "item2", "item3"]
        assert collected_texts == ["Text 1", "Text 2", "Text 3"]

    def test_iter_with_real_data_loading(self):
        """Test __iter__ with data loaded using load_lineated_text method."""
        loader = DataLoader()

        # Load some lineated text
        source = "Line 1\nLine 2\nLine 3"
        loader.load_lineated_text(source)

        # Test iteration
        datasets = list(loader)

        assert len(datasets) == 3

        from lexos.io.data_loader import Dataset

        # Check that all datasets are properly formed
        for i, dataset in enumerate(datasets):
            assert isinstance(dataset, Dataset)
            assert dataset.name == f"text{i + 1:03d}"
            assert dataset.path == "text_string"
            assert dataset.mime_type == "text/plain"
            assert dataset.text == f"Line {i + 1}"

    def test_iter_dataset_immutability(self):
        """Test that modifying a Dataset from iterator doesn't affect the loader."""
        loader = DataLoader()

        # Set individual attributes
        loader.names = ["original"]
        loader.paths = ["/original"]
        loader.mime_types = ["text/plain"]
        loader.texts = ["original text"]
        loader.errors = []

        # Get dataset and modify it
        dataset = next(iter(loader))
        original_name = dataset.name

        # Modify the dataset (this should not affect the loader's data)
        dataset.name = "modified"

        # Check that loader's data is unchanged
        assert loader.data["names"][0] == "original"

        # Get a fresh dataset from the loader
        fresh_dataset = next(iter(loader))
        assert fresh_dataset.name == "original"
