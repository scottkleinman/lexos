"""test_io_dataset.py.

Test file for the io.dataset module.

Need to test urls.
"""
import pytest

from lexos.io.dataset import Dataset, DatasetLoader

# Test data
test_dir = "../test_data"
raw_url = "https://raw.githubusercontent.com/scottkleinman/lexos/main/tests/test_data"

lineated_text = [
    (f"{test_dir}/datasets/zips", 8),
    (f"{test_dir}/datasets/lineated_text", 4),
    (f"{test_dir}/datasets/Austen.txt", 2),
    ([f"{test_dir}/datasets/Austen.txt", f"{test_dir}/datasets/Austen.txt"], 4),
    (
        "https://raw.githubusercontent.com/scottkleinman/lexos/main/tests/test_data/datasets/Austen.txt",
        2,
    ),
]

csv = [
    ([f"{test_dir}/datasets/Austen_no_headers.csv", ["title", "text"], None, None], 2),
    ([f"{test_dir}/datasets/Austen_valid_headers.csv", None, None, None], 2),
    ([f"{test_dir}/datasets/Austen_invalid_headers.csv", None, "label", "content"], 2),
    (
        [
            [
                f"{test_dir}/datasets/Austen_valid_headers.csv",
                f"{test_dir}/datasets/Austen_valid_headers.csv",
            ],
            None,
            None,
            None,
        ],
        4,
    ),
]

tsv = [
    (
        [
            f"{test_dir}/datasets/Austen_no_headers.tsv",
            ["title", "text"],
            None,
            None,
            "\t",
        ],
        2,
    ),
    ([f"{test_dir}/datasets/Austen_valid_headers.tsv", None, None, None, "\t"], 2),
    (
        [
            f"{test_dir}/datasets/Austen_invalid_headers.tsv",
            None,
            "label",
            "content",
            "\t",
        ],
        2,
    ),
]

json = [
    ([f"{test_dir}/datasets/Austen.json", None, None, None], 2),
    ([f"{test_dir}/datasets/Austen_invalid_fields.json", "label", "content", None], 2),
    ([f"{test_dir}/datasets/Austen_nl.jsonl", None, None, True], 2),
]

zip = [
    (f"{test_dir}/datasets/lineated_text.zip", 4),
]

# Test functions
@pytest.mark.parametrize("input, expected", lineated_text)
def test_load_lineated_text(input, expected):
    """Test loader with different types of datasets, as well as directories."""
    loader = DatasetLoader()
    loader.load(input)
    if isinstance(loader.datasets, list):
        data_len = 0
        for dataset in loader.datasets:
            data_len += len(dataset.data)
        assert data_len == expected
    else:
        assert len(loader.datasets.data) == expected


@pytest.mark.parametrize("input, expected", csv)
def test_load_csv(input, expected):
    """Test loader for csv."""
    path = input[0]
    columns = input[1]
    title_column = input[2]
    text_column = input[3]
    loader = DatasetLoader()
    loader.load(
        path,
        columns=columns,
        title_column=title_column,
        text_column=text_column,
    )
    if isinstance(loader.datasets, list):
        data_len = 0
        for dataset in loader.datasets:
            data_len += len(dataset.data)
        assert data_len == expected
    else:
        assert len(loader.datasets.data) == expected


@pytest.mark.parametrize("input, expected", tsv)
def test_load_tsv(input, expected):
    """Test loader for tsv."""
    path = input[0]
    columns = input[1]
    title_column = input[2]
    text_column = input[3]
    sep = input[4]
    loader = DatasetLoader()
    loader.load(
        path,
        columns=columns,
        title_column=title_column,
        text_column=text_column,
        sep=sep,
    )
    if isinstance(loader.datasets, list):
        data_len = 0
        for dataset in loader.datasets:
            data_len += len(dataset.data)
        assert data_len == expected
    else:
        assert len(loader.datasets.data) == expected


@pytest.mark.parametrize("input, expected", json)
def test_load_json(input, expected):
    """Test loader for json."""
    path = input[0]
    title_key = input[1]
    text_key = input[2]
    lines = input[3]
    loader = DatasetLoader()
    loader.load(
        path,
        title_key=title_key,
        text_key=text_key,
        lines=lines,
    )
    if isinstance(loader.datasets, list):
        data_len = 0
        for dataset in loader.datasets:
            data_len += len(dataset.data)
        assert data_len == expected
    else:
        assert len(loader.datasets.data) == expected


@pytest.mark.parametrize("input, expected", zip)
def test_load_zip(input, expected):
    """Test loader for zip files."""
    loader = DatasetLoader()
    loader.load(
        input,
    )
    if isinstance(loader.datasets, list):
        data_len = 0
        for dataset in loader.datasets:
            data_len += len(dataset.data)
        assert data_len == expected
    else:
        assert len(loader.datasets) == 2
