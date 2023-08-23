"""test_datasets.py."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from lexos.io.dataset import Dataset
from smart_open import open

LABELS = ["Text1", "Text2", "Text3", "Text4", "Text5"]


def to_csv(
    base: str, headers: Optional[str] = None, sep: str = ",", save: Optional[str] = None
) -> str:
    """Return the base in csv/tsv format."""
    base = "".join([f"{LABELS[i]},{line}" for i, line in enumerate(base.split("\n"))])
    if headers == "valid":
        csv = f"title{sep}text\n{base}"
    elif headers == "invalid":
        csv = f"label{sep}content\n{base}"
    else:
        csv = base
    if save:
        with open(save, "w") as f:
            f.write(csv)
    return csv


def to_json(
    base: str,
    headers: Optional[str] = None,
    sep: str = ",",
    lines: bool = False,
    save: Optional[str] = None,
) -> str:
    """Return the base in json or jsonl format."""
    base = "".join([f"{LABELS[i]},{line}" for i, line in enumerate(base.split("\n"))])
    if headers == "valid":
        base = f"title{sep}text\n{base}"
    elif headers == "invalid":
        base = f"label{sep}content\n{base}"
    else:
        raise Exception("Cannot make json without headers.")
    df = pd.read_csv(lines, sep=sep)
    if save:
        df.to_json(save, orient="records", lines=lines)
    return df.to_json(orient="records", lines=lines)


def make_test_dir(base: str, test_dir: str):
    """Make test dir."""
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    # Line-delimited no headers (base)
    with open(f"{test_dir}/test_file.txt", "w") as f:
        f.write(base)
    # CSV, no headers
    csv_no_headers = to_csv(base, save=f"{test_dir}/test_file.csv")
    # CSV, valid headers
    csv_valid = to_csv(base, headers="valid", save=f"{test_dir}/test_file_valid.csv")
    # CSV, invalid headers
    csv_invalid = to_csv(
        base, headers="invalid", save=f"{test_dir}/test_file_invalid.csv"
    )
    # TSV, no headers
    tsv_no_headers = to_csv(base, sep="\t", save=f"{test_dir}/test_file.tsv")
    # TSV, valid headers
    tsv_valid = to_csv(
        base, headers="valid", sep="\t", save=f"{test_dir}/test_file_valid.tsv"
    )
    # TSV, invalid headers
    tsv_invalid = to_csv(
        base, headers="invalid", sep="\t", save=f"{test_dir}/test_file_invalid.tsv"
    )
    # JSON, valid headers
    json_valid = to_csv(base, headers="valid", save=f"{test_dir}/test_file_valid.json")
    # JSON, invalid headers
    json_invalid = to_csv(
        base, headers="invalid", save=f"{test_dir}/test_file_invalid.json"
    )
    # JSONL, valid headers
    jsonl_valid = to_csv(
        base, headers="valid", lines=True, save=f"{test_dir}/test_file_valid.jsonl"
    )
    # JSON, invalid headers
    jsonl_invalid = to_csv(
        base, headers="invalid", lines=True, save=f"{test_dir}/test_file_invalid.jsonl"
    )


def make_test_set(base: str):
    """Make test set."""
    # CSV, no headers
    csv_no_headers = to_csv(base)
    # CSV, valid headers
    csv_valid = to_csv(base, headers="valid")
    # CSV, invalid headers
    csv_invalid = to_csv(base, headers="invalid")
    # TSV, no headers
    tsv_no_headers = to_csv(base, sep="\t")
    # TSV, valid headers
    tsv_valid = to_csv(base, headers="valid", sep="\t")
    # TSV, invalid headers
    tsv_invalid = to_csv(base, headers="invalid", sep="\t")
    # JSON, valid headers
    json_valid = to_csv(base, headers="valid")
    # JSON, invalid headers
    json_invalid = to_csv(base, headers="invalid")
    # JSONL, valid headers
    jsonl_valid = to_csv(base, headers="valid", lines=True)
    # JSON, invalid headers
    jsonl_invalid = to_csv(base, headers="invalid", lines=True)
    return [
        {"type": "string", "texts": base},
        {"type": "csv", "texts": csv_no_headers, "headers": None, "sep": ","},
        {"type": "csv", "texts": csv_valid, "headers": "valid", "sep": ","},
        {"type": "csv", "texts": csv_invalid, "headers": "invalid", "sep": ","},
        {"type": "csv", "texts": tsv_no_headers, "headers": None, "sep": "\t"},
        {"type": "csv", "texts": tsv_valid, "headers": "valid", "sep": "\t"},
        {"type": "csv", "texts": tsv_invalid, "headers": "invalid", "sep": "\t"},
        {"type": "json", "texts": json_valid, "headers": "valid", "lines": False},
        {"type": "json", "texts": json_invalid, "headers": "invalid", "lines": False},
        {"type": "json", "texts": jsonl_valid, "headers": "valid", "lines": True},
        {"type": "json", "texts": jsonl_invalid, "headers": "invalid", "lines": True},
    ]


def test_strings(test_set: List[dict]) -> None:
    """Test string representations."""
    for item in test_set:
        headers = item["headers"]
        sep = item["sep"]
        lines = item["lines"]
        if headers is None:
            names = LABELS
        else:
            names = None
        set_label = (item["type"], headers, sep, lines)
        try:
            if item["type"] == "string":
                dataset = Dataset.parse_string(test_set["texts"], names=names)
            elif item["type"] == "csv":
                dataset = Dataset.parse_csv(
                    test_set["texts"], names=names, headers=headers, sep=sep
                )
            elif item["type"] == "json":
                dataset = Dataset.parse_json(
                    test_set["texts"], names=names, lines=lines
                )
            print(f"Passed {set_label}")
        except Exception:
            raise Exception(f"Could not parse {set_label}.")


if __name__ == "__main__":
    base_file = "../test_data/datasets/base.txt"
    test_dir = "../test_data/dataset_test_dir"
    # Read the base
    with open(base_file) as f:
        base = f.read()

    # Make and test set and test it
    test_set = make_test_set(base)
    test_strings(test_set)

    # Make test dir
    # make_test_dir(base, test_dir)
    # files = Path(test_dir).glob('**/*')
    # files = [x for x in files if x.is_file()]
    # for file in files:
    #     test_file(file)
