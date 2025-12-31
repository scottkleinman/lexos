"""data_loader.py.

Last Update: 2025-06-29
Tested: 2025-06-29
"""

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Self

import pandas as pd
from pydantic import ConfigDict, validate_call
from smart_open import open

from lexos.exceptions import LexosException
from lexos.io.base_loader import BaseLoader
from lexos.util import _decode_bytes as decode


@dataclass
class Dataset:
    """Dataset class."""

    name: str
    path: str
    mime_type: str
    text: str


class DataLoader(BaseLoader):
    """DataLoader."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        """Initialize the DataLoader."""
        super().__init__()

    def __iter__(self) -> Generator[Dataset, None, None]:
        """Make the class iterable.

        Yields:
            Dataset: A Dataset object containing the name, path, mime_type, and text of each dataset item.

        Note: Overrides the BaseLoader's __iter__ method to yield Dataset objects.
        """
        for i in range(len(self.data["paths"])):
            yield Dataset(
                name=self.data["names"][i],
                path=self.data["paths"][i],
                mime_type=self.data["mime_types"][i],
                text=self.data["texts"][i],
            )

    def _update_data(
        self, path: Path | str, df: pd.DataFrame, mime_type: str = "text/plain"
    ) -> None:
        """Update the DataLoader.

        Args:
            path (Path | str): The path to the file.
            df (pd.DataFrame): The DataFrame to update with.
            mime_type (str): The mime type of the file.
        """
        self.names = self.names + df["name"].tolist()
        length = len(self.names)
        self.paths = self.paths + [str(path)] * length
        self.mime_types = self.mime_types + [mime_type] * length
        self.texts = self.texts + [decode(text) for text in df["text"].tolist()]

    @validate_call(config=model_config)
    def load_csv(
        self,
        path: io.StringIO | os.PathLike | Path | str,
        name_col: Optional[str] = "name",
        text_col: Optional[str] = "text",
        **kwargs,
    ) -> None:
        """Load a csv file.

        Args:
            path (io.StringIO | os.PathLike | Path | str): The path to the file.
            name_col (Optional[str]): The column name for the names.
            text_col (Optional[str]): The column name for the texts.
        """
        try:
            df = pd.read_csv(path, **kwargs)
        except BaseException as e:
            raise LexosException(e)
        if not isinstance(path, (Path, str)):
            path = "csv_string"
        if "sep" in kwargs and kwargs["sep"] == "\t":
            mime_type = "text/tab-separated-values"
        else:
            mime_type = "text/csv"
        if name_col:
            df = df.rename(columns={name_col: "name"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "name" not in df.columns or "text" not in df.columns:
            err = (
                "CSV and TSV files must contain headers named `name` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`name_col` and `text_col` parameters.",
            )
            raise LexosException("".join(err))
        self._update_data(path, df, mime_type)

    # @validate_call(config=model_config)
    def load_dataset(self, dataset: Self) -> None:
        """Load a dataset.

        Args:
            dataset (DataLoader): The dataset to load.

        Note: As of v2.10.5, Pydantic does not support recursive types (Self).
            As a result, this method performs its own check to see if the
            value of `dataset` is of type `DataSet`.
        """
        if not isinstance(dataset, DataLoader):
            raise LexosException("Invalid dataset type.")
        self.paths = self.paths + dataset.paths
        self.mime_types = self.mime_types + dataset.mime_types
        self.names = self.names + dataset.names
        self.texts = self.texts + dataset.texts

    # Skipped for coverage, same method as load_csv
    @validate_call(config=model_config)  # pragma: no cover
    def load_excel(  # pragma: no cover
        self, path: Path | str, name_col: str, text_col: str, **kwargs
    ) -> None:
        """Load an Excel file.

        Args:
            path (Path | str): The path to the file.
            name_col (str): The column name for the names.
            text_col (str): The column name for the texts.
        """
        try:
            df = pd.read_csv(path, **kwargs)
        except BaseException as e:
            raise LexosException(e)
        if not isinstance(path, (Path, str)):
            path = "buffer"
        if name_col:
            df = df.rename(columns={name_col: "name"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "name" not in df.columns or "text" not in df.columns:
            err = (
                "Excel files must contain headers named `name` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`name_col` and `text_col` parameters.",
            )
            raise LexosException("".join(err))
        self._update(
            path,
            df,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    @validate_call(config=model_config)
    def load_json(
        self,
        path: io.StringIO | os.PathLike | Path | str,
        name_field: Optional[str] = "name",
        text_field: Optional[str] = "text",
        **kwargs,
    ) -> None:
        """Load a JSON file.

        Args:
            path (io.StringIO | os.PathLike | Path | str): The path to the file.
            name_field (Optional[str] = ): The field name for the names.
            text_field (Optional[str] = ): The field name for the texts.
        """
        try:
            df = pd.read_json(path, **kwargs)
        except BaseException as e:
            raise LexosException(e)
        if not isinstance(path, (Path, str)):
            path = "json_string"
        if name_field:
            df = df.rename(columns={name_field: "name"})
        if text_field:
            df = df.rename(columns={text_field: "text"})
        if "name" not in df.columns or "text" not in df.columns:
            err = (
                "JSON files must contain fields named `name` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`name_field` and `text_field` parameters.",
            )
            raise LexosException("".join(err))
        self._update_data(path, df, "application/json")

    @validate_call(config=model_config)
    def load_lineated_text(
        self,
        path: io.StringIO | os.PathLike | Path | str,
        names: Optional[list[str]] = None,
        start: Optional[int] = 1,
        zero_pad: Optional[str] = "03",
    ) -> None:
        """Load a list of texts.

        Args:
            path (io.StringIO | os.PathLike | Path | str): The path to the file.
            names (Optional[list[str]]): The list of names for the texts.
            start (Optional[int]): The starting index for the names if no list is provided.
            zero_pad (Optional[str]): The zero padding for the names increments if no list is provided.
        """
        try:
            with open(path, "rb") as f:
                texts = f.readlines()
        except (FileNotFoundError, IOError, OSError):
            texts = path.split("\n")
        except BaseException as e:
            raise LexosException(e)
        if names is None:
            names = [f"text{i + start:{zero_pad}d}" for i in range(len(texts))]
        self.paths = ["text_string"] * len(texts)
        self.names = names
        self.mime_types = ["text/plain"] * len(texts)
        self.texts = [decode(text) for text in texts]
