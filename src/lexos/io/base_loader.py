"""base_loader.py.

Last Update: 2025-12-04
Tested: 2025-06-29
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call

from lexos.exceptions import LexosException


class BaseLoader(BaseModel, ABC):
    """BaseLoader."""

    paths: list = Field(default=[], description="The list of paths.")
    mime_types: list = Field(default=[], description="The list of text mime types.")
    names: list = Field(default=[], description="The list of text names.")
    texts: list = Field(default=[], description="The list of loaded texts.")
    errors: list = Field(default=[], description="The list of loading errors.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self) -> Generator[dict, None, None]:
        """Iterate through the records."""
        return (record for record in self.records)

    @property
    def data(self) -> dict[str, list]:
        """Get the data as a dictionary.

        Returns:
            dict[str, list]: A dictionary containing the paths, mime_types, names, texts, and errors.
        """
        return {
            "paths": self.paths,
            "mime_types": self.mime_types,
            "names": self.names,
            "texts": self.texts,
            "errors": self.errors,
        }

    @property
    def df(self) -> pd.DataFrame:
        """Get a pandas DataFrame of file records.

        Returns:
            pandas.DataFrame: A DataFrame containing file metadata and content.
        """
        return pd.DataFrame(self.records)

    @property
    def records(self) -> list[dict[str, str]]:
        """Get a list of file records.

        Returns:
            list[dict]: List of dictionaries containing file metadata and content.
            Each dict has keys: path, mime_type, name, text

        Raises:
            ValueError: If the lengths of paths, mime_types, names and texts don't match.

        Note:
            Validates that all lists have the same length before returning the records.
        """
        if not (
            len(self.paths)
            == len(self.mime_types)
            == len(self.names)
            == len(self.texts)
        ):
            raise LexosException("Mismatched lengths in file records data")

        return [
            {"name": name, "path": path, "mime_type": mime_type, "text": text}
            for name, path, mime_type, text in zip(
                self.names, self.paths, self.mime_types, self.texts
            )
        ]

    # Abstract method, skipped for coverage
    @validate_call(config=model_config)  # pragma: no cover
    @abstractmethod  # pragma: no cover
    def load_dataset(self, dataset) -> None:  # pragma: no cover
        """Load a dataset.

        Args:
            dataset (DataLoader): The dataset to load.
        """
        ...

    @validate_call(config=model_config)
    def dedupe(self, subset: Optional[list[str]] = None) -> pd.DataFrame:
        """Deduplicate a DataFrame.

        Args:
            subset (Optional[list[str]]): The columns to consider for deduplication.

        Returns:
            pd.DataFrame: The deduplicated DataFrame.
        """
        if not self.df.empty:
            df = self.df.copy()
            df.drop_duplicates(
                subset=subset, keep="first", inplace=True, ignore_index=True
            )
            self.paths = df["path"].tolist()
            self.mime_types = df["mime_type"].tolist()
            self.names = df["name"].tolist()
            self.texts = df["text"].tolist()

    @validate_call(config=model_config)
    def show_duplicates(
        self, subset: Optional[list[str]] = None
    ) -> pd.DataFrame | None:
        """Show duplicates in a DataFrame.

        Args:
            subset (Optional[list[str]] = None): The columns to consider for checking duplicates.

        Returns:
            pd.DataFrame: The DataFrame with duplicates.
        """
        if not self.df.empty:
            df = self.df.copy()
            return df[df.duplicated(subset=subset)]
        return None

    @validate_call(config=model_config)
    def to_csv(self, path: Path | str, **kwargs) -> None:
        """Save the data to a csv file.

        Args:
            path (Path | str): The path to save the csv file.
        """
        self.df.to_csv(path, **kwargs)

    @validate_call(config=model_config)
    def to_excel(self, path: Path | str, **kwargs) -> None:
        """Save the data to an Excel file.

        Args:
            path (Path | str): The path to save the csv file.
        """
        self.df.to_csv(path, **kwargs)

    @validate_call(config=model_config)
    def to_json(self, path: Path | str, **kwargs) -> None:
        """Save the data to a json file.

        Args:
            path (Path | str): The path to save the csv file.
        """
        self.df.to_json(path, **kwargs)

    @validate_call(config=model_config)
    def reset(self) -> None:
        """Reset the class attributes to empty lists."""
        self.paths = []
        self.mime_types = []
        self.names = []
        self.texts = []
        self.errors = []
