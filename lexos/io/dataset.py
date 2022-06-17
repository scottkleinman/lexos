"""dataset.py.

This class currently supports data in the following formats:

    - dicts, line-delimited texts, csv, tsv, Excel, dict, json, and jsonl formats
    - Input from strings, filepaths, or urls (except for dict and Excel data)
    - Lists of the above, provided that all items in the list are of the same format
"""

import io
import json
from typing import IO, AnyStr, Dict, List, Optional, Type, TypeVar

import pandas as pd
from pydantic import BaseModel
from smart_open import open

Model = TypeVar("Model", bound="BaseModel")


class Dataset(BaseModel):
    """Dataset class."""

    data: Optional[List[Dict[str, str]]] = None
    dataframe: Optional[pd.DataFrame] = None

    class Config:
        """Config class."""

        arbitrary_types_allowed = True

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the dataframe of the object data.

        Returns:
            pd.DataFrame: The dataframe of the object data.
        """
        return pd.DataFrame(self.data)

    @property
    def locations(self) -> List[str]:
        """Return the locations of the object data.

        Returns:
            List[str]: The locations of the object data.
        """
        if "locations" in self.df.columns:
            return self.df["locations"].values.tolist()
        else:
            return None

    @property
    def names(self) -> List[str]:
        """Return the names of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return self.df["title"].values.tolist()

    @property
    def texts(self) -> List[str]:
        """Return the texts of the object data.

        Returns:
            List[str]: The texts of the object data.
        """
        return self.df["text"].values.tolist()

    def df(self) -> pd.DataFrame:
        """Return the dataframe of the object data.

        Returns:
            pd.DataFrame: The dataframe of the object data.
        """
        return self.dataframe

    @classmethod
    def parse_csv(
        cls: Type["Model"],
        source: str,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse CSV/TSV texts into the Dataset object.

        Args:
            source (str): The string or path to file containing the texts to parse.
            title_col (Optional[str]): The column name to convert to "title".
            text_col (Optional[str]): The column name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        source = cls._get_file_like(source)
        df = pd.read_csv(source, **kwargs)
        if title_col:
            df = df.rename(columns={title_col: "title"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "CSV and TSV files must contain headers named `title` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`title_col` and `text_col` parameters.",
            )
            raise Exception(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_dict(cls: Type["Model"], source: dict,) -> "Model":
        """Alias for cls.parse_obj().

        Args:
            source (dict): The dict to parse.

        Returns:
            Model: A dataset object.
        """
        return cls.parse_obj({"data": source})

    @classmethod
    def parse_excel(
        cls: Type["Model"],
        source: str,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse Excel files into the Dataset object.

        Args:
            source (str): The path to the Excel file containing the texts to parse.
            title_col (Optional[str]): The column name to convert to "title".
            text_col (Optional[str]): The column name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        source = cls._get_file_like(source)
        df = pd.read_excel(source, **kwargs)
        if title_col:
            df = df.rename(columns={title_col: "title"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "Excel files must contain headers named `title` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`title_col` and `text_col` parameters.",
            )
            raise Exception(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_json(cls: Type["Model"], source: str,) -> "Model":
        """Parse JSON files or strings.

        Args:
            source (str): The json string to parse.

        Returns:
            Model: A dataset object.
        """
        try:
            with open(source) as f:
                doc = json.loads(f.read())
            return cls.parse_obj({"data": doc})
        except Exception:
            return cls.parse_obj({"data": source})

    @classmethod
    def parse_jsonl(
        cls: Type["Model"],
        source: str,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse lineated texts into the Dataset object.

        Args:
            source (str): The string or path to file containing the lines to parse.
            lines (Optional[str]): The lines to parse.
            title_field (Optional[str]): The field name to convert to "title".
            text_field (Optional[str]): The field name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        source = cls._get_file_like(source)
        df = pd.read_json(source, lines=True, **kwargs)
        if title_field:
            df = df.rename(columns={title_field: "title"})
        if text_field:
            df = df.rename(columns={text_field: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "JSON and JSONL files must contain fields named `title` and `text`. ",
                "You can convert the names of existing fields to these with the ",
                "`title_field` and `text_field` parameters.",
            )
            raise Exception(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_string(
        cls: Type["Model"],
        source: str,
        labels: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
    ) -> "Model":
        """Parse lineated texts into the Dataset object.

        Args:
            source (str): The string containing the lines to parse.
            labels (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.

        Returns:
            Model: A dataset object.
        """
        if not labels:
            raise Exception(
                "Please use the `labels` argument to provide a list of labels for each row in your data."
            )
        # Handle files
        try:
            with open(source) as f:
                source = f.readlines()
        # Handle strings
        except Exception:
            source = source.split("\n")
        if len(labels) != len(source):
            raise Exception(
                f"The number of labels ({len(labels)}) does not match the number of lines ({len(source)}) in your data."
            )
        else:
            data = [{"title": labels[i], "text": line} for i, line in enumerate(source)]
            if locations:
                if len(locations) == len(source):
                    for i, _ in enumerate(data):
                        data[i]["locations"] = locations[i]
                else:
                    raise Exception(
                        f"The number of locations ({len(locations)}) does not match the number of lines ({len(source)}) in your data."
                    )
            return cls.parse_obj({"data": data})

    @staticmethod
    def _get_file_like(source: str) -> IO[AnyStr]:
        """Read the source into a buffer.

        Args:
            source: str: A path or string containing the source.

        Returns:
            IO[AnyStr]: A file-like object containing the source.
        """
        try:
            with open(source) as f:
                source = f.read()
        except:
            pass
        return io.StringIO(source)
