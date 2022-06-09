"""dataset.py.

This class just wraps pandas.read_csv and pandas.read_json, which
efficiently load files or buffers in these formats. It also accepts
lineated text files.

To Do:

    - Needs better exceptions.
    - Currently supports lineated text files, .csv, .tsv, .json, .jsonl, and .zip.
      Also accepts GitHub directory urls and local directories.
"""

import zipfile
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pydantic import BaseModel
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException


class Dataset(BaseModel):
    """Dataset class."""

    path: str
    data: List[str]

    @property
    def df(self) -> pd.DataFrame:
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
        return [x["locations"] for x in self.data]

    @property
    def names(self) -> List[str]:
        """Return the names of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return [x["names"] for x in self.data]

    @property
    def texts(self) -> List[str]:
        """Return the texts of the object data.

        Returns:
            List[str]: The texts of the object data.
        """
        return [x["texts"] for x in self.data]


class DatasetLoader:
    """Load a csv, json, jsonl, or lineated text file."""

    def __init__(
        self, paths: Optional[Union[list, Path, str]] = None, **kwargs
    ) -> Union[Dataset, List[Dataset]]:
        """Instantiate loader class.

        Args:
            path (Optional[Union[list, Path, str]]): Path or url to the file.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.

        Returns:
            Union[Dataset, List[Dataset]]: A dataset or list of dataset objects.
        """
        self.datasets = []
        if not isinstance(paths, list):
            paths = [paths]
        for path in paths:
            self.load(path, **kwargs)

    def _decode(self, text: Union[bytes, str]) -> str:
        """Decode a text.

        Args:
            text (Union[bytes, str]): The text to decode.

        Returns:
            str: The decoded text.
        """
        return utils._decode_bytes(text)

    def _handle_zip(self, path: str) -> None:
        """Extract a zip file and add each text inside.

        Args:
            path (str): The path to the zip file.
        """
        with open(path, "rb") as f:
            with zipfile.ZipFile(f) as zip:
                namelist = [n for n in zip.namelist() if Path(n).suffix != ""]
                for info in namelist:
                    if not str(info).startswith("__MACOSX") and not str(
                        info
                    ).startswith(".ds_store"):
                        self.load(zip.read(info))

    def load(self, path: Union[list, Path, str], **kwargs) -> None:
        """Load a dataset file.

        Args:
            path (Union[list, Path, str]): The path to the file to load.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.
        """
        # Ensure self.datasets is a list
        if not isinstance(self.datasets, list):
            self.datasets = [self.datasets]
        if not isinstance(path, list):
            path = [str(path)]
        for p in path:
            try:
                if p.endswith(".csv") or p.endswith(".tsv"):
                    dataset = self._load_csv(p, **kwargs)
                elif p.endswith(".json") or p.endswith(".jsonl"):
                    dataset = self._load_json(p, **kwargs)
                elif "github.com" in str(path):
                    filepaths = utils.get_github_raw_paths(path)
                    for filepath in filepaths:
                        self.load(filepath)
                elif utils.is_dir(p):
                    filepaths = [f for f in Path(p).iterdir() if f.is_file()]
                    for filepath in filepaths:
                        self.load(filepath, **kwargs)
                elif p.endswith(".zip"):
                    self._handle_zip(p, **kwargs)
                else:
                    dataset = self._load_lineated_text(p, **kwargs)
                self.datasets.append(dataset)
                # Ensure that single datasets are returned as an object, not a list
                if len(self.datasets) == 1:
                    self.datasets = self.datasets[0]
            except LexosException:
                raise LexosException(f"Error loading file: {p}.")

    def _load_csv(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        title_column: Optional[str] = None,
        text_column: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        """Load a csv file.

        Args:
            path (str): The path to the file to load.
            columns (Optional[List[str]]): Names of all the columns to load in the csv file.
            title_column (Optional[str]): The name of the column containing the title of the text.
            text_column (Optional[str]): The name the column containing the text.
            **kwargs: Additional arguments to pass to pandas.read_csv.

        Returns:
            Dataset: A dataset object.
        """
        decode = True
        if "decode" in kwargs and kwargs["decode"] is False:
            decode = False
        if text_column and not title_column:
            raise ValueError(
                "You must supply both a `title_column` and a `text_column`."
            )
        if title_column and not text_column:
            raise ValueError(
                "You must supply both a `title_column` and a `text_column`."
            )
        try:
            # No headers: include a list of all columns, including "title" and "text"
            if columns:
                if "title" not in columns:
                    raise ValueError("One column must be named `title`.")
                if "text" not in columns:
                    raise ValueError("One column must be named `text`.")
                df = pd.read_csv(path, names=columns, **kwargs)
                title_column = "title"
                text_column = "text"
            # Headers contain "title" and "text"
            elif not title_column and not text_column:
                df = pd.read_csv(path, **kwargs)
                title_column = "title"
                text_column = "text"
            # User must specify which header is the title column and which is the text column
            elif text_column and title_column:
                df = pd.read_csv(path, **kwargs)
                df = df.rename(columns={title_column: "title", text_column: "text"})
            else:
                raise BaseException(f"Invalid keyword arguments.")
            # Decode the text
            if decode:
                df["text"] = df["text"].apply(lambda x: self._decode(x))
            # Assign dummy titles if necessary
            if not title_column:
                df["title"] = [Path(path).stem] * df.shape[1]
            # Add a location column
            df["locations"] = [path] * df.shape[1]
            # Rename columns and convert to a dictionary
            df = df.rename(columns={"title": "names", "text": "texts"})
            # Create dataset object
            return Dataset(path=path, data=df.to_dict(orient="records"))
        except BaseException as e:
            raise BaseException(f"Could not parse {path}: {e}")

    def _load_json(
        self,
        path: str,
        title_key: str,
        text_key: str,
        **kwargs,
    ) -> Dataset:
        """Load a json file.

        Args:
            path (str): The path to the file to load.
            title_key (str): The name of the field containing the title of the text.
            text_key (str): The name the field containing the text.
            **kwargs: Additional arguments to pass to pandas.read_json.

        Returns:
            Dataset: A dataset object.
        """
        decode = True
        if "decode" in kwargs and kwargs["decode"] is False:
            decode = False
        try:
            # JSON object must contain "title" and "text"
            if not title_key and not text_key:
                df = pd.read_json(path, **kwargs)
                columns = self.df.columns.tolist()
                if "title" not in columns:
                    raise ValueError(
                        "One field must be named `title` or you must convert an existing column with the `title_key` parameter."
                    )
                if "text" not in columns:
                    raise ValueError(
                        "One field must be named `text` or you must convert an existing column with the `title_key` parameter."
                    )
            # User must specify which field is the title field and which is the text field
            elif text_key and title_key:
                df = pd.read_json(path, **kwargs)
                df = df.rename(columns={title_key: "title", text_key: "text"})
            elif text_key and not title_key:
                raise ValueError("You must supply both a `title_key`.")
            elif title_key and not text_key:
                raise ValueError("You must supply both a `text_key`.")
            else:
                raise BaseException(f"Invalid keyword arguments.")
            # Decode the text
            if decode:
                df["text"] = df["text"].apply(lambda x: self._decode(x))
            # Assign dummy titles if necessary
            if not title_key:
                df["title"] = [Path(path).stem] * df.shape[1]
            # Add a location column
            df["locations"] = [path] * df.shape[1]
            # Rename columns and convert to a dictionary
            df = df.rename(columns={"title": "names", "text": "texts"})
            # Create dataset object
            return Dataset(path=path, data=df.to_dict(orient="records"))
        except BaseException as e:
            raise BaseException(f"Could not parse {path}: {e}")

    def _load_lineated_text(self, path: str, **kwargs) -> Dataset:
        """Load a plain text file with texts separated by line breaks.

        Args:
            path (str): The path to the file to load.
            **kwargs: Additional arguments which are ignored except `decode=False`.

        Returns:
            Dataset: A dataset object.
        """
        with open(path, encoding="utf-8") as f:
            if not "decode" in kwargs and not kwargs["decode"] is False:
                line = self._decode(line)
            data = [
                {
                    "names": Path(path).stem,
                    "locations": path,
                    "texts": line,
                }
                for line in f
            ]
        return Dataset(path=path, data=data)
