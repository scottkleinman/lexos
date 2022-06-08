"""dataset.py.

This class just wraps pandas.read_csv and pandas.read_json, which
efficiently load files or buffers in these formats. It also accepts
lineated text files.

To Do:

    - Needs better exceptions.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException


class DatasetLoader:
    """Load a csv, json, jsonl, or lineated text file."""

    def __init__(self, path: Optional[Union[list, Path, str]] = None, **kwargs):
        """Instantiate loader class.

        Args:
            path (Optional[Union[list, Path, str]]): Path or url to the file.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.


        Can take a str, path
        """
        self.path = []
        self.texts = []
        self.names = []
        self.locations = []
        self.df = None
        if path:
            if isinstance(path, list):
                self.path = path
            else:
                self.path = [str(path)]
            self.load(str(path), **kwargs)

    def _decode(self, text: Union[bytes, str]) -> str:
        """Decode a text.

        Args:
            text (Union[bytes, str]): The text to decode.

        Returns:
            str: The decoded text.
        """
        return utils._decode_bytes(text)

    def load(self, path: Union[list, Path, str], **kwargs) -> None:
        """Load a dataset file.

        Args:
            path (Union[list, Path, str]): The path to the file to load.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.
        """
        if not isinstance(path, list):
            path = [str(path)]
        for p in path:
            if p.endswith(".csv") or p.endswith(".tsv"):
                self.load_csv(p, **kwargs)
            elif p.endswith(".json") or p.endswith(".jsonl"):
                self.load_json(p, **kwargs)
            else:
                self.load_lineated_text(p)

    def load_csv(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        title_column: Optional[str] = None,
        text_column: Optional[str] = None,
        **kwargs,
    ):
        """Load a csv file.

        Args:
            path (str): The path to the file to load.
            columns (Optional[List[str]]): Names of all the columns to load in the csv file.
            title_column (Optional[str]): The name of the column containing the title of the text.
            text_column (Optional[str]): The name the column containing the text.
            **kwargs: Additional arguments to pass to pandas.read_csv.
        """
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
                self.df = pd.read_csv(path, names=columns, **kwargs)
                title_column = "title"
                text_column = "text"
            # Headers contain "title" and "text"
            elif not title_column and not text_column:
                self.df = pd.read_csv(path, **kwargs)
                title_column = "title"
                text_column = "text"
            # User must specify which header is the title column and which is the text column
            elif text_column and title_column:
                df = pd.read_csv(path, **kwargs)
                self.df = df.rename(
                    columns={title_column: "title", text_column: "text"}
                )
            else:
                raise BaseException(f"Invalid keyword arguments.")
            texts = self.df["text"].values.tolist()
            len_texts = len(texts)
            [self.texts.append(self._decode(text)) for text in texts]
            if title_column:
                self.names += self.df["title"].values.tolist()
            else:
                self.names += [Path(path).stem] * len_texts
            self.locations += [path] * len_texts
        except BaseException as e:
            raise BaseException(f"Could not parse {path}: {e}")

    def load_json(
        self,
        path: str,
        title_key: str,
        text_key: str,
        **kwargs,
    ):
        """Load a json file.

        Args:
            path (str): The path to the file to load.
            title_key (str): The name of the field containing the title of the text.
            text_key (str): The name the field containing the text.
            **kwargs: Additional arguments to pass to pandas.read_json.
        """
        try:
            # JSON object must contain "title" and "text"
            if not title_key and not text_key:
                self.df = pd.read_json(path, **kwargs)
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
                self.df = df.rename(columns={title_key: "title", text_key: "text"})
            elif text_key and not title_key:
                raise ValueError("You must supply both a `title_key`.")
            elif title_key and not text_key:
                raise ValueError("You must supply both a `text_key`.")
            else:
                raise BaseException(f"Invalid keyword arguments.")
            texts = self.df["text"].values.tolist()
            len_texts = len(texts)
            [self.texts.append(self._decode(text)) for text in texts]
            if title_key:
                self.names += self.df["title"].values.tolist()
            else:
                self.names += [Path(path).stem] * len_texts
            self.locations += [path] * len_texts
        except BaseException as e:
            raise BaseException(f"Could not parse {path}: {e}")

    def load_lineated_text(self, path: str) -> None:
        """Load a plain text file with texts separated by line breaks.

        Args:
            path (str): The path to the file to load.
        """
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.texts.append(self._decode(line))
                self.names.append(Path(path).stem)
                self.locations.append(path)
