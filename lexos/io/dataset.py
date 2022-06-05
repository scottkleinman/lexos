"""dataset.py.

This class just wraps pandas.read_csv and pandas.read_json, which
efficiently load files or buffers in these formats. It also accepts
lineated text files.

To Do:

    - Needs better exceptions.
"""

from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException


class DatasetLoader:
    """Load a csv, json, jsonl, or lineated text file."""

    def __init__(self, path: Union[BinaryIO, str, TextIO] = None, **kwargs):
        """Instantiate loader class.

        Args:
            path (Union[BinaryIO, str, TextIO]): Path, url, or file-like object.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.
        """
        self.path = path
        self.texts = []
        self.names = []
        self.locations = []
        self.df = None
        if "text_column" not in kwargs:
            kwargs["text_column"] = 1
        if "text_key" not in kwargs:
            kwargs["text_key"] = "text"
        self.text_column = kwargs["text_column"]
        self.text_key = kwargs["text_key"]
        self.load(path, **kwargs)

    def _decode(self, text: Union[bytes, str]) -> str:
        """Decode a text.

        Args:
            text (Union[bytes, str]): The text to decode.

        Returns:
            str: The decoded text.
        """
        return utils._decode_bytes(text)

    def load(self, path: str, **kwargs) -> None:
        """Load a dataset file.

        Args:
            path (str): The path to the file to load.
            **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.
        """
        if not "text_column" not in kwargs:
            kwargs["text_column"] = self.text_column
        if "text_key" not in kwargs:
            kwargs["text_key"] = self.text_key
        if not isinstance(path, list):
            path = [path]
        for p in path:
            if p.endswith(".csv") or p.endswith(".tsv"):
                self.load_csv(p, **kwargs)
            if p.endswith(".json") or p.endswith(".jsonl"):
                self.load_json(p, **kwargs)

    def load_csv(
        self, path: str, text_column: int = 1, name_column: int = None, **kwargs
    ) -> None:
        """Load a csv file.

        Args:
            path (str): The path to the file to load.
            text_column (int): 1-indexed reference the column number containing the text.
            name_column (int): 1-indexed reference the column number containing the name of the text.
            **kwargs: Additional arguments to pass to pandas.read_csv.
        """
        try:
            self.df = pd.read_csv(path, **kwargs)
            texts = self.df[text_column - 1].values.tolist()
            len_texts = len(texts)
            [self.texts.append(self._decode(text)) for text in texts]
            if name_column:
                self.names += self.df[name_column - 1].values.tolist()
            else:
                self.names += [Path(path).stem] * len_texts
            self.locations += [path] * len_texts

        except BaseException as e:
            raise BaseException(f"Could not parse {path}: {e}")

    def load_json(
        self, path: str, text_key: str = "text", name_key: str = "title", **kwargs
    ) -> None:
        """Load a json file.

        Args:
            path (str): The path to the file to load.
            text_key (str): Name of the json property containing the text.
            name_key (str): Name of the json property containing the name of the text.
            **kwargs: Additional arguments to pass to pandas.read_json.
        """
        try:
            self.df = pd.read_json(path, **kwargs)
            len_texts = len(texts)
            texts = self.df[text_key].values.tolist()
            [self.texts.append(self._decode(text)) for text in texts]
            if name_key:
                self.names += self.df[name_key].values.tolist()
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
