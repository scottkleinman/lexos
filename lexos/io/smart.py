"""smart.py.

This file contains the main logic for the Loader class. It is fairly
basic, but it will load a filepath, directory or URL (or list of those)
into a list of texts which can then be accessed by text processing tools.
Unlike the basic version, it accepts .docx, .pdf, and .zip files. It also
uses smart_open to open files and URLs, as opposed to using requests for
URLs.

The Loader class does not yet filter for file format.
"""

import io
import zipfile
from pathlib import Path
from typing import Any, List, Union

import docx2txt
from pdfminer.high_level import extract_text
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException

LANG = {
    "bad_source": "The source does not contain valid URL, file path, or directory path. Ensure that your source is a string, Path, or list of either.",
    "no_source": "No source provided. Please provide a source.",
}


class Loader:
    """Loader class.

    Handles the queue for assets to be pipelined from their sources to
    text processing tools.
    """

    def __init__(self):
        """__init__ method."""
        self.source = None
        self.names = []
        self.locations = []
        self.texts = []
        self.errors = []
        self.decode = True

    def _decode(self, text: Union[bytes, str]) -> str:
        """Decode a text.

        Args:
            text (Union[bytes, str]): The text to decode.

        Returns:
            str: The decoded text.
        """
        return utils._decode_bytes(text)

    def _add_text(self, path: str, text: Union[bytes, str]) -> None:
        """Decode and add a text.

        Args:
            path (str): The path to the text file.
            text (str): The text string.
        """
        if self.decode:
            self.texts.append(self._decode(text))
        else:
            self.texts.append(text)
        self.names.append(Path(path).stem)
        self.locations.append(path)

    def _handle_source(self, path: Union[Path, str]) -> None:
        """Add a text based on source type.

        Args:
            path (str): The path to the text file.
        """
        ext = Path(path).suffix
        path = str(path)
        if ext == ".zip":
            self._handle_zip(path)
        else:
            if ext in [".docx", ".pdf", ".zip"]:
                with open(path, "rb") as f:
                    bytes = io.BytesIO(f.read())
                if ext == ".docx":
                    self._add_text(path, docx2txt.process(bytes))
                elif ext == ".pdf":
                    self._add_text(path, extract_text(bytes))
                elif ext == ".zip":
                    self._handle_zip(path)
            else:
                with open(path, "rb") as f:
                    self._add_text(path, f.read())

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
                        self._add_text(path, zip.read(info))

    def _validate_source(self, source: Any) -> bool:
        """Validate a source.

        Args:
            source (Any): A source.

        Returns:
            bool: Whether the source is valid.
        """
        if not isinstance(source, str) and not isinstance(source, Path):
            self.errors.append(source)
            return False
        else:
            return True

    def load(
        self, source: Union[List[Union[Path, str]], Path, str], decode: bool = True
    ) -> None:
        """Load the source into a list of bytes and strings.

        Args:
            source (Union[List[Path, str], Path, str]): A source or list of sources.
            decode (bool): Whether to decode the source.

        Raises:
            LexosException: An error message.
        """
        if not source:
            raise LexosException(LANG["no_source"])
        else:
            self.source = source
            self.decode = decode
        if isinstance(self.source, str) or isinstance(self.source, Path):
            self.source = [self.source]

        for path in self.source:
            if self._validate_source(path):
                if "github.com" in str(path):
                    filepaths = utils.get_github_raw_paths(path)
                    for filepath in filepaths:
                        self._handle_source(filepath)
                elif utils.is_file(path) or utils.is_url(path):
                    self._handle_source(path)
                elif utils.is_dir(path):
                    for filepath in utils.ensure_path(path).rglob("*"):
                        self._handle_source(filepath)
                else:
                    raise LexosException(f'{LANG["bad_source"]}: {path}')
            else:
                pass

        if len(self.errors) > 0:
            print("The following items were not loaded:")
            for source in self.errors:
                print(f"Error: {source}")
