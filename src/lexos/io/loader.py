"""loader.py.

Last Update: 2025-09-13
Tested: 2025-09-13
"""

import mimetypes
import zipfile
from pathlib import Path
from typing import Optional, Self

import puremagic
from docx import Document
from pydantic import ConfigDict, validate_call
from pypdf import PdfReader
from smart_open import open

from lexos.constants import (
    DOCX_TYPES,
    FILE_START,
    MIN_ENCODING_DETECT,
    PDF_TYPES,
    TEXT_TYPES,
    ZIP_TYPES,
)

VALID_FILE_TYPES = {*TEXT_TYPES, *PDF_TYPES, *DOCX_TYPES, *ZIP_TYPES}
from lexos.exceptions import LexosException
from lexos.io.base_loader import BaseLoader
from lexos.io.data_loader import DataLoader
from lexos.util import _decode_bytes as decode
from lexos.util import ensure_list


class Loader(BaseLoader):
    """Loader."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        """Initialize the Loader."""
        super().__init__()

    def _get_mime_type(self, path: Path | str, file_start: str) -> str:
        """Get the mime type of a file.

        Args:
            path (Path | str): The path to the file.

        Returns:
            str: The mime type of the file.
        """
        if Path(path).suffix == ".pickle":
            return "application/vnd.python.pickle"
        results = puremagic.magic_string(file_start, path)
        if not results:
            return None
        else:
            mime_type = results[0].mime_type
            if mime_type == "":
                mime_type, _ = mimetypes.guess_type(path)
            return mime_type

    def _load_docx_file(self, path: Path | str) -> None:
        """Load a docx file.

        Args:
            path (Path | str): The path to the file.

        Note:
            Consider https://github.com/ShayHill/docx2python for greater coverage.
        """
        try:
            doc = Document(path)
            text = "\n".join([decode(p.text) for p in doc.paragraphs])
            self.names.append(Path(path).name)
            self.mime_types.append("application/docx")
            self.texts.append(text)
        except BaseException as e:
            self.errors.append(e)

    def _load_pdf_file(self, path: Path | str) -> None:
        """Load a pdf file.

        Args:
            path (Path | str): The path to the file.
        """
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = decode(page.extract_text())
                self.names.append(Path(path).name)
                self.mime_types.append("application/pdf")
                self.texts.append(text)
        except BaseException as e:
            self.errors.append(e)

    def _load_text_file(self, path: Path | str, mime_type: str) -> None:
        """Load a text file.

        Args:
            path (Path | str): The path to the file.
            mime_type (str): The mime type of the file.
        """
        try:
            with open(path, "rb") as f:
                text = decode(f.read())
                self.paths.append(Path(path).name)
                self.names.append(Path(path).stem)
                self.mime_types.append(mime_type)
                self.texts.append(text)
        except BaseException as e:
            self.errors.append(e)

    def _load_zip_file(self, path: Path | str) -> None:
        """Handle a zip file.

        Args:
            path (Path | str): The path to the file.
        """
        with open(path, "rb") as fin:
            with zipfile.ZipFile(fin) as zip:
                for info in zip.infolist():
                    try:
                        # Get the mime type of the file
                        file_bytes = zip.read(info.filename)
                        file_start = decode(file_bytes[:MIN_ENCODING_DETECT])
                        mime_type = self._get_mime_type(info.filename, file_start)
                    except (IOError, UnicodeDecodeError) as e:
                        self.errors.append(e)
                        mime_type = None
                    try:
                        if mime_type in VALID_FILE_TYPES:
                            text = decode(file_bytes)
                            self.paths.append(
                                Path(path).as_posix() + "/" + info.filename
                            )
                            self.names.append(Path(info.filename).stem)
                            self.mime_types.append(mime_type)
                            self.texts.append(text)
                        else:
                            self.errors.append(
                                f"Invalid MIME type: {mime_type} for file {info.filename}."
                            )
                    except BaseException as e:
                        self.errors.append(e)

    # @validate_call(config=model_config)
    def load_dataset(self, dataset: Self) -> None:
        """Load a dataset.

        Args:
            dataset (DataLoader): The dataset to load.

        Note: As of v2.10.5, Pydantic does not support recursive types (Self).
            As a result, this method performs its own check to see if the
            value of `dataset` is of type `DataLoader`.
        """
        if not isinstance(dataset, DataLoader):
            raise LexosException("Invalid dataset type.")
        self.paths = self.paths + dataset.paths
        self.mime_types = self.mime_types + dataset.mime_types
        self.names = self.names + dataset.names
        self.texts = self.texts + dataset.texts

    @validate_call(config=model_config)
    def load(self, paths: Path | str | list[Path | str]) -> None:
        """Load a list of paths.

        Args:
            paths (Path | str | list[Path | str]): The list of paths to load.
        """
        paths = ensure_list(paths)
        for path in paths:
            if Path(path).is_dir():
                paths = [p for p in Path(path).rglob("*")]
                self.load(paths)
            # Get the mime type of the file
            try:
                with open(path, "rb") as f:
                    file_start = f.read(FILE_START)
                mime_type = self._get_mime_type(path, file_start)
            except IOError as e:
                self.errors.append(e)
                mime_type = None
            if mime_type in TEXT_TYPES:
                self._load_text_file(path, mime_type)
            elif mime_type in PDF_TYPES:
                self._load_pdf_file(path)
            elif mime_type in DOCX_TYPES:
                self._load_docx_file(path)
            elif mime_type in ZIP_TYPES:
                self._load_zip_file(path)
            else:
                self.errors.append(f"Invalid MIME type: {mime_type} for file {path}.")

    @validate_call(config=model_config)
    def loads(
        self,
        texts: Optional[list[Path | str]] = None,
        names: Optional[list[str]] = None,
        start: Optional[int] = 1,
        zero_pad: Optional[str] = "03",
    ) -> None:
        """Load a list of texts.

        Args:
            texts (Optional[list[Path | str]]): The list of texts to load.
            names (Optional[list[str]]): The list of names for the texts.
            start (Optional[int]): The starting index for the names if no list is provided.
            zero_pad (Optional[str]): The zero padding for the names increments if no list is provided.
        """
        texts = ensure_list(texts)
        if names is None:
            names = [f"text{i + start:{zero_pad}d}" for i in range(len(texts))]
        for i, text in enumerate(texts):
            self.names.append(names[i])
            self.mime_types.append("text/plain")
            self.texts.append(text)
