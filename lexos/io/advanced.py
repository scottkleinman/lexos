"""basic.py.

This file contains the main logic for the Loader class. It is fairly
basic, but it will load a filepath, directory or URL (or list of those)
into a list of texts which can then be accessed by text processing tools.

The Loader class does not yet filter for file format.
"""

import io
from pathlib import Path
from typing import Any, List, Union

import docx2txt
import requests
from pdfminer.high_level import extract_text

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

    def _convert_doc_format(file: str) -> str:
        """Extract text from a .doc file.

        Args:
            file (str): The file to convert
        """
        import tempfile

        import docx

        # Read the .doc file
        doc = docx.Document(file)
        # Save it to a temporary directory and extract the text from there
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_filepath = f"{tmp_dir}/temp_file.docx"
            doc.save(temp_filepath)
            text = docx2txt.process(temp_filepath)
        return text

    def _decode(self, text: Union[bytes, str]) -> str:
        """Decode a text.

        Args:
            text (Union[bytes, str]): The text to decode.

        Returns:
            str: The decoded text.
        """
        return utils._decode_bytes(text)

    def _download_doc(self, url: str) -> str:
        """Download a .doc file from a url.

        Args:
            url (str): The url to download.

        Returns:
            str: The text downloaded from the url.
        """
        import tempfile

        import docx

        try:
            r = requests.get(url)
            r.raise_for_status()
            doc = io.BytesIO(r.content)
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_docpath = f"{tmp_dir}/temp_file.doc"
                temp_filepath = f"{tmp_dir}/temp_file.docx"
                with open(temp_docpath, "wb") as f:
                    f.write(doc)
                doc = docx.Document(temp_docpath)
                doc.save(temp_filepath)
                text = docx2txt.process(temp_filepath)
            return self._decode(docx2txt.process(docx))
        except requests.exceptions.HTTPError as e:
            raise LexosException(e.response.text)

    def _download_docx(self, url: str) -> str:
        """Download a docx from a url.

        Args:
            url (str): The url to download.

        Returns:
            str: The text downloaded from the url.
        """
        try:
            r = requests.get(url)
            r.raise_for_status()
            docx = io.BytesIO(r.content)
            return self._decode(docx2txt.process(docx))
        except requests.exceptions.HTTPError as e:
            raise LexosException(e.response.text)

    def _download_pdf(self, url: str) -> str:
        """Download a text from a url.

        Args:
            url (str): The url to download.

        Returns:
            str: The text downloaded from the url.
        """
        try:
            r = requests.get(url, headers={"User-Agent": "XY"}, stream=True)
            r.raise_for_status()
            pdf = io.BytesIO(r.content)
            return self._decode(extract_text(pdf))
        except requests.exceptions.HTTPError as e:
            raise LexosException(e.response.text)

    def _download_text(self, url: str) -> str:
        """Download a text from a url.

        Args:
            url (str): The url to download.

        Returns:
            str: The text downloaded from the url.
        """
        try:
            r = requests.get(url)
            r.raise_for_status()
            return self._decode(r.text)
        except requests.exceptions.HTTPError as e:
            raise LexosException(e.response.text)

    def _handle_file(self, file: str) -> None:
        """Handle file.

        Args:
            file (str): The file to download.
        """
        if utils.is_doc(file):
            text = self._convert_doc_format(file)
            self.texts.append(self._decode(text))
        elif utils.is_docx(file):
            self.texts.append(self._decode(docx2txt.process(file)))
        elif utils.is_pdf(file):
            self.texts.append(self._decode(extract_text(file)))
        else:
            with open(utils.ensure_path(file), "rb") as f:
                self.texts.append(self._decode(f.read()))
        self.names.append(Path(file).stem)
        self.locations.append(file)

    def _handle_url(self, url: str) -> None:
        """Handle url.

        Args:
            url (str): The url to download.
        """
        if utils.is_doc(url):
            self.texts.append(self._download_doc(url))
        elif utils.is_docx(url):
            self.texts.append(self._download_docx(url))
        elif utils.is_pdf(url):
            self.texts.append(self._download_pdf(url))
        else:
            self.texts.append(self._download_text(url))
        self.names.append(Path(url).stem)
        self.locations.append(url)

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
    ) -> List[str]:
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

        for item in self.source:
            if self._validate_source(item):
                # Handle url
                if utils.is_url(item):
                    self._handle_url(item)
                # Handle file
                elif utils.ensure_path(item).is_file():
                    self._handle_file(item)
                # Handle directory
                elif utils.ensure_path(item).is_dir():
                    for filepath in utils.ensure_path(item).rglob("*"):
                        self._handle_file(filepath)
                # Failsafe
                else:
                    raise LexosException(f'{LANG["bad_source"]}: {item}')
            else:
                pass

        if len(self.errors) > 0:
            print("The following items were not loaded:")
            for source in self.errors:
                print(f"Error: {source}")
