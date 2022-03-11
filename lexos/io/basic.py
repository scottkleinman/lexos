"""basic.py.

This file contains the main logic for the Loader class. It is fairly
basic, but it will load a filepath, directory or URL (or list of those)
into a list of texts which can then be accessed by text processing tools.

The Loader class does not yet filter for file format.
"""

from pathlib import Path
from typing import Any, List, Union

import requests

from lexos import utils
from lexos.exceptions import LexosException

LANG = {
    "bad_source": "The source does not contain valid URL, file path, or directory path. Ensure that your source is a string, Path, or list of either.",
    "no_source": "No source provided. Please provide a source.",
}

class Loader():
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
        """Decode a text."""
        return utils._decode_bytes(text)

    def _download_text(self, url: str) -> str:
        """Download a text from a url."""
        try:
            r = requests.get(url)
            r.raise_for_status()
            return self._decode(r.text)
        except requests.exceptions.HTTPError as e:
            raise LexosException(e.response.text)

    def _validate_source(self, source: Any) -> bool:
        """Validate a source."""
        if not isinstance(source, str) and not isinstance(source, Path):
            self.errors.append(source)
            return False
        else:
            return True

    def load(self,
             source: Union[List[Union[Path, str]], Path, str],
             decode: bool = True) -> List[str]:
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
                if utils.is_url(item):
                    self.texts.append(self._download_text(item))
                    self.names.append(Path(item).stem)
                    self.locations.append(item)
                elif utils.ensure_path(item).is_file():
                    with open(utils.ensure_path(item), "rb") as f:
                        self.texts.append(self._decode(f.read()))
                    self.names.append(Path(item).stem)
                    self.locations.append(item)
                elif utils.ensure_path(item).is_dir():
                    for filepath in utils.ensure_path(item).rglob("*"):
                        with open(filepath, "rb") as f:
                            self.texts.append(self._decode(f.read()))
                        self.names.append(Path(filepath).stem)
                        self.locations.append(filepath)
                # Failsafe
                else:
                    raise LexosException(f'{LANG["bad_source"]}: {item}')
            else:
                pass

        if len(self.errors) > 0:
            print("The following items were not loaded:")
            for source in self.errors:
                print(f"Error: {source}")
