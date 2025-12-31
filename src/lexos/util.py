"""utils.py.

This file contains helper functions used by multiple modules.

Last Updated: June 24, 2025
Lasty Tested: June 24, 2025
"""

from pathlib import Path
from typing import Any, Collection, TypeVar

import chardet
import spacy
from bs4 import (
    UnicodeDammit,  # type: ignore - this import is correct, but Pylance fails to recognize it
)
from pydantic_core import PydanticCustomError
from pydantic_extra_types.color import Color
from spacy.language import Language
from spacy.tokens import Doc

import lexos.constants as constants
from lexos.exceptions import LexosException

AnyVal = TypeVar("AnyVal")


def ensure_list(item: Any) -> list:
    """Ensure string is converted to a Path.

    Args:
        item (Any): Anything.

    Returns:
        The item inside a list if it is not already a list.
    """
    if not isinstance(item, list):
        item = [item]
    return item


def ensure_path(path: Any) -> Any:
    """Ensure string is converted to a Path.

    Args:
        path (Any): Anything. If string, it's converted to Path.

    Returns:
        Path or original argument.
    """
    if isinstance(path, str):
        return Path(path.replace("\\", "/"))
    else:
        return path


def get_paths(path: Path | str) -> list:
    """Get a list paths in a directory.

    Args:
        path (Path | str): The path to the directory.

    Returns:
        list: A list of file paths.
    """
    return list(Path(path).glob("**/*"))


def get_encoding(input_string: bytes) -> str:
    """Use chardet to return the encoding type of a string.

    Args:
        input_string (bytes): A bytestring.

    Returns:
        The string's encoding type.
    """
    encoding_detect = chardet.detect(input_string[: constants.MIN_ENCODING_DETECT])
    encoding_type = encoding_detect["encoding"]
    if encoding_type is None:
        encoding_type = "utf-8"
    return encoding_type


def is_valid_colour(color: str) -> bool:
    """Check if a string is a valid colour.

    Args:
        color: A string representing a colour.

    Returns:
        True if the string is a valid colour, False otherwise.

    Note: Implements Pydantic's Color type for validation.
    See https://docs.pydantic.dev/2.0/usage/types/extra_types/color_types/ for more information.
    """
    try:
        Color(color)
    except PydanticCustomError:
        return False
    return True


def load_spacy_model(model: Language | str) -> Language:
    """Load a spaCy language model.

    Args:
        model (Language | str): The spaCy model to load, either as a Language object or a string representing the model name.

    Returns:
        Language: The loaded spaCy language model.

    Raises:
        LexosException: If the model cannot be loaded or if the model type is incorrect.
    """
    if not isinstance(model, (Language, str)):
        raise LexosException("Model must be a string or a spaCy Language object.")

    if isinstance(model, Language):
        return model
    else:
        try:
            return spacy.load(model)
        except OSError:
            raise LexosException(
                f"Error loading model '{model}'. Please check the name and try again. You may need to install the model on your system."
            )


def normalize(raw_bytes: bytes | str) -> str:
    """Normalise a string to LexosFile format.

    Args:
        raw_bytes (bytes | str): The input bytestring.

    Returns:
        Normalised version of the input string.
    """
    s = _decode_bytes(raw_bytes)
    return s


def normalize_strings(strings: list[str]) -> list[str]:
    """Normalise a list of strings to LexosFile format.

    Args:
        strings (list[Path | str]): The list of input strings.

    Returns:
        A list of normalised versions of the input strings.
    """
    normalized_strings = []
    for s in strings:
        normalized_strings.append(normalize(s))
    return normalized_strings


def normalize_files(
    filepaths: list[Path | str], destination_dir: Path | str = "."
) -> None:
    """Normalise a list of files to LexosFile format and save the files.

    Args:
        filepaths (list[Path | str]): The list of paths to input files.
        destination_dir (Path | str): The path to the directory where the files.
            will be saved.
    """
    for filepath in filepaths:
        filepath = ensure_path(filepath)
        with open(filepath, "rb") as f:
            doc = f.read()
        with open(destination_dir / filepath.name, "w") as f:
            f.write(normalize(doc))


def normalize_file(filepath: Path | str, destination_dir: Path | str = ".") -> None:
    """Normalise a file to LexosFile format and save the file.

    Args:
        filepath (Path | str): The path to the input file.
        destination_dir (Path | str): The path to the directory where the files.
            will be saved.
    """
    # filepath = ensure_path(filepath)
    filepath = Path(filepath)
    destination_dir = ensure_path(destination_dir)
    with open(filepath, "rb") as f:
        doc = f.read()
    with open(destination_dir / Path(filepath.name), "w") as f:
        f.write(normalize(doc))


def _try_decode_bytes_(raw_bytes: bytes) -> str:
    """Try to decode raw bytes (helper function for decode_bytes().

    Args:
        raw_bytes (bytes): The bytes you want to decode to string.

    Returns:
        A decoded string.
    """
    # Detect the encoding with only the first couple of bytes
    encoding_detect = chardet.detect(raw_bytes[: constants.MIN_ENCODING_DETECT])
    # Get the encoding
    encoding_type = encoding_detect["encoding"]
    if encoding_type is None:
        encoding_detect = chardet.detect(raw_bytes)
        encoding_type = encoding_detect["encoding"]

    if encoding_type is None:
        encoding_type = "utf-8"

    try:
        # Try to decode the string using the original encoding
        decoded_string = raw_bytes.decode(encoding_type)

    except (UnicodeDecodeError, TypeError):
        # Try UnicodeDammit if chardet didn't work
        if encoding_type == "ascii":
            dammit = UnicodeDammit(
                raw_bytes, ["iso-8859-1", "iso-8859-15", "windows-1252"]
            )
        else:
            dammit = UnicodeDammit(raw_bytes)
        decoded_string = dammit.unicode_markup

    return decoded_string


def _decode_bytes(raw_bytes: bytes | str) -> str:
    """Decode raw bytes from a user's file into a string.

    Args:
        raw_bytes (bytes | str): The bytes to be decoded to a python string.

    Returns:
        The decoded string.
    """
    if isinstance(raw_bytes, bytes):
        try:
            decoded_str = _try_decode_bytes_(raw_bytes)

        except (UnicodeDecodeError, TypeError):
            raise LexosException(
                "Chardet failed to detect encoding of your "
                "file. Please make sure your file is in "
                "utf-8 encoding."
            )
    else:
        decoded_str = raw_bytes

    # Normalize line breaks - NB. Pylance fails in these lines
    # First handle "\r\n" -> "\n" (Windows line endings)
    if "\r\n" in decoded_str[: constants.MIN_NEWLINE_DETECT]:  # type: ignore
        decoded_str = decoded_str.replace("\r\n", "\n")  # type: ignore

    # Then handle remaining "\r" -> "\n" (Mac classic line endings)
    if "\r" in decoded_str[: constants.MIN_NEWLINE_DETECT]:  # type: ignore
        decoded_str = decoded_str.replace("\r", "\n")  # type: ignore

    return decoded_str  # type: ignore


def strip_doc(doc: Doc) -> Doc:
    """Strip leading and normalise trailing whitespace in a spaCy Doc.

    Args:
        doc: spaCy Doc to analyze

    Returns:
        Doc: the Doc with leading and trailing whitespace removed.

    Raises:
        ValueError: If Doc is empty or contains only whitespace.

    Note: If the final token has trailing whitespace, this will be preserved.
          You can remove the space with:

          ```python
          words = [t.text for t in doc]
          spaces = [t.whitespace_ for t in doc]
          spaces[-1] = ""
          doc = Doc(doc.vocab, words=words, spaces=spaces)

          But you will lose all entities and custom extensions. So it makes more
          sense to call doc.text.strip() when needed instead.
    """
    if not doc:
        raise LexosException("Document is empty.")

    # Find first non-whitespace token
    start_idx = 0
    for token in doc:
        if not token.is_space:
            start_idx = token.i
            break

    # Find last non-whitespace token
    end_idx = len(doc) - 1
    for i in range(len(doc) - 1, -1, -1):  # list(doc)[::-1]
        if not doc[i].is_space:
            end_idx = i
            break

    return doc[start_idx : end_idx + 1].as_doc()


def get_token_extension_names(doc: Doc) -> list[str]:
    """Get the names of token extensions from a spaCy Doc.

    Args:
        doc: spaCy Doc to analyze.

    Returns:
        list[str]: a list of token extensions.
    """
    return [ext for ext in doc[0]._.__dict__["_extensions"].keys()]


def to_collection(
    val: AnyVal | Collection[AnyVal],
    val_type: type[Any] | tuple[type[Any], ...],
    col_type: type[Any],
) -> Collection[AnyVal]:
    """Validate and cast a value or values to a collection.

    Args:
        val (AnyVal | Collection[AnyVal]): Value or values to validate and cast.
        val_type (type[Any] | tuple[type[Any], ...]): Type of each value in collection, e.g. ``int`` or ``(str, bytes)``.
        col_type (type[Any]): Type of collection to return, e.g. ``tuple`` or ``set``.

    Returns:
        Collection[AnyVal]: Collection of type ``col_type`` with values all of type ``val_type``.

    Raises:
        TypeError: An invalid value was passed.
    """
    if val is None:
        return []
    if isinstance(val, val_type):
        return col_type([val])
    elif isinstance(val, (tuple, list, set, frozenset)):
        if not all(isinstance(v, val_type) for v in val):
            raise TypeError(f"not all values are of type {val_type}")
        return col_type(val)
    else:
        # TODO: use standard error message, maybe?
        raise TypeError(
            f"values must be {val_type} or a collection thereof, not {type(val)}"
        )
