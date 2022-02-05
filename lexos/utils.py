"""utils.py.

    This file contains helper functions used by multiple modules.
"""
import re
import zipfile
from pathlib import Path
from time import sleep
from typing import Any, Collection, List, Tuple, Type, TypeVar, Union

import chardet
from bs4 import UnicodeDammit
from rich.progress import Progress

from lexos import constants
from lexos.exceptions import LexosException

AnyVal = TypeVar("AnyVal")

def ensure_list(item: Any) -> List:
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
        return Path(path.replace('\\', '/'))
    else:
        return path

def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return bool(re.match(
        r"(https?|ftp)://" # protocol
        r"(\w+(\-\w+)*\.)?" # host (optional)
        r"((\w+(\-\w+)*)\.(\w+))" # domain
        r"(\.\w+)*" # top-level domain (optional, can have > 1)
        r"([\w\-\._\~/]*)*(?<!\.)" # path, params, anchors, etc. (optional)
    , s))

def to_collection(
    val: Union[AnyVal, Collection[AnyVal]],
    val_type: Union[Type[Any], Tuple[Type[Any], ...]],
    col_type: Type[Any],
) -> Collection[AnyVal]:
    """
    Validate and cast a value or values to a collection.
    Args:
        val (object): Value or values to validate and cast.
        val_type (type): Type of each value in collection, e.g. ``int`` or ``(str, bytes)``.
        col_type (type): Type of collection to return, e.g. ``tuple`` or ``set``.
    Returns:
        Collection of type ``col_type`` with values all of type ``val_type``.
    Raises:
        TypeError
    """
    if val is None:
        return None
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

def unzip_archive(archive_path: str, extract_dir: str):
    """Extract a zip archive.

    For adding a progress indicator, see
    https://stackoverflow.com/questions/4006970/monitor-zip-file-extraction-python.

    Args:
        archive_path (str): The path to the archive file to be unzipped.
        extract_dir (str): The path to folder where the archive will be extracted.
    """
    zf = zipfile.ZipFile(archive_path, 'r')
    progress = Progress()
    with progress:
        for file in progress.track(zf.infolist(), description="Processing..."):
            zf.extract(file, path=extract_dir)
            sleep(0.1)


def zip_folder(source_dir: Path, archive_file: Path):
    """Zip a folder recursively with no extra root folder in the archive.

    Works with a progress indicator.

    Args:
        source_dir (Path): The path to the source directory.
        archive_file (Path): The path to the archive file to be created (including file extension).
    """
    progress = Progress()
    with zipfile.ZipFile(
            archive_file,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=7
        ) as zip:
        files = list(source_dir.rglob("*"))
        with progress:
            for file in progress.track(files, description="Processing..."):
                relative_path = file.relative_to(source_dir)
                zip.write(file, arcname=relative_path)
                sleep(0.1)

def get_encoding(input_string: bytes) -> str:
    """Use chardet to return the encoding type of a string.

    Args:
        input_string (bytes): A bytestring.

    Returns:
        The string's encoding type.
    """
    encoding_detect = chardet.detect(input_string[
                                     :constants.MIN_ENCODING_DETECT])
    encoding_type = encoding_detect['encoding']
    return encoding_type

def normalize(raw_bytes: Union[bytes, str]) -> str:
    """Normalise a string to LexosFile format.

    Args:
        raw_bytes (str): The input string.

    Returns:
        Normalised version of the input string.
    """
    s = _decode_bytes(raw_bytes)
    return s

def normalize_strings(strings: List[Union[bytes, str]]) -> str:
    """Normalise a list of strings to LexosFile format.

    Args:
        strings (list): The list of input strings.

    Returns:
        A list of normalised versions of the input strings.
    """
    normalized_strings = []
    for s in strings:
        normalized_strings.append(normalize(s))
    return normalized_strings

def normalize_files(filepaths: List[Union[Path, str]],
                    destination_dir: Union[Path, str] = '.') -> str:
    """Normalise a list of files to LexosFile format and save the files.

    Args:
        filepaths (list): The list of paths to input files.
        destination_dir (path, str): The path to the directory where the files
            will be saved.
    """
    for filepath in filepaths:
        filepath = ensure_path(filepath)
        with open(filepath, 'rb') as f:
            doc = f.read()
        with open(destination_dir / filepath.name, 'wb') as f:
            f.write(normalize(doc))

def normalize_file(filepath: Union[Path, str],
                    destination_dir: Union[Path, str] = '.') -> str:
    """Normalise a file to LexosFile format and save the file.

    Args:
        filepath (Path, str): The path to the input file.
        destination_dir (path, str): The path to the directory where the files
            will be saved.
    """
    filepath = ensure_path(filepath)
    with open(filepath, 'rb') as f:
        doc = f.read()
    with open(destination_dir / filepath.name, 'wb') as f:
        f.write(normalize(doc))


def _try_decode_bytes_(raw_bytes: bytes) -> str:
    """Try to decode raw bytes (helper function for decode_bytes().

    Args:
        raw_bytes (bytes): The bytes you want to decode to string.

    Returns:
        A decoded string.
    """
    # Detect the encoding with only the first couple of bytes
    encoding_detect = chardet.detect(
        raw_bytes[:constants.MIN_ENCODING_DETECT])
    # Get the encoding
    encoding_type = encoding_detect['encoding']
    if encoding_type is None:
        encoding_detect = chardet.detect(raw_bytes)
        encoding_type = encoding_detect['encoding']

    try:
        # Try to decode the string using the original encoding
        decoded_string = raw_bytes.decode(encoding_type)

    except (UnicodeDecodeError, TypeError):
        # Try UnicodeDammit if chardet didn't work
        if encoding_type == "ascii":
            dammit = UnicodeDammit(raw_bytes, ["iso-8859-1", "iso-8859-15", "windows-1252"])
        else:
            dammit = UnicodeDammit(raw_bytes)
        decoded_string = dammit.unicode_markup

    return decoded_string


def _decode_bytes(raw_bytes: Union[bytes, str]) -> str:
    """Decode raw bytes from a user's file into a string.

    Args
        raw_bytes (bytes, str): The bytes to be decoded to a python string.

    Returns:
        The decoded string.
    """
    if isinstance(raw_bytes, bytes):
        try:
            decoded_str = _try_decode_bytes_(raw_bytes)

        except (UnicodeDecodeError, TypeError):
            raise LexosException('Chardet failed to detect encoding of your '
                                 'file. Please make sure your file is in '
                                 'utf-8 encoding.')
    else:
        decoded_str = raw_bytes

    return decoded_str