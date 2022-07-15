"""merge.py.

Standalone function to merge a list of segments into a single string.

Segments can be either strings or spacy Doc objects, but not a mix.
To merge files, use the merge_files function or, if you have previously
used `Filesplit` to split the files, use `Filesplit.merge()`.
"""

import shutil
from typing import List, Union

import spacy
from smart_open import open
from spacy.tokens import Doc

from lexos.exceptions import LexosException


def merge(segments: Union[List[str], List[spacy.tokens.doc.Doc]], sep=None):
    """Merge a list of segments into a single string.

    Args:
        segments (_type_): The list of segments to merge.
        sep (str, optional): The separator to use when merging strings. Defaults to None.
    """
    if all(isinstance(segment, str) for segment in segments):
        if sep is None:
            sep = ""
        return sep.join(segments)
    elif all(isinstance(segment, spacy.tokens.doc.Doc) for segment in segments):
        return Doc.from_docs(segments)
    else:
        raise LexosException(
            "All segments must be either strings or spacy.tokens.doc.Doc."
        )


def merge_files(
    segment_files: List[str],
    output_file: str = "merged_files.txt",
    binary: bool = False,
) -> None:
    """Merge two files into a single string.

    Args:
        segment_files (List[str]): List of files to be merged.
        output_file (str, optional): The name of the output file.
        binary (bool, optional): Whether to read and write files as binary. Defaults to False.
    """
    if binary:
        read_mode = "rb"
        write_mode = "wb"
    else:
        read_mode = "r"
        write_mode = "w"
    with open("merged_file.txt", write_mode) as out_file:
        for file in segment_files:
            try:
                with open(output_file, read_mode) as f:
                    shutil.copyfileobj(file, out_file, 1024 * 1024 * 10)
            except Exception as e:
                raise LexosException(f"Error merging files: {e}.")

