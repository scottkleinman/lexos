"""mallet.py.

Last Updated: September 8, 2026
Last Tested: September 8, 2026

A fork of Maria Antoniak's Little Mallet Wrapper: https://github.com/maria-antoniak/little-mallet-wrapper.

Here is a rough summary of the changes:

- Some functions for importing training data from various sources.
- Formatting changes, type hinting, and Pydantic validation.
- A more object-oriented approach to keep track of paths and other metadata so that fewer arguments need to be passed to functions.
- Support for a fuller range of MALLET keyword arguments, including the output-state-file which is needed for generating PyLDAVis and Dfr-Browser visualizations.
- Option to choose between Java and Rust backends (using pyrmallet).
- Optional progress tracking during training.
- Topic clouds and termite plot visualisations.
- More parameters for customising the plotting functions.
"""

import glob
import json
import os
import re
import subprocess
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from pandas.io.formats.style import Styler
from pydantic import BaseModel, ConfigDict, Field, model_validator, validate_call
from spacy.tokens import Doc
from tqdm.auto import tqdm
from wasabi import msg
from wordcloud import WordCloud

from lexos.exceptions import LexosException
from lexos.util import ensure_list
from lexos.visualization.cloud import MultiCloud

# Get the path to the MALLET binary from the environment
load_dotenv()
MALLET_BINARY_PATH = str(Path(os.getenv("MALLET_BINARY_PATH") or "mallet").expanduser())
model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call
def read_file(file: Path | str) -> list[str]:
    r"""Import data from a single text file with one document per line.

    Args:
        file (Path | str): A file containing the documents to import.

    Returns:
        list[str]: The training data.

    Notes:
        This function uses an internal helper `_check_format` to validate and convert the input data to MALLET format. The helper accepts data with 1-3 tab-separated columns and normalizes it to the format: `id\\tlabel\\ttext`.
    """

    # Check the format of the input data and convert to MALLET format if necessary
    def _check_format(file: Path | str) -> list[str]:
        """Check the format of the input data and convert to MALLET format if necessary.

        Args:
            file (Path | str): The input file to check.

        Returns:
            list[str]: The training data in MALLET format.
        """
        df = pd.read_csv(file, sep="\t", header=None)
        if len(df.columns) == 1:
            df["label"] = ""
            df["id"] = df.index
            df = df[["id", "label", 0]]
        elif len(df.columns) == 2:
            df["id"] = df.index
            df["label"] = ""
            df = df[["id", "label", 1]]
        elif len(df.columns) >= 3:
            # Merge column 2 with all subsequent columns
            df[2] = df.iloc[:, 2:].apply(
                lambda x: " ".join(x.dropna().astype(str)), axis=1
            )
            df = df[[0, 1, 2]]
        else:
            raise ValueError("Input data must have between 1 and 3 columns.")
        df.columns = ["id", "label", "text"]
        return [
            f"{str(row['id']).strip()}\t{str(row['label']).strip()}\t{str(row['text']).strip()}"
            for row in df.to_dict(orient="records")
        ]

    # Validate the input
    if isinstance(file, bool):
        raise LexosException(
            "Invalid input for `file`. Expected a file path (str or Path), not a boolean."
        )

    # Retrieve the data from file
    try:
        return _check_format(file)
    except FileNotFoundError:
        raise LexosException(f"File {file} does not exist.")
    except IOError:
        raise LexosException(f"File {file} could not be read.")


def _validate_directory_path(directory: Path | str) -> Path:
    """Validate a directory argument and return a normalized Path object.

    Args:
        directory (Path | str): A directory path to validate.

    Returns:
        Path: The validated directory path.

    Raises:
        LexosException: If the path is a boolean, is not a path-like value, or does
            not exist.
    """
    if isinstance(directory, bool) or not isinstance(directory, (str, Path)):
        raise LexosException(
            f"Invalid directory argument '{directory}'. Expected a directory path (str or Path)."
        )

    path = Path(directory)
    if not path.is_dir():
        raise LexosException(f"Directory {directory} does not exist.")

    return path


def _read_txt_files_in_directory(directory: Path) -> list[str]:
    """Read all .txt files in a directory and return their contents.

    Args:
        directory (Path): The directory whose text files should be read.

    Returns:
        list[str]: The contents of each .txt file in the directory, in sorted file-name
            order.

    Notes:
        This intentionally avoids Path.glob() because iterating over the generator can
        disrupt tests that rely on a stable list order.
    """
    contents: list[str] = []
    for path in sorted(glob.glob(f"{directory}/*.txt")):
        file_path = Path(path)
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as file:
                contents.append(file.read())
    return contents


@validate_call
def read_dirs(dirs: Path | str | list[Path | str]) -> list[str]:
    """Import a directory or list of directories.

    Args:
        dirs (Path | str | list[Path | str]): A directory or list of directories to
            import.

    Returns:
        list[str]: The text contents of each .txt file found in the supplied directory or
            directories.
    """
    training_data: list[str] = []
    for directory in ensure_list(dirs):
        validated_path = _validate_directory_path(directory)
        training_data.extend(_read_txt_files_in_directory(validated_path))
    return training_data


@validate_call
def import_files(files: Path | str | list[Path | str]) -> list[str]:
    """Import the text content of a file or list of files.

    Args:
        files (Path | str | list[Path | str]): A file or list of files to read.

    Returns:
        list[str]: A list of file contents.
    """
    if isinstance(files, (Path, str)):
        files = [files]
    contents = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as fh:
                contents.append(fh.read())
        except FileNotFoundError:
            raise LexosException(f"File {file} does not exist")
        except IOError:
            raise LexosException(f"File {file} could not be read")
    return contents


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def import_docs(docs: list[str | Doc]) -> list[str]:
    """Import a list of document strings or spaCy Docs.

    Args:
        docs (list[str | Doc]): List of documents.

    Returns:
        list[str]: List of document texts.
    """
    training_data = []
    for doc in docs:
        if isinstance(doc, Doc):
            training_data.append(doc.text)
        else:
            training_data.append(doc)
    return training_data


class Mallet(BaseModel):
    """A class for training and using MALLET topic models."""

    backend: str = "java"

    # IMPORTANT: The class initializes with only the `model_directory` key.
    # Functions will add canonical metadata entries as needed (e.g.
    # 'path_to_topic_distributions', 'path_to_term_weights', 'path_to_topic_keys').
    # Legacy synonyms are not used; code reads canonical keys only.

    path_to_mallet: str = MALLET_BINARY_PATH
    # Accept either a string or a Path for `model_dir` to allow intuitive usage
    model_dir: Optional[Path | str] = Field(
        None,
        description="The directory where the model is stored.",
    )
    metadata: dict[str, Any] = Field(
        {},
        description="A dict containing metadata generated by the class instance.",
    )

    model_config = model_config

    def __new__(cls, *args: Any, backend: Optional[str] = None, **kwargs: Any):
        """Route to the requested backend while preserving the Java-backed default as the public API."""
        if cls is not Mallet:
            return super().__new__(cls)

        target_backend = (backend or kwargs.get("backend") or "java").lower()
        if target_backend == "pyrmallet":
            from lexos.topic_modeling.mallet.pyrmallet import PyRMallet

            return object.__new__(PyRMallet)
        if target_backend == "java":
            return object.__new__(cls)
        raise ValueError(f"Unknown MALLET backend: {target_backend!r}")

    # Canonical metadata keys used consistently across methods for common
    # training outputs. To preserve backward compatibility when loading
    # metadata produced by older flows, a set of synonyms is still supported
    # but all internal methods should rely only on the canonical keys below.
    # The synonyms list is used for migration to canonical form.
    CANONICAL_DOC_TOPIC_KEY: ClassVar[str] = "path_to_topic_distributions"
    # Canonical key names
    CANONICAL_DOC_TOPIC_KEY: ClassVar[str] = "path_to_topic_distributions"
    CANONICAL_TERM_WEIGHTS_KEY: ClassVar[str] = "path_to_term_weights"
    CANONICAL_TOPIC_KEYS_KEY: ClassVar[str] = "path_to_topic_keys"
    CANONICAL_INFERENCER_KEY: ClassVar[str] = "path_to_inferencer"

    def __init__(self, **data: Any):
        """Initialize the Mallet class.

        Args:
            **data (Any): Arbitrary keyword arguments for initialization.
        """
        super().__init__(**data)
        # Save the path to MALLET in the metadata for reference
        self.metadata["path_to_mallet"] = self.path_to_mallet

        # Ensure model_dir is a Path object if provided as a string
        if self.model_dir and isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

        if self.model_dir:
            self.metadata["model_directory"] = str(self.model_dir)

        # If the model directory exists, attempt to load existing metadata from meta.json
        if self.model_dir and self.model_dir.exists():
            meta_path = self.model_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        loaded_metadata = json.load(f)
                        # Update the metadata dictionary with loaded values
                        self.metadata.update(loaded_metadata)
                        # Ensure model_directory in metadata matches the object property
                        self.metadata["model_directory"] = str(self.model_dir)
                except (json.JSONDecodeError, IOError) as e:
                    raise LexosException(
                        f"Failed to load metadata from {meta_path}: {e}"
                    )

    @model_validator(mode="after")
    def _normalize_backend(self) -> "Mallet":
        """Store the selected backend in the instance for downstream introspection."""
        self.backend = str(self.backend or "java").lower()
        return self

    def _metadata_get(self, keys: list[str]) -> str | None:
        """Return the first metadata value present among the provided keys or None.

        The method assumes callers pass canonical key names; no synonym
        translation is performed.
        """
        # Only accept the canonical key for each category. If a synonym key is
        # present (legacy metadata), raise an error instructing users to use
        # the canonical key. This ensures a single canonical name per category.
        for k in keys:
            if k in self.metadata and self.metadata[k]:
                return self.metadata[k]
        return None

    def _metadata_has(self, keys: list[str]) -> bool:
        return self._metadata_get(keys) is not None

    # No metadata canonicalization: initialization should only set model_directory
    # and functions will add canonical keys as necessary.

    def _is_sparse_distribution(self, raw_values: list[str]) -> bool:
        """Return whether a distribution line is encoded in sparse topic:probability form.

        Args:
            raw_values (list[str]): The non-header values from a MALLET distribution line.

        Returns:
            bool: True if the values are sparse topic:probability pairs; otherwise False.
        """
        return (len(raw_values) == 1 and ":" in raw_values[0]) or (
            len(raw_values) > 0 and all(":" in value for value in raw_values)
        )

    def _parse_sparse_distribution(self, raw_values: list[str]) -> list[float]:
        """Parse a sparse MALLET distribution using topic:probability pairs.

        Args:
            raw_values (list[str]): A list of sparse values such as ["0:0.2", "1:0.8"] or
                ["0:0.2 1:0.8"].

        Returns:
            list[float]: A dense list of probabilities keyed by topic index.

        Raises:
            LexosException: If a topic:probability pair is malformed.
        """
        probability_map: dict[int, float] = {}
        max_topic = -1
        pairs = raw_values[0].split() if len(raw_values) == 1 else raw_values

        for pair in pairs:
            try:
                topic_text, probability_text = pair.split(":")
                topic_index = int(topic_text)
                probability = float(probability_text)
            except (ValueError, IndexError) as exc:
                raise LexosException(f"Malformed topic:prob pair: {pair}") from exc

            probability_map[topic_index] = probability
            if topic_index > max_topic:
                max_topic = topic_index

        return [
            float(probability_map.get(index, 0.0)) for index in range(max_topic + 1)
        ]

    def _parse_dense_distribution(self, raw_values: list[str]) -> list[float]:
        """Parse a dense MALLET distribution from whitespace or tab-delimited floats.

        Args:
            raw_values (list[str]): The float values from a distribution line.

        Returns:
            list[float]: The parsed probability values.

        Raises:
            LexosException: If one or more values cannot be converted to floats.
        """
        try:
            return [float(value) for value in raw_values]
        except ValueError as exc:
            raise LexosException(
                f"Failed to parse float from distribution: {exc}"
            ) from exc

    def _parse_distribution_line(self, line: str) -> list[float]:
        """Parse a single MALLET distribution line from either dense or sparse format.

        Args:
            line (str): A MALLET topic distribution line.

        Returns:
            list[float]: A dense list of document-topic probabilities.

        Raises:
            LexosException: If the distribution line is malformed or cannot be parsed.
        """
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 3:
            parts = re.split(r"\s+", line)

        if len(parts) < 3:
            if len(parts) == 2 and ":" in parts[1]:
                raw_values = parts[1:]
            else:
                raise LexosException(f"Malformed line: {line}")
        else:
            raw_values = parts[2:]

        if self._is_sparse_distribution(raw_values):
            return self._parse_sparse_distribution(raw_values)

        return self._parse_dense_distribution(raw_values)

    @model_validator(mode="after")
    def _validate_mallet_path(self) -> "Mallet":
        """Expand tilde and handle directory paths for MALLET."""
        if self.path_to_mallet:
            # Expand ~ to the user's home directory
            p = Path(self.path_to_mallet).expanduser()

            # If the path points to a directory, append 'mallet'
            if p.is_dir():
                p = p / "mallet"

            self.path_to_mallet = str(p)

        return self

    def _resolve_model_dir_value(self) -> Path | str | None:
        """Resolve the model directory from metadata when it is not already set.

        Returns:
            Path | str | None: The model directory value if one is available; otherwise
                None.
        """
        if self.model_dir is None and isinstance(self.metadata, dict):
            if "model_directory" in self.metadata:
                self.model_dir = self.metadata["model_directory"]
        return self.model_dir

    def _ensure_valid_model_dir(self, model_dir_value: Path | str) -> None:
        """Validate a model directory and create it if needed.

        Args:
            model_dir_value (Path | str): The directory path to validate.

        Raises:
            LexosException: If the provided value is a boolean, or if the path exists and
                is a file instead of a directory.
        """
        if isinstance(model_dir_value, bool):
            raise LexosException(
                "Invalid `model_dir` argument: expected a path (str or Path), not a boolean."
            )

        model_dir_str = (
            str(model_dir_value)
            if isinstance(model_dir_value, Path)
            else model_dir_value
        )
        path = Path(model_dir_str)

        if path.exists() and path.is_file():
            raise LexosException(
                f"The specified `model_dir` ({model_dir_str}) exists and is a file, expected a directory."
            )

        path.mkdir(parents=True, exist_ok=True)
        self.metadata["model_directory"] = model_dir_str

    @model_validator(mode="after")
    def _validate_model_dir(self) -> "Mallet":
        """Validate and create the model directory for this instance.

        Returns:
            Mallet: The validated model instance.
        """
        model_dir = self._resolve_model_dir_value()
        if model_dir is not None:
            self._ensure_valid_model_dir(model_dir)
        return self

    @cached_property
    def distributions(self) -> list[list[float]]:
        """Get the topic distributions for each document in the model.

        Returns:
            list[list[float]]: A list of topic distributions for each document.
        """
        distro_path = self._metadata_get([self.CANONICAL_DOC_TOPIC_KEY])
        if distro_path is None:
            raise LexosException("No topic distributions set.")

        topic_distributions = []
        with open(distro_path, "r") as f:
            for line in f:
                # Skip header and blank lines
                if not line.strip() or line.startswith("#"):
                    continue
                topic_distributions.append(self._parse_distribution_line(line))
        return topic_distributions

    @property
    def num_docs(self) -> int:
        """Get the number of docs in the model."""
        if "num_docs" in self.metadata:
            return self.metadata["num_docs"]
        else:
            return 0

    @property
    def mean_num_tokens(self) -> int:
        """Get the mean number of tokens per document in the model."""
        if "mean_num_tokens" in self.metadata:
            v = self.metadata["mean_num_tokens"]
            try:
                return v.item()
            except Exception:
                return int(v)
        else:
            return 0

    @property
    def model_directory(self) -> str:
        """Return the model_directory from metadata or raise LexosException if missing."""
        if isinstance(self.metadata, dict) and "model_directory" in self.metadata:
            return self.metadata["model_directory"]
        raise LexosException(
            "No model directory has been set; provide one or set 'model_directory' in metadata."
        )

    @cached_property
    def topic_keys(self) -> list[list[str]]:
        """Get the keys of the model.

        Returns:
            list[list[str]]: A list of topics where each topic is a sublist containing the topic index, topic weight, and a space-separated list of keywords.
        """
        topic_keys_path = self._metadata_get([self.CANONICAL_TOPIC_KEYS_KEY])
        if not topic_keys_path:
            raise LexosException(
                f"No topic keys have been set. Please designate a path for `{self.CANONICAL_TOPIC_KEYS_KEY}` when you train your topic model."
            )
        with open(self.metadata[self.CANONICAL_TOPIC_KEYS_KEY], "r") as f:
            results = []
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                # Ensure at least 3 columns for consistency (index, weight, keywords)
                while len(parts) < 3:
                    parts.append("")

                results.append(parts)
            return results

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of documents in the model."""
        if "vocab_size" in self.metadata:
            return self.metadata["vocab_size"]
        else:
            return 0

    def _resolve_training_file_paths(
        self, path_to_training_data: Optional[str] = None
    ) -> tuple[str, str]:
        """Resolve the raw and formatted training-data file paths for MALLET.

        Args:
            path_to_training_data (Optional[str]): The raw training-data path to use. If
                not provided, a default file is created inside the model directory.

        Returns:
            tuple[str, str]: The raw training-data path and the MALLET-formatted output path.
        """
        raw_path = (
            path_to_training_data
            if path_to_training_data is not None
            else str(Path(self.model_dir) / "training_data.txt")
        )
        formatted_path = str(Path(self.model_dir) / "training_data.mallet")
        return raw_path, formatted_path

    def _write_training_data_file(
        self,
        training_data: list[str],
        path_to_training_data: str,
        training_ids: Optional[list[int]] = None,
    ) -> tuple[int, set[str]]:
        """Write raw training documents to disk and collect vocabulary statistics.

        Args:
            training_data (list[str]): The document texts to write.
            path_to_training_data (str): The path to the raw training-data file.
            training_ids (Optional[list[int]]): Optional IDs to attach to each document.

        Returns:
            tuple[int, set[str]]: The total token count and the document vocabulary set.
        """
        total_tokens = 0
        vocab: set[str] = set()

        with open(path_to_training_data, "w", encoding="utf-8") as training_data_file:
            for i, doc in enumerate(training_data):
                doc = re.sub("[\r\n]+", " ", doc).strip()
                document_id = training_ids[i] if training_ids else i
                training_data_file.write(f"{document_id}\tno_label\t{doc}\n")

                tokens = doc.split()
                total_tokens += len(tokens)
                vocab.update(tokens)

        return total_tokens, vocab

    def _build_import_command(
        self,
        path_to_training_data: str,
        path_to_formatted_training_data: str,
        keep_sequence: bool = True,
        remove_stopwords: bool = True,
        preserve_case: bool = True,
        use_pipe_from: Optional[str] = None,
    ) -> list[str]:
        """Build the MALLET import command used to format training documents.

        Args:
            path_to_training_data (str): Path to the raw text training data.
            path_to_formatted_training_data (str): Path for the formatted MALLET file.
            keep_sequence (bool): Whether to preserve token order.
            remove_stopwords (bool): Whether to remove stopwords during import.
            preserve_case (bool): Whether to preserve original casing.
            use_pipe_from (Optional[str]): Optional MALLET pipe file to reuse.

        Returns:
            list[str]: The MALLET import command arguments.
        """
        cmd = [
            self.path_to_mallet or "mallet",
            "import-file",
            "--input",
            path_to_training_data,
            "--output",
            path_to_formatted_training_data,
        ]
        if keep_sequence:
            cmd.append("--keep-sequence")
        if remove_stopwords:
            cmd.append("--remove-stopwords")
        if preserve_case:
            cmd.append("--preserve-case")
        if use_pipe_from:
            cmd.extend(["--use-pipe-from", use_pipe_from])
        return cmd

    def _import_training_data(
        self,
        training_data: list[str],
        path_to_training_data: Optional[str] = None,
        keep_sequence: bool = True,
        remove_stopwords: bool = True,
        preserve_case: bool = True,
        use_pipe_from: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
    ) -> None:
        """Import training data from a list of documents.

        Args:
            training_data (list[str]): A list of documents to import.
            path_to_training_data (Optional[str]): The raw text file to write before MALLET
                import. If None, a default path is created inside the model directory.
            keep_sequence (bool): Whether to keep the word sequence in the documents.
            remove_stopwords (bool): Whether to remove stopwords from the documents.
            preserve_case (bool): Whether to preserve the case of the documents.
            use_pipe_from (Optional[str]): Path to a MALLET pipe file to use for importing.
            training_ids (Optional[list[int]]): A list of document ids designating a subset
                of the dataset. If None, the entire dataset is imported.
        """
        raw_path, formatted_path = self._resolve_training_file_paths(
            path_to_training_data
        )
        total_tokens, vocab = self._write_training_data_file(
            training_data, raw_path, training_ids
        )

        self.metadata["path_to_training_data"] = raw_path
        self.metadata["path_to_formatted_training_data"] = formatted_path

        num_docs = len(training_data)
        self.metadata["num_docs"] = num_docs
        self.metadata["mean_num_tokens"] = (
            total_tokens / num_docs if num_docs > 0 else 0
        )
        self.metadata["vocab_size"] = len(vocab)

        with open(Path(self.model_dir) / "meta.json", "w") as file:
            file.write(json.dumps(self.metadata))

        cmd = self._build_import_command(
            raw_path,
            formatted_path,
            keep_sequence,
            remove_stopwords,
            preserve_case,
            use_pipe_from,
        )
        msg.info(" ".join(cmd))
        subprocess.run(cmd, check=True)

    @validate_call(config=model_config)
    def import_data(
        self,
        training_data: list[str],
        path_to_training_data: str = None,
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
    ) -> None:
        """Convenience wrapper to import a list of documents and format them for MALLET.

        Args:
            training_data (list[str]): List of document texts.
            path_to_training_data (str): Path to write raw training text file. If None, will default to model directory.
            keep_sequence (bool): Keep token sequence.
            preserve_case (bool): Preserve case.
            remove_stopwords (bool): Remove stopwords.
            use_pipe_from (Optional[str]): Pipe filename for MALLET import.
            training_ids (Optional[list[int]]): Optional training IDs mapping.
        """
        # Validate training_data is a list of strings
        if isinstance(training_data, bool) or not isinstance(training_data, list):
            raise LexosException(
                "Invalid `training_data` argument: expected a list of document strings."
            )
        for doc in training_data:
            if isinstance(doc, bool) or not isinstance(doc, str):
                raise LexosException(
                    "Invalid `training_data` element: expected document text (str) for each item."
                )

        # Determine output paths if not provided
        if not path_to_training_data:
            model_base = Path(self.model_dir) if self.model_dir else Path.cwd()
            path_to_training_data = str(model_base / "training_data.txt")
        self._import_training_data(
            training_data,
            path_to_training_data,
            keep_sequence,
            remove_stopwords,
            preserve_case,
            use_pipe_from,
            training_ids,
        )

    def _setup_wordcloud(
        self, round_mask, max_terms, **kwargs: dict[str, Any]
    ) -> WordCloud:
        """Set up the word cloud object.

        Args:
            round_mask (bool): Whether to use a round mask for the word cloud.
            max_terms (int): The maximum number of keywords to display.
            **kwargs (dict[str, Any])): Additional keyword arguments for the WordCloud object.

        Returns:
            WordCloud: A configured WordCloud object.
        """
        # Define a mask to make the word cloud round (just some eye candy)
        if round_mask:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > 130**2
            mask = 255 * mask.astype(int)
        else:
            mask = None

        # Configure the word cloud object
        options = {
            "background_color": "white",
            "mask": mask,
            "contour_width": 0.1,
            "contour_color": "white",
            "max_words": max_terms,
            "min_font_size": 10,
            "max_font_size": 150,
            "random_state": 42,
            "colormap": "Dark2",
        }
        for k, v in kwargs.items():
            options[k] = v

        return WordCloud(**options)

    def _format_topic_key_row(
        self, topic: list[str], num_keys: int
    ) -> tuple[str, str, str]:
        """Format a single topic row for display.

        Args:
            topic (list[str]): The raw topic row from the MALLET topic-keys file.
            num_keys (int): The maximum number of keyword tokens to display.

        Returns:
            tuple[str, str, str]: The formatted topic label, weight, and keywords.
        """
        keywords = " ".join(topic[2].split()[:num_keys])
        custom_labels = self.metadata.get("topic_labels")
        topic_label = topic[0]
        if custom_labels and str(topic[0]) in custom_labels:
            topic_label = custom_labels[str(topic[0])]
        return str(topic_label), str(topic[1]), keywords

    def _validate_topic_index_list(
        self, topics: list[int], num_available_topics: int
    ) -> None:
        """Ensure requested topic indices fall within the available topic range.

        Args:
            topics (list[int]): Requested topic indices.
            num_available_topics (int): Number of topics in the current model.

        Raises:
            IndexError: If any requested topic index is out of range.
        """
        for index in topics:
            if index < 0 or index >= num_available_topics:
                raise IndexError(
                    f"Topic index {index} is out of range. Valid indices are 0 to {num_available_topics - 1}."
                )

    def _resolve_topic_keys(
        self, num_topics: int = None, topics: list[int] = None
    ) -> list[list[str]]:
        """Resolve the topic rows to display and validate requested indices.

        Args:
            num_topics (int): The maximum number of topics to return when no explicit list is
                provided.
            topics (list[int]): The explicit topic indices to display.

        Returns:
            list[list[str]]: The selected topic rows.

        Raises:
            IndexError: If a requested topic index is outside the valid range.
        """
        num_available_topics = len(self.topic_keys)
        if num_topics is not None and not topics:
            if num_topics > num_available_topics:
                raise IndexError(
                    f"Requested num_topics={num_topics}, but only {num_available_topics} topics are available."
                )
            return self.topic_keys[:num_topics]

        if topics is not None:
            self._validate_topic_index_list(topics, num_available_topics)
            return [self.topic_keys[i] for i in topics]

        return self.topic_keys

    def _build_topic_key_dataframe(
        self, topic_keys: list[list[str]], num_keys: int
    ) -> pd.DataFrame:
        """Build the DataFrame used for the styled topic-key output.

        Args:
            topic_keys (list[list[str]]): The selected topic rows.
            num_keys (int): The maximum number of keyword tokens to display.

        Returns:
            pd.DataFrame: A DataFrame with topic labels, weights, and keywords.
        """
        rows = []
        for topic in topic_keys:
            topic_label, weight, keywords = self._format_topic_key_row(topic, num_keys)
            rows.append({"Topic": topic_label, "Weight": weight, "Keywords": keywords})
        return pd.DataFrame(rows)

    def _style_topic_key_dataframe(self, dataframe: pd.DataFrame) -> Styler:
        """Apply notebook-friendly styling to the topic-key DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to style.

        Returns:
            Styler: A styled DataFrame with left-aligned keyword text.
        """
        show_index = True
        offset = 2 if show_index else 1
        nth = dataframe.columns.get_loc("Keywords") + offset

        css = [
            {
                "selector": f"thead th:nth-child({nth})",
                "props": [("text-align", "left")],
            },
            {
                "selector": f"td.col{dataframe.columns.get_loc('Keywords')}",
                "props": [("text-align", "left")],
            },
        ]

        return dataframe.style.set_table_styles(css).set_properties(
            subset=["Keywords"], **{"text-align": "left"}
        )

    def _start_training_process(self, mallet_cmd: list[str]):
        """Start the MALLET training subprocess and capture its output.

        Args:
            mallet_cmd (list[str]): The MALLET command to run.

        Returns:
            subprocess.Popen: The running training process.
        """
        return subprocess.Popen(
            mallet_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )

    def _update_training_progress(
        self,
        pbar: tqdm,
        line_str: str,
        num_iterations: int,
        last_iter: int,
    ) -> int:
        """Parse a training line and advance the progress bar when a new iteration is seen.

        Args:
            pbar (tqdm): The active progress bar.
            line_str (str): The line emitted by MALLET.
            num_iterations (int): Total optimization iterations.
            last_iter (int): The last iteration already reported.

        Returns:
            int: The updated iteration value.
        """
        prog = re.compile(r"(?:\<|Iteration\s+)(\d+)(?:\>|:)")
        try:
            match = prog.search(line_str)
            if not match:
                return last_iter
            this_iter = int(match.group(1))
            if this_iter <= last_iter:
                return last_iter
            pbar.n = min(this_iter, num_iterations)
            pbar.refresh()
            if num_iterations and this_iter >= num_iterations:
                pbar.set_description("Saving model files")
            return this_iter
        except (AttributeError, ValueError):
            return last_iter

    def _track_progress(
        self, mallet_cmd: list[str], num_iterations: int, verbose: bool = True
    ) -> None:
        """Track the progress of the modeling run and update the tqdm bar.

        Args:
            mallet_cmd (list[str]): The MALLET command to run as a list of strings.
            num_iterations (int): The number of iterations for the model.
            verbose (bool): Whether to print the MALLET output to the terminal.
        """
        pbar = tqdm(total=num_iterations or 0, desc="Training model", leave=True)

        try:
            process = self._start_training_process(mallet_cmd)
            last_iter = -1

            if process.stdout:
                for line_str in process.stdout:
                    if verbose:
                        tqdm.write(line_str.rstrip())
                    last_iter = self._update_training_progress(
                        pbar, line_str, num_iterations, last_iter
                    )

            process.wait()

            if process.returncode == 0:
                pbar.n = num_iterations
                pbar.set_description("Complete")
                pbar.refresh()
            else:
                raise subprocess.CalledProcessError(process.returncode, mallet_cmd)
        finally:
            pbar.close()

    @validate_call(config=model_config)
    def get_keys(
        self,
        num_topics: int = None,
        topics: list[int] = None,
        num_keys: int = 10,
        as_df: bool = False,
    ) -> str | Styler:
        """Get a string representation of the topic keys of the model.

        Args:
            num_topics (int): The number of topics to get keys for. If None, get keys for all topics.
            topics (list[int]): A list of topic indices to get keys for. If None, get keys for all topics.
            num_keys (int): The number of keys to output for each topic.
            as_df (bool): Whether to return the result as a pandas DataFrame instead of a string.

        Returns:
            str | Styler: A string or DataFrame representation of the topic keys. The DataFrame is styled for presentation in a Jupyter notebook to prevent clipping of the keywords in a Jupyter notebook. If you need an actual `DataFrame` object, reference `df.data`.
        """
        selected_topics = self._resolve_topic_keys(num_topics, topics)
        output = ""
        for topic in selected_topics:
            topic_label, weight, keywords = self._format_topic_key_row(topic, num_keys)
            output += f"Topic {topic_label}\t{weight}\t{keywords}\n"

        if as_df:
            dataframe = self._build_topic_key_dataframe(selected_topics, num_keys)
            return self._style_topic_key_dataframe(dataframe)

        return output

    def _validate_topic_index(self, topic: int, num_topics: int) -> int:
        """Validate that a topic index is in range for the current model.

        Args:
            topic (int): The topic index to validate.
            num_topics (int): The number of topics available in the model.

        Returns:
            int: The validated integer topic index.

        Raises:
            ValueError: If the topic index is not an integer or is outside the valid range.
        """
        try:
            normalized_topic = int(topic)
        except (TypeError, ValueError) as exc:
            raise ValueError("Topic index must be an integer") from exc

        if not (0 <= normalized_topic < num_topics):
            raise ValueError(
                f"Invalid topic index {normalized_topic}. Valid topic indices are 0..{num_topics - 1} (0-based)."
            )
        return normalized_topic

    def _read_metadata_num_topics(self) -> int | None:
        """Read the declared topic count from the model metadata, when available.

        Returns:
            int | None: The metadata-declared topic count, or None if no valid value is
                available.
        """
        if "num_topics" not in self.metadata:
            return None

        try:
            return int(self.metadata["num_topics"])
        except (TypeError, ValueError):
            return None

    def _read_distribution_topic_count(self) -> int | None:
        """Read the topic count implied by document-topic distributions, when available.

        Returns:
            int | None: The inferred topic count from the distribution vectors, or None if
                there are no distributions.

        Raises:
            LexosException: If document-topic distributions use inconsistent lengths.
        """
        if len(self.distributions) == 0:
            return None

        lengths = {len(distribution) for distribution in self.distributions}
        if len(lengths) > 1:
            raise LexosException(
                "Topic distribution lengths are inconsistent across documents; check `path_to_topic_distributions` format."
            )
        return next(iter(lengths))

    def _resolve_num_topics(self) -> int:
        """Determine the number of topics from metadata, topic keys, or distributions.

        Returns:
            int: The number of topics declared by the model.

        Raises:
            LexosException: If the model does not contain enough topic information to
                determine the count.
        """
        num_topics = self._read_metadata_num_topics()
        if num_topics is None:
            try:
                num_topics = len(self.topic_keys)
            except Exception:
                num_topics = None

        distribution_len = self._read_distribution_topic_count()
        if distribution_len is not None and num_topics is None:
            num_topics = distribution_len

        if num_topics is None:
            raise LexosException(
                "Model does not have topic information yet. Train or load a model first."
            )

        if distribution_len is not None and distribution_len != num_topics:
            raise LexosException(
                f"Mismatch between declared number of topics ({num_topics}) and distribution vector length ({distribution_len}). Check your training outputs."
            )

        return num_topics

    def _read_training_documents(self) -> list[str]:
        """Read the raw training data file and return the document texts.

        Returns:
            list[str]: The training documents in their original order.

        Raises:
            LexosException: If the model has not recorded a training data path.
        """
        if "path_to_training_data" not in self.metadata:
            raise LexosException(
                "No training data has been set. Please designate a path for `path_to_training_data` when you train your topic model."
            )

        with open(
            self.metadata["path_to_training_data"], "r", encoding="utf-8"
        ) as file:
            training_data = file.readlines()
        return [line.split("\t")[2].strip() for line in training_data]

    def _build_top_docs_frame(
        self, topic: int, training_data: list[str], metadata: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Build the DataFrame of top document scores for a given topic.

        Args:
            topic (int): The topic index to inspect.
            training_data (list[str]): The document texts loaded from the training data file.
            metadata (pd.DataFrame): Optional metadata aligned to the document order.

        Returns:
            pd.DataFrame: A frame containing document distributions and the optional metadata.
        """
        distribution_data = [
            (_distribution[topic], _document)
            for _distribution, _document in zip(self.distributions, training_data)
        ]
        frame = pd.DataFrame(distribution_data, columns=["Distribution", "Document"])
        frame.index.name = "Doc ID"

        if metadata is not None:
            frame = pd.concat([frame, metadata], axis=1)
        return frame

    @validate_call(config=model_config)
    def get_top_docs(
        self, topic=0, n=10, metadata: pd.DataFrame = None, as_str: bool = False
    ) -> pd.DataFrame | str:
        """Get the top n documents for a given topic.

        Args:
            topic (int): Topic number.
            n (int): Number of top documents to return.
            metadata (pd.DataFrame): Dataframe with the metadata in the same order as the training data (optional).
            as_str (bool): Whether to return the result as a string instead of a dataframe.

        Returns:
            A pd.DataFrame or str: A dataframe with the top n documents for the given topic, or a string representation of the dataframe.

        Notes:
            - The metadata must be in the same order as the training data.
            - The document text will get ellided by the maximum width of a pandas column. An easy way to see the full text is to set `as_str=True` and output the result with a print statement. You can also use the pandas API to extract the information with something like `top_docs.Document.tolist()`.
        """
        if not self._metadata_has([self.CANONICAL_DOC_TOPIC_KEY]):
            raise LexosException(
                "No topic distributions have been set. Please designate a path to the doc-topic distributions (e.g. `path_to_topic_distributions`) when you train your topic model."
            )

        training_data = self._read_training_documents()
        num_topics = self._resolve_num_topics()
        topic = self._validate_topic_index(topic, num_topics)

        frame = self._build_top_docs_frame(topic, training_data, metadata)
        sorted_frame = frame.sort_values(by="Distribution", ascending=False).head(n)

        if as_str:
            return sorted_frame.to_string()
        return sorted_frame

    def _select_topic_term_rows(
        self,
        topic_term_probability_dict: dict[int, dict[str, float]],
        topics: Optional[int | list[int]] = None,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Build the rows used for term-probability output and DataFrame export.

        Args:
            topic_term_probability_dict (dict[int, dict[str, float]]): The loaded topic-term
                probabilities keyed by topic index.
            topics (Optional[int | list[int]]): Topic index or indices to include.
            n (int): The number of terms to include per topic.

        Returns:
            list[dict[str, Any]]: The serialized rows for every selected topic.
        """
        if isinstance(topics, int):
            topics = [topics]

        rows: list[dict[str, Any]] = []
        for _topic, _term_probability_dict in topic_term_probability_dict.items():
            if topics is not None and _topic not in topics:
                continue
            for _term, _probability in sorted(
                _term_probability_dict.items(), key=lambda x: x[1], reverse=True
            )[:n]:
                rows.append(
                    {
                        "Topic": _topic,
                        "Term": _term,
                        "Probability": _probability,
                    }
                )
        return rows

    def _format_topic_term_string(
        self,
        topic_term_probability_dict: dict[int, dict[str, float]],
        topics: Optional[int | list[int]] = None,
        n: int = 5,
    ) -> str:
        """Format the legacy string view of topic-term probabilities.

        Args:
            topic_term_probability_dict (dict[int, dict[str, float]]): The loaded topic-term
                probabilities keyed by topic index.
            topics (Optional[int | list[int]]): Topic index or indices to include.
            n (int): The number of terms to include per topic.

        Returns:
            str: The legacy string output format used by the MALLET API.
        """
        result = ""
        for _topic, _term_probability_dict in topic_term_probability_dict.items():
            if topics is not None and _topic not in topics:
                continue
            result += f"Topic {_topic}\n"
            for _term, _probability in sorted(
                _term_probability_dict.items(), key=lambda x: x[1], reverse=True
            )[:n]:
                result += f"\t{_term}: {_probability}\n"
            result += "\n"
        return result

    @validate_call(config=model_config)
    def get_topic_term_probabilities(
        self, topics: Optional[int | list[int]] = None, n: int = 5, as_df: bool = False
    ) -> str | pd.DataFrame:
        """Get a string representation of the term distribution for a given topic.

        Args:
            topics (int | list[int]): Topic number. If None, get the probabilities for all topics.
            n (int): The number of keywords to display.
            as_df (bool): Whether to display the result as a string or a pandas DataFrame.

        Returns:
            str: A string representation of the term distribution for the given topic.
        """
        topic_term_probability_dict = self.load_topic_term_distributions()
        rows = self._select_topic_term_rows(topic_term_probability_dict, topics, n)

        if as_df:
            return pd.DataFrame(rows)
        return self._format_topic_term_string(topic_term_probability_dict, topics, n)

    def _prepare_termite_components(
        self,
        topics: Optional[int | list[int]] = None,
    ) -> tuple[pd.DataFrame, list[int]]:
        """Load and validate the topic-term data used for termite plotting.

        Args:
            topics (Optional[int | list[int]]): Topic index or indices to include.

        Returns:
            tuple[pd.DataFrame, list[int]]: The selected topic-term matrix and the list of
                requested topic indices.

        Raises:
            LexosException: If no topic-term probabilities are available.
            ValueError: If any requested topic is not available in the model.
        """
        topic_term_probability_dict = self.load_topic_term_distributions()
        components = (
            pd.DataFrame.from_dict(topic_term_probability_dict, orient="columns")
            .fillna(0.0)
            .sort_index()
        )

        if components.empty:
            raise LexosException("No topic-term probabilities are available to plot.")

        if isinstance(topics, int):
            topics = [topics]

        available_topics = list(components.columns)
        selected_topics = topics if topics is not None else sorted(available_topics)
        missing_topics = [
            topic for topic in selected_topics if topic not in available_topics
        ]
        if missing_topics:
            raise ValueError(
                f"Requested topics {missing_topics} are not available. "
                f"Available topics: {sorted(available_topics)}"
            )

        return components.loc[:, selected_topics], selected_topics

    def _resolve_highlight_labels(
        self,
        components: pd.DataFrame,
        highlight_topics: Optional[int | str | list[int | str]],
    ) -> list[str] | None:
        """Resolve highlighted topic labels for termite plotting.

        Args:
            components (pd.DataFrame): The selected topic-term matrix with topic labels as
                columns.
            highlight_topics (Optional[int | str | list[int | str]]): Topic labels or indices
                to highlight.

        Returns:
            list[str] | None: The resolved label list, or None when no highlights are set.

        Raises:
            ValueError: If a requested highlight topic is not in the selected data.
        """
        if highlight_topics is None:
            return None

        custom_labels = self.metadata.get("topic_labels", {})
        highlight_labels = []
        for topic in ensure_list(highlight_topics):
            if isinstance(topic, int):
                highlight_labels.append(custom_labels.get(str(topic), f"Topic {topic}"))
            else:
                highlight_labels.append(topic)

        missing_highlights = [
            topic for topic in highlight_labels if topic not in components.columns
        ]
        if missing_highlights:
            raise ValueError(
                f"Highlighted topics {missing_highlights} are not available in the selected data. "
                f"Available topics: {list(components.columns)}"
            )

        return highlight_labels

    @validate_call(config=model_config)
    def plot_termite(
        self,
        topics: Optional[int | list[int]] = None,
        highlight_topics: Optional[int | str | list[int | str]] = None,
        n_terms: int = 25,
        rank_terms_by: str = "max",
        sort_terms_by: str = "seriation",
        output_path: Optional[str] = None,
        rc_params: Optional[dict[str, Any]] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot a termite chart from MALLET topic-term outputs using textacy.

        Args:
            topics (Optional[int | list[int]]): Topic index or indices to include.
                If None, all available topics are used.
            highlight_topics (Optional[int | str | list[int | str]]): Topic labels
                or indices to highlight in the plot.
            n_terms (int): Number of top terms to include in the plot.
            rank_terms_by (str): Metric used by textacy to rank terms.
            sort_terms_by (str): Method used by textacy to sort selected terms.
            output_path (Optional[str]): If provided, save the figure to this path.
            rc_params (Optional[dict[str, Any]]): Matplotlib rc params passed to
                textacy's plotting helper.
            show (bool): Whether to show the plot.
            title (Optional[str]): Figure title.

        Returns:
            Any: A matplotlib axis containing the termite plot.

        Raises:
            LexosException: If textacy isn't installed or topic-term data is unavailable.
            ValueError: If requested topics or highlighted topics are invalid.
        """
        try:
            from textacy.viz.termite import termite_df_plot
        except Exception as e:
            raise LexosException(
                "textacy is required for termite plots. Please install textacy and try again."
            ) from e

        components, _ = self._prepare_termite_components(topics)

        custom_labels = self.metadata.get("topic_labels", {})
        components.columns = [
            custom_labels.get(str(topic), f"Topic {int(topic)}")
            for topic in components.columns
        ]

        highlight_labels = self._resolve_highlight_labels(components, highlight_topics)

        axis = termite_df_plot(
            components=components,
            highlight_topics=highlight_labels,
            n_terms=n_terms,
            rank_terms_by=rank_terms_by,
            sort_terms_by=sort_terms_by,
            save=output_path or False,
            rc_params=rc_params,
        )

        if title:
            axis.set_title(title, pad=20)

        if show:
            plt.show()
            return None

        return axis

    def _resolve_plotly_topic_selection(
        self,
        components: pd.DataFrame,
        topics: Optional[int | list[int]] = None,
    ) -> list[int]:
        """Select and validate the topic indices used for the Plotly termite plot.

        Args:
            components (pd.DataFrame): The topic-term matrix loaded from the model.
            topics (Optional[int | list[int]]): Topic index or indices to include.

        Returns:
            list[int]: The selected topic indices.

        Raises:
            ValueError: If any requested topic is not available in the data.
        """
        if isinstance(topics, int):
            topics = [topics]

        available_topics = list(components.columns)
        selected_topics = topics if topics is not None else sorted(available_topics)
        missing_topics = [
            topic for topic in selected_topics if topic not in available_topics
        ]
        if missing_topics:
            raise ValueError(
                f"Requested topics {missing_topics} are not available. "
                f"Available topics: {sorted(available_topics)}"
            )
        return selected_topics

    def _resolve_plotly_highlights(
        self,
        components: pd.DataFrame,
        highlight_topics: Optional[int | str | list[int | str]],
    ) -> set[str]:
        """Resolve highlighted topic labels for the Plotly termite plot.

        Args:
            components (pd.DataFrame): The selected topic-term matrix with topic labels as
                columns.
            highlight_topics (Optional[int | str | list[int | str]]): Topic labels or indices
                to highlight.

        Returns:
            set[str]: The set of highlighted labels.

        Raises:
            ValueError: If any highlight target is not in the selected data.
        """
        custom_labels = self.metadata.get("topic_labels", {})
        highlight_labels = set()
        if highlight_topics is None:
            return highlight_labels

        for topic in ensure_list(highlight_topics):
            if isinstance(topic, int):
                highlight_labels.add(custom_labels.get(str(topic), f"Topic {topic}"))
            else:
                highlight_labels.add(topic)

        missing_highlights = [
            topic for topic in highlight_labels if topic not in components.columns
        ]
        if missing_highlights:
            raise ValueError(
                f"Highlighted topics {missing_highlights} are not available in the selected data. "
                f"Available topics: {list(components.columns)}"
            )

        return highlight_labels

    def _sort_plotly_termites(
        self,
        components: pd.DataFrame,
        sort_terms_by: str,
    ) -> pd.DataFrame:
        """Sort the selected terms for the Plotly termite plot according to the chosen mode.

        Args:
            components (pd.DataFrame): The selected topic-term matrix.
            sort_terms_by (str): The requested sorting mode.

        Returns:
            pd.DataFrame: The sorted term matrix.
        """
        if sort_terms_by == "alphabetical":
            return components.sort_index()
        if sort_terms_by == "index":
            return components.sort_index(kind="stable")
        if sort_terms_by == "seriation":
            weights = components.values
            similarity = weights @ (weights - weights.min()).T
            laplacian = np.diag(similarity.sum(axis=1)) - similarity
            vals, vecs = np.linalg.eigh(laplacian)
            fiedler_idx = np.argsort(vals)[1]
            return components.iloc[np.argsort(vecs[:, fiedler_idx])]
        return components.loc[components.max(axis=1).sort_values(ascending=False).index]

    def _validate_plotly_termite_inputs(
        self,
        n_terms: int,
        marker_scale: float,
        rank_terms_by: str,
        sort_terms_by: str,
    ) -> tuple[str, str]:
        """Validate Plotly termite inputs and normalize case for sorting/ranking."""
        if n_terms <= 0:
            raise ValueError("`n_terms` must be greater than 0.")
        if marker_scale <= 0:
            raise ValueError("`marker_scale` must be greater than 0.")

        rank_terms_by = rank_terms_by.lower()
        sort_terms_by = sort_terms_by.lower()
        if rank_terms_by not in {"max", "mean", "var"}:
            raise ValueError("`rank_terms_by` must be one of: 'max', 'mean', 'var'.")
        if sort_terms_by not in {"weight", "alphabetical", "index", "seriation"}:
            raise ValueError(
                "`sort_terms_by` must be one of: 'weight', 'alphabetical', 'index', 'seriation'."
            )
        return rank_terms_by, sort_terms_by

    def _prepare_plotly_termite_components(
        self,
        topics: Optional[int | list[int]] = None,
    ) -> tuple[pd.DataFrame, list[int]]:
        """Load and prepare the topic-term matrix for a Plotly termite plot."""
        topic_term_probability_dict = self.load_topic_term_distributions()
        components = (
            pd.DataFrame.from_dict(topic_term_probability_dict, orient="columns")
            .fillna(0.0)
            .sort_index()
        )
        if components.empty:
            raise LexosException("No topic-term probabilities are available to plot.")

        selected_topics = self._resolve_plotly_topic_selection(components, topics)
        components = components.loc[:, selected_topics]
        custom_labels = self.metadata.get("topic_labels", {})
        components.columns = [
            custom_labels.get(str(topic), f"Topic {int(topic)}")
            for topic in components.columns
        ]
        return components, selected_topics

    def _build_plotly_termite_figure(
        self,
        components: pd.DataFrame,
        selected_topics: list[int],
        highlight_labels: set[str],
        n_terms: int,
        rank_terms_by: str,
        sort_terms_by: str,
        marker_scale: float,
        title: Optional[str],
        output_path: Optional[str],
        go: Any,
    ) -> Any:
        """Construct the Plotly termite figure from preprocessed topic-term data."""
        top_terms = (
            components.agg(rank_terms_by, axis=1)
            .sort_values(ascending=False)
            .head(n_terms)
            .index
        )
        components = components.loc[top_terms]
        components = self._sort_plotly_termites(components, sort_terms_by)

        df_melted = components.reset_index().melt(
            id_vars="index", var_name="Topic", value_name="Probability"
        )
        df_melted = df_melted.rename(columns={"index": "Term"})
        df_melted = df_melted[df_melted["Probability"] > 0]

        max_prob = df_melted["Probability"].max()
        term_order = components.index.tolist()
        topic_labels = list(components.columns)
        colors = [
            "#2596be" if topic in highlight_labels else "#d3d3d3"
            for topic in df_melted["Topic"]
        ]
        ticktext = [
            f'<span style="color:#2596be">{label}</span>'
            if label in highlight_labels
            else label
            for label in topic_labels
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_melted["Topic"],
                y=df_melted["Term"],
                mode="markers",
                marker={
                    "size": df_melted["Probability"],
                    "sizemode": "area",
                    "sizeref": max_prob / (marker_scale**2) if max_prob > 0 else 1,
                    "color": colors,
                    "line": {"color": "grey", "width": 1},
                    "sizemin": 2,
                },
                customdata=df_melted["Probability"],
                hovertemplate="Topic: %{x}<br>Term: %{y}<br>Probability: %{customdata:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
            xaxis_tickangle=-45,
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=max(400, n_terms * 33 + 150),
            width=max(400, len(selected_topics) * 60 + 150),
            margin={"l": 120, "r": 50, "t": 150, "b": 50},
            xaxis=dict(
                showgrid=True,
                gridcolor="lightgrey",
                side="top",
                tickmode="array",
                tickvals=topic_labels,
                ticktext=ticktext,
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
                mirror=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="lightgrey",
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
                mirror=True,
            ),
        )
        fig.update_yaxes(
            autorange="reversed",
            type="category",
            categoryorder="array",
            categoryarray=term_order,
        )

        if output_path:
            fig.write_html(output_path)
        return fig

    @validate_call(config=model_config)
    def plot_termite_plotly(
        self,
        topics: Optional[int | list[int]] = None,
        highlight_topics: Optional[int | str | list[int | str]] = None,
        n_terms: int = 25,
        rank_terms_by: str = "max",
        sort_terms_by: str = "weight",
        marker_scale: float = 25.0,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Any:
        """Create an interactive termite plot with Plotly.

        Args:
            topics (Optional[int | list[int]]): Topic index or indices to include.
                If None, all available topics are used.
            highlight_topics (Optional[int | str | list[int | str]]): Topic labels
                or indices to highlight in the plot.
            n_terms (int): Number of terms to include in the plot.
            rank_terms_by (str): Metric used to select top terms. Supported
                values are "max", "mean", and "var".
            sort_terms_by (str): Method used to order selected terms on the y-axis.
                Supported values are "weight", "alphabetical", "index", and "seriation".
            marker_scale (float): Multiplier used to map probabilities to marker size.
            title (str): Figure title.
            output_path (Optional[str]): If provided, save the plot to this path.

        Returns:
            Any: A Plotly Figure object containing the termite plot.

        Raises:
            LexosException: If plotly isn't installed or no topic-term data is available.
            ValueError: If inputs are invalid.
        """
        rank_terms_by, sort_terms_by = self._validate_plotly_termite_inputs(
            n_terms, marker_scale, rank_terms_by, sort_terms_by
        )
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise LexosException(
                "plotly is required for interactive termite plots. Please install plotly and try again."
            ) from e

        components, selected_topics = self._prepare_plotly_termite_components(topics)
        highlight_labels = self._resolve_plotly_highlights(components, highlight_topics)
        return self._build_plotly_termite_figure(
            components,
            selected_topics,
            highlight_labels,
            n_terms,
            rank_terms_by,
            sort_terms_by,
            marker_scale,
            title,
            output_path,
            go,
        )

    @validate_call(config=model_config)
    def import_dir(
        self,
        data_source: str | list[str],
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
    ) -> None:
        """Read training data from directories and save formatted training data file.

        Args:
            data_source (str | list[str]): A directory or list of directories to import.
            keep_sequence (bool): Whether to keep the word sequence in the documents.
            preserve_case (bool): Whether to preserve the case of the documents.
            remove_stopwords (bool): Whether to remove stopwords from the documents.
            use_pipe_from (Optional[str]): Path to a MALLET pipe file to use for importing.
            training_ids: Optional[list[int]]: A list of document ids designating a subset of the entire data set. If None, the entire dataset will be imported.
        """
        # Explicitly validate data_source to reject booleans
        if isinstance(data_source, bool):
            raise LexosException(
                "Invalid `data_source` argument: expected a directory path or list of paths, not a boolean."
            )
        training_data = read_dirs(ensure_list(data_source))
        self._import_training_data(
            training_data,
            path_to_training_data=None,
            keep_sequence=keep_sequence,
            remove_stopwords=remove_stopwords,
            preserve_case=preserve_case,
            use_pipe_from=use_pipe_from,
            training_ids=training_ids,
        )

    @validate_call(config=model_config)
    def import_docs(
        self,
        data_source: str | list[str],
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
    ) -> None:
        """Read training data from docs and save formatted training data file.

        Args:
            data_source (str | list[str]): A doc or list of docs to import.
            keep_sequence (bool): Whether to keep the word sequence in the documents.
            preserve_case (bool): Whether to preserve the case of the documents.
            remove_stopwords (bool): Whether to remove stopwords from the documents.
            use_pipe_from (Optional[str]): Path to a MALLET pipe file to use for importing.
            training_ids: Optional[list[int]]: A list of document ids designating a subset of the entire data set. If None, the entire dataset will be imported.
        """
        if isinstance(data_source, bool):
            raise LexosException(
                "Invalid `data_source` argument: expected a doc or list of docs, not a boolean."
            )
        docs = ensure_list(data_source)
        training_data = [
            f"{i}\t\t{doc.text}" if isinstance(doc, Doc) else f"{i}\t\t{doc}"
            for i, doc in enumerate(docs)
        ]
        self._import_training_data(
            training_data,
            path_to_training_data=None,
            keep_sequence=keep_sequence,
            remove_stopwords=remove_stopwords,
            preserve_case=preserve_case,
            use_pipe_from=use_pipe_from,
            training_ids=training_ids,
        )

    @validate_call(config=model_config)
    def import_file(
        self,
        data_source: str | list[str],
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
    ) -> None:
        """Read training data from file and save formatted training data file.

        Args:
            data_source (str | list[str]): A file or list of files to import.
            keep_sequence (bool): Whether to keep the word sequence in the documents.
            preserve_case (bool): Whether to preserve the case of the documents.
            remove_stopwords (bool): Whether to remove stopwords from the documents.
            use_pipe_from (Optional[str]): Path to a MALLET pipe file to use for importing.
            training_ids: Optional[list[int]]: A list of document ids designating a subset of the entire data set. If None, the entire dataset will be imported.
        """
        if isinstance(data_source, bool):
            raise LexosException(
                "Invalid `data_source` argument: expected a file path or list of paths, not a boolean."
            )
        data_sources = ensure_list(data_source)
        training_data = []
        for source in data_sources:
            training_data.extend(read_file(source))
        self._import_training_data(
            training_data,
            path_to_training_data=None,
            keep_sequence=keep_sequence,
            remove_stopwords=remove_stopwords,
            preserve_case=preserve_case,
            use_pipe_from=use_pipe_from,
            training_ids=training_ids,
        )

    def _read_term_weight_rows(
        self, term_weight_path: str
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Read and validate the raw term-weight rows from the model output file.

        Args:
            term_weight_path (str): The path to the MALLET term-weight file.

        Returns:
            tuple[dict[str, dict[str, float]], dict[str, float]]: The raw topic-term weights and
                the per-topic totals used to normalize them into probabilities.

        Raises:
            ValueError: If a row is malformed or contains an invalid numeric weight.
        """
        topic_term_weight_dict: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        topic_sum_dict: dict[str, float] = defaultdict(float)

        with open(term_weight_path, "r") as file:
            for line in file:
                if not line.strip():
                    continue

                parts = line.strip().split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        f"Malformed line in term weights file: '{line.strip()}'"
                    )

                topic, term, weight = parts
                try:
                    weight_value = float(weight)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid weight value '{weight}' in line: '{line.strip()}'"
                    ) from exc

                topic_term_weight_dict[topic][term] = weight_value
                topic_sum_dict[topic] += weight_value

        return topic_term_weight_dict, topic_sum_dict

    def load_topic_term_distributions(self) -> dict[str, float]:
        """Load the topic-term distributions from a file.

        Returns:
            dict[str, float]: A dictionary of all topic-term distributions.
        """
        term_weight_path = self._metadata_get([self.CANONICAL_TERM_WEIGHTS_KEY])
        if term_weight_path is None:
            raise LexosException(
                f"No term weights have been set. Please designate a path to the term weights file (e.g. `{self.CANONICAL_TERM_WEIGHTS_KEY}`) when you train your topic model."
            )

        try:
            topic_term_weight_dict, topic_sum_dict = self._read_term_weight_rows(
                term_weight_path
            )
        except FileNotFoundError:
            raise

        topic_term_probability_dict = defaultdict(lambda: defaultdict(float))
        for topic, term_weight_dict in topic_term_weight_dict.items():
            for term, weight in term_weight_dict.items():
                topic_term_probability_dict[int(topic)][term] = (
                    weight / topic_sum_dict[topic]
                )

        return topic_term_probability_dict

    def _normalize_boxplot_topics(
        self, topics: Optional[int | list[int]], num_topics: int
    ) -> list[int]:
        """Normalize topic selections for category boxplots.

        Args:
            topics (Optional[int | list[int]]): The selected topic index or indices.
            num_topics (int): The number of available topics.

        Returns:
            list[int]: The list of topic indices to plot.
        """
        if topics is None:
            return list(range(num_topics))
        if isinstance(topics, int):
            return [topics]
        return topics

    def _build_boxplot_dataframe(
        self,
        categories: list[str],
        distributions: list[list[float]],
        topic: int,
        topic_header: str,
        target_labels: Optional[list[str]],
    ) -> pd.DataFrame:
        """Build the per-topic dataframe used for a category boxplot.

        Args:
            categories (list[str]): Category labels aligned with the distributions.
            distributions (list[list[float]]): The topic distributions for each category.
            topic (int): The topic index used for plotting.
            topic_header (str): The visual title for the topic.
            target_labels (Optional[list[str]]): Optional category filters.

        Returns:
            pd.DataFrame: The data prepared for seaborn boxplot drawing.
        """
        rows = []
        for label, distribution in zip(categories, distributions):
            if target_labels and label not in target_labels:
                continue
            rows.append(
                {
                    "Probability": float(distribution[topic]),
                    "Category": label,
                    "Topic": topic_header,
                }
            )
        return pd.DataFrame(rows)

    def _render_boxplot_overlay(
        self,
        ax: Any,
        overlay: Optional[str],
        df_to_plot: pd.DataFrame,
        overlay_kws: Optional[dict[str, Any]],
    ) -> None:
        """Render the optional strip/swarm overlay of raw data points on a boxplot.

        Args:
            ax (Any): The matplotlib axes to draw on.
            overlay (Optional[str]): The overlay mode to use.
            df_to_plot (pd.DataFrame): The boxplot data.
            overlay_kws (Optional[dict[str, Any]]): Extra arguments for the overlay plot.
        """
        if overlay not in ("strip", "swarm", "none", None):
            raise LexosException(
                "Invalid `overlay` argument: expected 'strip', 'swarm', or 'none'."
            )

        overlay_kws = dict(overlay_kws or {})
        try:
            if overlay == "strip" or overlay is None:
                sns.stripplot(
                    data=df_to_plot,
                    x="Category",
                    y="Probability",
                    color=overlay_kws.pop("color", "black"),
                    size=overlay_kws.pop("size", 4),
                    jitter=overlay_kws.pop("jitter", True),
                    ax=ax,
                    **overlay_kws,
                )
            elif overlay == "swarm":
                sns.swarmplot(
                    data=df_to_plot,
                    x="Category",
                    y="Probability",
                    color=overlay_kws.pop("color", "black"),
                    size=overlay_kws.pop("size", 4),
                    ax=ax,
                    **overlay_kws,
                )
        except Exception:
            pass

    def _resolve_topic_header(
        self,
        topic: int,
        topic_keys: list[list[str]],
        num_keys: int,
    ) -> str:
        """Build the display title for a topic using either custom labels or default labels.

        Args:
            topic (int): The topic index.
            topic_keys (list[list[str]]): The model's topic-key rows.
            num_keys (int): The number of keywords to include in the title.

        Returns:
            str: A label including the topic header and the leading keywords.
        """
        keywords = " ".join(topic_keys[topic][2].split()[:num_keys])
        custom_labels = self.metadata.get("topic_labels")
        topic_label = str(topic)
        if custom_labels and topic_label in custom_labels:
            return f"{custom_labels[topic_label]}: {keywords}"
        return f"Topic {topic}: {keywords}"

    def _save_plot_figure(
        self, fig: Figure, output_path: Optional[str], topic: int
    ) -> None:
        """Save a figure with a topic-specific suffix when an output path is provided.

        Args:
            fig (Figure): The figure to save.
            output_path (Optional[str]): The root output path.
            topic (int): The topic index used in the filename.
        """
        if not output_path:
            return
        path = Path(output_path)
        save_path = f"{path.parent / path.stem}_topic{topic}{path.suffix}"
        fig.savefig(save_path)

    def _plot_boxplot_for_topic(
        self,
        categories: list[str],
        distributions: list[list[float]],
        topic: int,
        topic_keys: list[list[str]],
        target_labels: Optional[list[str]],
        output_path: Optional[str],
        num_keys: int,
        figsize: Optional[tuple[int, int]],
        font_scale: Optional[float],
        color: Optional[ColorType],
        show: Optional[bool],
        title: Optional[str],
        overlay: Optional[str],
        overlay_kws: Optional[dict[str, Any]],
    ) -> Figure:
        """Render a single topic boxplot and return its matplotlib figure."""
        topic_header = self._resolve_topic_header(topic, topic_keys, num_keys)
        df_to_plot = self._build_boxplot_dataframe(
            categories,
            distributions,
            topic,
            topic_header,
            target_labels,
        )

        sns.set_theme(style="ticks", font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
        sns.boxplot(
            data=df_to_plot,
            x="Category",
            y="Probability",
            color=color,
            ax=ax,
            showmeans=True,
        )
        self._render_boxplot_overlay(ax, overlay, df_to_plot, overlay_kws)
        sns.despine()
        plt.xticks(rotation=45, ha="right")
        if title is None:
            ax.set_title(topic_header)
        else:
            fig.suptitle(title)
        plt.tight_layout()
        self._save_plot_figure(fig, output_path, topic)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    @validate_call(config=model_config)
    def plot_categories_by_topic_boxplots(
        self,
        categories: list[str],
        topics: Optional[int | list[int]] = None,
        output_path: Optional[str] = None,
        target_labels: Optional[list[str]] = None,
        num_keys: int = 5,
        figsize: Optional[tuple[int, int]] = (6, 6),
        font_scale: Optional[float] = 1.2,
        color: Optional[ColorType] = "lightblue",
        show: Optional[bool] = True,
        title: Optional[str] = None,
        overlay: Optional[str] = "strip",
        overlay_kws: Optional[dict[str, Any]] = None,
        topic_distributions: Optional[list[list[float]]] = None,
    ) -> Figure | list[Figure]:
        """Plot boxplots showing the distribution of topic probabilities for each category.

        Args:
            categories (list[str]): The labels to use for the categories.
            topics (int | list[int]): The index of the topic to plot.
            output_path (str): The path to save the figure.
            target_labels (list[str]): Unique labels for categories to classify.
            num_keys (int): The number of keywords to display.
            figsize: (Optional[tuple[int, int]]): The dimensions of the figure.
            font_scale (Optional[float]): The font scale for the figure.
            color (Optional[ColorType]): The color to use for the heatmap boxes. A matplotlib ColorType name or object.
            show (Optional[bool]): Whether to show the figure.
            title (Optional[str]): Optional figure title. If not supplied, each plot will use a default title of
                `Topic {topic}: {keywords}`.
            overlay (Optional[str]): How to display the individual points overlaid on each boxplot. Supported
                values are 'strip' (default), 'swarm', or 'none'.
            overlay_kws (Optional[dict]): Keyword arguments passed to the chosen overlay plotting method
                (`seaborn.stripplot` or `seaborn.swarmplot`).

        Returns:
            Figure | list[Figure]: The boxplot showing the topic associations by category.
        """
        topic_keys = self.topic_keys
        topics = self._normalize_boxplot_topics(topics, len(topic_keys))
        target_labels = target_labels or list(set(categories))
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )
        figs = []

        for topic in topics:
            figs.append(
                self._plot_boxplot_for_topic(
                    categories,
                    distributions,
                    topic,
                    topic_keys,
                    target_labels,
                    output_path,
                    num_keys,
                    figsize,
                    font_scale,
                    color,
                    show,
                    title,
                    overlay,
                    overlay_kws,
                )
            )

        if show:
            return None
        return figs[0] if len(figs) == 1 else figs

    def _resolve_heatmap_topic_label(
        self,
        topic_index: int,
        topic_keys: list[list[str]],
        num_keys: int,
    ) -> str:
        """Build the display label used for a topic in a heatmap.

        Args:
            topic_index (int): The topic index.
            topic_keys (list[list[str]]): The topic keyword rows.
            num_keys (int): The number of keywords to include.

        Returns:
            str: The display label for the topic column.
        """
        keywords = (
            ""
            if topic_index >= len(topic_keys)
            else " ".join(topic_keys[topic_index][2].split()[:num_keys])
        )
        custom_labels = self.metadata.get("topic_labels")
        topic_display_name = (
            custom_labels.get(str(topic_index), f"Topic {topic_index}")
            if custom_labels and str(topic_index) in custom_labels
            else f"Topic {topic_index}"
        )

        if num_keys and keywords:
            return f"{topic_display_name}: {keywords}"
        return topic_display_name

    def _build_heatmap_rows(
        self,
        categories: list[str],
        distributions: list[list[float]],
        topic_keys: list[list[str]],
        target_labels: Optional[list[str]],
        num_keys: int,
    ) -> list[dict[str, float | str]]:
        """Build the rows used for the topic-by-category heatmap.

        Args:
            categories (list[str]): The category labels.
            distributions (list[list[float]]): The per-category probability vectors.
            topic_keys (list[list[str]]): The topic-key rows.
            target_labels (Optional[list[str]]): Optional filters for categories.
            num_keys (int): The number of term labels to include in each topic label.

        Returns:
            list[dict[str, float | str]]: The row data for the heatmap DataFrame.
        """
        rows: list[dict[str, float | str]] = []
        for category_label, distribution in zip(categories, distributions):
            if target_labels and category_label not in target_labels:
                continue
            for topic_index, probability in enumerate(distribution):
                rows.append(
                    {
                        "Probability": float(probability),
                        "Category": category_label,
                        "Topic": self._resolve_heatmap_topic_label(
                            topic_index, topic_keys, num_keys
                        ),
                    }
                )
        return rows

    def _sort_heatmap_columns(self, df_norm_col: pd.DataFrame) -> pd.DataFrame:
        """Sort the topic columns in a heatmap by topic index when possible.

        Args:
            df_norm_col (pd.DataFrame): The normalized heatmap DataFrame.

        Returns:
            pd.DataFrame: The reordered heatmap DataFrame.
        """

        def _topic_key(col):
            try:
                match = re.match(r"Topic\s+(\d+)", str(col))
                if match:
                    return (0, int(match.group(1)))
            except Exception:
                pass
            return (1, str(col))

        try:
            return df_norm_col[sorted(list(df_norm_col.columns), key=_topic_key)]
        except Exception:
            return df_norm_col

    @validate_call(config=model_config)
    def plot_categories_by_topics_heatmap(
        self,
        categories: list[str],
        output_path: Path | str = None,
        target_labels: list[str] = None,
        num_keys: int = 5,
        figsize: Optional[tuple[int, int]] = None,
        font_scale: Optional[float] = 1.2,
        cmap: Optional[ColorType] = sns.cm.rocket_r,
        show: Optional[bool] = True,
        title: Optional[str] = None,
        topic_distributions: Optional[list[list[float]]] = None,
    ) -> Figure:
        """Plot heatmap showing topics by category.

        Args:
            categories (list[str]): The categories to use to classify topics.
            output_path (Path | str): The path to save the figure.
            target_labels (list[str]): Unique labels for categories to classify.
            num_keys (int): The number of keywords to display.
            figsize: (Optional[tuple[int, int]]): The dimensions of the figure.
            font_scale (Optional[float]): The font scale for the figure.
            cmap (Optional[ColorType]): The colormap to use for the heatmap. A matplotlib colormap name or object, or list of colors.
            show (Optional[bool]): Whether to show the figure.
            title (Optional[str]): Optional title for the figure. If not supplied, defaults to "Topics by Category (N=x)".

        Returns:
            Figure: The heatmap showing the topic associations by category.
        """
        topic_keys = self.topic_keys
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )

        rows = self._build_heatmap_rows(
            categories,
            distributions,
            topic_keys,
            target_labels,
            num_keys,
        )
        df_to_plot = pd.DataFrame(rows)
        df_wide = df_to_plot.pivot_table(
            index="Category", columns="Topic", values="Probability"
        )
        df_norm_col = (df_wide - df_wide.mean()) / df_wide.std()
        df_norm_col = self._sort_heatmap_columns(df_norm_col)

        sns.set_theme(style="ticks", font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
        ax = sns.heatmap(df_norm_col, cmap=cmap, ax=ax)

        if title is None:
            try:
                num_topics = len(df_norm_col.columns)
            except Exception:
                num_topics = None
            if num_topics is not None:
                title = f"Topics by Category ({num_topics} Topics)"
            else:
                title = "Topics by Category"
        fig.suptitle(title)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=30, ha="left")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if output_path:
            plt.savefig(output_path)
        if show:
            plt.show()
            return None
        plt.close()
        return fig

    def _resolve_cloud_round_radius(self, round_mask: Any) -> int:
        """Normalize the round-mask option into the integer radius expected by MultiCloud.

        Args:
            round_mask (Any): A boolean or integer-like radius specification.

        Returns:
            int: The normalized round-mask radius.

        Raises:
            LexosException: If the value cannot be interpreted as a boolean or integer radius.
        """
        if isinstance(round_mask, bool):
            return 120 if round_mask else 0

        try:
            return int(round_mask) if round_mask is not None else 0
        except Exception as exc:
            raise LexosException(
                "Invalid `round_mask` argument: expected a boolean or integer radius."
            ) from exc

    def _resolve_cloud_labels(self, df: pd.DataFrame) -> list[str]:
        """Build the display labels for each topic cloud.

        Args:
            df (pd.DataFrame): The topic-term probability matrix with one row per topic.

        Returns:
            list[str]: The topic labels used by MultiCloud.
        """
        custom_labels = self.metadata.get("topic_labels")
        labels = []
        for i in range(len(df)):
            topic_id = str(i)
            labels.append(
                custom_labels[topic_id]
                if custom_labels and topic_id in custom_labels
                else f"Topic {i}"
            )
        return labels

    def _resolve_cloud_title(self, df: pd.DataFrame, title: Optional[str]) -> str:
        """Resolve the title for the topic-cloud display.

        Args:
            df (pd.DataFrame): The topic-term probability matrix.
            title (Optional[str]): An explicitly provided title.

        Returns:
            str: The final title for the MultiCloud figure.
        """
        if title is not None:
            return title
        try:
            num_topics = len(df)
        except Exception:
            return "Topic Clouds"
        return f"Topic Clouds ({num_topics} topics)" if num_topics else "Topic Clouds"

    @validate_call(config=model_config)
    def topic_clouds(
        self,
        topics: Optional[int | list[int]] = None,
        max_terms: Optional[int] = 30,
        figsize: Optional[tuple[int, int]] = (10, 10),
        output_path: Optional[str] = None,
        show: Optional[bool] = True,
        round_mask: Any = True,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Figure:
        """Get a `MultiCloud` object for the topic-term distributions.

        This method converts the internal topic-term probability dictionary
        to a DataFrame (topics as rows) and constructs a `lexos.visualization.cloud.MultiCloud`
        instance for visualization.

        Parameters:
            topics (Optional[int | list[int]]): Topics to include (rows). If None, show all.
            max_terms (Optional[int]): Maximum number of top keywords to display per topic. Maps
                to the `limit` parameter of `MultiCloud` and `max_words` in `opts` when not set.
            figsize (Optional[tuple[int, int]]): Size of the overall figure.
            output_path (Optional[str]): If provided, the MultiCloud figure will be saved to this path.
            show (Optional[bool]): If True, the figure will be displayed in the current environment.
            round_mask (bool|int|str): Either a boolean indicating whether to use a default circular mask
                (True maps to radius 120; False disables mask), or an integer radius to use for a custom
                mask. Strings containing integer values will be converted. Passing invalid values will
                raise a `LexosException`.
            title (Optional[str]): Optional title for the overall MultiCloud figure. If None, a default
                of "Topic Clouds (N topics)" will be used.
            **kwargs (Any): Additional keyword arguments. Use `opts` to pass wordcloud options for each cloud.

        Returns:
            Figure: If `show` is False, returns a Matplotlib Figure object created by `MultiCloud`.
            Otherwise returns None after displaying the figure.

        Notes:
            The labels displayed above each word cloud will be of the form `Topic 0`,
            `Topic 1`, etc.; keywords are not included in the labels to keep the
            display uncluttered.
        """
        sns.set_theme()

        topic_term_probability_dict = self.load_topic_term_distributions()
        df = pd.DataFrame.from_dict(topic_term_probability_dict, orient="index").fillna(
            0
        )
        if topics is not None:
            df = df.iloc[ensure_list(topics)]

        opts = kwargs.get("opts", {})
        opts.setdefault("background_color", "white")
        if "max_words" not in opts and max_terms is not None:
            opts["max_words"] = max_terms

        round_radius = self._resolve_cloud_round_radius(round_mask)
        labels = self._resolve_cloud_labels(df)

        figure_opts = kwargs.get("figure_opts", {})
        figure_opts.setdefault("facecolor", "white")

        mc = MultiCloud(
            data=df,
            limit=max_terms,
            figsize=figsize,
            opts=opts,
            round=round_radius,
            labels=labels,
            figure_opts=figure_opts,
            title=self._resolve_cloud_title(df, title),
        )

        if output_path:
            mc.save(output_path)

        if show:
            mc.show()
            return None
        return mc.fig

    def _validate_time_series_inputs(
        self,
        times: list,
        distributions: Optional[list[list[float]]],
        topic_index: int,
    ) -> None:
        """Validate the inputs needed to render a time-series topic plot.

        Args:
            times (list): Time points corresponding to each document.
            distributions (Optional[list[list[float]]]): The topic distributions per document.
            topic_index (int): The topic to plot.

        Raises:
            LexosException: If there are no distributions or the length does not match.
            ValueError: If the topic index is negative.
        """
        if distributions is None or len(distributions) == 0:
            raise LexosException("No topic distributions available to plot.")
        if topic_index < 0:
            raise ValueError("topic_index must be a non-negative integer")
        if len(times) != len(distributions):
            raise LexosException(
                "Length mismatch: 'times' must be the same length as topic_distributions"
            )

    def _build_time_series_rows(
        self,
        times: list,
        distributions: list[list[float]],
        topic_index: int,
    ) -> pd.DataFrame:
        """Build the DataFrame used for the topic-over-time line plot.

        Args:
            times (list): Time points corresponding to each document.
            distributions (list[list[float]]): The topic probabilities for each document.
            topic_index (int): The topic index to plot.

        Returns:
            pd.DataFrame: The rows used in the time-series plot.

        Raises:
            LexosException: If no documents contain the requested topic.
        """
        rows = []
        for j, distribution in enumerate(distributions):
            if len(distribution) <= topic_index:
                continue
            rows.append({"Probability": distribution[topic_index], "Time": times[j]})
        if len(rows) == 0:
            raise LexosException(f"No data found for topic index {topic_index}")
        return pd.DataFrame(rows)

    def _resolve_time_series_title(
        self,
        topic_keys: list[list[str]],
        topic_index: int,
        title: Optional[str],
    ) -> Optional[str]:
        """Resolve the title for the topic-over-time plot.

        Args:
            topic_keys (list[list[str]]): The topic-key rows.
            topic_index (int): The topic index.
            title (Optional[str]): An explicit title override.

        Returns:
            Optional[str]: The final title or a simple topic fallback.
        """
        if title is not None:
            return title

        custom_labels = self.metadata.get("topic_labels", {})
        try:
            topic_id = str(topic_keys[topic_index][0])
            topic_label = custom_labels.get(topic_id, f"Topic {topic_id}")
            if len(topic_keys[topic_index]) < 3:
                return f"Topic {topic_index}"
            keywords = " ".join(topic_keys[topic_index][2].split()[:5])
            return f"{topic_label}: {keywords}"
        except Exception:
            return custom_labels.get(str(topic_index), f"Topic {topic_index}")

    @validate_call(config=model_config)
    def plot_topics_over_time(
        self,
        times: list,
        topic_index: int,
        topic_distributions: Optional[list[list[float]]] = None,
        topic_keys: Optional[list[list[str]]] = None,
        output_path: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = (7, 2.5),
        font_scale: Optional[float] = 1.2,
        color: Optional[ColorType] = "cornflowerblue",
        show: Optional[bool] = True,
        title: Optional[str] = None,
    ) -> Figure | None:
        """Plot the probability of a topic over time.

        Args:
            times (list): List of time points corresponding to each document (must be same length as topic_distributions).
            topic_index (int): The index of the topic to plot.
            topic_distributions (Optional[list[list[float]]]): If provided, a list of topic distributions per document. If None, uses `self.distributions`.
            topic_keys (Optional[list[list[str]]]): If provided, a list of topic keys; otherwise uses `self.topic_keys`.
            output_path (Optional[str]): Path to save the output plot. If None the plot is shown but not saved.
            figsize (Optional[tuple[int,int]]): Figure size.
            font_scale (Optional[float]): Seaborn font_scale.
            color (Optional[ColorType]): Line color.
            show (Optional[bool]): Whether to display the figure.
            title (Optional[str]): Optional figure title. Will default to the topic's keywords if not supplied.

        Returns:
            Figure | None: The matplotlib figure if `show=False`, otherwise None.
        """
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )
        topic_keys = topic_keys if topic_keys is not None else self.topic_keys

        self._validate_time_series_inputs(times, distributions, topic_index)
        data_df = self._build_time_series_rows(times, distributions, topic_index)

        title = self._resolve_time_series_title(topic_keys, topic_index, title)

        sns.set_theme(style="ticks", font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=data_df, x="Time", y="Probability", color=color, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Topic Probability")

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        sns.despine()
        if output_path:
            fig.savefig(output_path)
        if show:
            plt.show()
            return None
        return fig

    def _normalize_train_flag_value(self, value: Any) -> Optional[str]:
        """Normalize a MALLET training flag value to a CLI-safe string.

        Args:
            value (Any): The raw flag value.

        Returns:
            Optional[str]: The normalized value, or None if the flag is unset.
        """
        if not value:
            return None
        if isinstance(value, str) and len(Path(value).parts) == 1:
            return str(Path(self.model_dir) / value)
        return str(value)

    def _record_train_output_metadata(self, key: str, value: str) -> None:
        """Persist canonical metadata locations for key output files.

        Args:
            key (str): The MALLET flag name.
            value (str): The output file path.
        """
        mapping = {
            "output-doc-topics": self.CANONICAL_DOC_TOPIC_KEY,
            "topic-word-weights-file": self.CANONICAL_TERM_WEIGHTS_KEY,
            "output-topic-keys": self.CANONICAL_TOPIC_KEYS_KEY,
            "inferencer-filename": self.CANONICAL_INFERENCER_KEY,
        }
        if key in mapping:
            self.metadata[mapping[key]] = value

    def _build_train_command(
        self,
        num_topics: int,
        num_iterations: Optional[int],
        optimize_interval: Optional[int],
        path_to_state: Optional[str],
        path_to_topic_keys: Optional[str],
        path_to_topic_distributions: Optional[str],
        path_to_term_weights: Optional[str],
        path_to_diagnostics: Optional[str],
        path_to_inferencer: Optional[str],
    ) -> list[str]:
        """Build the MALLET train-topics command and record canonical output metadata.

        Args:
            num_topics (int): The number of topics to train.
            num_iterations (Optional[int]): The number of training iterations.
            optimize_interval (Optional[int]): The optimization interval.
            path_to_state (Optional[str]): State output path.
            path_to_topic_keys (Optional[str]): Topic-key output path.
            path_to_topic_distributions (Optional[str]): Document-topic output path.
            path_to_term_weights (Optional[str]): Topic-word weights output path.
            path_to_diagnostics (Optional[str]): Diagnostics output path.
            path_to_inferencer (Optional[str]): Inferencer output path.

        Returns:
            list[str]: The full MALLET command to run.
        """
        path_to_formatted_training_data = str(
            Path(self.model_dir) / "training_data.mallet"
        )
        cmd = [self.path_to_mallet or "mallet", "train-topics"]
        flags = {
            "input": path_to_formatted_training_data,
            "num-topics": num_topics,
            "num-iterations": num_iterations,
            "output-state": path_to_state
            or str(Path(self.model_dir) / "topic-state.gz"),
            "output-topic-keys": path_to_topic_keys
            or str(Path(self.model_dir) / "topic-keys.txt"),
            "output-doc-topics": path_to_topic_distributions
            or str(Path(self.model_dir) / "doc-topic.txt"),
            "topic-word-weights-file": path_to_term_weights
            or str(Path(self.model_dir) / "topic-weights.txt"),
            "diagnostics-file": path_to_diagnostics
            or str(Path(self.model_dir) / "diagnostics.xml"),
            "inferencer-filename": path_to_inferencer
            or str(Path(self.model_dir) / "inferencer.mallet"),
            "optimize-interval": optimize_interval,
        }

        for key, value in flags.items():
            normalized_value = self._normalize_train_flag_value(value)
            if normalized_value is None:
                continue
            cmd.extend([f"--{key}", normalized_value])
            self._record_train_output_metadata(key, normalized_value)

        return cmd

    def _record_train_metadata(
        self,
        flags: dict[str, Any],
        cmd: list[str],
        num_topics: int,
        num_iterations: Optional[int],
        optimize_interval: Optional[int],
    ) -> None:
        """Persist the training metadata used by downstream inference and inspection.

        Args:
            flags (dict[str, Any]): The training flags passed to MALLET.
            cmd (list[str]): The command executed to train the model.
            num_topics (int): Number of topics trained.
            num_iterations (Optional[int]): Training iterations.
            optimize_interval (Optional[int]): Optimization interval.
        """
        mapping = {
            "output-doc-topics": self.CANONICAL_DOC_TOPIC_KEY,
            "topic-word-weights-file": self.CANONICAL_TERM_WEIGHTS_KEY,
            "output-topic-keys": self.CANONICAL_TOPIC_KEYS_KEY,
            "inferencer-filename": self.CANONICAL_INFERENCER_KEY,
        }
        for key, value in flags.items():
            if key not in ["num-topics", "optimize-interval"]:
                if key in mapping:
                    continue
                self.metadata[f"path_to_{key.replace('-', '_')}"] = value
        self.metadata["training_command"] = cmd
        self.metadata["num_topics"] = num_topics
        self.metadata["num_iterations"] = num_iterations
        self.metadata["optimize_interval"] = optimize_interval

        with open(self.model_dir / "meta.json", "w") as f:
            f.write(json.dumps(self.metadata))

    @validate_call(config=model_config)
    def train(
        self,
        num_topics: int = 20,
        num_iterations: Optional[int] = 100,
        optimize_interval: Optional[int] = 10,
        verbose: Optional[bool] = True,
        # Common output paths: caller may pass canonical keys or path_to_* names
        path_to_state: Optional[str] = None,
        path_to_topic_keys: Optional[str] = None,
        path_to_topic_distributions: Optional[str] = None,
        path_to_term_weights: Optional[str] = None,
        path_to_diagnostics: Optional[str] = None,
        path_to_inferencer: Optional[str] = None,
    ) -> None:
        """Train the topic model using MALLET.

        Args:
            num_topics (int): The number of topics to train.
            num_iterations (int): The number of iterations to train for.
            optimize_interval (int): The interval at which to optimize the model.
            verbose (bool): Whether to print the MALLET output.
            path_to_state (Optional[str]): Optional output filename for saving the topic state file. If not provided, defaults to `model_dir/topic-state.gz`.
            path_to_topic_keys (Optional[str]): Optional output filename for saving the topic keys file. If not provided, defaults to `model_dir/topic-keys.txt`.
            path_to_topic_distributions (Optional[str]): Optional output filename for saving the document-topic distributions. If not provided, defaults to `model_dir/doc-topic.txt`.
            path_to_term_weights (Optional[str]): Optional output filename for saving the topic-word weights. If not provided, defaults to `model_dir/topic-weights.txt`.
            path_to_diagnostics (Optional[str]): Optional output filename for saving the diagnostics file. If not provided, defaults to `model_dir/diagnostics.xml`.
            path_to_inferencer (Optional[str]): Optional output filename for saving a trained inferencer object
                that can be used with `mallet infer-topics`. If not provided, defaults to
                `model_dir/inferencer.mallet`.
        """
        flags = {
            "input": str(Path(self.model_dir) / "training_data.mallet"),
            "num-topics": num_topics,
            "num-iterations": num_iterations,
            "output-state": path_to_state
            or str(Path(self.model_dir) / "topic-state.gz"),
            "output-topic-keys": path_to_topic_keys
            or str(Path(self.model_dir) / "topic-keys.txt"),
            "output-doc-topics": path_to_topic_distributions
            or str(Path(self.model_dir) / "doc-topic.txt"),
            "topic-word-weights-file": path_to_term_weights
            or str(Path(self.model_dir) / "topic-weights.txt"),
            "diagnostics-file": path_to_diagnostics
            or str(Path(self.model_dir) / "diagnostics.xml"),
            "inferencer-filename": path_to_inferencer
            or str(Path(self.model_dir) / "inferencer.mallet"),
            "optimize-interval": optimize_interval,
        }
        cmd = self._build_train_command(
            num_topics,
            num_iterations,
            optimize_interval,
            path_to_state,
            path_to_topic_keys,
            path_to_topic_distributions,
            path_to_term_weights,
            path_to_diagnostics,
            path_to_inferencer,
        )
        self._track_progress(cmd, num_iterations, verbose)
        self._record_train_metadata(
            flags, cmd, num_topics, num_iterations, optimize_interval
        )
        msg.good("Complete")

    def _validate_inference_docs(
        self, docs: list[str] | Path | str
    ) -> tuple[str, Optional[str]]:
        """Validate incoming inference docs and resolve raw/input paths."""
        if isinstance(docs, (Path, str)) and Path(docs).is_file():
            return str(docs), None

        if isinstance(docs, bool) or not isinstance(docs, list):
            raise LexosException(
                "Invalid `docs` argument: expected a list of strings or a path to a file."
            )

        input_file = str(Path(self.model_dir) / "infer_input.txt")
        with open(input_file, "w", encoding="utf-8") as fh:
            for i, doc in enumerate(docs):
                if isinstance(doc, bool) or not isinstance(doc, str):
                    raise LexosException(
                        "Invalid `docs` element: expected document text (str) for each item."
                    )
                fh.write(f"{i}\tno_label\t{doc.replace('\n', ' ')}\n")
        return input_file, input_file

    def _build_inference_import_command(
        self,
        input_file: str,
        output_file: str,
        keep_sequence: bool,
        preserve_case: bool,
        remove_stopwords: bool,
        use_pipe_from: Optional[str | Path],
    ) -> list[str]:
        """Construct the MALLET import-file command for inference."""
        cmd_import = [
            self.path_to_mallet or "mallet",
            "import-file",
            "--input",
            input_file,
            "--output",
            output_file,
        ]
        if keep_sequence:
            cmd_import.append("--keep-sequence")
        if remove_stopwords:
            cmd_import.append("--remove-stopwords")
        if preserve_case:
            cmd_import.append("--preserve-case")
        if use_pipe_from:
            cmd_import.extend(["--use-pipe-from", str(use_pipe_from)])
        return cmd_import

    def _prepare_inference_input(
        self,
        docs: list[str] | Path | str,
        keep_sequence: bool,
        preserve_case: bool,
        remove_stopwords: bool,
        use_pipe_from: Optional[str | Path],
    ) -> str:
        """Prepare a MALLET-formatted input file for inference.

        Args:
            docs (list[str] | Path | str): Either a document file path or a list of documents.
            keep_sequence (bool): Whether to retain sequence information in the import step.
            preserve_case (bool): Whether to preserve case in the import step.
            remove_stopwords (bool): Whether to remove stopwords in the import step.
            use_pipe_from (Optional[str | Path]): A pipe file to reuse for formatting.

        Returns:
            str: The path to the MALLET-formatted input file.

        Raises:
            LexosException: If the supplied docs list is invalid or contains non-string items.
        """
        output_file = str(Path(self.model_dir) / "infer_input.mallet")
        input_file, _ = self._validate_inference_docs(docs)
        cmd_import = self._build_inference_import_command(
            input_file,
            output_file,
            keep_sequence,
            preserve_case,
            remove_stopwords,
            use_pipe_from,
        )
        subprocess.run(cmd_import, check=True)
        return output_file

    def _resolve_inference_paths(
        self,
        path_to_inferencer: Optional[str | Path],
        output_path: Optional[str | Path],
    ) -> tuple[str, str]:
        """Resolve the inferencer and output paths for inference.

        Args:
            path_to_inferencer (Optional[str | Path]): The inferencer to use.
            output_path (Optional[str | Path]): Optional output path for document-topic probabilities.

        Returns:
            tuple[str, str]: The inferencer path and the output doc-topics path.

        Raises:
            LexosException: If no inferencer is configured.
        """
        if not path_to_inferencer:
            path_to_inferencer = self._metadata_get([self.CANONICAL_INFERENCER_KEY])
        if not path_to_inferencer:
            raise LexosException(
                "No inferencer has been set. Provide `path_to_inferencer` or set it in metadata when training."
            )

        if output_path is None:
            output_path = str(Path(self.model_dir) / "infer-doc-topics.txt")
        else:
            output_path = str(output_path)
        return str(path_to_inferencer), output_path

    @validate_call(config=model_config)
    def infer(
        self,
        docs: list[str] | Path | str,
        path_to_inferencer: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str | Path] = None,
        show: bool = False,
    ) -> list[list[float]] | None:
        """Infer topic distributions for new documents using a saved MALLET inferencer.

        Args:
            docs (list[str] | Path | str): The documents to infer topics for or a path to a file with documents.
            path_to_inferencer (Optional[str | Path]): Path to the MALLET inferencer file. If None, use metadata.
            output_path (Optional[str | Path]): Path to write the output doc-topics file. If None, it defaults to model_dir/infer-doc-topics.txt
            keep_sequence (bool): Whether to keep the sequence in the import-file step.
            preserve_case (bool): Whether to preserve case in the import-file step.
            remove_stopwords (bool): Whether to remove stopwords in the import-file step.
            use_pipe_from (Optional[str | Path]): Optional pipe file to reuse for formatting.
            show (bool): If True, display the returned distributions (no-op in headless).

        Returns:
            list[list[float]] | None: The inferred topic distributions (list of lists), or None if `show` is True.
        """
        if use_pipe_from:
            use_pipe_from = str(use_pipe_from)
        path_to_formatted = self._prepare_inference_input(
            docs,
            keep_sequence,
            preserve_case,
            remove_stopwords,
            use_pipe_from,
        )

        path_to_inferencer, output_path = self._resolve_inference_paths(
            path_to_inferencer,
            output_path,
        )

        cmd = [
            self.path_to_mallet or "mallet",
            "infer-topics",
            "--inferencer",
            path_to_inferencer,
            "--input",
            path_to_formatted,
            "--output-doc-topics",
            output_path,
        ]
        subprocess.run(cmd, check=True)

        distributions = []
        try:
            with open(output_path, "r") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    distributions.append(self._parse_distribution_line(line))
        except FileNotFoundError:
            raise LexosException(
                f"Inferred doc-topic output file not found: {output_path}"
            )

        if show:
            return None
        return distributions

    def set_metadata(self, parameter: str, value: Any) -> None:
        """Set the model parameters from the metadata.

        Args:
            parameter (str): The name of the parameter to set.
            value (Any): The value to set for the parameter.
        """
        self.metadata[parameter] = value
        with open(Path(self.model_dir) / "meta.json", "w") as f:
            f.write(json.dumps(self.metadata))


JavaMallet = Mallet
