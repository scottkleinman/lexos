"""__init__.py.

Last Updated: December 4, 2025
Last Tested: November 26, 2025

A fork of Maria Antoniak's Little Mallet Wrapper: https://github.com/maria-antoniak/little-mallet-wrapper.

Here is a rough summary of the changes:

- Some functions for importing training data from various sources.
- Formatting changes, type hinting, and Pydantic validation.
- A more object-oriented approach to keep track of paths and other metadata so that fewer arguments need to be passed to functions.
- Support for a fuller range of MALLET keyword arguments, including the output-state-file which is needed for generating PyLDAVis and Dfr-Browser visualizations.
- Optional progress tracking during training.
- Topic clouds visualisation.
- More parameters for customising the plotting functions.
"""

import glob
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
from pydantic import BaseModel, ConfigDict, Field, validate_call
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from spacy.tokens import Doc
from wasabi import msg
from wordcloud import WordCloud

from lexos.exceptions import LexosException
from lexos.util import ensure_list
from lexos.visualization.cloud import MultiCloud

# Get the path to the MALLET binary from the environment
load_dotenv()
MALLET_BINARY_PATH = os.getenv("MALLET_BINARY_PATH") or "mallet"


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


@validate_call
def read_dirs(dirs: Path | str | list[Path | str]) -> list[str]:
    """Import a directory or list of directories.

    Args:
        dirs (Path | str | list[Path | str]): A directory or list of directories to import.

    Returns:
        list[str]: The training data.
    """
    # Ensure dirs is a list
    dirs = ensure_list(dirs)

    # Retrieve file paths or raise an error if the directory does not exist
    training_data = []
    for dir in dirs:
        # Validate the argument type here to provide a clear error message
        if isinstance(dir, bool) or not isinstance(dir, (str, Path)):
            raise LexosException(
                f"Invalid directory argument '{dir}'. Expected a directory path (str or Path)."
            )
        if not Path(dir).is_dir():
            raise LexosException(f"Directory {dir} does not exist.")
        else:
            # NOTE: Cannot use Path.glob() here because it returns a generator, which disrupts testing.
            filepaths = glob.glob(f"{dir}/*.txt")
            for path in filepaths:
                if Path(path).is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        training_data.append(f.read())

    return training_data


# Backwards-compatible aliases for older function names/tests
import_data_file = read_file
import_dirs = read_dirs


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Canonical metadata keys used consistently across methods for common
    # training outputs. To preserve backward compatibility when loading
    # metadata produced by older flows, a set of synonyms is still supported
    # but all internal methods should rely only on the canonical keys below.
    # The synonyms list is used for migration to canonical form.
    CANONICAL_DOC_TOPIC_KEY: ClassVar[str] = "path_to_topic_distributions"
    # canonical key names
    CANONICAL_DOC_TOPIC_KEY: ClassVar[str] = "path_to_topic_distributions"
    CANONICAL_TERM_WEIGHTS_KEY: ClassVar[str] = "path_to_term_weights"
    CANONICAL_TOPIC_KEYS_KEY: ClassVar[str] = "path_to_topic_keys"
    CANONICAL_INFERENCER_KEY: ClassVar[str] = "path_to_inferencer"

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

    def __init__(self, **data) -> None:
        """Initialize the Mallet class."""
        super().__init__(**data)
        # Assign the model directory if provided via constructor (model_dir)
        # or via incoming metadata. Create the directory if it is provided
        # to maintain a predictable filesystem state for later operations.
        if self.model_dir is None and isinstance(self.metadata, dict):
            # If the user provided a model_directory via metadata, accept it.
            if "model_directory" in self.metadata:
                self.model_dir = self.metadata["model_directory"]
        # If we now have a model_dir, validate and create it
        if self.model_dir is not None:
            # Validate that model_dir is not a boolean
            if isinstance(self.model_dir, bool):
                raise LexosException(
                    "Invalid `model_dir` argument: expected a path (str or Path), not a boolean."
                )
            # Convert Path objects to str
            if isinstance(self.model_dir, Path):
                model_dir_str = str(self.model_dir)
            else:
                model_dir_str = self.model_dir
            # Ensure the model_dir is not a file
            p = Path(model_dir_str)
            if p.exists() and p.is_file():
                raise LexosException(
                    f"The specified `model_dir` ({model_dir_str}) exists and is a file, expected a directory."
                )
            # Create the directory if it does not exist
            p.mkdir(parents=True, exist_ok=True)
            # Set metadata `model_directory` if not already set
            if not isinstance(data.get("model_dir"), type(None)):
                self.metadata["model_directory"] = model_dir_str
        # Do not alter user-supplied metadata keys outside this logic.

    @cached_property
    def distributions(self) -> list[str]:
        """Get the topic distributions of the model."""
        # Resolve the distribution path using canonical metadata resolution
        distro_path = self._metadata_get([self.CANONICAL_DOC_TOPIC_KEY])
        if distro_path is None:
            raise LexosException(
                "No topic distributions have been set. Please designate a path for `path_to_topic_distributions` when you train your topic model."
            )

        topic_distributions = []
        with open(distro_path, "r") as f:
            for line in f.readlines():
                # Skip blank lines
                if not line.strip():
                    continue
                if line.split()[0] != "#doc":
                    # Try tab-delimited format first: id \t docid \t val1 \t val2 ...
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        raw_dist = parts[2:]
                        # If the distribution token contains topic:prob pairs as a single
                        # token (e.g. '0:0.1 1:0.9'), parse it out into a dense vector
                        if len(raw_dist) == 1 and ":" in raw_dist[0]:
                            token = raw_dist[0]
                            pairs = re.split(r"\s+", token)
                            if all(":" in p for p in pairs):
                                tp_map = {}
                                max_topic = -1
                                for p in pairs:
                                    try:
                                        t, prob = p.split(":")
                                        t_i = int(t)
                                        prob_f = float(prob)
                                        tp_map[t_i] = prob_f
                                        if t_i > max_topic:
                                            max_topic = t_i
                                    except Exception:
                                        raise LexosException(
                                            f"Topic:prob pair malformed in: {p}"
                                        )
                                raw_dist = [
                                    str(tp_map.get(i, 0.0))
                                    for i in range(max_topic + 1)
                                ]
                    else:
                        # If no tabs, try whitespace-separated: id docid val1 val2 ...
                        parts_ws = re.split(r"\s+", line.strip())
                        if len(parts_ws) >= 3:
                            raw_dist = parts_ws[2:]
                        else:
                            # If there is a single token containing topic:prob pairs like
                            # '0:0.1 1:0.2 2:0.7', split and parse those
                            token = line.strip().split()[-1]
                            pairs = re.split(r"\s+", token)
                            if all(":" in p for p in pairs):
                                # rebuild distribution as a full dense vector if possible
                                # create a dict mapping topic idx to prob
                                tp_map = {}
                                max_topic = -1
                                for p in pairs:
                                    try:
                                        t, prob = p.split(":")
                                        t_i = int(t)
                                        prob_f = float(prob)
                                        tp_map[t_i] = prob_f
                                        if t_i > max_topic:
                                            max_topic = t_i
                                    except Exception:
                                        raise LexosException(
                                            f"Topic:prob pair malformed in: {p}"
                                        )
                                # convert mapping to list of floats (dense vector)
                                raw_dist = [
                                    str(tp_map.get(i, 0.0))
                                    for i in range(max_topic + 1)
                                ]
                            else:
                                raise LexosException(
                                    f"The line '{line.strip()}' in the topic distributions file is not formatted correctly."
                                )
                    try:
                        distribution = [float(p) for p in raw_dist]
                    except Exception:
                        raise LexosException(
                            f"Unable to parse distribution values from line: '{line.strip()}'"
                        )
                    topic_distributions.append(distribution)

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
            return [line.strip().split("\t") for line in f.readlines()]

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of documents in the model."""
        if "vocab_size" in self.metadata:
            return self.metadata["vocab_size"]
        else:
            return 0

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
            keep_sequence (bool): Whether to keep the word sequence in the documents.
            remove_stopwords (bool): Whether to remove stopwords from the documents.
            preserve_case (bool): Whether to preserve the case of the documents.
            use_pipe_from (Optional[str]): Path to a MALLET pipe file to use for importing.
            training_ids: Optional[list[int]]: A list of document ids designating a subset of the entire data set. If None, the entire dataset will be imported.
        """
        # Save the training data file
        path_to_training_data = (
            path_to_training_data
            if path_to_training_data is not None
            else os.path.join(self.model_dir, "training_data.txt")
        )
        path_to_formatted_training_data = os.path.join(
            self.model_dir, "training_data.mallet"
        )
        training_data_file = open(path_to_training_data, "w", encoding="utf-8")
        for i, doc in enumerate(training_data):
            # Remove newlines and carriage returns from the document
            doc = re.sub("[\r\n]+", " ", doc).strip()
            if training_ids:
                training_data_file.write(f"{training_ids[i]}\tno_label\t{doc}\n")
            else:
                training_data_file.write(f"{i}\tno_label\t {doc}\n")
        training_data_file.close()
        self.metadata["path_to_training_data"] = path_to_training_data
        self.metadata["path_to_formatted_training_data"] = (
            path_to_formatted_training_data
        )
        self.metadata["num_docs"] = len(training_data)
        # WARNING: Tokenisation relies on whitespace, so it may not be accurate for all languages
        self.metadata["mean_num_tokens"] = np.mean(
            [len(doc.split()) for doc in training_data]
        ).item()
        self.metadata["vocab_size"] = len(
            list(set([token for doc in training_data for token in doc.split()]))
        )

        # Build and execute the command to format the training data for MALLET
        cmd = f"{self.path_to_mallet or 'mallet'} import-file --input {path_to_training_data} --output {path_to_formatted_training_data}"
        if keep_sequence:
            cmd += " --keep-sequence"
        if remove_stopwords:
            cmd += " --remove-stopwords"
        if preserve_case:
            cmd += " --preserve-case"
        if use_pipe_from:
            cmd += f" --use-pipe-from {use_pipe_from}"
        msg.info(cmd)
        os.system(cmd)

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
            model_base = self.model_dir if self.model_dir else os.getcwd()
            path_to_training_data = os.path.join(model_base, "training_data.txt")
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

    def _track_progress(
        self, mallet_cmd: str, num_iterations: int, verbose: bool = True
    ) -> None:
        """Track the progress of the modeling.

        Args:
            mallet_cmd (str): The MALLET command to run.
            num_iterations (int): The number of iterations for the model.
            verbose (bool): Whether to print the MALLET output.

        Notes:
            - Prints MALLET output and updates the progress bar in 10% increments.
        """
        console = Console()
        # NOTE: This is a hack to make Jupyter notebooks in VS Code display all lines
        # in the same cell. It may cause undesirable results in other environments and
        # needs further testing. See https://github.com/Textualize/rich/issues/3483.
        if verbose:
            console.is_jupyter = False

        # Create a progress display with rich
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            # Create a task with a total of 100 (percentage)
            task = progress.add_task("[blue]Training model...", total=100)

            # Run the MALLET command
            p = subprocess.Popen(
                mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )

            # Regex to match progress information
            prog = re.compile(r"\<([^\)]+)\>")

            # Track the last reported progress percentage to avoid duplicate updates
            last_progress = -1

            # Process the output line by line
            while p.poll() is None:
                line = p.stdout.readline().decode()
                if verbose:
                    # Print MALLET output without disrupting progress
                    console.print(line, end="")

                # Keep track of modeling progress
                try:
                    # A float indicating the percentage, which is output by MALLET
                    this_iter = float(prog.match(line).groups()[0])
                    current_progress = int(100.0 * this_iter / num_iterations)

                    # Only update on 10% multiples and avoid duplicate updates
                    if current_progress % 10 == 0 and current_progress > last_progress:
                        # Update to the current progress percentage
                        progress.update(task, completed=this_iter)
                        last_progress = current_progress
                    if current_progress == 100:
                        progress.update(
                            task, description="[green]Complete", completed=100
                        )
                except AttributeError:  # Not every line will match.
                    pass

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
        num_available_topics = len(self.topic_keys)
        if num_topics and not topics:
            if num_topics > num_available_topics:
                raise IndexError(
                    f"Requested num_topics={num_topics}, but only {num_available_topics} topics are available."
                )
            topic_keys = self.topic_keys[:num_topics]
        elif topics:
            # Validate all indices
            for i in topics:
                if i < 0 or i >= num_available_topics:
                    raise IndexError(
                        f"Topic index {i} is out of range. Valid indices are 0 to {num_available_topics - 1}."
                    )
            topic_keys = [self.topic_keys[i] for i in topics]
        else:
            topic_keys = self.topic_keys
        output = ""
        for topic in topic_keys:
            keywords = " ".join(topic[2].split()[:num_keys])
            output += f"Topic {topic[0]}\t{topic[1]}\t{keywords}\n"
        if as_df:
            data = []
            for topic in topic_keys:
                keywords = " ".join(topic[2].split()[:num_keys])
                data.append(
                    {"Topic": topic[0], "Label": topic[1], "Keywords": keywords}
                )
            df = pd.DataFrame(data)
            show_index = True  # or False
            offset = 2 if show_index else 1
            nth = df.columns.get_loc("Keywords") + offset

            css = [
                # header cell of Keywords column
                {
                    "selector": f"thead th:nth-child({nth})",
                    "props": [("text-align", "left")],
                },
                # the column cells
                {
                    "selector": f"td.col{df.columns.get_loc('Keywords')}",
                    "props": [("text-align", "left")],
                },
            ]

            styled_df = df.style.set_table_styles(css).set_properties(
                subset=["Keywords"], **{"text-align": "left"}
            )
            return styled_df
        return output

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
        # Ensure that the path to doc-topic distributions exists (resolved via canonical keys)
        if not self._metadata_has([self.CANONICAL_DOC_TOPIC_KEY]):
            raise LexosException(
                "No topic distributions have been set. Please designate a path to the doc-topic distributions (e.g. `path_to_topic_distributions`) when you train your topic model."
            )

        if "path_to_training_data" not in self.metadata:
            raise LexosException(
                "No training data has been set. Please designate a path for `path_to_training_data` when you train your topic model."
            )

        # Read the training data file
        with open(self.metadata["path_to_training_data"], "r", encoding="utf-8") as f:
            training_data = f.readlines()
        training_data = [
            line.split("\t")[2].strip() for line in training_data
        ]  # Skip the id and label

        # Validate topic index against model's known number of topics (0-based)
        try:
            topic = int(topic)
        except Exception:
            raise ValueError("Topic index must be an integer")

        num_topics = None
        # Try the reliable metadata if present
        if "num_topics" in self.metadata:
            try:
                num_topics = int(self.metadata["num_topics"])
            except Exception:
                num_topics = None
        # Fall back to topic_keys if available
        if num_topics is None:
            try:
                num_topics = len(self.topic_keys)
            except Exception:
                num_topics = None
        # As a last resort, infer from distributions
        distribution_len = None
        if len(self.distributions) > 0:
            # ensure all distributions have the same length; otherwise raise
            lengths = set(len(d) for d in self.distributions)
            if len(lengths) > 1:
                raise LexosException(
                    "Topic distribution lengths are inconsistent across documents; check `path_to_topic_distributions` format."
                )
            distribution_len = next(iter(lengths))
            if num_topics is None:
                num_topics = distribution_len
        if num_topics is None:
            raise LexosException(
                "Model does not have topic information yet. Train or load a model first."
            )
        # If we have both a metadata num_topics and inferred distribution length, they should match.
        if (
            distribution_len is not None
            and num_topics is not None
            and distribution_len != num_topics
        ):
            raise LexosException(
                f"Mismatch between declared number of topics ({num_topics}) and distribution vector length ({distribution_len}). Check your training outputs."
            )

        if not (0 <= topic < num_topics):
            raise ValueError(
                f"Invalid topic index {topic}. Valid topic indices are 0..{num_topics - 1} (0-based)."
            )

        # Combine the distribution and training data, then convert to a dataframe
        distribution_data = [
            (_distribution[topic], _document)
            for _distribution, _document in zip(self.distributions, training_data)
        ]
        df = pd.DataFrame(distribution_data, columns=["Distribution", "Document"])
        df.index.name = "Doc ID"

        # If metadata is provided, concatenate it to the dataframe
        if metadata is not None:
            df = pd.concat([df, metadata], axis=1)

        # Sort the dataframe by distribution and return the top n documents
        if as_str:
            return (
                df.sort_values(by="Distribution", ascending=False).head(n).to_string()
            )
        return df.sort_values(by="Distribution", ascending=False).head(n)

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
        if isinstance(topics, int):
            topics = [topics]
        topic_term_probability_dict = self.load_topic_term_distributions()
        # Build either a string (legacy behavior) or a DataFrame with columns
        # Topic | Term | Probability based on the `as_df` parameter.
        if as_df:
            rows = []
            for _topic, _term_probability_dict in topic_term_probability_dict.items():
                if topics is None or _topic in topics:
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
            df = pd.DataFrame(rows)
            return df
        result = ""
        for _topic, _term_probability_dict in topic_term_probability_dict.items():
            if topics is None or _topic in topics:
                result += f"Topic {_topic}\n"
                for _term, _probability in sorted(
                    _term_probability_dict.items(), key=lambda x: x[1], reverse=True
                )[:n]:
                    result += f"\t{_term}: {_probability}\n"
                result += "\n"
        return result

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
        training_data = read_file(ensure_list(data_source))
        self._import_training_data(
            training_data,
            path_to_training_data=None,
            keep_sequence=keep_sequence,
            remove_stopwords=remove_stopwords,
            preserve_case=preserve_case,
            use_pipe_from=use_pipe_from,
            training_ids=training_ids,
        )

    def load_topic_term_distributions(self) -> dict[str, float]:
        """Load the topic-term distributions from a file.

        Returns:
            dict[str, float]: A dictionary of all topic-term distributions.
        """
        # Ensure that the path to a term weights file has been set.
        term_weight_path = self._metadata_get([self.CANONICAL_TERM_WEIGHTS_KEY])
        if term_weight_path is None:
            raise LexosException(
                f"No term weights have been set. Please designate a path to the term weights file (e.g. `{self.CANONICAL_TERM_WEIGHTS_KEY}`) when you train your topic model."
            )
        topic_term_weight_dict = defaultdict(lambda: defaultdict(float))
        topic_sum_dict = defaultdict(float)
        try:
            with open(term_weight_path, "r") as f:
                for _line in f:
                    if not _line.strip():
                        continue
                    parts = _line.strip().split("\t")
                    if len(parts) != 3:
                        # Malformed line
                        raise ValueError(
                            f"Malformed line in term weights file: '{_line.strip()}'"
                        )
                    _topic, _term, _weight = parts
                    try:
                        weight_f = float(_weight)
                    except Exception:
                        raise ValueError(
                            f"Invalid weight value '{_weight}' in line: '{_line.strip()}'"
                        )
                    topic_term_weight_dict[_topic][_term] = weight_f
                    topic_sum_dict[_topic] += weight_f
        except FileNotFoundError:
            # Surface file not found as filesystem error
            raise

        topic_term_probability_dict = defaultdict(lambda: defaultdict(float))
        for _topic, _term_weight_dict in topic_term_weight_dict.items():
            for _term, _weight in _term_weight_dict.items():
                topic_term_probability_dict[int(_topic)][_term] = (
                    _weight / topic_sum_dict[_topic]
                )

        return topic_term_probability_dict

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
        # Load topic_keys
        topic_keys = self.topic_keys

        # Ensure that topics is a list
        if topics is None:
            topics = list(range(len(topic_keys)))
        elif isinstance(topics, int):
            topics = [topics]

        # Ensure there are topic_labels
        if not target_labels:
            target_labels = list(set(categories))

        # Combine the labels and distributions into a dataframe.
        figs = []
        import os

        # Use user-provided topic_distributions if given, else default to self.distributions
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )

        for topic in topics:
            keywords = " ".join(topic_keys[topic][2].split()[:num_keys])

            dicts_to_plot = []
            for _label, _distribution in zip(categories, distributions):
                if not target_labels or _label in target_labels:
                    dicts_to_plot.append(
                        {
                            "Probability": float(_distribution[topic]),
                            "Category": _label,
                            "Topic": keywords,
                        }
                    )
            df_to_plot = pd.DataFrame(dicts_to_plot)

            # Validate overlay option
            if overlay not in ("strip", "swarm", "none", None):
                raise LexosException(
                    "Invalid `overlay` argument: expected 'strip', 'swarm', or 'none'."
                )

            # Show the final plot
            sns.set_theme(style="ticks", font_scale=font_scale)
            # Create a figure/axes so we can overlay points for small datasets
            if figsize:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
            sns.boxplot(
                data=df_to_plot,
                x="Category",
                y="Probability",
                color=color,
                ax=ax,
                showmeans=True,
            )
            # Overlay data points so users can see the raw values when there are
            # too few observations to form a full box
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
                # if overlay == 'none', do nothing
            except Exception:
                # Overlay plotting is optional; ignore any backend failures
                pass
            sns.despine()
            plt.xticks(rotation=45, ha="right")
            # Set either the provided title or a sensible default including topic index and top keys
            if title is None:
                ax.set_title(f"Topic {topic}: {keywords}")
            else:
                # Use a figure-level title to avoid per-subplot clobbering
                fig.suptitle(title)
            plt.tight_layout()
            # Save each plot to a unique file if output_path is set
            if output_path:
                base, ext = os.path.splitext(output_path)
                save_path = f"{base}_topic{topic}{ext}"
                fig.savefig(save_path)
            figs.append(fig)
            if show:
                plt.show()
            plt.close(fig)
        if show:
            return None
        # If this function only generated a single figure, return it.
        if len(figs) == 1:
            return figs[0]
        return figs

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
        # Load topic_keys
        topic_keys = self.topic_keys

        # Use user-provided topic_distributions if given, else default to self.distributions
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )

        dicts_to_plot = []
        for _category_label, _distribution in zip(categories, distributions):
            if not target_labels or _category_label in target_labels:
                for _topic, _probability in enumerate(_distribution):
                    keywords = " ".join(topic_keys[_topic][2].split()[:num_keys])
                    if num_keys:
                        if keywords:
                            _topic_label = f"Topic {_topic}: {keywords}"
                        else:
                            _topic_label = f"Topic {_topic}"
                    else:
                        _topic_label = f"Topic {_topic}"
                    dicts_to_plot.append(
                        {
                            "Probability": float(_probability),
                            "Category": _category_label,
                            "Topic": _topic_label,
                        }
                    )

        # Create a dataframe, format it for the heatmap function, and normalize the columns.
        df_to_plot = pd.DataFrame(dicts_to_plot)
        df_wide = df_to_plot.pivot_table(
            index="Category", columns="Topic", values="Probability"
        )
        df_norm_col = (df_wide - df_wide.mean()) / df_wide.std()

        # Ensure the columns are ordered by numeric topic index where available (natural sort)
        def _topic_key(col):
            # Match 'Topic <num>' possibly followed by ': ...'
            try:
                m = re.match(r"Topic\s+(\d+)", str(col))
                if m:
                    return (0, int(m.group(1)))
            except Exception:
                pass
            return (1, str(col))

        try:
            ordered_cols = sorted(list(df_norm_col.columns), key=_topic_key)
            df_norm_col = df_norm_col[ordered_cols]
        except Exception:
            # If columns are not iterable or sorting fails (e.g., custom objects),
            # we leave the DataFrame as-is rather than raising an exception.
            pass

        # Show the final plot
        sns.set_theme(style="ticks", font_scale=font_scale)
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
        ax = sns.heatmap(df_norm_col, cmap=cmap, ax=ax)
        # Set either provided title or a sensible default that indicates the content and the number of topics
        if title is None:
            try:
                num_topics = len(df_norm_col.columns)
            except Exception:
                num_topics = None
            if num_topics is not None:
                title = f"Topics by Category ({num_topics} Topics)"
            else:
                title = "Topics by Category"
        if title:
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
        else:
            plt.close()
            return fig

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

        # Load topic-term probabilities and convert to DataFrame with topics as rows
        topic_term_probability_dict = self.load_topic_term_distributions()
        df = pd.DataFrame.from_dict(topic_term_probability_dict, orient="index").fillna(
            0
        )

        # Filter the DataFrame to include only the specified topics (rows)
        if topics is not None:
            df = df.iloc[ensure_list(topics)]

        # Build options dict for MultiCloud
        opts = kwargs.get("opts", {})
        # Default to a white background unless overridden
        opts.setdefault("background_color", "white")
        # Ensure `max_words` is present if not provided, mapping from max_terms
        if "max_words" not in opts and max_terms is not None:
            opts["max_words"] = max_terms

        # Convert round_mask boolean or int into the radius integer expected by MultiCloud
        if isinstance(round_mask, bool):
            round_radius = 120 if round_mask else 0
        else:
            try:
                round_radius = int(round_mask) if round_mask is not None else 0
            except Exception:
                raise LexosException(
                    "Invalid `round_mask` argument: expected a boolean or integer radius."
                )

        # Build simple numeric labels for each topic to avoid clutter
        labels = [f"Topic {i}" for i in range(len(df))]

        # Build figure_opts forwarding and set a white facecolor by default
        figure_opts = kwargs.get("figure_opts", {})
        figure_opts.setdefault("facecolor", "white")

        # Create the MultiCloud object with updated args compatible with the class
        # If no explicit title supplied, create a helpful default
        if title is None:
            try:
                num_topics = len(df)
            except Exception:
                num_topics = None
            if num_topics is not None:
                title = f"Topic Clouds ({num_topics} topics)"
            else:
                title = "Topic Clouds"

        mc = MultiCloud(
            data=df,
            limit=max_terms,
            figsize=figsize,
            opts=opts,
            round=round_radius,
            labels=labels,
            figure_opts=figure_opts,
            title=title,
        )

        # Save the file if requested
        if output_path:
            mc.save(output_path)

        # Show the file if requested
        if show:
            mc.show()
            return None
        else:
            return mc.fig

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
        # Use provided distributions / keys or fall back to instance data
        distributions = (
            topic_distributions
            if topic_distributions is not None
            else self.distributions
        )
        topic_keys = topic_keys if topic_keys is not None else self.topic_keys

        if distributions is None or len(distributions) == 0:
            raise LexosException("No topic distributions available to plot.")

        if topic_index < 0:
            raise ValueError("topic_index must be a non-negative integer")

        if len(times) != len(distributions):
            raise LexosException(
                "Length mismatch: 'times' must be the same length as topic_distributions"
            )

        data_dicts = []
        for j, _distribution in enumerate(distributions):
            if len(_distribution) <= topic_index:
                # skip documents that don't cover the requested topic index
                continue
            data_dicts.append(
                {"Probability": _distribution[topic_index], "Time": times[j]}
            )

        if len(data_dicts) == 0:
            raise LexosException(f"No data found for topic index {topic_index}")

        data_df = pd.DataFrame(data_dicts)

        sns.set_theme(style="ticks", font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=data_df, x="Time", y="Probability", color=color, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Topic Probability")

        # Default title
        if title is None:
            try:
                keywords = " ".join(topic_keys[topic_index][2].split()[:5])
                title = f"Topic {topic_index}: {keywords}"
            except Exception:
                title = f"Topic {topic_index}"
        if title:
            fig.suptitle(title)

        plt.tight_layout()
        sns.despine()
        if output_path:
            fig.savefig(output_path)
        if show:
            plt.show()
            return None
        else:
            return fig

    @validate_call(config=model_config)
    def train(
        self,
        num_topics: int = 20,
        num_iterations: Optional[int] = 100,
        optimize_interval: Optional[int] = 10,
        verbose: Optional[bool] = True,
        # Common output paths: caller may pass canonical keys or path_to_* names
        path_to_model: Optional[str] = None,
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
            path_to_inferencer (Optional[str]): Optional output filename for saving a trained inferencer object
                that can be used with `mallet infer-topics`. If not provided, defaults to
                `model_dir/inferencer.mallet`.
        """
        path_to_formatted_training_data = os.path.join(
            self.model_dir, "training_data.mallet"
        )

        # Build the MALLET command
        cmd = f"{self.path_to_mallet or 'mallet'} train-topics"
        flags = {
            "input": path_to_formatted_training_data,
            "num-topics": num_topics,
            "num-iterations": num_iterations,
            "output-state": path_to_state
            or os.path.join(self.model_dir, "topic-state.gz"),
            "output-topic-keys": path_to_topic_keys
            or os.path.join(self.model_dir, "topic-keys.txt"),
            "output-doc-topics": path_to_topic_distributions
            or os.path.join(self.model_dir, "doc-topic.txt"),
            "topic-word-weights-file": path_to_term_weights
            or os.path.join(self.model_dir, "topic-weights.txt"),
            "diagnostics-file": path_to_diagnostics
            or os.path.join(self.model_dir, "diagnostics.xml"),
            # Optional inferencer filename path to save a trained inferencer for later inference
            "inferencer-filename": path_to_inferencer
            or os.path.join(self.model_dir, "inferencer.mallet"),
            "optimize-interval": optimize_interval,
        }

        for k, v in flags.items():
            if v:
                # Save file names in the model directory if they are not absolute paths
                if isinstance(v, str) and len(Path(v).parts) == 1:
                    v = f"{self.metadata['model_directory']}/{v}"
                cmd += f" --{k} {v}"
                # Set canonical metadata keys for common outputs so consumers can
                # rely on a single key. Map train flags directly to the
                # canonical metadata keys.
                if k == "output-doc-topics":
                    self.metadata[self.CANONICAL_DOC_TOPIC_KEY] = v
                if k == "topic-word-weights-file":
                    self.metadata[self.CANONICAL_TERM_WEIGHTS_KEY] = v
                if k == "output-topic-keys":
                    self.metadata[self.CANONICAL_TOPIC_KEYS_KEY] = v
                if k == "inferencer-filename":
                    self.metadata[self.CANONICAL_INFERENCER_KEY] = v

        # Train the model
        msg.good("Training topics...")
        self._track_progress(cmd, num_iterations, verbose)
        self.metadata["num_topics"] = num_topics
        self.metadata["optimize_interval"] = optimize_interval
        # For flags we don't have a canonical mapping for, provide a path_to_ entry
        # to preserve other easily accessible metadata entries. Do not set legacy
        # keys when we are mapping to a canonical key.
        mapping = {
            "output-doc-topics": self.CANONICAL_DOC_TOPIC_KEY,
            "topic-word-weights-file": self.CANONICAL_TERM_WEIGHTS_KEY,
            "output-topic-keys": self.CANONICAL_TOPIC_KEYS_KEY,
            "inferencer-filename": self.CANONICAL_INFERENCER_KEY,
        }
        for k, v in flags.items():
            if k not in ["num-topics", "optimize-interval"]:
                if k in mapping:
                    # canonical keys already set earlier in the loop
                    continue
                self.metadata[f"path_to_{k.replace('-', '_')}"] = v
        self.metadata["training_command"] = cmd
        msg.good("Complete")

    @validate_call(config=model_config)
    def infer(
        self,
        docs: list[str] | Path | str,
        path_to_inferencer: Optional[str] = None,
        output_path: Optional[str] = None,
        keep_sequence: bool = True,
        preserve_case: bool = True,
        remove_stopwords: bool = True,
        use_pipe_from: Optional[str] = None,
        show: bool = False,
    ) -> list[list[float]] | None:
        """Infer topic distributions for new documents using a saved MALLET inferencer.

        Args:
            docs (list[str] | Path | str): The documents to infer topics for or a path to a file with documents.
            path_to_inferencer (Optional[str]): Path to the MALLET inferencer file. If None, use metadata.
            output_path (Optional[str]): Path to write the output doc-topics file. If None, it defaults to model_dir/infer-doc-topics.txt
            keep_sequence (bool): Whether to keep the sequence in the import-file step.
            preserve_case (bool): Whether to preserve case in the import-file step.
            remove_stopwords (bool): Whether to remove stopwords in the import-file step.
            use_pipe_from (Optional[str]): Optional pipe file to reuse for formatting.
            show (bool): If True, display the returned distributions (no-op in headless).

        Returns:
            list[list[float]] | None: The inferred topic distributions (list of lists), or None if `show` is True.
        """
        # Accept a single file path or list of documents
        if isinstance(docs, (Path, str)) and Path(docs).is_file():
            # it's an input file
            input_file = str(docs)
            # ensure we have a formatted mallet file if not provided
            path_to_formatted = os.path.join(self.model_dir, "infer_input.mallet")
            # import-file to format the input for mallet
            cmd_import = f"{self.path_to_mallet or 'mallet'} import-file --input {input_file} --output {path_to_formatted}"
            if keep_sequence:
                cmd_import += " --keep-sequence"
            if remove_stopwords:
                cmd_import += " --remove-stopwords"
            if preserve_case:
                cmd_import += " --preserve-case"
            if use_pipe_from:
                cmd_import += f" --use-pipe-from {use_pipe_from}"
            # msg.info(cmd_import)
            os.system(cmd_import)
        else:
            # assume a list of document strings
            if isinstance(docs, bool) or not isinstance(docs, list):
                raise LexosException(
                    "Invalid `docs` argument: expected a list of strings or a path to a file."
                )
            # Write a temporal input file
            path_to_plain = os.path.join(self.model_dir, "infer_input.txt")
            with open(path_to_plain, "w", encoding="utf-8") as fh:
                for i, doc in enumerate(docs):
                    if isinstance(doc, bool) or not isinstance(doc, str):
                        raise LexosException(
                            "Invalid `docs` element: expected document text (str) for each item."
                        )
                    fh.write(f"{i}\tno_label\t{doc.replace('\n', ' ')}\n")
            # format it with import-file
            path_to_formatted = os.path.join(self.model_dir, "infer_input.mallet")
            cmd_import = f"{self.path_to_mallet or 'mallet'} import-file --input {path_to_plain} --output {path_to_formatted}"
            if keep_sequence:
                cmd_import += " --keep-sequence"
            if remove_stopwords:
                cmd_import += " --remove-stopwords"
            if preserve_case:
                cmd_import += " --preserve-case"
            if use_pipe_from:
                cmd_import += f" --use-pipe-from {use_pipe_from}"
            # msg.info(cmd_import)
            os.system(cmd_import)

        # Determine the inferencer file to use
        if not path_to_inferencer:
            path_to_inferencer = self._metadata_get([self.CANONICAL_INFERENCER_KEY])
        if not path_to_inferencer:
            raise LexosException(
                "No inferencer has been set. Provide `path_to_inferencer` or set it in metadata when training."
            )

        path_to_formatted = path_to_formatted
        if output_path is None:
            output_path = os.path.join(self.model_dir, "infer-doc-topics.txt")

        cmd = f"{self.path_to_mallet or 'mallet'} infer-topics --inferencer {path_to_inferencer} --input {path_to_formatted} --output-doc-topics {output_path}"
        # msg.info(cmd)
        os.system(cmd)

        # Read the output file and return distributions
        distributions = []
        try:
            with open(output_path, "r") as f:
                for line in f.readlines():
                    # Skip blank lines
                    if not line.strip():
                        continue
                    if line.split()[0] != "#doc":
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            raw_dist = parts[2:]
                            # If the distribution token contains topic:prob pairs as a single
                            # token (e.g. '0:0.1 1:0.9'), parse it out into a dense vector
                            if len(raw_dist) == 1 and ":" in raw_dist[0]:
                                token = raw_dist[0]
                                pairs = re.split(r"\s+", token)
                                if all(":" in p for p in pairs):
                                    tp_map = {}
                                    max_topic = -1
                                    for p in pairs:
                                        try:
                                            t, prob = p.split(":")
                                            t_i = int(t)
                                            prob_f = float(prob)
                                            tp_map[t_i] = prob_f
                                            if t_i > max_topic:
                                                max_topic = t_i
                                        except Exception:
                                            raise LexosException(
                                                f"Topic:prob pair malformed in: {p}"
                                            )
                                    raw_dist = [
                                        str(tp_map.get(i, 0.0))
                                        for i in range(max_topic + 1)
                                    ]
                        else:
                            parts_ws = re.split(r"\s+", line.strip())
                            if len(parts_ws) >= 3:
                                raw_dist = parts_ws[2:]
                            else:
                                # parse compressed topic:prob pairs
                                token = line.strip().split()[-1]
                                pairs = re.split(r"\s+", token)
                                if all(":" in p for p in pairs):
                                    tp_map = {}
                                    max_topic = -1
                                    for p in pairs:
                                        t, prob = p.split(":")
                                        t_i = int(t)
                                        prob_f = float(prob)
                                        tp_map[t_i] = prob_f
                                        if t_i > max_topic:
                                            max_topic = t_i
                                    raw_dist = [
                                        str(tp_map.get(i, 0.0))
                                        for i in range(max_topic + 1)
                                    ]
                                else:
                                    raise LexosException(
                                        f"The line '{line.strip()}' in the inferred doc-topics file is not formatted correctly."
                                    )
                        try:
                            distribution = [float(p) for p in raw_dist]
                        except Exception:
                            raise LexosException(
                                f"Unable to parse distribution values from line: '{line.strip()}'"
                            )
                        distributions.append(distribution)
        except FileNotFoundError:
            raise LexosException(
                f"Inferred doc-topic output file not found: {output_path}"
            )

        if show:
            # user wants to display; we return None in this case for parity with other methods
            return None
        return distributions
