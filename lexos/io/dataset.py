"""dataset.py.

This class reads files (or folders of files) in which multiple texts are stored.
Texts may be line-delimited, or the files may be in .csv, .tsv, .json, or .jsonl format.
If the file is .csv, .tsv, .json, or .jsonl, it is loaded using pandas.read_csv or
pandas.read_json. In this case, it may be necessary to specify the delimiter and/or
extra metadata. Other keywords from the pandas API may also be supplied for these
two methods.

To Do:

    - Needs better exceptions.

Pydantic version testing:
  - Done: .csv, .tsv, .json, .jsonl, url, lines, directories of files
  - .zip files
  - Need to find a way to check if the path is a dir and then redirect to parse_dir;
    maybe the way to go is to include a parse() method which autodirects to a format-
    specific parse method.
"""

import io
import itertools
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from IPython.display import display  # Remove when debugging is complete
from pydantic import BaseModel
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException
from lexos.io import validators

Model = TypeVar("Model", bound="BaseModel")


class Dataset(BaseModel):
    """Dataset class."""

    path: Optional[str]
    data: Optional[List[Dict[str, str]]]

    @property
    def df(self) -> pd.DataFrame:
        """Return the dataframe of the object data.

        Returns:
            pd.DataFrame: The dataframe of the object data.
        """
        return pd.DataFrame(self.data)

    @property
    def locations(self) -> List[str]:
        """Return the locations of the object data.

        Returns:
            List[str]: The locations of the object data.
        """
        return [x["locations"] for x in self.data]

    @property
    def names(self) -> List[str]:
        """Return the names of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return [x["names"] for x in self.data]

    @property
    def texts(self) -> List[str]:
        """Return the texts of the object data.

        Returns:
            List[str]: The texts of the object data.
        """
        return [x["texts"] for x in self.data]

    @classmethod
    def parse_lines(
        cls: Type["Model"],
        path: Optional[str] = None,
        lines: Optional[str] = None,
        names: Optional[Union[List[List[str]], List[str]]] = None,
        locations: Optional[List[str]] = None,
        dir_items: Optional[Tuple[int]] = None,
    ) -> "Model":
        """Parse lineated texts into the Dataset object.

        Args:
            path (Optional[str]): The path to the file containg the lines to parse.
            lines (Optional[str]): The lines to parse.
            names (Optional[Union[List[List[str]], List[str]]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            dir_items (Optional[Tuple[int]]): The number of file number and the number of lines in the file.

        Returns:
            Model: A dataset object.
        """
        if not path and not lines:
            raise LexosException("You must include either a `path` or a `lines` value.")
        elif path and not lines:
            lines = path
        # If the lines string is a path or url, read the file
        if validators.is_path_or_url(lines):
            path = lines
            # Fails if path is a zip file, not a text file
            try:
                with open(path) as f:
                    lines = f.readlines()
            except Exception:
                raise LexosException(f"The file {path} is not a lineated text file.")
        # Read from a string
        else:
            lines = lines.split("\n")
            path = ""
        # Convert the lines into a list of dicts
        texts = [{"title": "", "text": line} for line in lines]
        # Add locations if specified
        if locations:
            if len(locations) != len(lines):
                raise Exception(
                    "The number of locations must equal the number of lines."
                )
            else:
                for i, location in enumerate(locations):
                    texts[i]["location"] = location
        # Add names if specified
        if names:
            # If the source is a directory, parse the names list for the current file
            if dir_items:
                if len(names) != dir_items[1] * 2:
                    raise Exception(
                        "The number of names is each list must equal the number of lines in each file."
                    )
                else:
                    if dir_items[0] == 0:
                        start = 0
                    else:
                        start = dir_items[0] * 2
                    names = names[start : start + 2]
            # Otherwise, just check the naes for the current file
            else:
                if len(names) != len(lines):
                    raise Exception(
                        f"The number of names {len(names)} must equal the number of lines {len(lines)}."
                    )
            for i, name in enumerate(names):
                texts[i]["title"] = name
        # Create the dict to be parsed to an object
        obj = {"path": path, "data": texts}
        return cls.parse_obj(obj)

    @classmethod
    def parse_csv(
        cls: Type["Model"],
        path: str,
        names: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        location_col: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse csv into the Dataset object.

        Args:
            path (str): The path to parse.
            names (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            title_col (str): The name of the column containing the titles.
            text_col (str): The name of the column containing the texts.
            location_col (str): The name of the column containing the locations.

        Returns:
            Model: A dataset object.
        """
        # Read from path or url
        if validators.is_path_or_url(path):
            df = pd.read_csv(path, **kwargs)
        # Read from a string
        else:
            df = pd.read_csv(io.StringIO(path), **kwargs)
            path = ""
        # Populate the locations if specified
        df = check_locations_titles(df, locations, names)
        # Rename columns if headers are specified
        df = rename_columns(df, title_col, text_col, location_col)
        # Convert the data to list of dictionaries
        texts = df.to_dict(orient="records")
        # Create the dict to be parsed to an object
        obj = {"path": path, "data": texts}
        return cls.parse_obj(obj)

    @classmethod
    def parse_json(
        cls: Type["Model"],
        path: str,
        names: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse json into the Dataset object.

        Args:
            path (str): The path to parse.
            names (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_field (str): The name of the field containing the locations.

        Returns:
            Model: A dataset object.
        """
        # Read from path or url
        if validators.is_path_or_url(path):
            df = pd.read_json(path, orient="records", **kwargs)
        # Read from a string
        else:
            df = pd.read_json(io.StringIO(path), orient="records", **kwargs)
            path = ""
        # Populate the locations if specified
        df = check_locations_titles(df, locations, names)
        # Rename columns if headers are specified
        df = rename_columns(df, title_field, text_field, location_field)
        # Convert the data to list of dictionaries
        texts = df.to_dict(orient="records")
        # Create the dict to be parsed to an object
        obj = {"path": path, "data": texts}
        return cls.parse_obj(obj)

    @classmethod
    def parse_zip(
        cls: Type["Model"],
        path: str,
        names: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse json into the Dataset object.

        Args:
            path (str): The path to parse.
            names (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_field (str): The name of the field containing the locations.

        Returns:
            Model: A dataset object.
        """
        # Read from path or url (extra context manager required for smart_open)
        with open(path, "rb") as f:
            with zipfile.ZipFile(f) as zip:
                with tempfile.TemporaryDirectory() as tempdir:
                    zip.extractall(tempdir)
                    filepaths = list(Path(tempdir).glob("**/*"))
                    filepaths = [path for path in filepaths if path.is_file()]
        # This will be a list of dataframes, one for each file
        print(filepaths)
        tmp_datasets = cls._process_directory(
            filepaths,
            names,
            locations,
            title_field,
            text_field,
            location_field,
            **kwargs,
        )
        # Merge the data from the list of datasets into a single dict
        data = [text_data for dataset in tmp_datasets for text_data in dataset.data]
        # Create the dict to be parsed to an object
        obj = {"path": path, "data": data}
        return cls.parse_obj(obj)

    @classmethod
    def parse_dir(
        cls: Type["Model"],
        path: str,
        user: Optional[str] = None,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        names: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse a directory into the Dataset object.

        Args:
            path (str): The path to parse.
            user(str): The user name of the GitHub repository.
            repo(str): The repository name of the GitHub repository.
            branch(str): The branch of the GitHub repository.
            names (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_field (str): The name of the field containing the locations.

        Returns:
            Model: A dataset object.
        """
        # Check if the path is a local directory or GitHub directory
        if "github.com" in path:
            filepaths = utils.get_github_raw_paths(path, user, repo, branch)
        elif Path(path).is_dir():
            filepaths = list(Path(path).glob("**/*"))
        else:
            raise LexosException("Path is not a valid directory.")
        tmp_datasets = cls._process_directory(
            filepaths,
            names,
            locations,
            title_field,
            text_field,
            location_field,
            **kwargs,
        )
        # Merge the data from the list of datasets into a single dict
        data = [text_data for dataset in tmp_datasets for text_data in dataset.data]
        # Create the dict to be parsed to an object
        obj = {"path": path, "data": data}
        return cls.parse_obj(obj)

    @classmethod
    def _process_directory(
        cls,
        filepaths: List[Path],
        names: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs,
    ) -> List["Model"]:
        """Parse a directory into the Dataset object.

        Args:
            filepaths (List[Path]): The list of paths to parse.
            names (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_field (str): The name of the field containing the locations.

        Returns:
            Model: A list of dataset objects.
        """
        tmp_datasets = []
        # Flatten names if it is a list of lists
        if names and any(isinstance(el, list) for el in names):
            lines_per_file = len(names)
            names = list(itertools.chain(*names))
            dir_items = True
        else:
            dir_items = None
        for i, path in enumerate(filepaths):
            if dir_items:
                dir_items = (i, lines_per_file)
            if Path(path).suffix in [".csv", ".tsv"]:
                tmp_datasets.append(
                    cls.parse_csv(
                        str(path),
                        names,
                        locations,
                        title_field,
                        text_field,
                        location_field,
                        **kwargs,
                    )
                )
            elif Path(path).suffix in [".json", ".jsonl"]:
                tmp_datasets.append(
                    cls.parse_json(
                        str(path),
                        names,
                        locations,
                        title_field,
                        text_field,
                        location_field,
                        **kwargs,
                    )
                )
            else:
                tmp_datasets.append(
                    cls.parse_lines(
                        path=str(path),
                        names=names,
                        locations=locations,
                        dir_items=dir_items,
                        **kwargs,
                    )
                )
        return tmp_datasets


# Helper functions
def check_locations_titles(
    df: pd.DataFrame,
    locations: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Check if the locations and titles are specified.

    Args:
        df (pd.DataFrame): The dataframe to check.
        locations (Optional[List[str]]): The locations of the texts.
        names (Optional[List[str]]): The names of the texts.
    Returns:
        pd.DataFrame: The dataframe with the locations and titles.
    """
    if locations:
        try:
            df["location"] = locations
        except:
            raise Exception("The number of locations must equal the number of lines.")
    if names:
        try:
            df["name"] = names
        except:
            raise Exception("The number of names must equal the number of lines.")
    return df


def rename_columns(
    df: pd.DataFrame,
    title: Optional[str] = None,
    text: Optional[str] = None,
    location: Optional[str] = None,
) -> pd.DataFrame:
    """Rename specified to column or field to "title" and "text".

    Args:
        df (pd.DataFrame): The dataframe to check.
        title (Optional[str]): The column or field to rename "title".
        text (Optional[str]): The column or field to rename "text".
        location (Optional[str]): The column or field to rename "location".
    Returns:
        pd.DataFrame: The dataframe with the renamed columns.
    """
    if title:
        df = df.rename(columns={title: "title"})
    if text:
        df = df.rename(columns={text: "text"})
    if location:
        df = df.rename(columns={location: "location"})
    return df


"""Deprecated code below."""
# class DatasetLoader:
#     """Load a csv, json, jsonl, or lineated text file."""

#     def __init__(
#         self, paths: Optional[Union[list, Path, str]] = None, **kwargs
#     ) -> Union[Dataset, List[Dataset]]:
#         """Instantiate loader class.

#         Args:
#             path (Optional[Union[list, Path, str]]): Path or url to the file.
#             **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.

#         Returns:
#             Union[Dataset, List[Dataset]]: A dataset or list of dataset objects.
#         """
#         self.datasets = []
#         if paths:
#             if not isinstance(paths, list):
#                 paths = [paths]
#             for path in paths:
#                 self.load(path, **kwargs)

#     def _decode(self, text: Union[bytes, str]) -> str:
#         """Decode a text.

#         Args:
#             text (Union[bytes, str]): The text to decode.

#         Returns:
#             str: The decoded text.
#         """
#         return utils._decode_bytes(text)

#     def _handle_zip(self, path: str) -> List[Dataset]:
#         """Extract a zip file and add each text inside.

#         Args:
#             path (str): The path to the zip file.

#         Returns:
#             List[Dataset]: A list of dataset objects.
#         """
#         with open(path, "rb") as f:
#             with zipfile.ZipFile(f) as zip:
#                 with tempfile.TemporaryDirectory() as tempdir:
#                     zip.extractall(tempdir)
#                     tmp_datasets = []
#                     for tmp_path in Path(tempdir).glob("**/*"):
#                         if (
#                             tmp_path.is_file()
#                             and not tmp_path.suffix == ""
#                             and not str(tmp_path).startswith("__MACOSX")
#                             and not str(tmp_path).startswith(".ds_store")
#                         ):
#                             tmp_datasets.append(self.load(tmp_path))
#         return tmp_datasets

#     def load(self, path: Union[list, Path, str], **kwargs) -> None:
#         """Load a dataset file.

#         Args:
#             path (Union[list, Path, str]): The path to the file to load.
#             **kwargs: Additional arguments to pass pandas.read_csv or pandas.read_json.
#         """
#         if not isinstance(path, list):
#             path = [str(path)]
#         for p in path:
#             try:
#                 dataset = None
#                 # Ensure self.datasets is a list
#                 if not isinstance(self.datasets, list):
#                     self.datasets = [self.datasets]
#                 if p.endswith(".csv") or p.endswith(".tsv"):
#                     dataset = self._load_csv(p, **kwargs)
#                 elif p.endswith(".json") or p.endswith(".jsonl"):
#                     dataset = self._load_json(p, **kwargs)
#                 elif "github.com" in str(path):
#                     filepaths = utils.get_github_raw_paths(path)
#                     for filepath in filepaths:
#                         self.load(filepath)
#                 elif utils.is_dir(p):
#                     filepaths = [f for f in Path(p).iterdir() if f.is_file()]
#                     for filepath in filepaths:
#                         self.load(filepath, **kwargs)
#                 elif p.endswith(".zip"):
#                     dataset = self._handle_zip(p, **kwargs)
#                 else:
#                     dataset = self._load_lineated_text(p, **kwargs)
#                 if isinstance(dataset, Dataset):
#                     self.datasets.append(dataset)
#                 if len(self.datasets) == 1:
#                     self.datasets = self.datasets[0]
#             except LexosException:
#                 raise LexosException(f"Error loading file: {p}.")

#     def _load_csv(
#         self,
#         path: str,
#         columns: Optional[List[str]] = None,
#         title_column: Optional[str] = None,
#         text_column: Optional[str] = None,
#         **kwargs,
#     ) -> Dataset:
#         """Load a csv file.

#         Args:
#             path (str): The path to the file to load.
#             columns (Optional[List[str]]): Names of all the columns to load in the csv file.
#             title_column (Optional[str]): The name of the column containing the title of the text.
#             text_column (Optional[str]): The name the column containing the text.
#             **kwargs: Additional arguments to pass to pandas.read_csv.

#         Returns:
#             Dataset: A dataset object.
#         """
#         decode = True
#         if "decode" in kwargs and kwargs["decode"] is False:
#             decode = False
#         if text_column and not title_column:
#             raise ValueError(
#                 "You must supply both a `title_column` and a `text_column`."
#             )
#         if title_column and not text_column:
#             raise ValueError(
#                 "You must supply both a `title_column` and a `text_column`."
#             )
#         try:
#             # No headers: include a list of all columns, including "title" and "text"
#             if columns:
#                 if "title" not in columns:
#                     raise ValueError("One column must be named `title`.")
#                 if "text" not in columns:
#                     raise ValueError("One column must be named `text`.")
#                 df = pd.read_csv(path, names=columns, **kwargs)
#                 title_column = "title"
#                 text_column = "text"
#             # Headers contain "title" and "text"
#             elif not title_column and not text_column:
#                 try:
#                     df = pd.read_csv(path, **kwargs)
#                     if "title" in df.columns and "text" in df.columns:
#                         title_column = "title"
#                         text_column = "text"
#                     else:
#                         msg = (
#                             "The csv file must contain columns named `title` and `text`. ",
#                             "You can convert the names of existing column to these with ",
#                             "the `title_column` and `text_column` parameters. If your ",
#                             "file has no column headers, you can supply a list with the ",
#                             "`columns` parameter.",
#                         )
#                         raise LexosException(msg)
#                 except BaseException as e:
#                     raise BaseException(f"Could not parse file: {e}.")
#             # User must specify which header is the title column and which is the text column
#             elif text_column and title_column:
#                 df = pd.read_csv(path, **kwargs)
#             else:
#                 raise BaseException(f"Invalid keyword arguments.")
#             # Rename the columns
#             df = df.rename(columns={title_column: "title", text_column: "text"})
#             # Decode the text
#             if decode:
#                 df["text"] = df["text"].apply(lambda x: self._decode(x))
#             # Assign dummy titles if necessary
#             if not title_column:
#                 df["title"] = [Path(path).stem] * df.shape[1]
#             # Add a location column
#             df["locations"] = [path] * df.shape[1]
#             # Rename columns and convert to a dictionary
#             df = df.rename(columns={"title": "names", "text": "texts"})
#             # Create dataset object
#             return Dataset(path=path, data=df.to_dict(orient="records"))
#         except BaseException as e:
#             raise BaseException(f"Could not parse {path}: {e}")

#     def _load_json(
#         self,
#         path: str,
#         title_key: Optional[str] = None,
#         text_key: Optional[str] = None,
#         **kwargs,
#     ) -> Dataset:
#         """Load a json file.

#         Args:
#             path (str): The path to the file to load.
#             title_key (Optional[str]): The name of the field containing the title of the text.
#             text_key (Optional[str]): The name the field containing the text.
#             **kwargs: Additional arguments to pass to pandas.read_json.

#         Returns:
#             Dataset: A dataset object.
#         """
#         decode = True
#         if "decode" in kwargs and kwargs["decode"] is False:
#             decode = False
#         try:
#             # JSON object must contain "title" and "text"
#             if not title_key and not text_key:
#                 df = pd.read_json(path, orient="records", **kwargs)
#                 columns = df.columns.tolist()
#                 if "title" not in columns:
#                     raise ValueError(
#                         "One field must be named `title` or you must convert an existing column with the `title_key` parameter."
#                     )
#                 if "text" not in columns:
#                     raise ValueError(
#                         "One field must be named `text` or you must convert an existing column with the `title_key` parameter."
#                     )
#             # User must specify which field is the title field and which is the text field
#             elif text_key and title_key:
#                 df = pd.read_json(path, orient="records", **kwargs)
#                 df = df.rename(columns={title_key: "title", text_key: "text"})
#             elif text_key and not title_key:
#                 raise ValueError("You must supply both a `text_key` and a `title_key`.")
#             elif title_key and not text_key:
#                 raise ValueError("You must supply both a `text_key` and a `title_key`.")
#             else:
#                 raise BaseException(f"Invalid keyword arguments.")
#             # Decode the text
#             if decode:
#                 df["text"] = df["text"].apply(lambda x: self._decode(x))
#             # Assign dummy titles if necessary
#             if not title_key:
#                 df["title"] = [Path(path).stem] * df.shape[1]
#             # Add a location column
#             df["locations"] = [path] * df.shape[1]
#             # Rename columns and convert to a dictionary
#             df = df.rename(columns={"title": "names", "text": "texts"})
#             # Create dataset object
#             return Dataset(path=path, data=df.to_dict(orient="records"))
#         except BaseException as e:
#             raise BaseException(f"Could not parse {path}: {e}")

#     def _load_lineated_text(self, path: str, decode: bool = True, **kwargs) -> Dataset:
#         """Load a plain text file with texts separated by line breaks.

#         Args:
#             path str: The path to the file to load.
#             decode (bool): Whether to decode the text.
#             **kwargs: Additional arguments which are ignored except `decode=False`.

#         Returns:
#             Dataset: A dataset object.
#         """
#         if "decode" in kwargs:
#             decode = kwargs["decode"]
#         with open(path, encoding="utf-8") as f:
#             if decode:
#                 data = [
#                     {
#                         "names": Path(path).stem,
#                         "locations": path,
#                         "texts": self._decode(line)[0:100],
#                     }
#                     for line in f.readlines()
#                 ]
#             else:
#                 data = [
#                     {
#                         "names": Path(path).stem,
#                         "locations": path,
#                         "texts": line,
#                     }
#                     for line in f.readlines()
#                 ]
#         return Dataset(path=path, data=data)
