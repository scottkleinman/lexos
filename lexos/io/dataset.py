"""dataset.py.

This class currently supports data in the following formats:
  - dicts, line-delimited texts, csv, tsv, Excel, dict, json, and jsonl formats
  - Input from strings, filepaths, or urls (except for dict and Excel data)
  - Lists of the above, provided that all items in the list are of the same format

Tested:
  - string and file inputs for line-delimited texts, csv, tsv, Excel, dict, json, and jsonl formats
  - local directories with files of the same format (DatasetLoader)
  - zip archives (DatasetLoader)

To do:
  - Test dict format
  - Test remote directories (DatasetLoader)

Notes:
  - The Dataset class accepts only single files as input. To load lists of files or zipped files,
    use the DatasetLoader class.
  - There could be some issues reading in encoding.
  - Also, the class does not currently decode dataset data.
"""

import io
import itertools
import tempfile
import zipfile
from pathlib import Path
from typing import IO, Any, AnyStr, Dict, Iterable, List, Optional, Type, TypeVar, Union

import pandas as pd
from pydantic import BaseModel
from smart_open import open

from lexos import utils
from lexos.exceptions import LexosException

Model = TypeVar("Model", bound="BaseModel")


class Dataset(BaseModel):
    """Dataset class."""

    data: Optional[List[Dict[str, str]]] = None

    class Config:
        """Config class."""

        arbitrary_types_allowed = True

    @property
    def locations(self) -> List[str]:
        """Return the locations of the object data.

        Returns:
            List[str]: The locations of the object data.
        """
        if any("location" in item for item in self.data):
            return [item["locations"] for item in self.data]
        else:
            return None

    @property
    def names(self) -> List[str]:
        """Return the names of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return [item["title"] for item in self.data]

    @property
    def texts(self) -> List[str]:
        """Return the texts of the object data.

        Returns:
            List[str]: The texts of the object data.
        """
        return [item["text"] for item in self.data]

    def __iter__(self) -> Iterable:
        """Iterate over the dataset.

        Returns:
            Iterable: The dataset.
        """
        for item in iter(self.data):
            yield item

    def __getitem__(self, item: int) -> Dict[str, str]:
        """Get an item from dataset.

        Args:
            item: The index of the item to get.

        Returns:
            Dict[str, str]: The item at the given index.
        """
        return self.data[item]

    def df(self) -> pd.DataFrame:
        """Return the dataframe of the object data.

        Returns:
            pd.DataFrame: The dataframe of the object data.
        """
        return pd.DataFrame(self.data)

    @classmethod
    def parse_csv(
        cls: Type["Model"],
        source: str,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse CSV/TSV texts into the Dataset object.

        Args:
            source (str): The string or path to file containing the texts to parse.
            title_col (Optional[str]): The column name to convert to "title".
            text_col (Optional[str]): The column name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        source = cls._get_file_like(source)
        df = pd.read_csv(source, **kwargs)
        if title_col:
            df = df.rename(columns={title_col: "title"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "CSV and TSV files must contain headers named `title` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`title_col` and `text_col` parameters.",
            )
            raise LexosException("".join(err))
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_dict(cls: Type["Model"], source: dict,) -> "Model":
        """Alias for cls.parse_obj().

        Args:
            source (dict): The dict to parse.

        Returns:
            Model: A dataset object.
        """
        return cls.parse_obj({"data": source})

    @classmethod
    def parse_excel(
        cls: Type["Model"],
        source: str,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse Excel files into the Dataset object.

        Args:
            source (str): The path to the Excel file containing the texts to parse.
            title_col (Optional[str]): The column name to convert to "title".
            text_col (Optional[str]): The column name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        try:
            df = pd.read_excel(source, **kwargs)
        except Exception as e:
            raise LexosException(f"Could not read {source}: {e}")
        if title_col:
            df = df.rename(columns={title_col: "title"})
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "Excel files must contain headers named `title` and `text`. ",
                "You can convert the names of existing headers to these with the ",
                "`title_col` and `text_col` parameters.",
            )
            raise LexosException(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_json(
        cls: Type["Model"],
        source: str,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse JSON files or strings.

        Args:
            source (str): The json string to parse.
            title_field (Optional[str]): The field name to convert to "title".
            text_field (Optional[str]): The field name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        try:
            with open(source) as f:
                df = pd.read_json(f, **kwargs)
        except Exception:
            df = pd.read_json(io.StringIO(source), **kwargs)
        if title_field:
            df = df.rename(columns={title_field: "title"})
        if text_field:
            df = df.rename(columns={text_field: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "JSON files must contain fields named `title` and `text`. ",
                "You can convert the names of existing fields to these with the ",
                "`title_field` and `text_field` parameters.",
            )
            raise LexosException(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_jsonl(
        cls: Type["Model"],
        source: str,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> "Model":
        """Parse lineated texts into the Dataset object.

        Args:
            source (str): The string or path to file containing the lines to parse.
            title_field (Optional[str]): The field name to convert to "title".
            text_field (Optional[str]): The field name to convert to "text".

        Returns:
            Model: A dataset object.
        """
        source = cls._get_file_like(source)
        df = pd.read_json(source, lines=True, **kwargs)
        if title_field:
            df = df.rename(columns={title_field: "title"})
        if text_field:
            df = df.rename(columns={text_field: "text"})
        if "title" not in df.columns or "text" not in df.columns:
            err = (
                "JSON and JSONL files must contain fields named `title` and `text`. ",
                "You can convert the names of existing fields to these with the ",
                "`title_field` and `text_field` parameters.",
            )
            raise LexosException(err)
        return cls.parse_obj({"data": df.to_dict(orient="records")})

    @classmethod
    def parse_string(
        cls: Type["Model"],
        source: str,
        labels: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
    ) -> "Model":
        """Parse lineated texts into the Dataset object.

        Args:
            source (str): The string containing the lines to parse.
            labels (Optional[List[str]]): The names of the texts.
            locations (Optional[List[str]]): The locations of the texts.

        Returns:
            Model: A dataset object.
        """
        if not labels:
            raise LexosException(
                "Please use the `labels` argument to provide a list of labels for each row in your data."
            )
        # Handle files
        try:
            with open(source, "r", encoding="utf-8") as f:
                source = f.readlines()
        # Handle strings
        except Exception:
            source = source.split("\n")
        if len(labels) != len(source):
            raise LexosException(
                f"The number of labels ({len(labels)}) does not match the number of lines ({len(source)}) in your data."
            )
        else:
            data = [{"title": labels[i], "text": line} for i, line in enumerate(source)]
            if locations:
                if len(locations) == len(source):
                    for i, _ in enumerate(data):
                        data[i]["locations"] = locations[i]
                else:
                    raise LexosException(
                        f"The number of locations ({len(locations)}) does not match the number of lines ({len(source)}) in your data."
                    )
            return cls.parse_obj({"data": data})

    @staticmethod
    def _get_file_like(source: str) -> IO[AnyStr]:
        """Read the source into a buffer.

        Args:
            source: str: A path or string containing the source.

        Returns:
            IO[AnyStr]: A file-like object containing the source.
        """
        if utils.is_file(source) or utils.is_github_dir(source) == False:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    source = f.read()
            except:
                pass
            return io.StringIO(source)
        else:
            raise LexosException(f"{source} is not a valid file path or input string.")


class DatasetLoader:
    """Loads a dataset.

    Usage:
        loader = DatasetLoader(source)
        dataset = loader.dataset

    Notes:
      - Different types of data may require different keyword parameters. Error messages
        provide some help in identifying what keywords are required.
      - The class will handle lists of sources, but errors may occur if the sources are
        of different formats or require different arguments or argument values.
    """

    def __init__(
        self,
        source: Any,
        labels: List[str] = None,
        locations: Optional[List[str]] = None,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_col: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> Union[Dataset, List[Dataset]]:
        """Initialise the loader.

        Args:
            source (Any): The source type to detect.
            labels (List[str]): The labels to use.
            locations (Optional[List[str]]): The locations of the texts.
            title_col (str): The name of the column containing the titles.
            text_col (str): The name of the column containing the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_col (str): The name of the column containing the locations.
            location_field (str): The name of the field containing the locations.

        Returns:
            Dataset: A Dataset or list of Dataset object.
        """
        if isinstance(source, list):
            new_data = [
                self.load(
                    item,
                    labels,
                    locations,
                    title_col,
                    text_col,
                    title_field,
                    text_field,
                    location_col,
                    location_field,
                    **kwargs,
                ).data
                for item in source
            ]
            # Data a Dataset with the flattened list of dicts
            self.data = Dataset(data=list(itertools.chain(*new_data)))
        else:
            self.data = self.load(
                source,
                labels,
                locations,
                title_col,
                text_col,
                title_field,
                text_field,
                location_col,
                location_field,
                **kwargs,
            )

    @property
    def locations(self) -> List[str]:
        """Return the locations of the object data.

        Returns:
            List[str]: The locations of the object data.
        """
        if any("location" in item for item in self.data):
            return [item["locations"] for item in self.data]
        else:
            return None

    @property
    def names(self) -> List[str]:
        """Return the names of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return [item["title"] for item in self.data]

    @property
    def texts(self) -> List[str]:
        """Return the texts of the object data.

        Returns:
            List[str]: The names of the object data.
        """
        return [item["text"] for item in self.data]

    def __iter__(self) -> Iterable:
        """Iterate over the dataset.

        Returns:
            Iterable: The dataset.
        """
        for item in iter(self.data):
            yield item

    def __getitem__(self, item: int) -> Dict[str, str]:
        """Get an item from dataset.

        Args:
            item: The index of the item to get.

        Returns:
            Dict[str, str]: The item at the given index.
        """
        return self.data[item]

    def load(
        self,
        source: Any,
        labels: List[str] = None,
        locations: Optional[List[str]] = None,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_col: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> Dataset:
        """Load the given file.

        Args:
            source (Any): The source the data to load.
            labels (List[str]): The labels to use.
            locations (Optional[List[str]]): The locations of the texts.
            title_col (str): The name of the column containing the titles.
            text_col (str): The name of the column containing the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_col (str): The name of the column containing the locations.
            location_field (str): The name of the field containing the locations.

        Returns:
            Dataset: A Data object.
        """
        if not utils.is_dir(source) and not utils.is_github_dir(
            source
        ):  # and not utils.is_url(source):
            ext = Path(source).suffix
            if ext == "" or ext == ".txt":
                return Dataset.parse_string(source, labels, locations)
            elif ext == ".csv":
                return Dataset.parse_csv(source, title_col, text_col, **kwargs)
            elif ext == ".tsv":
                return Dataset.parse_csv(source, title_col, text_col, **kwargs)
            elif ext == ".xlsx":
                return Dataset.parse_excel(source, title_col, text_col, **kwargs)
            elif ext == ".json":
                return Dataset.parse_json(source, title_field, text_field, **kwargs)
            elif ext == ".jsonl":
                return Dataset.parse_jsonl(source, title_field, text_field, **kwargs)
            elif ext == ".zip":
                return self._load_zip(
                    source,
                    labels,
                    locations,
                    title_col,
                    text_col,
                    title_field,
                    text_field,
                    location_col,
                    location_field,
                    *kwargs,
                )
        elif utils.is_dir(source) or utils.is_github_dir(source):
            new_data = []
            if utils.is_github_dir(source):
                paths = utils.get_github_raw_paths(source)
            else:
                paths = utils.get_paths(source)
            for path in paths:
                new_data.append(
                    self.load(
                        path,
                        labels,
                        locations,
                        title_col,
                        text_col,
                        title_field,
                        text_field,
                        location_col,
                        location_field,
                        **kwargs,
                    )
                )
            # Return a Dataset with the flattened list of dicts
            return Dataset(data=list(itertools.chain(*new_data)))
        else:
            raise LexosException(
                f"{source} is an unknown source type or requires different arguments than the other sources in the directory."
            )

    def _load_zip(
        self,
        file_path: str,
        labels: List[str] = None,
        locations: Optional[List[str]] = None,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        title_field: Optional[str] = None,
        text_field: Optional[str] = None,
        location_col: Optional[str] = None,
        location_field: Optional[str] = None,
        **kwargs: Dict[str, str],
    ) -> Dataset:
        """
        Load a zip file.

        Args:
            file_path (str): The path to the file to load.
            source (Any): The source the data to load.
            labels (List[str]): The labels to use.
            locations (Optional[List[str]]): The locations of the texts.
            title_col (str): The name of the column containing the titles.
            text_col (str): The name of the column containing the texts.
            title_field (str): The name of the field containing the titles.
            text_field (str): The name of the field containing the texts.
            location_col (str): The name of the column containing the locations.
            location_field (str): The name of the field containing the locations.

        Returns:
            Dataset: A Data object.
        """
        new_data = []
        with open(file_path, "rb") as f:
            with zipfile.ZipFile(f) as zip:
                with tempfile.TemporaryDirectory() as tempdir:
                    zip.extractall(tempdir)
                    for tmp_path in Path(tempdir).glob("**/*"):
                        if (
                            tmp_path.is_file()
                            and not tmp_path.suffix == ""
                            and not str(tmp_path).startswith("__MACOSX")
                            and not str(tmp_path).startswith(".ds_store")
                        ):
                            new_data.append(
                                self.load(
                                    tmp_path,
                                    labels,
                                    locations,
                                    title_col,
                                    text_col,
                                    title_field,
                                    text_field,
                                    location_col,
                                    location_field,
                                    **kwargs,
                                ).data
                            )
        # Return a Dataset with the flattened list of dicts
        return Dataset(data=list(itertools.chain(*new_data)))
