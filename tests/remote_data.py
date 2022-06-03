"""remote_data.py.


This file contains a class for retrieving urls of test data in the GitHub repo.

Usage:

    ```python
    # Instantiate the class with the default "tests/test_data", another path,
    # or a list of paths.
    data = RemoteData()
    data = RemoteData(source="tests/test_pdf")
    data = RemoteData(source=["tests/test_data", "tests/test_pdf"])

    # Retrieve the data.
    print(data.files)
    print(data.folders)
"""

from pathlib import Path
from typing import List, Union

import requests


class RemoteData:
    """Class for retrieving locations of test data on GitHub."""

    def __init__(
        self,
        source: Union[List[str], str] = "tests/test_data/txt",
        user: str = "scottkleinman",
        repo: str = "lexos",
        branch: str = "main",
        recursive: bool = True,
    ):
        """Instantiate the object.

        Args:
            source (Union[List[str], str]): The path relative to the data directory (relative to the repo's root).
            user (str): The GitHub username.
            repo (str): The name of the GitHub repository.
            branch (str): The name of the repository branch to be accessed.
            recursive (bool): Whether or not to return a recursive tree
        """
        self.urls = []
        self.tree_url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}"
        self.raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}"
        if isinstance(source, str):
            self.source = [source]
        else:
            self.source = source
        self.queries = [f"{self.tree_url}:{s}" for s in self.source]
        if recursive:
            self.queries = [f"{q}?recursive=1" for q in self.queries]
        for query in self.queries:
            r = requests.get(query)
            result = r.json()
            [self.urls.append(p["path"]) for p in result["tree"]]

    @property
    def files(self) -> list:
        """List files returned from the source path."""
        return self.get_files()

    @property
    def folders(self) -> list:
        """List folders returned from the source path."""
        return self.get_folders()

    def get_files(self) -> list:
        """Get all the files returned from the source path."""
        return [f"{self.raw_url}/{x}" for x in self.urls if Path(x).suffix]

    def get_folders(self) -> list:
        """Get all the folders returned from the source path."""
        return [x for x in self.urls if not Path(x).suffix]