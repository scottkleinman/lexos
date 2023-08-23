"""analyse_dependencies.py.

Generates a table showing versions of all installed dependencies,
and, optionally the latest versions available on pip.
"""

from typing import List, Union
import argparse
import requests
import tomllib
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table


def version_check():
    """Make sure that we are running Python 3.11 or above for `pyproject.toml`."""
    try:
        assert sys.version_info >= (3, 11)
    except AssertionError:
        raise (
            "This script can only parse `pyproject.toml` files if you are using Python 3.11 or above. Try using `requirements.txt` instead."
        )


class AnalyseDependencies:
    """Analyse dependencies."""

    def __init__(
        self,
        source: Union[List[str], str] = "pyproject.toml",
        latest: bool = False,
        dependencies: bool = True,
        optional: bool = True,
    ):
        """Construct object."""
        self.source = source
        self.latest = latest
        self.dependencies = dependencies
        self.optional = optional
        self.process_source()
        if self.latest:
            self.get_latest(self.dependencies, self.optional)

    def process_source(self):
        """Process source file."""
        self.dependencies = {}
        self.optional_dependencies = {}
        error_msg = "Sorry, at the moment, I can only parse `pyproject.toml` and `requirements.txt` files."
        if isinstance(self.source, str):
            if self.source.endswith("pyproject.toml"):
                version_check()
                self.process_pyproject()
            elif self.source.endswith("requirements.txt"):
                self.process_requirements()
            else:
                raise Exception(error_msg)
        elif isinstance(self.source, list):
            self.process_requirements()
        else:
            raise Exception(error_msg)

    def process_pyproject(self):
        """Process pyproject.toml file."""
        with open(self.source, "rb") as f:
            data = tomllib.load(f)
        self.dependencies = {
            f"{x.split()[0]}": {"current": "".join(x.split()[1:])}
            for x in data["project"]["dependencies"]
        }
        for k, v in data["project"]["optional-dependencies"].items():
            if isinstance(v, list):
                for item in v:
                    package = item.split()[0]
                    current = "".join(item.split()[1:])
                    self.optional_dependencies[package] = {"current": current}
            else:
                self.optional_dependencies[k] = {"current": v}

    def process_requirements(self):
        """Process requirements.txt file."""

        def read(file: str) -> List[str]:
            """Read a requirements.txt file.

            Args:
                line: A line from the source.

            Returns:
                A list of lines.
            """
            with open(file, "r") as f:
                lines = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() != "" and not line.strip().startswith("#")
                ]
            return lines

        def get_package_version(line: str):
            """Get package and version from lines.

            Args:
                line: A line from the source.

            Returns:
                The package and version.
            """
            package, version = line.split("==")
            return package.strip(), version.strip()

        if isinstance(self.source, str):
            files = [self.source]
        else:
            files = self.source
        for file in files:
            lines = read(file)
            for line in lines:
                package, version = get_package_version(line)
                if file.split("/")[-1] == "requirements.txt":
                    self.dependencies[package] = {"current": version}
                else:
                    self.optional_dependencies[package] = {"current": version}

    def get_latest(self, dependencies: bool = True, optional: bool = True):
        """Get the latest version of a package.

        Args:
            dependencies: Whether to return versions of dependencies.
            optional: Whether to return versions of optional dependencies.
        """

        def make_request(package: str) -> str:
            """Request the latest version for an individual package.

            Args:
                dependency: A dict in which the key is the name of the package.

            Returns:
                The latest version of the package
            """
            try:
                response = requests.get(f"https://pypi.org/pypi/{package}/json")
                latest = response.json()["info"]["version"]
            except ValueError:
                latest = "unknown"
            return latest

        if dependencies:
            for package in self.dependencies:
                latest = make_request(package)
                self.dependencies[package]["latest"] = latest
        if optional:
            for package in self.optional_dependencies:
                latest = make_request(package)
                self.optional_dependencies[package]["latest"] = latest

    def to_df(self, dependencies: bool = True, optional: bool = True) -> pd.DataFrame:
        """Convert the object to a dataframe.

        Args:
            dependencies: Whether to return versions of dependencies.
            optional: Whether to return versions of optional dependencies.

        Returns:
            A pandas dataframe.
        """
        data = []
        latest = False
        if dependencies and len(self.dependencies) == 0:
            dependencies = False
        if optional and len(self.optional_dependencies) == 0:
            optional = False
        if dependencies:
            first_val = list(self.dependencies.values())[0]
            if "latest" in first_val.keys():
                latest = True
            for k, v in self.dependencies.items():
                if "latest" in v.keys():
                    data.append(
                        {"package": k, "current": v["current"], "latest": v["latest"]}
                    )
                else:
                    data.append({"package": k, "current": v["current"]})
        if optional:
            first_val = list(self.optional_dependencies.values())[0]
            if "latest" in first_val.keys():
                latest = True
            for k, v in self.optional_dependencies.items():
                if "latest" in v.keys():
                    data.append(
                        {"package": k, "current": v["current"], "latest": v["latest"]}
                    )
                else:
                    data.append({"package": k, "current": v["current"]})
        if latest:
            columns = ["package", "current", "latest"]
        else:
            columns = ["package", "current"]
        return pd.DataFrame(data, columns=columns)

    def to_console(self) -> None:
        """Print the analysis to the console."""
        df = self.to_df()
        table = Table(
            title="Dependency Comparison",
            title_style="bold white",
            header_style="bold white",
        )
        styles = ["bold cyan", "bold magenta", "bold green"]
        for i, col in enumerate(df.columns):
            table.add_column(col, style=styles[i])
        rows = df.to_records(index=False).tolist()
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)


def run(
    source: Union[List[str], str],
    get_latest: bool = False,
    dependencies: bool = True,
    optional_dependencies: bool = True,
):
    """Run from the command line."""
    console = Console()
    console.print("\nAnalysing...\n", style="bold green")
    a = AnalyseDependencies(source, get_latest, dependencies, optional_dependencies)
    a.to_console()
    console.print("\nDone!\n", style="bold green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Figure out how to accept multiple sources as a list.
    # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument(
        "source", nargs="+", help="Path(s) to the `pyproject.toml` file."
    )
    parser.add_argument(
        "--get_latest", help="Get the latest version numbers.", action="store_true"
    )
    parser.add_argument(
        "--no_dependencies",
        help="Do not analyse required dependencies.",
        action="store_true",
    )
    parser.add_argument(
        "--no_optional_dependencies",
        help="Do not analyse optional dependencies.",
        action="store_true",
    )
    args = parser.parse_args()
    dependencies = True
    optional = True
    get_latest = False
    if isinstance(args.source, list) and len(args.source) == 1:
        source = args.source[0]
    else:
        source = args.source
    if args.get_latest:
        get_latest = True
    if args.no_dependencies:
        dependencies = False
    if args.no_optional_dependencies:
        optional = False
    run(source, get_latest, dependencies, optional)
