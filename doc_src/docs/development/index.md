# Development Notes

- Dependency management is handled with [Poetry](https://python-poetry.org/).
- The code is tested with [pytest](https://docs.pytest.org/en/7.0.x/) as a pre-commit hook.
- Before commit, I generally run [isort](https://pycqa.github.io/isort/) and [interrogate](https://interrogate.readthedocs.io/en/latest/) to ensure consistent imports and docstrings, but these are not currently implemented as pre-commit hooks.
- Docstrings are given an fully as possible in Google style, with as much type hinting as possible. Docstrings are used by [mkdocs](https://www.mkdocs.org/) to auto-generate the documentation through the magic of [mkdocstrings](https://mkdocstrings.github.io/).
- A number of test scripts have been implemented, and they are used by the continuous integration process. However, the scripts are incomplete and intended primarily for quick testing from the command line. A fuller test suite is intended once the API is more complete.

More information for developers will be added soon. In the meantime, see the <a href="https://github.com/scottkleinman/lexos/wiki" target="_blank">GitHub Wiki</a> for information on getting started.
