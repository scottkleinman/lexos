# Introduction

The Lexos API is a library of methods for programmatically implementating and extending the functionality found in the <a href="http://lexos.wheatoncollege.edu/" target="_blank">Lexos</a> web app. Eventually, the web app will be rewritten to use the API directly. The goal of this stage of development is to reproduce (and in some cases extend) the functionality of the current web app.

For the moment, much of the thinking behind the API's architecture is explained in the [Tutorial](tutorial).

## Current Status (v0.0.1)

So far, only the basic architecture of the API has been built. The `Loader` class will accept any local file, regardless of format, and it will also download text from URLs. Obviously, there are some security features that need to be added. It would also be nice to load from different file formats (json, docx, zip, etc.), which is not currently supported.

All of the functionality of the Lexos app's `scrubber` module has been ported over, and the basic `tokenizer` module works. However, there needs to be some error checking in both modules.

The `cutter` module will need some consideration, as it will probably require a combination of features from `scrubber` and `tokenizer`, depending on whether the user wants to cut based on some pattern or cut by token ranges.

## Development Notes

- Dependency management is handled with [Poetry](https://python-poetry.org/).
- The code is tested with [pytest](https://docs.pytest.org/en/7.0.x/) as a pre-commit hook.
- Before commit, I generally run [isort](https://pycqa.github.io/isort/) and [interrogate](https://interrogate.readthedocs.io/en/latest/) to ensure consistent imports and docstrings, but these are not currently implemented as pre-commit hooks.
- Docstrings are given an fully as possible in Google style, with as much type hinting as possible. Docstrings are used by [mkdocs](https://www.mkdocs.org/) to auto-generate the documentation through the magic of [mkdocstrings](https://mkdocstrings.github.io/).
- A number of test scripts have been implemented, and they are used by the continuous integration process. However, the scripts are incomplete and intended primarily for quick testing from the command line. A fuller test suite is intended once the API is more complete.
