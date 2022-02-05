# Installation

During the development process, dependency management and packaging are handled using <a href="https://python-poetry.org/" target="_blank">Poetry</a>. When the API is released, you will be able to install it with `pip install lexos-api`. Other methods of installation will be added to this page at a later date.

## Installing the Lexos API with Poetry

A Mac-centric approach to installing Poetry is as follows:

```linux
$ brew install poetry
$ poetry --version
Poetry version 1.1.11
```

The Poetry website contains extensive documentation on [how to install it](https://python-poetry.org/docs/#installation) on different operating systems.

My running theory is that they way to install the Lexos API prior to publication is to complete following steps:

1. Install Poetry.
2. Clone the repo.
3. `cd` to the local repo directory.
4. Run `poetry init`.

This is as yet untested.
