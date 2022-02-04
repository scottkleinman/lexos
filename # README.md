# README

This file provides instructions for getting up and running with Poetry and the Lexos API.

## Install Poetry

The Poetry website contains extensive documentation on [how to install it](https://python-poetry.org/docs/#installation) on different operating systems.

A Mac-centric approach is:

```linux
$ brew install poetry
$ poetry --version
Poetry version 1.1.11
```

## Create the Package Directory

In the command line, go to the location where you would like to create a new package. Do not create the package directory (Poetry will do that). For a package called "Lexos", initialise a new package with

```linux
First, initialize a new library project with Poetry:
$ poetry new lexos
Created package lexos in lexos
```

Next, `cd` into the package directory and run `poetry install`. This will create a new virtual environment and install the initial set of dependencies.

## Set Up the GitHub Repo

Start by initialising git and committing all the generated files.

```linux
[lexos] $ git init
[lexos] $ git checkout -b main
[lexos] $ git add *
[lexos] $ git commit -m "Initial commit"
```

Create a repo on GitHub, add this repository as a remote to your local Git repository, and push all changes (but see below if you want it private):

```linux
[lexos] $ git remote add origin git@github.com:<YOUR_GITHUB_USERNAME>/lexos.git
[lexos] $ git branch -M main
[lexos] $ git push -u origin main
```

For information on how to make the repo private, see "Using Libraries in a Private Organisation" below.

## Add Some Code and Tests

Open the generated file `test/test_lexos.py` and add a first unit-test:

```python
def test_initials():
    from lexos.initials import initials
    assert initials('Guide van Rossum') == 'GvR'
```

Next, run the unit-tests (using Poetry):

```linux
[lexos] $ poetry run pytest
```

This will fail because there is no function called initials. To fix the test add the following code to a new file: `lexos/initials.py`:

```python
def initials(fullname: str) -> str:
    return 'GvR'
```

Next, run the unit-tests again to check they all pass.

## Set Up Continuous Integration

To have the unit-tests run on every code push you can setup a simple workflow in GitHub Actions (there are specific Actions for Poetry available that you can use.)

Create a new file in you repository called `ci.yml` in the directory `.github/workflows` with the following content:

```yaml
name: 'CI'
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
    - name: 'Checkout'
      uses: actions/checkout@v2
    - name: 'Set up Python'
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: 'Set up Poetry'
      uses: snok/install-poetry@v1
    - name: 'Install dependencies'
      run: poetry install --no-interaction --no-root
    - name: 'Run unit-tests'
      run: poetry run pytest
```

Add all new files to Git and push your changes to GitHub. On the “Actions” tab of your repository in GitHub you should now see the first run of the CI pipeline.

## Using Libraries in a Private Organisation

!!! warning
    Working with private repositories creates a lot of headaches.

After running `poetry install`, add the dependency to the main branch of the GitHub lexos repository:

```linux
[lexos] $ poetry add git+https://github.com/<YOUR_GITHUB_USERNAME>/lexos.git#main
```

The # suffix can be used to select branches, tags or even specific commit hashes, in this case the main branch is selected.

If your repositories are private use `git+ssh://` links and make sure everyone on the team (and your CI/CD pipelines) can access the repository using private keys.

To check the library was included successfully, start a shell inside your Poetry virtual environment and make a call to it:

```linux
[lexos] $ poetry shell
Spawning shell within /../lexos-2UmXNoUn-py3.9
$ python
Python 3.9.7 (default, Sep  3 2021, 12:37:55)
>>> from lexos import initials
>>> initials('Guido van Rossum')
'GvR'
```

## Publish to PyPi (Eventually)

Poetry has two commands that make publishing to the central Python package repository a breeze: `build` and `publish`

(Before publishing a library to the open source community, make sure your code and documentation follow the best practices from the [Python Packaging User Guide](https://packaging.python.org/) ✨)

First you need to build the distribution builds:

```linux
[lexos] $ poetry build
Building lexos (1.0.0)
  - Building sdist
  - Built lexos-1.0.0.tar.gz
  - Building wheel
  - Built lexos-1.0.0-py3-none-any.whl
```

The directory `dist` now contains a source distribution (`sdist`) file called `lexos-1.0.0.tar.gz` and a pre-built distribution (wheel) file called `lexos-1.0.0-py3-none-any.whl`.

Next you can publish both distribution builds to the Python package repository:

```linux
[lexos] poetry publish
Publishing lexos (1.0.0) to PyPI
 - Uploading lexos-1.0.0-py3-none-any.whl 100%
 - Uploading lexos-1.0.0.tar.gz 100%
```

And, voilà, that’s it, shipped! 🚀

Adapted from Rob van der Leek, [Avoid the Snake Pit of Python Package Management With Poetry](https://betterprogramming.pub/avoid-the-snake-pit-of-python-package-management-with-poetry-54ab186cf2a4).