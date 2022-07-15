![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/scottkleinman/lexos?sort=semver)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python wheels](https://img.shields.io/badge/wheels-%E2%9C%93-4c1.svg?longCache=true&style=flat-square&logo=python&logoColor=white)](https://github.com/explosion/wheelwright/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![license](https://img.shields.io/github/license/scottkleinman/lexos)](https://img.shields.io/github/license/scottkleinman/lexos)

## Introduction

The Lexos API is a library of methods for programmatically implementing and extending the functionality in the <a href="http://lexos.wheatoncollege.edu/" target="_blank">Lexos</a> text analysis tool. Eventually, the web app will be rewritten to use the API directly. The goal of this alpha stage of development is to reproduce (and in some cases extend) the functionality of the current web app.

For the moment, much of the thinking behind the API's architecture is explained in the [Tutorial](tutorial).

**Current Status:** v0.0.1-alpha

## Features

<li>Loads texts from a variety of sources.</li>
<li>Manages a corpus of texts.</li>
<li>Performs text pre-processing ("scrubbing") and splitting ("cutting").</li>
<li>Performs tokenization and trains language models using <a href="https://spacy.io/" target="_blank">spaCy</a>.</li>
<li>Creates assorted visualizations of term vectors.</li>
<li>Generates topic models and topic model visualizations using <a href="https://github.com/mimno/Mallet" target="_blank">MALLET</a> and <a href="https://github.com/agoldst/dfr-browser" target="_blank">dfr-browser</a>.</li>

An expanded set of features is planned for the future.
