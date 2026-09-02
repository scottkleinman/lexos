# The Lexos Python Library

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/scottkleinman/lexos?sort=semver)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31212/)
[![Python wheels](https://img.shields.io/badge/wheels-%E2%9C%93-4c1.svg?longCache=true&style=flat-square&logo=python&logoColor=white)](https://github.com/scottkleinman/lexos/releases)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/ambv/ruff)
[![license](https://img.shields.io/github/license/scottkleinman/lexos)](https://img.shields.io/github/license/scottkleinman/lexos)
[![Coverage](https://img.shields.io/badge/coverage-97%25-blue)](https://img.shields.io/badge/coverage-97%25-blue)

The Lexos Python library reproduces and extends most of the text analysis tools in the [Lexos web app](http://lexos.wheatoncollege.edu/). Lexos is designed to implement many common text analysis procedures in a way that saves the user having to re-invent the wheel or figure out how to combine multiple Python packages to achieve a given result. It is intended to be used as a library in other projects to build backend functions for applications, but it can be used in standalone scripts or in Jupyter notebooks. As with the original web app, it is designed to accessible to entry-level users whilst offering power functionality for students and researchers, particularly in the Humanities. It is also designed to be as language-agnostic as possible so that it can be used for a wide variety of historical and under-resourced languages.

## 📖 Documentation

A full discussion of the use of the API can be found on the website [Documentation website](https://scottkleinman.github.io/lexos/).

## ⭐️ Major Features

- Loads texts from a variety of sources into a common data structure.
- Manages a corpus of texts and generates stastics about the corpus.
- Performs text pre-processing ("scrubbing") and splitting ("cutting").
- Performs tokenization and trains language models using [spaCy](https://spacy.io/).
- Creates assorted visualizations of term vectors.
- Performs hierarchical and kmeans clustering with a variety of visualizations for text comparisons..
- Performs classification using [spaCy](https://spacy.io/), [scikit-learn](https://scikit-learn.org/stable/), and custom architectures.
- Generates topic models and topic model visualizations using [MALLET](https://github.com/mimno/Mallet) and [DFR Browser 2](https://github.com/scottkleinman/dfr-browser2).

And more!

### What's New in V0.2.0-Beta

- Performance optimizations and bug fixes througout.
- A simplified public API.
- Better HTML and XML tag handling in Scrubber.
- New and improved clustering visualizations.
- Improved plotting functions in Bootstrap Consensus clustering.
- Silhouette score analysis for KMeans clustering.
- More advanced topic modelling features, including termite plots, customisable labels, and LLM-generated labels with local or cloud-based models.
- A new Classification module that performs text classification using a variety of architectures.
- A new Structural Stylometry module that allows you to assess the significance of punctuation and whitespace patterns in your documentation.
- Language model training features.
- Improved documentation and tutorials for new users (now in a separate [lexos-docs](https://github.com/scottkleinman/lexos-docs) repository).

## ⏳ Installation

```bash
pip install lexos
```

To update to the latest version, use

```bash
pip install -U lexos
```

Or, if you are using `uv`:

```bash
uv add lexos
```

Lexos uses [spaCy](https://spacy.io/) language models to obtain language-specific information about texts. By default, it comes with spaCy's multi-language model `xx_sent_ud_sm` and its small English-language `en_core_web_sm`.

If you are working in another language or need a larger language model, you can download instructions for additional models from the [spaCy models](https://spacy.io/models) page. Use the following command:

```bash
python -m spacy download en_core_web_md # Replace with the name of your model
```

## 🚦Project Status

The Lexos API is currently in beta. Most of the core functionality of the Lexos web app, along with new features, has been implemented and documented. As of January 2026, the API is considered feature complete and stable for general use, but some rough edges remain. Feedback is welcome.

I will continue to fix bugs and improve the documentation as issues arise, but no major new features are planned at this time. The beta release coincides with at a time when the landscape of digital tools is rapidly evolving and AI-assisted coding is becoming more prevalent. I am waiting to see whether there is significant adoption of the Lexos library before investing more time in developing new features. If you like Lexos, you can help by requesting new features in the [GitHub issues](https://github.com/scottkleinman/lexos/issues) (labelled as "enhancement") or [contributing](https://scottkleinman.github.io/lexos/#contributing) them yourself.

## 💝 Contribute

- If are looking for help using Lexos, please post you question on the [GitHub Discussions board](https://github.com/scottkleinman/lexos/discussions).
- Bug reports and feature requests: Please use [GitHub issues](https://github.com/scottkleinman/lexos/issues).
- For other types of contributions see the [Documentation website](https://scottkleinman.github.io/lexos/development/).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation Information

Kleinman, S., (2026). Lexos. v0.2.0-beta https://github.com/scottkleinman/lexos. [doi:10.5281/zenodo.1403869](https://doi.org/10.5281/zenodo.18112380).
