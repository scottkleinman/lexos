# The Lexos Python Library

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/scottkleinman/lexos?sort=semver)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31212/)
[![Python wheels](https://img.shields.io/badge/wheels-%E2%9C%93-4c1.svg?longCache=true&style=flat-square&logo=python&logoColor=white)](https://github.com/scottkleinman/lexos/releases)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/ambv/ruff)
[![license](https://img.shields.io/github/license/scottkleinman/lexos)](https://img.shields.io/github/license/scottkleinman/lexos)
[![Coverage](https://img.shields.io/badge/coverage-97%25-blue)](https://img.shields.io/badge/coverage-97%25-blue)

The Lexos Python library reproduces and extends most of the text analysis tools in the <a href="http://lexos.wheatoncollege.edu/" target="_blank">Lexos web app</a>. Lexos is designed to implement many common text analysis procedures in a way that saves the user having to re-invent the wheel or figure out how to combine multiple Python packages to achieve a given result. It is intended to be used as a library in other projects to build backend functions for applications, but it can be used in standalone scripts or in Jupyter notebooks. As with the original web app, it is designed to accessible to entry-level users whilst offering power functionality for students and researchers, particularly in the Humanities. It is also designed to be as language-agnostic as possible so that it can be used for a wide variety of historical and under-resourced languages.

## 📖 Documentation

A full discussion of the use of the API can be found on the website <a href="https://scottkleinman.github.io/lexos/" target="_blank">Documentation website</a>.

## ⭐️ Major Features

- Loads texts from a variety of sources into a common data structure.
- Manages a corpus of texts and generates stastics about the corpus.
- Performs text pre-processing ("scrubbing") and splitting ("cutting").
- Performs tokenization and trains language models using <a href="https://spacy.io/" target="_blank">spaCy</a>.
- Creates assorted visualizations of term vectors.
- Performs hierarchical and kmeans clustering.
- Generates topic models and topic model visualizations using <a href="https://github.com/mimno/Mallet" target="_blank">MALLET</a> and <a href="https://github.com/scottkleinman/dfr-browser2" target="_blank">DFR Browser 2</a>.

And more!

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

Lexos uses <a href="https://spacy.io/" target="_blank">spaCy</a> language models to obtain language-specific information about texts. By default, it comes with spaCy's multi-language model `xx_sent_ud_sm` and its small English-language `en_core_web_sm`.

If you are working in another language or need a larger language model, you can download instructions for additional models from the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page. Use the following command:

```bash
python -m spacy download en_core_web_md # Replace with the name of your model
```

## 🚦Project Status

The Lexos API is currently in beta. Most of the core functionality of the Lexos web app, along with new features, has been implemented and documented. As of January 2026, the API is considered feature complete and stable for general use, but some rough edges remain. Feedback is welcome.

I will continue to fix bugs and improve the documentation as issues arise, but no major new features are planned at this time. The beta release coincides with at a time when the landscape of digital tools is rapidly evolving and AI-assisted coding is becoming more prevalent. I am waiting to see whether there is significant adoption of the Lexos library before investing more time in developing new features. If you like Lexos, you can help by requesting new features in the <a href="https://github.com/scottkleinman/uv_lexos/issues" target="_blank">GitHub issues</a> (labelled as "enhancement") or [contributing](development/index.md) them yourself.

## 💝 Contribute

- If are looking for help using Lexos, please post you question on the <a href="https://github.com/scottkleinman/lexos/discussions" target="_blank">GitHub Discussions board</a>.
- Bug reports and feature requests: Please use [GitHub issues](https://github.com/scottkleinman/lexos/issues).
- For other types of contributions see the <a href="https://scottkleinman.github.io/lexos/development/" target="_blank">Documentation website</a>.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation Information

Kleinman, S., (2026). Lexos. v0.1.0b3 https://github.com/scottkleinman/lexos. <a href="https://doi.org/10.5281/zenodo.18112380" target="_blank">doi:10.5281/zenodo.1403869</a>.
