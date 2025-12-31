# Installation

The information below describes how to install Lexos as a user. If you are interested in contributing to the Lexos source code or documentation, see the separate [Development](development/index.md) documentation.

## Installing Python

Lexos requires Python 3.12 or greater. Our development environment is <code><a href="https://docs.astral.sh/uv/" target="blank">uv</a></code>, and Lexos should work in a Python virtual environment created using that tool. If you are using a different Python environment, you can install Lexos using `pip`.

## Installing the Lexos Package

If using uv, run

```bash
uv add lexos
```

Otherwise, you can install Lexos using pip:

```bash
pip install lexos
```

This will install the Lexos API and all of its dependencies.

By default, `uv` installs the latest version of Lexos. To update to the latest version with `pip`, use

```bash
pip install -U lexos
```

## Downloading Language Models

Many features of Lexos use language models created for the Python <a href="https://spacy.io/" target="_blank">spaCy</a> natural language processing library. When you install Lexos, spaCy's multi-language model <code><a href="https://spacy.io/models/xx#xx_sent_ud_sm" target="_blank">xx_sent_ud_sm</a></code> and small English model <code><a href="https://spacy.io/models/en#en_core_web_sm" target="_blank">en_core_web_sm</a></code> are installed. For information on how Lexos uses language models, see [Tokenizing Texts](user_guide/tokenizing_texts.md).

## Downloading Additional Language Models (Optional)

The `xx_sent_ud_sm` model is a minimal model that can be used for sentence and token segmentation in a variety of languages, while the `en_core_web_sm` model is specifically for English text. If you are working in another language or need a larger language, you may need to download additional language models. You can find information on available models on the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page.

To download a model (for instance, the small Chinese model `zh_core_web_sm`), you can run the following commands in your terminal.

If you are using `uv`, run:

```bash
uv run python -m spacy download zh_core_web_sm
```

or, if you are not using `uv`, you can run:

```bash
python -m spacy download zh_core_web_sm
```

## Verify Installation

To verify that Lexos is installed correctly, you can run the following command in your terminal:

If you are using `uv`:

```bash
uv run python -m lexos --version
```

or, if you are not using `uv`:

```bash
python -m lexos --version
```

If you are using a Jupyter notebook, you can also check the installation by running the following code in a cell:

```python
import lexos
print(lexos.__version__)
```

This should display the version of Lexos that is installed. If you see an error, please check your installation steps or refer to the [Troubleshooting](#troubleshooting) section below.

## Troubleshooting

Below are some common issues and solutions you may encounter during installation:

### 1. Python Version Error

**Issue:** You see an error like `ModuleNotFoundError` or `SyntaxError` when installing or running Lexos.

**Solution:**
Lexos requires Python 3.12 or greater. Check your Python version with:

```bash
python --version
```

If your version is lower than 3.12, please install a compatible version and create a new virtual environment.

### 2. `uv` or `pip` Not Found

**Issue:** The terminal says `uv: command not found` or `pip: command not found`.

**Solution:**
Make sure you have installed <code><a href="https://docs.astral.sh/uv/" target="blank">uv</a></code> or <code><a href="https://pip.pypa.io/en/stable/installation/" target="blank">pip</a></code>. If not, follow the official installation instructions for your platform.

### 3. Permission Denied Errors

**Issue:** You see `Permission denied` when installing packages.

**Solution:**
Always use a virtual environment for Lexos. If you must install globally, you may need to use `sudo` on Mac and Linux systems, but this is not recommended. Prefer using a virtual environment to avoid permission issues.

### 4. spaCy Model Not Found

**Issue:** You see an error like `OSError: [E050] Can't find model 'en_core_web_sm'` or similar when running Lexos features that use spaCy.

**Solution:**
Install the required spaCy model using one of the following commands:

```bash
uv run python -m spacy download en_core_web_sm
# or, if not using uv:
python -m spacy download en_core_web_sm
```

### 5. Lexos Not Found After Installation

**Issue:** Running `python -m lexos --version` or `import lexos` fails with `ModuleNotFoundError`.

**Solution:**
Ensure you are in the correct virtual environment where Lexos was installed. Activate your environment and try again. If the problem persists, reinstall Lexos using `uv add lexos` or `pip install lexos`.

### 6. Outdated `pip` or `uv`

**Issue:** Installation fails with errors about incompatible or missing dependencies.

**Solution:**
Update your package manager:

```bash
pip install --upgrade pip
# or
uv pip install --upgrade uv
```

If you encounter any other problems not covered here, please consider reaching out to the Lexos community Discussion forum on GitHub or checking the <a href="https://github.com/scottkleinman/lexos/issues" target="_blank">GitHub Issues page</a> for assistance.
