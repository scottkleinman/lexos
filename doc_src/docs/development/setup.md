# Setting Up Your Development Environment

To make changes to the Lexos source code or documentation, you will need to have a development environment consisting of a Python 3.12+, (preferably) <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank">uv</a> to manage dependencies and your virtual environment, and <a href="https://git-scm.com" target="_blank">git</a> installed. The steps below detail how to set up your development environment step by step.

## Install `uv` Globally

We recommend `uv` for dependency management. If you haven't already, install `uv` according to the <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank">official documentation</a>, follow these steps:

**For Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you have Anaconda installed, you can run the command in a new Anaconda Prompt.

**For macOS/Linux (Bash/Zsh):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Some Mac users have reported issues with the above command. `uv` appears to install correctly, but you can't run `uv` commands. If you encounter this problem, you can try <a href="https://brew.sh/" target="_blank">homebrew</a> instead (you may need to install homebrew first):

```bash
brew install uv
```

Close and reopen your terminal, then run:

```bash
uv --version
```

or

```bash
uv --help
```

This will verify the installation.

Most of the instructions in this manual assume that you are using `uv` to manage dependencies.

## Install Python

Lexos requires **Python 3.12** or greater. If you do not already have it installed, you can install it from the command-line by running

```bash
uv python install 3.12
```

However, if you already have it installed using a distribution like <a href="https://www.anaconda.com/download" target="_blank">Anaconda</a>, `uv` will detect that installation, so there is no need for a fresh install.

!!! important
    We recommend installing Anaconda, which comes with some dependencies needed by Lexos pre-installed. If you use `uv` to download a fresh installation of Python, it may not have these dependencies, and you may need to installed them independently.

## Install Git

**Git** is used for version control. If you don't have it installed, you can download it from <a href="https://git-scm.com" target="_blank">git</a>.

## Install Visual Studio Code (VS Code) (Recommended)

We recommend using <a href="https://code.visualstudio.com/" target="_blank">Visual Studio Code</a> as your code editor. It has excellent support for Python and Git, and you can install extensions for code linting and formatting. Although you are not required to use it, the discussion in this documentation assumes that you are working in VS Code.

The Lexos repo has a file called `.vscode.json`, where you can configure the path to your Python interpreter in your local virtual environment. If you are on Windows, you need to change that path to the appropriate one on your computer, probably something like `C:\\Users\\Your_Name\\Documents\\uv_lexos\\.venv\\Scripts\\Python.exe`. If you are on a Mac or Linux, it will be something like `/Users/Your_Name/Documents/uv_lexos/.venv/bin/python`. This will point VS Code towards the Python interpreter installed for Lexos.

!!! important
    Don't forget to configure this path. Without it, VS Code may not recognize your virtual environment correctly, and you may encounter issues running Python code or Jupyter notebooks.

We also recommend installing the following VS Code extensions for Python development:

- Even Better TOML: For better syntax highlighting and formatting of `pyproject.toml`.
- Jupyter: If you plan to work with Jupyter notebooks.
- Markdownlint: For linting Markdown files.
- Pylance: Provides rich type information and IntelliSense for Python.
- Python: Official extension for Python development.
- Ruff: For linting and formatting Python code (see below for further instructions).

You can install these extensions from the VS Code marketplace or by searching for them in the Extensions view (`Ctrl+Shift+X`).

Lexos uses the <code><a href="https://docs.astral.sh/ruff/" target="_blank">ruff</a></code> code formatter and linter to produce readable code. Althouh you can run `ruff` from the command-line, it is advisable to set up your code editor to use it. In VS Code, you can use the Ruff extension. Add the following to your `settings.json` to use `ruff` for formatting and auto-format your files on save:

```json
{
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "ruff.linting.enabled": true,
    "ruff.linting.run": "onType"
}
```

## Get the Code

### Fork and Clone the Repository

Start by forking the project on GitHub to your own account. Then clone your fork locally. On the command line, run

```bash
git clone https://github.com/your-username/lexos.git
cd lexos
```

Alternatively, if you are using VS Code or a client like GitHub Desktop, go to the GitHub repository page, click on the green "Code" button, and copy the HTTPS URL. Use this URL with your client's clone feature to clone the repository.

### Navigate into the Project Directory

Use whatever path leads to the `lexos` directory.

```bash
cd lexos
```

### Create a Virtual Environment and Install the Project Dependencies

From the `lexos` project root:

```bash
uv venv
uv sync
```

This creates a `.venv` directory and installs all dependencies listed in `pyproject.toml`.

### Installing SpaCy

Lexos relies on the <a href="https://spacy.io/" target="_blank">spaCy</a> for Natural Language Processing library for much of its functionality. SpaCy itself is written in Cython, which compiles Python code into C or C++ for better memory management. However, Cython does not come pre-installed in vanilla downloads of Python, and, as of June 2025, the spaCy installer wheels cannot themselves install all of Cython's dependencies (or cannot do so for all common operating systems and processors). This, at least, is our theory of why installation of spaCy fails when you call `uv sync` in a vanilla installation of Python. To remedy the problem, we recommend that you install <a href="https://www.anaconda.com/download" target="_blank">Anaconda</a>, which is distributed with Cython. This should allow spaCy to install correctly.

The alternative is to install Cython's dependencies, and then Cython, independently. Cython requires a GCC-compatible C compiler to be present on your system. We have not thoroughly tested the following procedure, but it has worked in the a linux environment running on Windows 11 with an ARM64 processor (a challenging setup).

```bash
sudo apt-get install build-essential python3-dev
uv pip install cython
```

The first command will install `build-essential`, which provides the C compiler and other development tools, along with the Python development headers. We have read that there may be some discrepancy between the installation paths used by `uv` and `pip`. To be safe, we suggest trying to install Cython using `pip` as shown in the command above. You may need to install `pip` in your environment first.

Once you have a working version of Cython, `uv sync` should properly install spaCy.

### Installing SpaCy Models

SpaCy itself is installed as a dependency package via `uv`; however, its language models are downloaded as a separate process from urls. The two default models, "xx_sent_ud_sm" and "en_core_web_sm", are downloaded and installed automatically from commands in `pyproject.toml` when you run `uv sync`. If for any reason this fails, you can manually download the models. From your activated virtual environment in the project root, run:

```bash
uv run python -m spacy download xx_sent_ud_sm
uv run python -m spacy download en_core_web_sm
```

You can also use these commands to download additional models, if required.

## Activate the Virtual Environment

  `uv` commands will intelligently activate the virtual environment when you run them. However, for other commands (like `python` or `pip`), you need to activate the virtual environment manually. So it's a good idea to do this every time you start a new terminal session.

**Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
 ```

Your terminal prompt should now show `(lexos)` or `(.venv)` at the beginning.

---

Your local development environment should now be up and running.
