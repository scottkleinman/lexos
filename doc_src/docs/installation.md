# Installation

During the development process, dependency management and packaging are handled using <a href="https://python-poetry.org/" target="_blank">Poetry</a>. When the API is released, you will be able to install it with `pip install lexos-api`. Other methods of installation will be added to this page at a later date.

## Installing Python and Anaconda

We recommend installing Python through Anaconda, even if you already have a version of Python installed on your computer. The Lexos installation process has been tested using Anaconda, and we have run into some issues trying to do it without Anaconda in the past.

1. Visit the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page to download the Anaconda installer for your operating system.
2. Once it has finished downloading, run the Anaconda Installer and follow the instructions.
3. To verify Python has been installed correctly, open the Anaconda Powershell Prompt and enter `python -V`.  
The response should be "Python version number" e.x. `Python 3.9.12`

## Installing the Lexos API with Poetry

A Mac-centric approach to installing Poetry is as follows:

```linux
$ brew install poetry
$ poetry --version
Poetry version 1.1.11
```

The approach for Windows is as follows:
>Make sure to run the following commands in the **Anaconda Powershell Prompt**

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
poetry --version
Sample Output: Poetry version 1.1.11
```

The Poetry website contains extensive documentation on [how to install it](https://python-poetry.org/docs/#installation) on different operating systems.

My running theory is that the best way to install the Lexos API prior to release is to complete following steps:

1. Install Poetry.
2. Clone the repo.
> To clone the repo, open the Anaconda Powershell Prompt and `cd` to the folder you want to clone the repo in. Then run the command `git clone https://github.com/scottkleinman/lexos.git`
3. `cd` to the local repo directory.
4. Run `poetry build`
5. Run `poetry install`.

This has not been thoroughly tested. If you are only evaluating the API and not modifying the code base, you can probably just clone the repo and run the library locally by following the instructions below.

## Running the Development Library Locally

Although you can call the Lexos API from command-line scripts or the Python command line, the easiest method is to call it from a Jupyter notebook. However, you may find that Jupyter has trouble finding the module if it has not been added to your Python environment. There is an easy workaround for this.

1. `cd` to the directory containing the API. If you cloned the repository, this will be the `lexos` folder (not the `lexos` subfolder inside it).
2. Then fire up `jupyter notebook` or `jupyter lab`.
3. In the first cell of your notebook, run the following code:

```python
import os
import sys
LEXOS_PATH = "lexos"
if "NOTEBOOK_INITIATED_FLAG" not in globals():
    NOTEBOOK_INITIATED_FLAG = True
    try:
        module_path = os.path.join(os.path.dirname(__file__), os.pardir)
    except:
        module_path = os.path.abspath(os.path.join(LEXOS_PATH))
        %cd lexos
        %pwd
    if module_path not in sys.path:
        sys.path.append(module_path)
```

If you are starting your notebook from another directory, you can modify `LEXOS_PATH` to point to the `lexos` subfolder.

You should now be able to import the `lexos` API module from your local directory.
