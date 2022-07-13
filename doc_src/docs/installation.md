# Installation

## Installing Python and Anaconda

We recommend installing using the Anaconda Python distribution, even if you already have a version of Python installed on your computer. The Lexos installation process has been tested using Anaconda, and we have run into some issues trying to do it without Anaconda in the past.

<ol>
<li>Visit the <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda Distribution</a> page to download the Anaconda installer for your operating system.</li>
<li>Once it has finished downloading, run the Anaconda Installer and follow the instructions.</li>
<li>To verify Python has been installed correctly, open the Anaconda Powershell Prompt and enter `python -V`.
The response should be a Python version number,  e.g. "Python 3.9.12".</li>
</ol>

## Installing the Lexos API

To install the Lexos API, run

```bash
pip install lexos
```

To update to the latest version, use

```bash
pip install -U lexos
```

This will install the Lexos API and all of its dependencies, including spaCy's multilanguage model and its small English language model. If you are working in another language or need a larger language model, you can download additional models from the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page.
