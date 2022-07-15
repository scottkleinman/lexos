# Installation

## Installing Python and Anaconda

We recommend installing using the Anaconda Python distribution, even if you already have a version of Python installed on your computer. The Lexos installation process has been tested using Anaconda, and we have run into some issues trying to do it without Anaconda in the past.

<ol>
<li>Visit the <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda Distribution</a> page to download the Anaconda installer for your operating system.</li>
<li>Once it has finished downloading, run the Anaconda Installer and follow the instructions.</li>
<li>To verify Python has been installed correctly, open the Anaconda prompt and enter `python -V`.
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

This will install the Lexos API and all of its dependencies.

## Downloading the Default Language Model (Required)

Before using Lexos, you will want to install its default language model:

```bash
python -m spacy download xx_sent_ud_sm
```

For information on how Lexos uses language models, see [Tokenizing Texts](tutorial/tokenizing_texts.md).

## Downloading Additional Language Models (Optional)

This is a minimal model that performs sentence and token segmentation for a variety of languages. If you want a model for a specific language, such as English, download it by providing the name of the model:

```bash
python -m spacy download en_core_web_sm
```

If you are working in another language or need a larger language model, you can download instructions for additional models from the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page.
