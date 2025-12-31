## Overview

Lexos is a library for constructing text analysis workflows. This normally means a step-by-step pipeline of collecting, processing, analyzing, and visualizating data. (The distinction between the analysis and visualization, however, is often blurred because most visualizations require some form of analysis.) Lexos offers different modules for performing these steps. The `Loader` and `Corpus` modules collect and create containers for storing and accessing data. The `Scrubber` module enables you to perform preprocessing steps on the texts in your data, such as normalizing whitespace or removing certain character patterns. The `Tokenizer` module uses Natural Language Processing (NLP) tools to extract features from your data &mdash; most importantly, countable tokens. This can be transformed into a document-term matrix with the `DTM` module. A typical workflow is shown below (dotted lines indicate optional steps).

``` mermaid
flowchart LR
   id1{Data} --> id2(((Loader))) & id3[(Corpus)]-. Preprocessing .-> id4{Scrubber}-. Feature Recognition .-> id5{Tokenizer} --> id6{DTM}
```

The `DTM` module allows you to extract basic statistics which you can use to interpret your data.

Lexos modules do not always have to be used in a strict sequential order. For instance, you can feed scrubbed or tokenized texts back into a corpus. You can also split your data at any time in the workflow with the `Cutter` module.

The workflow above might be supplemented by another leading to analysis and visualization.

``` mermaid
flowchart LR
   id1{DTM}-.->id2(Analysis) & id3([Visualization]) & id4{Export}
```

Admittedly, the line between analysis and visualization can be blurred because visualizations are often produced by computational transformations of your data. In some cases, these tools may bypass the DTM by accepting data in other formats. Conceptually, however, Lexos is designed to provide tools for you to implement an end-to-end workflow whilst allowing you to export your data for use with other tools, if needed.

## Before You Get Started

Before you get started, make sure that you have [installed Lexos](../installation.md).

## Basic Usage

Lexos workflows can be run conveniently in Jupyter notebooks simply by importing the relevant module (or the required functions and classes from the module). For instance, you can import the `Loader` with

```python
# Import the Lexos Loader class
from lexos.io.loader import Loader

# Instantiate a Loader object
loader = Loader()

# Load a text file
loader.load("myfile.txt")
```

This will work in a standalone script as well. Any errors will be printed to your notebook or console.

!!! note "Note for Developers"
    If you are designing an app that uses Lexos "under the hood", it is good practice to import the `LexosException` class and re-write the last line above in a `try...except` clause:

    ```python
    from lexos.exceptions import LexosException

    try:
        loader.load("myfile.txt")
    except LexosException as e:
        print(e)
    ```

    This will enable your application to handle errors without stopping the program.

To learn about each of the individual Lexos modules, browse through the pages in this guide. More examples of the use of Lexos functions can be found in the [Tutorial notebooks](../tutorials/index.md), and a full descriptions can be found in the [API documentation](../api/index.md).
