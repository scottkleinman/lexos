# MALLET Topic Models in Lexos

Topic modelling is a widely-used method of exploring the semantic and discursive concepts, or "topics", within collections of texts. Wikipedia defines a topic model as follows:

> In machine learning and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body. Intuitively, given that a document is about a particular topic, one would expect particular words to appear in the document more or less frequently: "dog" and "bone" will appear more often in documents about dogs, "cat" and "meow" will appear in documents about cats, and "the" and "is" will appear approximately equally in both. A document typically concerns multiple topics in different proportions; thus, in a document that is 10% about cats and 90% about dogs, there would probably be about 9 times more dog words than cat words. The "topics" produced by topic modeling techniques are clusters of similar words. A topic model captures this intuition in a mathematical framework, which allows examining a set of documents and discovering, based on the statistics of the words in each, what the topics might be and what each document's balance of topics is.

<a href="https://mimno.github.io/Mallet/" target="_blank">MALLET</a> is the most widely used topic modelling tool in the Digital Humanities, both because it is very performant and because its implementation of the Latent Dirichlet Allocation (LDA) algorithm tends to produce quality topics. MALLET is a command-line tool written in Java. It is independent of Lexos and must be installed separately. User-friendly instructions for installing and using MALLET can be found in the _Programming Historian_ tutorial [Getting Started with Topic Modeling and MALLET](https://programminghistorian.org/en/lessons/topic-modeling-and-mallet).

One of the difficulties of using MALLET is that its output is relatively difficult to manipulate into data structures useful for visualization. This tutorial is a proof of concept for how we might use Lexos to generate a MALLET topic model and then use the model to create a visualization with Andrew Goldstone's <a href="http://agoldst.github.io/dfr-browser/" target="_blank">dfr-browser</a>.

!!! note
    Much of the legwork for this procedure was done for the <a href="https://we1s.ucsb.edu/" target="_blank">WhatEvery1Says Project</a>, which established the basic workflow.

## Before Getting Started

Before you get started, make sure that you have a working installation of MALLET by following the instructions in the _Programming Historian_ tutorial [Getting Started with Topic Modeling and MALLET](https://programminghistorian.org/en/lessons/topic-modeling-and-mallet). Make sure that you know the path to the MALLET binary file.

Next, make a new folder for your topic model. In this tutorial, we will locate our new folder at `../topic_model`, which indicates that the folder is at the same level as the Lexos API's `lexos` folder.

## Import Some Data

For this tutorial, we'll use the MALLET English-language sample data. This a very small dataset and should run very quickly. It should have been downloaded when you installed MALLET.

Change the `data_path` to wherever your data is located. The code below simply reads a folder of text files and adds each text to a list.

```python
data_path = "C:/mallet/mallet-2.0.8/sample-data/web/en"

data = []
for file in os.listdir(data_path):
    with open(f"{data_path}/{file}", "r") as f:
        data.append(f.read())
```

## Create Metadata

Although not required for topic modelling, metadata is required to generate a dfr-browser. Dfr-browser was originally designed for displaying models of journal articles in the JSTOR database, so you need to supply metadata fields with the categories it expects. These categories are `id`, `title`, `publication`, `authors`, `volume`, `issue`, `year`, and `pagerange`. If these categories are not appropriate to your data, you can leave them blank (as an empty string). You can also include additional fields (e.g. `file` or `url`), although they may not be displayed in the dfr-browser. Further information on customizing metadata can be found in the <a href="https://github.com/agoldst/dfr-browser#adapting-this-project-to-other-kinds-of-documents-or-models" target="_blank">dfr-browser documentation</a>.

Metadata should be stored in a CSV file with no headings called `meta.csv`.

## Scrub the Data

Now we will use Lexos to scrub the data. We import the `Scrubber` components, make a pipeline, and run the pipeline on each text. The components here are just random samples of the possible options.

```python
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import scrubber_components, load_components

emails, new_lines, pattern = load_components(("emails", "new_lines", "pattern"))

scrub = make_pipeline(
    emails,
    new_lines,
    pipe(pattern, pattern="\'")
)

data = [scrub(item) for item in data]
```

## Tokenize the Data

We will import the Lexos tokenizer and create a list of spaCy docs. In the example below, we use spaCy's "en_core_web_sm" language model, and we'll add "gorillas" as an arbitrary extra stop word.

Keep in mind that each token in the doc is annotated with its part of speech, whether or not it is a stop word, and whether or not it is a punctuation mark (to name a few examples). We will use these properties below.

Note that because tokenization also involves adding these annotations, it may take a long time for large datasets.

```python
from lexos import tokenizer

docs = tokenizer.make_docs(
    data,
    model="en_core_web_sm",
    add_stopwords=["gorillas"]
)
```

## Topic Modelling

We are now ready to create the topic model. We start by creating a `Mallet` object, pointing it to a directory where we would like the model to be saved, and supplying the path to our MALLET installation.

```python
from lexos.topic_model.mallet import Mallet

model = Mallet(
    model_dir="../topic_model",
    mallet_path="C:/mallet/mallet-2.0.8/bin"
)
```

### Import the Data

We use our `Mallet` object to import our tokenized docs. In the example below, we will import only tokens labelled as nouns. The default behaviour is to skip stop words and punctuation.

This process creates two files in the model directory. The first is called `data.txt`. This file contains all our doc tokens with one doc per line. Each doc is a bag of words (meaning token order is lost). The second file is called `import.mallet`. This contains the information in `data.txt`, imported into a binary format. It will also have the method's default settings overridden by any MALLET parameters you supply. However, we'll stick with the defaults below.

```python
model.import_data(docs, allowed=["NOUN"])
```

!!! note
    You can override the default settings by creating a dict of keyword-value pairs based on MALLET settings and then passing the dict to the `import_data()` function. For instance, say you wanted to use an external stop word file called `stoplist.txt`:

    ```python
    opts = {
        "remove-stopwords": True,
        "stoplist-file": "stoplist.txt"
    }

    model.import_data(docs, **opts)
    ```

    This feature is currently not fully developed, but it should work for some basic procedures.

### Train the Model

If we've followed the procedure above, we can simply call `model.train()`. If for some reason, we need to re-instantiate the `Mallet` object, we can do so and skip the import step above. In this case, we would call `model.train(mallet_file="import.mallet")`.

```python
model.train()
```

!!! important
    The progress of the modelling task is monitored by continuous output to the console. If you are running the process in a Jupyter notebook, you may wish to put `%%capture` at the beginning of the cell so that the progress output is not printed to the cell's output, which may overwhelm the memory buffer. Eventually, we will create a progress bar option to avoid this issue.

Once the model is complete (which may take a long time if you have a lot of data but should take seconds for the MALLET sample data), it is worth inspecting the model. Navigate to your models `keys.txt` file and open it. If some topics have no keywords, that is a sign that something has gone wrong with your model. If everything looks good, you're ready for the next step.

### Create the Topic Scale File

dfr-browser requires an additional CSV file containing topic scaling data. To produce this, we just need to call `model.scale()`

```python
model.scale()
```

## Dfr-Browser

We can now generate a dfr-browser from our topic model. We import the `DfrBrowser` class and create a `DfrBrowser` object. This will create a `dfr_browser` directory in your model's folder where all the necessary files are housed. Make sure that your `meta.csv` file is in the root of your topic model folder.

```python
from lexos.topic_model.dfr_browser import DfrBrowser

browser = DfrBrowser(model_dir="../topic_model")
```

### Open the Dfr-Browser

When the process is complete, you will need to start a local server. Open a command prompt and `cd` to your model's `dfr_browser` folder. Then type `python -m http.server 8080`. If you are already running a local server on port 8080, you can change it to something else. Then point your browser to [http://localhost:8080/](http://localhost:8080/), and the dfr-browser should load.

Note that some features of dfr-browser may not work if you do not have appropriate metadata.

When you are finished, remember to go back to the command prompt and type `Ctr+C` to shutdown the server.
