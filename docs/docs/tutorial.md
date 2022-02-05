# Tutorial

This page is a rough overview of the usage of the API.

!!! important

    Aspects of the API may change before the tutorial is updated. At this stage,
    the tutorial should only be taken as a general guideline to the API's usage.

## Getting Started

Begin by importing in some modules in the Lexos API.

```python
from lexos.io.basic import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import scrubber_components, load_components
```

Here are some explanations:

1. The `io` module contains IO functions. Right now, there is a "basic" Loader class that takes a file path, url, list of file paths or urls, or a directory name indicating where the source data is. More sophisticated loaders can be created later.
2. The `scrubber` module consists of thematic "components": `normalize`, `remove`, `replace`, and so on. Each component has a number of functions, such as converting to lower case, removing digits, stripping tags, etc.
3. Component functions are registered in a registry. They can be loaded into memory as needed and applied to texts in any order.

## Loading Texts

A typical workflow would create a [lexos.io.basic.Loader][] object and call [lexos.io.basic.Loader.load][] to load the data from disk or download it from the internet. You can access all loaded texts by calling `Loader.texts`.

!!! note

    It is more efficient simply to use Python's `open()` to load texts into a list _if_ you know the file's encoding. Currently, the main advantage of the `Loader` class is that it automatically coerces the data to Unicode.

At this stage of development, the user or application developer is expected to maintain their data folders and know their file locations. More sophisticated project/corpus management methods could be added to the API at a later date.

Here is a sample of the code for loading a single text file:

```python
# Data source
data = "tests/test_data/Austen_Pride.txt"

# Create the loader and load the data
loader = Loader()
loader.load(data)

# Print the first text in the Loader
text = loader.texts[0]
print(text)
```

## Scrubbing Texts

Scrubber is now explicitly defined as a _destructive_ preprocessor. In other words, it changes the text as loaded in ways that potentially make mapping the results onto the original text potentially impossible. It is therefore best used before other procedures so that the scrubbed text is essentially treated as the "original" text. The importance of this will be seen below when we see the implementation of the tokeniser. But, to be short, Scrubber does not play a role in tokenisation by separating tokens by whitespace.

Scrubbing works by applying a single function or a pipeline of functions to the text. We have already loaded the Scrubber components registry above, so let's load the components we want. We can load them individually, as in the first example below, or we can specify multiple components in a tuple, as in the second example. In both cases, the returned variable is a function, which we can then feed to a scrubbing pipeline.

```python
# Load a component from the registry
lower_case = scrubber_components.get("lower_case")

# Or, if you want to do several at once...
title_case, remove_digits = load_components(("title_case", "remove_digits"))
```

Now let's make the pipeline. We simply feed our component function names into the [make_pipeline()][lexos.scrubber.pipeline.make_pipeline] function in the order we want them to be implemented. Notice that the remove_digits function has to be passed through the [pipe()][lexos.scrubber.pipeline.make_pipeline] function, which enables arguments to be passed.

The value returned is a function that implements the full pipeline when called on a text, as shown below.

```python
# Make the pipeline
scrub = make_pipeline(
    lower_case,
    title_case,
    pipe(remove_digits, only=["1"])
)

# Scrub the text
scrubbed_text = scrub("Lexos is the number 12 text analysis tool.")
```

This will return "Lexos Is The Number 2 Text Analysis Tool".

You can also call component functions without a pipeline. For instance,

```python
scrubbed_text = remove_digits("Lexos123", only=["2", "3"])
```

This will return "Lexos1".

### Custom Scrubbing Components

The `title_case` function in the code in the pipeline above will not work because `title_case` is a custom component. To use it, we need to add it to the registry.

```python
# Define the custom function
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title()

# Register the custom function
scrubber_components.register("title_case", func=title_case)
```

Users can add whatever scrubbing functions they want. For development purposes, we can start by creating custom functions, and, if we use them a lot, migrate them to the permanent registry.

!!! important

    To use a custom scrubbing function, you must register it _before_ you call [lexos.scrubber.registry.load_components][].

## Tokenising Texts

The `tokenizer` module is a big change for Lexos, as it formally separates tokenisation from preprocessing. At present, the Lexos user uses Scrubber to massage the text into shape using his or her implicit knowledge about the text's language. The text is then separated into tokens on whitespace by <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank">sklearn.feature_extraction.text.CountVectorizer</a></code>. The API instead uses a language model to tokenize the text. The language model formalises the implicit rules the user supplies and thus automates the process. Built-in procedures appropriate to specific languages can often do a better job of tokenisation than the current Lexos approach. There some other advantages as well, which we'll see below.

So the new Lexos tokenizer will load a language model and use its rules and predictions to separate the text into a list of tokens. Many filtering procedures will then be easy to implement with list comprehensions like

```python
no_stopwords = [token for token in tokens if token not in stopwords]
```

Using language models will have other benefits. If the language of the text is English, for instance, an English language model can be loaded which allows tokens to be annotated automatically with labels for parts of speech information, lemmas, stop words, and other grammatical information at the time the text is tokenised. It then becomes possible to filter by part of speech, for instance, with something like

```python
only_nouns = [token for token in tokens if token.pos_ == "NOUN"]
```

If no language model exists for the text's language, it will only be possible to tokenise using general rules, and it will not be possible to add other labels (at the tokenisation stage). But new language models, including models for historical languages, are being produced all the time, and this is a growing area of interest in DH.

The Lexos API wraps the <a href="https://spacy.io/" target="_blank">spaCy</a> NLP library for loading models and tokenising texts. Because spaCy has excellent documentation and fairly wide acceptance in the DH community, it is a good tool to use under the bonnet. But it should be possible to add procedures for using libraries like NLTK. As a side note, the architecture of the scrubber module is built on top of the preprocessing functions in <a href="https://github.com/chartbeat-labs/textacy/" target="_blank">Textacy</a>, which also accesses and extends spaCy.

So let's see an example of how we tokenise a text.

```
from lexos import tokenizer

doc = tokenizer.make_doc(text)
```

In spaCy, sequences of tokens are stored in a Doc object. So the Lexos tokenizer's [make_doc()][lexos.tokenizer.make_doc] function creates a spaCy Doc (or just "doc"). Note that by default the tokenizer uses a small language model that has been trained for tokenisation and sentence segmentation on multiple languages. If you were making a document from a text in a language with a more sophisticated model, you would specify the model to be used. For instance, to use spaCy's small English model trained on web texts, you would call

```python
doc = tokenizer.make_doc(text, model="en_core_web_sm")
```

The following are some examples of iterations that can be made over a spaCy doc.

```python
# Get a list of tokens
tokens = [token.text for token in doc]

# Get a list of non-punctuation tokens
non_punct_tokens = [token.text for token in doc if not token.is_punct]
```

The spaCy Doc is _non-destructive_ because it preserves the original text alongside the list of tokens and their attributes. Indeed, calling `doc.to_json()` will return a JSON representation which gives the start and end position of each token in the original text!

SpaCy docs are produced by a pipeline of components which can annotated with labels like parts of speech if the information is available in the language model. Each of these annotations is stored in the document's attributes. It is possible to extend spaCy's Doc object with its extension attribute. The Lexos API has a sample `is_fruit` extension, which is illustrated below. Note that extensions are accessed via the underscore prefix, as shown.

```python
# Indicate whether the token is labelled as fruit
for token in doc:
    print(token._.is_fruit)
```

In addition, there is a [LexosDoc][lexos.tokenizer.lexosdoc.LexosDoc] class, which provides a wrapper for spaCy docs. Its use is illustrated below.

```python
from lexos.tokenizer.lexosdoc import LexosDoc

lexos_doc = LexosDoc(doc)
tokens = lexos_doc.get_tokens()
```

The example above just returns `[token.text for token in doc]` but it can be useful for producing clean code. In other cases, it might be useful to manipulate spaCy docs with methods that do not access their built-in or extended attributes or method. For instance, there is a method to check what attributes are available for tokens in the doc and a method for exporting the tokens and their attributes in a pandas dataframe.

!!! note
    Notice in the code above that `tokenizer` has a [make_docs()][lexos.tokenizer.make_docs] function to parse a list of texts into spaCy docs.

## Summary

Here is a summary of the procedure so far.

```python

# Create the loader and load the data
loader = Loader()
loader.load(data)

# Load Scrubber components, make a pipeline, and scrub the texts
lower_case, remove_digits = load_components(
    ('lower_case', 'remove_digits')
)
scrub = make_pipeline(
    lower_case,
    pipe(remove_digits, only=["1"])
)
scrubbed_texts = [scrub(text) for text in loader.texts]

# Tokenise the texts
docs = tokenizer.make_docs(scrubbed_texts)
```

## Creating a Document-Term Matrix

The function to generate a DTM from these docs has not yet been written, but it could already be done easily with Textacy as shown below (notice the `LexosDoc(doc).get_tokens()` is from the Lexos API — the rest is Textacy's <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/representations.html#textacy.representations.vectorizers.Vectorizer" target="_blank">Vectorizer</a></code>, which is pretty sophisticated).

```python
from textacy.representations.vectorizers import Vectorizer

vectorizer = Vectorizer(
    tf_type="linear",
    idf_type="smooth",
    norm="l2",
    min_df=3,
    max_df=0.95
)

tokenised_docs = (LexosDoc(doc).get_tokens() for doc in docs)
doc_term_matrix = vectorizer.fit_transform(tokenised_docs)
```

The main difference here from the current procedure in Lexos, is that the text is pre-tokenised, rather than relying on <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank">sklearn.feature_extraction.text.CountVectorizer</a></code> to do the dirty work.
