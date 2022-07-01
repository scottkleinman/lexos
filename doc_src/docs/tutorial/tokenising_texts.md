## Language Models

The `tokenizer` module is a big change for Lexos, as it formally separates tokenisation from preprocessing. In the Lexos app, the user employs Scrubber to massage the text into shape using his or her implicit knowledge about the text's language. Tokenisation then takes place by splitting the text according to a regular expression pattern (normally whitespace). By contrast, the Lexos API uses a language model that formalises the implicit rules and thus automates the tokenisation process. Language models can implement both rule-based and probabilistic strategies for separating document strings into tokens. Because they have built-in procedures appropriate to specific languages, language models can often do a better job of tokenisation than the approach used in the Lexos app.

!!! important
    There are some trade-offs to using language models. Because the algorithm does more than split strings, processing times can be greater. In addition, tokenisation is no longer (explicitly) language agnostic. A language model is "opinionated" and it may overfit the data. At the same time, if no language model exist for the language being tokenised, the results may not be satisfactory. The Lexos API strategy for handling this situation is described below.

## Tokenised Documents

A tokenised document can be defined as a text split into tokens in which each token is stored with any number of annotations assigned by the model. These annotations are token "attributes". The structure of a tokenised document can then be conceived in theory as a list of tuples like

```python
[
    (text="The", part_of_speech="noun", is_stopword="True"),
    (text="end", part_of_speech="noun", is_stopword="False"),

]
```

It is then a simple matter to iterate through the document and retrieve all the tokens that are not stopwords using a Python list comprehension. Many filtering procedures are easy to implement in this way.

For languages such as Modern English, language models exist that can automatically annotate tokens with information like part of speech, lemmas, stop words, and other information. However, token attributes can also be set after the text has been tokenised.

If no language model exists for the text's language, it will only be possible to tokenise using general rules, and it will not be possible to add other labels (at the tokenisation stage). But new language models, including models for historical languages, are being produced all the time, and this is a growing area of interest in DH.

## spaCy Docs

The Lexos API wraps the <a href="https://spacy.io/" target="_blank">spaCy</a> NLP library for loading language models and tokenising texts. Because spaCy has excellent documentation and fairly wide acceptance in the DH community, it is a good tool to use under the bonnet. spaCy has a growing number of language models in a number of languages, as well as wrappers for loading models from other common NLP libraries such as Stanford Stanza.

!!! note
    The architecture of the Scrubber module is partially built on top of the preprocessing functions in <a href="https://github.com/chartbeat-labs/textacy/" target="_blank">Textacy</a>, which also accesses and extends spaCy.

In spaCy, texts are parsed into `spacy.Doc` objects consisting of sequences of tokens.

!!! note
    In order to formalise the difference between a text string that has been scrubbed and one that has been tokenised, we refer wherever possible to the string as a "text" and to the tokenised `spacy.Doc` object as a "document" (or just "doc"). We continue to refer to the individual items as "documents" if we are not concerned with their data type.

Each token is `spacy.Token` object which stores all the token's attributes.

The Lexos API wraps this procedure in the [lexos.tokenizer.make_doc][] function:

```python
from lexos import tokenizer

doc = tokenizer.make_doc(text)
```

This returns a `spacy.Doc` object.

By default the tokenizer uses spaCy's ["xx_sent_ud_sm"](https://github.com/explosion/spacy-models/releases/tag/xx_sent_ud_sm-3.2.0) language model, which has been trained for tokenisation and sentence segmentation on multiple languages. This model performs statistical sentence segmentation and possesses general rules for token segmentation.

If you were making a document from a text in a language for which a more language-specific model, you would specify the model to be used. For instance, to use spaCy's small English model trained on web texts, you would call

```python
doc = tokenizer.make_doc(text, model="en_core_web_sm")
```

`tokenizer` also has a [lexos.tokenizer.make_docs][lexos.tokenizer.make_docs] function to parse a list of texts into spaCy docs.

!!! important
    Tokenisation using spaCy uses a lot of memory. For a small English-language model, the parser and named entity recogniser (NER) can require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the memory limit with the `max_length` parameter. The limit is in number of characters (the default is set to 2000000 for Lexos), so you can check whether your inputs are too long by checking `len(text)`. If you are not using RAM-hungry pipeline components, you can disable or exclude them to avoid errors an increase efficiency (see the discussion on the spaCy pipeline below). In some cases, it may also be possible to cut the texts into segments before tokenisation.

A list of individual tokens can be obtained by iterating over the spaCy doc:

```python
# Get a list of tokens
tokens = [token.text for token in doc]
```

Here the `text` attribute stores the original text form of the token. SpaCy docs are _non-destructive_ because they preserve the original text alongside the list of tokens and their attributes. You can access the original text of the entire doc by calling `doc.text` (assuming you have assigned the `Doc` object to the `doc` variable). Indeed, calling `doc.to_json()` will return a JSON representation which gives the start and end position of each token in the original text!

As mentioned above, you can use a Python list comprehension to filter the the contents of the doc using information in the document's attributes. For instance:

```python
# Get a list of non-punctuation tokens
non_punct_tokens = [token.text for token in doc if not token.is_punct]
```

The example above leverages the built-in `is_punct` attribute to indicate whether the token is defined as (or predicted to be) a punctuation mark in the language model. SpaCy docs have a number of built-in attributes, which are described in the <a href="https://spacy.io/api/doc#attributes" target="_blank">spaCy API reference</a>.

!!! note
    It is possible to extend spaCy's Doc object with its extension attribute. The Lexos API has a sample `is_fruit` extension, which is illustrated below. Note that extensions are accessed via the underscore prefix, as shown.

    ```python
    # Indicate whether the token is labelled as fruit
    for token in doc:
        print(token._.is_fruit)
    ```

    The sample extension can be found in [lexos.tokenizer.extensions][extensions].

### The spaCy Pipeline

Once spaCy tokenises a text, it normally passes the resulting document to a pipeline of functions to parse it for other features. Typically, these functions will perform actions such as part-of-speech tagging, labelling syntactic dependencies, and identifying named entities. Processing times can be increased by disabling pipeline components if they are unavailable in the language model or not needed for the application's purposes. [lexos.tokenizer.make_doc][] and [lexos.tokenizer.make_docs][] will automatically run all pipeline components in the model unless they are disabled or excluded with the `disable` or `exclude` parameter. Check the model's documentation for the names of the components it includes.

It is also possible to include custom pipeline components, which can be inserted at any point in the pipeline order. Custom components are supplied with the `pipeline_components` parameter, which takes a dictionary containing the keyword "custom". The value is a list of dictionaries where each dictionary contains information about the component as described at https://spacy.io/api/language/#add_pipe.

!!! note
    The `pipeline_components` dict also contains `disable` and `exclude` keywords. The values are lists of components which will be merged with any components supplied in the `disable` or `exclude` paramaters of [lexos.tokenizer.make_doc][] and [lexos.tokenizer.make_docs][].

The ability to add custom pipeline components is valuable for certain language- or application-specific scenarios. However, it also opens Lexos up to the wealth of third-part pipeline components available through the <a href="https://spacy.io/universe/category/pipeline" target="_blank">spaCy Universe</a>.

### Handling Stop Words

Every token in a spaCy doc has an `is_stop` attribute. Most language models will have a list of default stop words, and this list is used to set the `is_stop` attribute `True` for every token when the document is parsed. It is possible to add stop words to the default list by passing a list to [lexos.tokenizer.make_doc][] or [lexos.tokenizer.make_docs][] with the `add_stopwords` argument:

```python
doc = tokenizer.make_doc(
    text,
    model="en_core_web_sm",
    add_stopwords=["yes", "no", "maybe"]
)
```

The `remove_stopwords` argument removes stop words from the default list. If `remove_stopwords=True`, all stop words are removed.

!!! important
    `add_stopwords` and `remove_stopwords` do not remove stop word tokens from the doc; rather, they modify the stop word list used to set the `is_stop` attribute of individual tokens. To get a list of tokens without stop words, you must do something like `[token for token in doc if not token.is_stop]`. If you a are producing a corpus of documents in which the documents will be processed by different models, it is most efficient to process the documents in batches, one batch for each model.

## LexosDocs

The Lexos API also has a [LexosDoc][lexos.tokenizer.lexosdoc.LexosDoc] class, which provides a wrapper for spaCy docs. Its use is illustrated below.

```python
from lexos.tokenizer.lexosdoc import LexosDoc

lexos_doc = LexosDoc(doc)
tokens = lexos_doc.get_tokens()
```

This example just returns `[token.text for token in doc]`, so it is not strictly necessary. But using the `LexosDoc` wrapper can be useful for producing clean code. In other cases, it might be useful to manipulate spaCy docs with methods that do not access their built-in or extended attributes or method. For instance, [lexos.tokenizer.lexosdoc.LexosDoc.get_token_attrs][lexos.tokenizer.lexosdoc.LexosDoc.get_token_attrs] shows what attributes are available for tokens in the doc and [lexos.tokenizer.lexosdoc.LexosDoc.to_dataframe][lexos.tokenizer.lexosdoc.LexosDoc.to_dataframe] exports the tokens and their attributes to a pandas dataframe.

## Ngrams

Both texts and documents can be parsed into sequences of two or more tokens called ngrams. Many spaCy models can identify syntactic units such as noun chunks. These capabilities are not covered here since they are language specific. Instead, the section below describe how to obtain more general ngram sequences.

### Generating Word Ngrams

The easiest method of obtaining ngrams from a text is to create a spaCy doc and then call Textacy's `textacy.extract.basics.ngrams` method:

```python
import spacy
import textacy.extract.basics.ngrams as ng
nlp = spacy.load("xx_sent_ud_sm")
text = "The end is nigh."
doc = nlp(text)
ngrams = list(ng(doc, 2, min_freq=1))
```

This will produce `[The end, end is, is nigh]`. The output is a list of spaCy tokens. (An additional `[token.text for token in ngrams]` is required to ensure that you have quoted strings: `["The end", "end is", "is nigh"]`).

Textacy has a lot of additional options, which are documented in the Textacy API reference under <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/extract.html#textacy.extract.basics.ngrams" target="_blank">textacy.extract.basics.ngrams</a></code>. However, if you do not need these options, you can use Lexos' helper function [lexos.tokenizer.ngrams_from_doc][]:

```python
import spacy
nlp = spacy.load("xx_sent_ud_sm")
text = "The end is nigh."
doc = nlp(text)
ngrams = tokenizer.ngrams_from_doc(doc, size=2)
```

Notice that in both cases, the output will be a list of overlapping ngrams generated by a rolling window across the pre-tokenised document. If you want your document to contain ngrams _as_ tokens, you will need to create a new document using the [lexos.tokenizer.doc_from_ngrams][] function:

```python
doc = tokenizer.doc_from_ngrams(ngrams, strict=True)
```

!!! note
    Setting `strict=False` will preserve all the whitespace in the ngrams; otherwise, your language model may modify the output by doing things like splitting punctuation into separate tokens.

There is also a [lexos.tokenizer.docs_from_ngrams][] function to which you can feed multiple lists of ngrams.

A possible workflow might call Textacy directly to take advantage of some its filters, when generating ngrams and then calling [lexos.tokenizer.doc_from_ngrams][] to pipe the extracted tokens back into a doc. <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/extract.html#textacy.extract.basics.ngrams" target="_blank">textacy.extract.basics.ngrams</a></code> has sister functions that do things like extract noun chunks (if available in the language model), making this a very powerful approach generating ngrams with semantic information.

### Generating Character Ngrams

Character ngrams at their most basic level split the _untokenised_ string every N characters. So "The end is nigh." would produce something like `["Th", "e ", "nd", " i", "s ", "ni", "gh", "."]` (if we wanted to preserve the whitespace). Lexos does this with the [lexos.tokenizer.generate_ngrams]() function:

```python
text = "The end is nigh."
ngrams = tokenizer.generate_character_ngrams(text, 2, drop_whitespace=False)
```

This will produce the output shown above. If we wanted to output `["Th", "en", "di", "sn", "ig", "h."]`, we would use `drop_whitespace=True` (which is the default).

!!! note
    `lexos.tokenizer.generate_character_ngrams` is a wrapper for Python's `textwrap.wrap` method, which can also be called directly.

Once you have produced a list of ngrams, you can create a doc from them using [lexos.tokenizer.ngrams_from_doc][], as shown above.

Use [lexos.tokenizer.generate_character_ngrams][] (a) when you simply want a list of non-overlapping ngrams, or (b) when you want to produce docs with non-overlapping ngrams as tokens.

Note that your language model may not be able to apply labels effectively to ngram tokens, so working with character ngrams is primarily useful if you are planning to work with the token forms only, or if the ngram size you use maps closely to character lengths of words in the language you are working in.

## Summary of Workflow

Tokenisation can be considered the end of the text preparation workflow, although, in its use of language models, it already begins to introduce some elements of analysis. Before proceeding to the analysis stage, it is useful to have summary of what a minimal text preparation workflow might look like:

```python

# Create the loader and load the data
loader = Loader()
loader.load(data)

# Load Scrubber components, make a pipeline, and scrub the texts
lower_case, remove_digits = load_components(
    ("lower_case", "remove_digits")
)
scrub = make_pipeline(
    lower_case,
    pipe(remove_digits, only=["1"])
)
scrubbed_texts = [scrub(text) for text in loader.texts]

# Tokenise the texts
docs = tokenizer.make_docs(scrubbed_texts)
```
