# Tokenizing Texts

Many computational methods of studying texts require the text to be split into smaller, countable units called **tokens**. These tokens can be words, phrases, or even characters, depending on the method being used. The process of splitting a text into tokens is called **tokenization**.

A tokenized document can be defined as a text split into tokens. This can be represented by a simple list of token strings. However, each token may also be represented as dictionary in which the token string is stored along with additional annotations. Below, we will refer to these annotations as token **attributes**. Here is an example of a list of token dictionaries conntaining attributes to indicate the token's part of speech and whether or not it is a stop word.

```python
tokenized_doc = [
    {"token_text": "The", "part_of_speech": "noun", "is_stopword": True},
    {"token_text": "end", "part_of_speech": "noun", "is_stopword": False}
]
```

It is then a simple matter to iterate through the document and retrieve all the tokens that are not stopwords using a Python list comprehension.

```python
for token in tokenized_doc:
    if not token["is_stopword"]: # If is_stopword is False
        print(token)
```

Or you might want to save the tokens to a new list with a list comprehension:

```python
non_stopwords = [
    token for token in tokenized_doc
    if not token["is_stopword"]
]
```

Many filtering procedures are easy to implement in this way.

However, a list of dictionaries is not the only way to represent a tokenized document; it is used here purely to introduce the concept. The strategy employed by Lexos will be discussed further below.

## Language Models

The easiest method for splitting a text into tokens is to use a simple rule-based approach, such as splitting the text on whitespace. However, this method is not always sufficient, especially for languages with complex morphology or syntax or where whitespace is not used to separate words (typical of Chinese, Japanese, and Korean, for instance). In these cases, it is often necessary to use a more sophisticated approach that takes into account the language's grammar and structure.

Lexos uses language models to automate the tokenization process. A **language model** is a statistical model that has been trained on a large corpus of texts in a specific language. It can be used to predict the likelihood of a sequence of words, which can help in identifying the boundaries between tokens. Language models can implement both rule-based and probabilistic strategies for separating document strings into tokens. The Lexos [`tokenizer`](../api/tokenizer/index.md) module enables you to choose a language model appropriate to your data in order to split your texts into tokens.

!!! note
    The `tokenizer` module is a big change for Lexos, as it formally separates tokenization from preprocessing. In the Lexos web app, users employ Scrubber to massage the text into shape using their *implicit* knowledge about the text's language. Tokenization then takes place by splitting the text according to a regular expression pattern (normally whitespace). By contrast, the Lexos `tokenizer` module uses a language model that formalizes the implicit rules and probabilities needed to tokenize the text. Because they have built-in *explicit* procedures appropriate to specific languages, language models can often do a better job of tokenization than the approach used in the Lexos web app.

For languages such as Modern English, language models exist that can both split texts into tokens and automatically annotate tokens with attributes like parts of speech, lemmas, stop words, and other information. This procedure is often referred to as **Natural Language Processing (NLP)**.

If no language model exists for the text's language, it will only be possible to tokenize using general rules, and it will not be possible to add other annotations (at the tokenization stage). But new language models, including models for historical languages, are being produced all the time, and this is a growing area of interest in the Digital Humanities.

There are some trade-offs to using language models. Because the algorithm does more than split strings, processing times can be greater. In addition, the tokenization procedure is not completely language agnostic. A language model is "opinionated" and it may overfit the data. At the same time, if no language model exists for the language being tokenized, the results may not be satisfactory. The Lexos strategy for handling this situation is described below.

!!! note "Using Other Tokenizers"
    Many machine-learning tools &mdash; including the Lexos web app &mdash; deploy `scikit-learn`'s `CountVectorizer` (and similar) classes to perform tokenizations. Such tools combine the process of tokenizing and counting tokens, whereas the Lexos `tokenizer` module keeps them separate. Moreover, tools like `CountVectorizer` use simple regular expression patterns to divide texts into tokens and do not leverage the capabilities of a language model. If you need a tool like `CountVectorizer` to perform tokenization but still want to make use of the NLP capabilities of the Lexos Tokenizer, there is an example of how to do it on the [Document-Term Matrix](the_document_term_matrix.md#advanced-usage-with-scikit-learn-vectorizers) page.

## SpaCy `Docs`

Lexos uses the <a href="https://spacy.io/" target="_blank">spaCy</a> Natural Language Processing (NLP) library for loading language models and tokenizing texts. Because spaCy has excellent documentation and fairly wide acceptance in the Digital Humanities community, it is a good tool to use under the bonnet. SpaCy has a growing number of language models for a variety of languages, as well as wrappers for loading models from other common NLP libraries such as Stanford Stanza.

In spaCy, texts are parsed into spaCy `Doc` objects consisting of sequences of annotated tokens.

!!! note
    In order to formalize the difference between a text string that has been scrubbed and one that has been tokenized, we refer wherever possible to the string as a "text" and to the tokenized `Doc` object as a "document" (or just "doc"). We continue to refer to the individual items as "documents" if we are not concerned with their data type.

Each token is spaCy `Token` object which stores all the token's attributes.

### Creating a SpaCy `Doc` Object

The Lexos API wraps this procedure in the `Tokenizer.make_doc()` method:

```python
from lexos.tokenizer import Tokenizer

tokenizer = Tokenizer()
doc = tokenizer.make_doc("This is a test.")
```

This returns a `Doc` object.

By default the tokenizer uses spaCy's "<a href="https://spacy.io/models/xx#xx_sent_ud_sm" target="_blank">xx_sent_ud_sm</a>" language model, which has been trained for tokenization and sentence segmentation on multiple languages. This model performs statistical sentence segmentation and possesses general rules for token segmentation that work well for a variety of languages. The default model has been chosen to be as language-agnostic as possible, so it can be used for many languages without requiring a specific model. However, it is not guaranteed to work well for all languages.

If you were making a document from a text in a language which rquired a more language-specific model, you would specify the model to be used. For instance, to use spaCy's small English model trained on web texts, instantiate the `Tokenizer` class and use the `model` keyword argument to specify the model (it must be installed in your Python environment):

```python
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("This is a test.")
```

!!! note
    The `tokenizer` module is a wrapper for the spaCy library, so you can also use spaCy directly to create a `Doc` object since spaCy is installed with Lexos. Lexos is designed to make it easier to work with spaCy's functionality, but it is not necessary to use the Lexos API to work with spaCy. The equivalent of the above using spaCy's API is

    ```python
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a test.")
    ```

    In our documentation, we use the Lexos `Tokenizer` class because it automatically loads the default model.

Be sure that the model you specify is installed in your Python environment. You can install spaCy models using the command line, for example:

```bash
# Install the small French model
python -m spacy download fr_core_news_sm
```

The `Tokenizer` class also has a `make_docs()` method to parse a list of texts into a list of spaCy docs.

!!! important
    Tokenization with spaCy uses a lot of memory. For a small English-language model, the parser and named entity recognizer (NER) can require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the memory limit with the `max_length` parameter in `make_doc()` or `makes_docs()`. The limit is in number of characters (the default is set to 2,000,000 for Lexos), so you can check whether your inputs are too long by checking `len(text)`. If you are not using RAM-hungry pipeline components, you can disable or exclude them to avoid errors an increase efficiency (see the discussion on the spaCy pipeline below). In some cases, it may also be possible to cut the texts into segments before tokenization. See [Cutting Documents](cutting_documents.md) for more information.

### Working with SpaCy `Docs`

A list of individual tokens can be obtained by iterating over the spaCy doc:

```python
# Use a for loop to iterate over the tokens
for token in doc:
    print(token.text, token.pos_, token.is_stop)

# Get a list of tokens with a list comprehension
tokens = [token.text for token in doc]
```

Here the `text` attribute stores the original text form of the token. Tokenizing texts into SpaCy docs is *non-destructive* because original text is preserved alongside the list of tokens and their attributes. You can access the original text of the entire doc by calling `doc.text` (assuming you have assigned the `Doc` object to the `doc` variable). Indeed, calling `doc.to_json()` will return a JSON representation which gives the start and end position of each token in the original text!

As mentioned above, you can use a Python list comprehension to filter the the contents of the doc using information in the document's attributes. For instance:

```python
# Get a list of non-punctuation tokens
non_punct_tokens = [token.text for token in doc if not token.is_punct]
```

The example above leverages the built-in `is_punct` attribute to indicate whether the token is defined as (or predicted to be) a punctuation mark in the language model. SpaCy docs have a number of built-in attributes, which are described in the <a href="https://spacy.io/api/doc#attributes" target="_blank">spaCy API reference</a>.

!!! note
    It is possible to extend spaCy's Doc object with its extension attribute. For instance, if you wanted to have an `is_fruit` attribute, you could create an extension and then access it using the underscore prefix, as shown below:

    ```python
    # Indicate whether the token is labelled as fruit
    for token in doc:
        print(token._.is_fruit)
    ```

    For information on creating custom extensions, see the <a href="https://spacy.io/usage/processing-pipelines#custom-components-attributes" target="_blank">spaCy documentation</a>.

### Handling Stop Words

**Stop words** are tokens that are often filtered out in text processing because they do not carry significant meaning for the intended task. Examples include grammatical function words like "the", "is", "in", and "and". In many cases, it is useful to remove stop words from the text before performing further analysis, such as frequency counts or topic modeling. Many language models come with a predefined list of stop words that are commonly used in the language. These stop words are used to set the `is_stop` attribute for each stop word token to `True` when a document is tokenized. It is also possible to add stop words to or remove stop words from tokenizer using the `add_stopwords()` and `remove_stopwords()` method:

```python
tokenizer = Tokenizer(model="en_core_web_sm")
tokenizer.add_stopwords(["yes", "no", "maybe"])
tokenizer.remove_stopwords(["and", "or", "for"])
doc = tokenizer.make_doc(text)
```

A search of the web can often yield a list of stop words for the language you are working in, and you will often have to add stop words to obtain satisfactory results. If you are using a language model that does not have a predefined stop word list, you can use the `add_stopwords()` method to add your own stop words.

!!! important
    `add_stopwords()` and `remove_stopwords()` do not add or remove tokens from the `Doc`; instead, they modify the stop word list used to set the `is_stop` attribute of individual tokens to `True` or `False`. To get a list of tokens without stop words, you must **filter** them with something like `[token for token in doc if not token.is_stop]`. If you are producing a corpus of documents in which the documents will be processed by different models, it is most efficient to process the documents in batches, one batch for each model.

### Modifying the SpaCy Pipeline

Once spaCy tokenizes a text, it normally passes the resulting document to a pipeline of functions to parse it for other features. Typically, these functions will perform actions such as part-of-speech tagging, labelling syntactic dependencies, and identifying named entities (named entity recognition, or NER). Processing times can be increased by disabling pipeline components if they are unavailable in the language model or not needed for the application's purposes. `make_doc()` and `make_docs()` will automatically run all pipeline components in the model unless they are disabled with the `disable` parameter.

```python
doc = tokenizer.make_doc(text, disable=["parser", "ner"])
```

Check the model's documentation for the names of the components it includes by default.

It is also possible to include custom pipeline components, which can be inserted at any point in the pipeline order. Custom components are supplied with the `pipeline_components` parameter, which takes a dictionary containing the keyword "custom". The value is a list of dictionaries where each dictionary contains information about the component as described in <a href="https://spacy.io/api/language/#add_pipe" target="_blank">spaCy's documentation</a>.

!!! note
    The `pipeline_components` dict also contains `disable` and `exclude` keywords. The values are lists of components which will be merged with any components supplied in the `disable` or `exclude` paramaters of `make_doc()` and `make_docs()`.

The ability to add custom pipeline components is valuable for certain language- or application-specific scenarios. However, it also opens Lexos up to the wealth of third-part pipeline components available through the <a href="https://spacy.io/universe/category/pipeline" target="_blank">spaCy Universe</a>.

## Custom Tokenizers

Sometimes using a language model to perform tokenization is not appropriate or is overkill for the desired output. Lexos has two tokenizer classes that operate on strings and return lists of strings. They mostly illustrate how you can produce your own tokenizer class if required.

`SliceTokenizer` slices the text into tokens of `n` characters. The constructor takes two arguments: `n`, which is the number of characters  in each output token, and `drop_ws`, a modifier that controls whether to drop whitespace or keep it.

```python
from lexos.tokenizer import SliceTokenizer
test_text = "Cut me up into tiny pieces!"
slicer = SliceTokenizer(n=4, drop_ws=True)
slices = slicer(test_text)
print(slices)
```

`WhitespaceTokenizer` simply slices a text into tokens on whitespace, similarly to Python's built-in `split()` method.

```python
from lexos.tokenizer import WhitespaceTokenizer
test_text = "Split me up by whitespace!"
neatSlicer = WhitespaceTokenizer()
slices = neatSlicer(test_text)
print(slices)
```

!!! note
    The NLP community are largely concerned with extracting *linguistic* features from text strings where whitespace is typically ignored. But scholars in the Humanities have pointed out that in the texts they study, whitespace itself can be meaningful. Lexos has an experimental `WhitespaceCounter` tokenizer that extends the `Tokenizer` class by counting runs of spaces and line breaks. See the [Whitespace Counter](../api/tokenizer/whitespace_counter.md) documentation for more information.

    To use it, instantiate the `WhitespaceCounter` class and call it with a text string:

    ```python
    from lexos.tokenizer.whitespace_counter import WhitespaceCounter

    wc = WhitespaceCounter()
    tokens = wc("This  is   a\n\n test.")
    for token in tokens:
        print(token, token._.width)
    ```

## Generating Ngrams

Both texts and documents can be parsed into sequences of two or more tokens called **ngrams**. Many spaCy models can identify syntactic units such as noun chunks. These capabilities are not covered here since they are language specific. Instead, the section below describe how to obtain more general ngram sequences.

The easiest method of obtaining ngrams from a text is to create a spaCy doc and then call the [`ngrams_from_doc()`](../api/tokenizer/ngrams.md#lexos.tokenizer.ngrams.Ngrams.from_doc) method:

```python
import spacy
from lexos.tokenizer.ngrams import Ngrams as ng

nlp = spacy.load("xx_sent_ud_sm")
text = "The end is nigh."
doc = nlp(text)

ng = Ngrams()

ngrams = ng.from_doc(doc, size=2)
for ngram in ngrams:
    print(ngram.text)
# The end
# end is
# is nigh
# nigh .
```

The `from_doc()` function yields a generator, so, if you wish to view it as a list, you need to call `list(ngrams)` on the output shown above. The size of the ngrams is specified by the `size` parameter, which defaults to 2. Setting it to 3, for instance, will result in the ngrams "The end is", "end is nigh", "is nigh ."

!!! note
    The `from_doc()` function is a wrapper for the `textacy.extract.basics.ngrams` method, which is part of the <a href="https://textacy.readthedocs.io/en/latest/" target="_blank">Textacy</a> library. You can call Textacy directly as shown below:

    ```python
    import textacy.extract.basics.ngrams as ng
    ngrams = ng(doc, 2, min_freq=2)
    ```

    The `min_freq` parameter removes ngrams that do not occur at least two times. This can cut down on the size of the generated ngrams. Textacy has a lot of additional options, which are documented in the Textacy API reference under <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/extract.html#textacy.extract.basics.ngrams" target="_blank">textacy.extract.basics.ngrams</a></code>. The Lexos `Ngrams.from_doc()` method accepts the same parameters as Textacy's method with a few additional options (see the [API documentation](../api/tokenizer/ngrams.md)).

There is also a `Ngrams.from_docs()` method that accepts a list of `Doc` objects and returns a list of ngram generators.

If you do not want to use a language model, the `Ngrams` class also accepts input data in the form of text strings with `Ngrams.from_text()` and `Ngrams.from_texts()`. By default, your text(s) will be processed using the `WhitespaceTokenizer` before the ngrams are generated, although you can swap it for another tokenizer. If you use the `SliceTokenizer`, you will produce **character ngrams**. For instance, the text "Hello world" will produce the bigrams "He, el, lo, o , w, wo, or, rl, ld". (You can also generate character ngrams by calling `SliceTokenizer` directly: `ngrams = SliceTokenizer(text, n=2)`). Note that your language model may not be able apply labels effectively to ngram tokens, so working with character ngrams is primarily useful if you are planning to work with the token forms only, or if the ngram size you use maps closely to character lengths of words in the language you are working in.

If you have a list of pre-tokenized strings, you can use the `Ngrams.from_tokens()` method. For instance, `ngrams = ng.from_tokens(["Hello", "world", "how", "are", "you"], n=3)` will generate "Hello world how, world how are, how are you".

In some cases, you may wish to generate a `Doc` object with ngrams as tokens. This can be done by calling spaCy's `Doc.from_docs()` method, which takes a list or tuple of ngrams and returns a new spaCy `Doc`:

```python
from spacy.tokens import Doc
new_doc = Doc.from_docs(ngrams)
```
