## Cutting Documents

`Cutter` is a module that divides files, texts, or documents into segments. At present, it is highly experimental. There are two classes for cutting documents into segments, one for working with spaCy documents (codename `Ginsu`) and one for working with raw texts (codename `Machete`). A third class `FileSplit` (codename `Chainsaw`) can be used to split files based on bytesize.

### `Ginsu`

The `Ginsu` class is used for splitting spaCy documents (pre-tokenised texts).

<iframe style="width: 560px; height: 315px; margin: auto;" src="https://www.youtube.com/embed/Sv_uL1Ar0oM" title="YouTube video player -- Ginsu knives" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

`Ginsu` is the preferred method for creating segments because it can access information supplied by the language model.

`Ginsu` has the following features:

- [`split()`][lexos.cutter.ginsu.Ginsu.split]: Split by number of tokens (i.e. every N token).
- [`splitn()`][lexos.cutter.ginsu.Ginsu.splitn]: Split by number of segments (i.e. return a predetermined number of segments).
- [`split_on_milestones()`][lexos.cutter.ginsu.Ginsu.split_on_milestones]: Split on milestone tokens.

`Ginsu.split()` and `Ginsu.splitn()` both have the ability to merge the list segment into the preceding one if it falls under a customisable threshold. Likewise, they can generate overlapping segments. All three methods return a list of lists, where each item in the sublist is a spaCy document.

With `Ginsu.split_on_milestones()`, the user can choose whether or not to preserve the milestone token at the beginning of each segment. A milestone must be a single token and will generally match the token's `text` attribute. However, it can also match other attributes of the token if they are available in the language model used to produce the spaCy document. There is also an elaborate query language for fine-grained matching.

#### Splitting Documents by Number of Tokens

To split documents every N tokens, use the `Ginsu.split` method. Here are some examples:

```python
from lexos.cutter.ginsu import Ginsu

cutter = Ginsu()

segments = cutter.split(doc)

segments = cutter.split([doc1, doc2])
```

By default, the document will be split every 1000 tokens. Here is an example with the full range of options:

```python
segments = cutter.split(doc, n=500, overlap=5, merge_threshold=0.5)
```

This will split the document every 500 tokens with each segment overlapping by 5 tokens. If the final segment is less than half the length of `n`, it will be merged with the previous segment (0.5 is the default).

#### Splitting Documents into a Pre-Determined Number of Segments

To split documents into a pre-determined number of segments, use the `Ginsu.split` method. Here are some examples:

```python
from lexos.cutter.ginsu import Ginsu

cutter = Ginsu()

segments = cutter.splitn(doc, n=10)

segments = cutter.splitn([doc1, doc2], n=10, overlap=5, merge_threshold=0.5)
```

This will split the document(s) into 10 segments. The `overlap` and `merge_threshold` flags work exactly as they do for the `Ginsu.split()` method.

#### Splitting Documents on Milestones

A milestone is a pattern that serves as a boundary between segments of a document. The `Ginsu.split_on_milestones()` method accepts a pattern to match tokens to milestones. Typically a milestone pattern will be a simple string, and Cutter will split the spaCy doc whenever the `text` attribute of a token matches the pattern. As we will see below, more complex pattern matching methods are possible.

```python
from lexos.tokenizer import make_doc
from lexos.cutter.ginsu import Ginsu

text = """
    It is a truth universally acknowledged, that a single man
    in possession of a good fortune, must be in want of a wife.
"""
doc = make_doc(text.strip())

cutter = Ginsu()
segments = cutter.split_on_milestones(doc, milestone=",")
print(segments)
"""
[
    "It is a truth universally acknowledged",
    "that a single man in possession of a good fortune",
    "must be in want of a wife."
]
```

By default, the milestone is deleted from the result. If you wish to preserve it, use `preserve_milestones=True`. This milestone token will be preserved at the beginning of each segment.

If the the `milestone` parameter is supplied with a list of strings, each item in the list will be treated as a milestone:

```python
segments = cutter.split_on_milestones(
        doc,
        milestone=["be", "is"],
        preserve_milestones=True
)
print(segments)
"""
[
    "It",
    "is a truth universally acknowledged, that a single man in possession of a good fortune, must",
    "be in want of a wife."
]
```

Milestone patterns can also be matched using a query language supplied in the form of a dict, where the keyword is the name of a spaCy token attribute (e.g. `text`, `pos_`, `lemma_`, etc.) and the value is the value to match. By default a token matches a milestone if its value for the attribute is equivalent the value given in the milestone dict. In other words, if the dict has the form `{"text": "chapter"}`, any token in the document for which `token.text` returns "chapter" will be treated as a milestone. As result, the following commands are functionally equivalent:

```python
segments = cutter.split_on_milestones(doc, milestone=",")

segments = cutter.split_on_milestones(doc, milestone={"text": ","})
```

However, the following will treat all varieties of "chapter" ("chapters", "chapter's", "Chapter", etc.) as milestones:

```python
segments = cutter.split_on_milestones(doc, milestone={"lemma_": "chapter"})
```

!!! note
    For further information on spaCy token attributes, see [Tokenising Texts](../tokenising_texts/).

Milestone dict values can also be given as tuples consisting of a pattern and an operator. The operator is the method used to perform the match. Currently, the following operators are available:

- `in`: If the pattern is given as a list of strings, any item in the list will be treated as a milestone.
- `not_in`: If the pattern is given as a list of strings, any item not in the list will be treated as a milestone.
- `starts_with`: Any item starting with the string pattern will be treated as a milestone (uses the Python `startswith()` function).
- `ends_with`: Any item ending with the string pattern will be treated as a milestone (uses the Python `endswith()` function).
- `re_match`: Any item starting with the regular expression pattern will be treated as a milestone (uses the Python `re.match()` function).
- `re_search`: Any item containing with the regular expression pattern will be treated as a milestone (uses the Python `re.search()` function).

For example, the following command will treat any adjective or noun as a milestone.

```python
segments = cutter.split_on_milestones(
        doc,
        milestone={"pos_": (["ADJ", "NOUN"], "in")}
)
```

More complex queries can be built using the `and` and `or` keywords. For instance, the following pattern is the equivalent of the previous example.

```python
segments = cutter.split_on_milestones(
        doc,
        milestone={
            "and": {"pos_": "ADJ"},
            "or": {"pos_": "NOUN"}
        }
)

In the example below, the word "can" would be treated as a milestone only when it functions as a verb:

```python
segments = cutter.split_on_milestones(
        doc,
        milestone={
            "and": {"text": "can"},
            "and": {"pos_": "VERB"}
        }
)

!!! important
    Filtering based on parts of speech, lemmas, and some other attributes is only possible if you have tokenised your documents using a language model that contains the relevant attribute.

#### The `Milestone` Class

The `Ginsu` split functions match tokens to milestones patterns and cut documents on the fly. However, it is also possible to use the [lexos.cutter.milestones.Milestones][] class to preprocess documents. This adds the custom extension `token._.is_milestone` (by default `False`) to each token in the document and uses the same query language to allow the user to match tokens where the value should be `True`. If documents are pre-processed in this way, the [lexos.cutter.Ginsu.split_on_milestones][lexos.cutter.Ginsu.split_on_milestones] method can leverage that information.

```python
from lexos.tokenizer import make_doc
from lexos.cutter.ginsu import Ginsu
from lexos.cutter.milestones import Milestones

text = """
    It is a truth universally acknowledged, that a single man
    in possession of a good fortune, must be in want of a wife.
"""
doc = make_doc(text.strip())

# Set commas as milestones
Milestones().set(doc, ",")

cutter = Ginsu()
segments = cutter.split_on_milestones(doc, milestone={"is_milestone": True})
print(segments)
"""
[
    "It is a truth universally acknowledged",
    "that a single man in possession of a good fortune",
    "must be in want of a wife."
]
```

!!! note
    Custom extension attributes like `token._.is_milestone` are normally referenced with the `_.` prefix. However, in milestone dicts, only the name "is_milestone" is given, parallelling built-in attributes like "is_punct". The same is true for any other custom extension available in the token's attributes.

An obvious advantage of preprocessing milestones is that the custom attribute can be saved. If the user chooses to preserve milestones when using `Ginsu.split_on_milestones`, the milestone will appear at the beginning of each document, but will not have the `_.is_milestone` attribute. In the future, it is hoped that we will incorporate the ability to set milestones on the fly.

### Machete

`Machete` is a cruder method of cutting raw text into segments without the benefit of a language model. It may be particularly valueable as a standalone method of segmenting texts for outside applications.

#### The `Machete` Tokenizer
`Machete` works in a manner similar to `Ginsu` and has all the same functionality. However, before splitting the text it applies a makeshift tokenizer function and then splits the text based on the resulting list of tokens.

The Lexos API has three tokenizer functions in the `cutter` function registry: "whitespace" (the default), "character", and "linebreaks". A `Machete` object can be initialised with one of the tokenizers or the tokenizer can be passed to the `Machete.split()`, `Machete.splitn()`, and `Machete.split_on_milestones()` methods using the `tokenizer` parameter.

!!! information "What if I don't like the tokenizer?"
    You can supply a custom function after first adding it to the registry. Here is an example:

    ```python
    from lexos.cutter.machete import Machete
    import lexos.cutter.registry

    def custom_punctuation_tokenizer(text: str) -> str:
        """Split the text on punctuation or whitespace."""
        return re.split(r"(\W+)", text)

    # Register the custom function
    registry.tokenizers.register("custom_punctuation_tokenizer", func=custom_punctuation_tokenizer)

    # Create a `Machete` object
    machete = Machete()

    # Split the texts into 5 segments
    result = machete.splitn(texts, n=5, tokenizer="custom_punctuation_tokenizer")
    ```

#### Splitting Documents with `Machete`

`Machete.split()`, `Machete.splitn()`, and `Machete.split_on_milestones()` return a list of lists, where each item in the outer list corresponds to a text and each sublist contains the text's segments. `Machete.split()`, `Machete.splitn()` take the same `merge_threshold` and `overlap` parameters as in the `Ginsu` class. `Machete.split_on_milestones()` is more limited than its `Ginsu` equivalent. Milestone patterns are evaluated as regular expressions and searched from the beginning of the token string using Python's `re.match()` function.


By default, all three methods return segments as lists of strings. In the example below, we get results with the default `as_string=True`:

```python
from lexos.cutter.machete import Machete

text = """
    It is a truth universally acknowledged, that a single man
    in possession of a good fortune, must be in want of a wife.
"""

cutter = Machete()
segments = cutter.split_on_milestones(text.strip(), n=2)
print(segments)
"""
[
    [
    "It is a truth universally acknowledged, that a single man in possession",
    "of a good fortune, must be in want of a wife."
]
```

The basis for the division can be seen by displaying the segments as lists of tokens using `as_string=False`:

```python
segments = cutter.split_on_milestones(text.strip(), n=2, as_string=False)
print(segments)
"""
[
    ['It ', 'is ', 'a ', 'truth ', 'universally ', 'acknowledged, ', 'that ', 'a ', 'single ', 'man\n    ', 'in ', 'possession '],
    ['of ', 'a ', 'good ', 'fortune, ', 'must ', 'be ', 'in ', 'want ', 'of ', 'a ', 'wife.\n']
]
```

The default "whitespace" tokenizer does not strip whitespace around words and does not treat punctuation as separate functions. This makes it very easy to reconstitute the text as a string, but it may not be the desired behaviour for all applications. You may need to apply a custom tokenizer as described above. For specific languages, more reliable results may be obtained by using a language model as described in [Tokenising Texts](../tokenising_texts/).

#### Splitting Lists of Tokens with `Machete`

Sometimes data will be available as lists of tokens, rather than as strings (for instance, if you have already tokenised your texts using a tool like <a href="https://www.nltk.org/" target="_blank">NLTK</a>). In this case, you can cut your texts using the `Machete.split_list()` method. It works just like `Machete.split()`, except that it takes a list of tokens as input.

def split_list(
        self,
        texts: List[str],
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
        as_strin

```python
from lexos.cutter.machete import Machete

text = [
        "It", "is", "a", "truth", "universally", "acknowledged", "that",
        "a", "single", "man", "in", "possession", "of", "a", "good",
        "fortune", "must", "be", "in", "want", "of", "a", "wife"
    ]

# Pad each token with a following space
text = " ".join(text).split()

cutter = Machete()
segments = cutter.split_list(text, n=12)
print(segments)
"""
[
    [
    "It is a truth universally acknowledged that a single man in possession",
    "of a good fortune must be in want of a wife."
]
```

!!! note
    `Machete` assumes that spaces and punctuation between words are preserved, which makes it easier to return the segments as a human-readable string. In the example above, our token list did not preserve spaces or punctuation between words. As a result, it was necessary to take the extra step of padding each token (although the resulting segments have still lost the original punctuation). Steps like this may be necessary when using lists of tokens, depending on how they are created.

It should be noted that, whilst of the Machete functions will accept either a single text string or a list of text strings, `Machete.split_list()` accepts only a list of token strings. If you have multiple lists, you should handle them in a loop (or equivalent list comprehension) as follows:

```python
from lexos.cutter.machete import Machete

texts = [
            ["It", "is", "a", "truth", "universally", "acknowledged", "that", "a", "single", "man"],
            ["in", "possession", "of", "a", "good", "fortune", "must", "be", "in", "want", "of", "a", "wife"]
    ]

# Pad each token with a following space
texts = [" ".join(text).split() for text in texts]

cutter = Machete()
all_segments = [cutter.split_list(text, n=6) for text in texts]
print(all_segments)
"""
[
    [
        "It is a truth universally acknowledged",
        "that a single man in possession"
    ], [
        "of a good fortune must be",
        "in want of a wife."
    ]
]
```

### Filesplit (codename `Chainsaw`)

The [lexos.cutter.filesplit.Filesplit][] class allows the user to cut binary files into numbered file segments with the format `filename_1.txt`, `filename_2.txt`, etc. The source file is divided by number of bytes. This class would typically be used as a first step in a workflow if a large file needs to be divided into many smaller files.

The class is a fork of Ram Prakash Jayapalan's <a href="https://github.com/ram-jayapalan/filesplit/releases/tag/v3.0.2" target="_blank">filesplit</a> module with a few minor tweaks. The most important is that the `split` function takes a `sep` argument to allow the user to specify the separator between the filename and number in each generated file.

Typical usage is as follows:

```python
from lexos.cutter.filesplit import Filesplit

fs = Filesplit()

fs.split(
    file="/filesplit_test/longfile.txt",
    split_size=30000000,
    output_dir="/filesplit_test/splits/"
)
```

There are several options available, for which see the API documentation.

The class generates a manifest file called `fs_manifest.csv` in the output directory. This can be used to re-merge the files, if desired:

```python
fs.merge("/filesplit_test/splits/", cleanup=True)
```
