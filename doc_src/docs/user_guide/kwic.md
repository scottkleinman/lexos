# Keywords in Context (KWIC)

## Overview

Keywords in Context (KWIC) is a common method of finding all the examples of a term in a document in the context of the text immediately before and after the term. Lexos provides sophisticated methods of searching for keywords that make the process easy.

### Basic Usage

The basic procedure is as follows:

```python
# Import the Kwic class
from lexos.kwic import Kwic

# Define a text
text = "It is a truth universally acknowledged, that a single  man in possession of a good fortune must be in want of a wife."

# Define a pattern
pattern = "universally"

# Create an instance of a Kwic object
kwic = Kwic()

# Pass the object the desired parameters
kwic(docs=text, patterns=pattern, window=10)
```

This will display

|    | doc   | context_before   | keyword     | context_after   |
|---:|:------|:-----------------|:------------|:----------------|
|  0 | Doc 1 | s a truth        | universally | acknowled       |

You will notice that the keywords `docs` and `patterns` are plural. This is because the `Kwic` class will also accept multiple documents and lists of patterns. A document, in the case, can be a raw text string, but spaCy `Doc` objects are also accepted. Patterns may be raw strings or regex patterns.

The `window` keyword will by default provide a window of *n* characters around each keyword found.

The standard output is a pandas DataFrame. You can either display this DataFrame directly, or assign it to a variable for further processing:

```python
df = kwic(docs=text, patterns=pattern, window=10)
```

By default, your documents will be labelled "Doc 1", "Doc 2", "Doc3", etc. However, you can supply a list of labels with the `labels` keyword.

Here are some other useful parameters:

- `case_sensitive`: If set to `False`, the `Kwic` class will perform case-insensitive searches.
- `sort_by`: If you wish to sort the DataFrame, use this keyword to set the column for sorting.
- `ascending`: Set to `True` or `False` to set the order for sorting the DataFrame.
- `as_df`: If set to `False`, the output will be a list of tuples, where the first item is the "before" window, the second item is the matched pattern, and the third item is the "after" window.

### Searching Token Patterns

If you have spaCy `Docs`, you may wish to search for token patterns. A simple way to do this is to set the matcher to "tokens".

Consider the example above. If we have a spaCy `Doc`, we can re-write it as

```python
kwic(docs=doc, labels=None, patterns=patterns, window=5, matcher="tokens")
```

|    | doc   | context_before       | keyword    | context_after          |
|---:|:------|:---------------------|:-----------|:-----------------------|
|  0 | Doc 1 | that a single man in | possession | of a good fortune must |

Now the `Kwic` class will search for the *token* "universally", and the "before" and "after" windows will be counted in tokens, rather than characters.

We can use the `use_regex` keyword to use a regular expression for our pattern:

```python
patterns = r".ingle"

kwic(docs=doc, labels=None, patterns=patterns, window=5, matcher="tokens", use_regex=True, case_sensitive=False)
```

This will find any token containing a character followed by "ingle" (and the search will be case insensitive).

|    | doc   | context_before                   | keyword   | context_after          |
|---:|:------|:---------------------------------|:----------|:-----------------------|
|  0 | Doc 1 | universally acknowledged, that a | single    | man in possession of a |

We can also perform more sophisticated token-based searches using spaCy's rule-matching syntax. To use it, we set the `matcher` parameter to "rule". See the <a href="https://spacy.io/usage/rule-based-matching#matcher" target="_blank">spaCy documentation</a> for details of how to construct rules for token-based matching.

```python
pattern1 = [{"LOWER": "truth"}, {"LOWER": "universally"}, {"LOWER": "acknowledged"}]
pattern2 = [{"TEXT": "possession"}]
patterns = [pattern1, pattern2]

kwic(docs=doc, patterns=patterns, window=5, matcher="rule")
```

|    | doc   | context_before       | keyword                        | context_after          |
|---:|:------|:---------------------|:-------------------------------|:-----------------------|
|  0 | Doc 1 | It is a              | truth universally acknowledged | , that a single man    |
|  1 | Doc 1 | that a single man in | possession                     | of a good fortune must |

Finally, we can also search multi-token patterns using spaCy's `PhraseMatcher` by setting `matcher` to "phrase".

```python
patterns = ["truth universally acknowledged", "possession"]

kwic(docs=doc, patterns=patterns, window=5, matcher="phrase")
```

|    | doc   | context_before       | keyword                        | context_after          |
|---:|:------|:---------------------|:-------------------------------|:-----------------------|
|  0 | Doc 1 | It is a              | truth universally acknowledged | , that a single man    |
|  1 | Doc 1 | that a single man in | possession                     | of a good fortune must |

This differs from the previous example because the patterns are first pre-processed into `Doc` objects, which can be significantly faster to process if you have large numbers of patterns.
