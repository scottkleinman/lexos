# Topwords Analysis

## Overview

The `lexos.topwords` module provides tools for identifying statistically significant words in a corpus by comparing target documents against background documents. It enables comparative text analysis to discover which terms best distinguish one set of texts from another.

The `topwords` module helps you answer questions like:

- What words make Document A unique compared to all other documents?
- What terms distinguish Shakespeare's plays from Marlowe's plays?
- Which words are most characteristic of scientific papers versus news articles?

A "topword" is a term that is highly ranked according to some statistical metric. Since there are numerous ways to measure statistical significance, Lexos provides three different Python classes for extracting topwords: `KeyTerms`, `ZTest`, `MannWhitney`.

- `KeyTerms` extracts significant keywords from individual documents using graph-based ranking algorithms.
- `ZTest` identifies statistically over-represented words in target documents compared to comparison documents by calculating their <a href="https://en.wikipedia.org/wiki/Standard_score" target="_blank">z-score</a>.
- `MannWhitney` implements the <a href="https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test" target="_blank">Mann-Whitney U test</a> (also called the Wilcoxon rank-sum test) to compare two groups of documents to determine if they differ significantly.

!!! Note
    Each of these classes inherits from a basic `TopWords` class to define a common API for all topwords methods, creating a plugin-like architecture for additional methods to be added. Currently, this plugin structure is a bit limited. Each class is expected to have a `to_df()` method, but no other common elements are expected.

There is also a `Compare` class which implements provides a workflow for compare documents across a corpus or comparing documents belonging to different classes. Any test that inherits from the base `TopWords` class can be used with `Compare`.

Each of these classes is implemented as a submodule of `topwords`.

## Submodules

### KeyTerms

The `KeyTerms` module extracts significant keywords from documents using graph-based ranking algorithms. Ranking algorithms build a graph where words are nodes and their co-occurrences create edges, then rank words by their importance in the network. Unlike statistical tests that compare document sets, `KeyTerms` analyzes individual documents to identify their most "central" or important terms.

**Common Use Cases:**

1. **Document summarization** - Extract key concepts from articles
2. **Metadata generation** - Auto-generate tags for documents
3. **Topic identification** - Understand what a document is about
4. **Search indexing** - Identify important terms for search
5. **Content analysis** - Analyze themes in qualitative data

**Available Algorithms:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `textrank` | PageRank applied to words (default) | General purpose, most documents |
| `sgrank` | Statistical selection, positional ranking | Longer documents, academic papers |
| `scake` | Single candidate keyword extraction | Short documents, tweets |
| `yake` | Yet Another Keyword Extractor | Multilingual documents |

!!! Note
    `KeyTerms` implements these algorithms using the Python <a href="https://textacy.readthedocs.io/en/latest/api_reference/extract.html#keyterms" target="_blank">Textacy</a> library. Further documentation about their use can be found there.

The `KeyTerms` class accepts either a string or a spaCy `Doc`. Here is an example:

```python
from lexos.topwords.keyterms import KeyTerms

# Basic usage with a single document
text = """
Machine learning is a subset of artificial intelligence that focuses on
algorithms that learn from data. Deep learning uses neural networks with
multiple layers to process complex patterns.
"""

kt = KeyTerms(
    document=text,
    method="textrank",
    topn=10,
    model="xx_sent_ud_sm",
    ngrams=1,
    normalize="lemma"
)

# Access keyterms directly
print(kt.keyterms)
# [('learning', 0.35), ('machine', 0.28), ('neural', 0.22), ...]

# Get results as a pandas DataFrame
results_df = kt.to_df()
```

The code above shows the default settings. If you wish to use ngrams, you can set `ngrams=2` (or 3, 4, etc.). You can also count multiple ngrams with a tuple. For instance, `ngrams=(1, 2)` will count both single terms and bigrams. Note, however, that the `scake` method does not accept ngrams.

The default `normalize` setting counts all variant forms of a lemma (if the language model supports lemma). You can also set `normalize="lower"` to make counts case-insensitive. You can also turn off normalization with `normalize=None`.

!!! Note
    The `KeyTerms` class accepts a string or spaCy `Doc` (automatically converting strings to `Docs` using the chosen language model). However, it does not currently accept lists of strings or `Docs`. Since, strings are internally converted to `Docs`, you may find it more efficient to preprocess string data with the Lexos `tokenizer` module.

---

### ZTest

Identifies statistically over-represented words in target documents compared to comparison documents using a two-proportion z-test.

**How it works:**

- Calculates the proportion of each term in target vs. comparison documents
- Computes a z-score for the difference in proportions
- Returns terms with the highest z-scores, filtering out terms with a z-score of 0.0

These terms are deemed to be the most distinctive of the target documents.

!!! Note
    There is a separate implementation of z-term calculation in the `corpus` module. If you are using `corpus`, you may find it easier to perform z-score testing directly in that module.

Here is an example of how the `ZTest` class is used:

```python
from lexos.topwords.ztest import ZTest

# Basic usage with strings
target = ["This is Shakespeare's unique style.", "More Shakespeare text."]
comparison = ["This is Marlowe's writing.", "More Marlowe text."]

# Perform the test and return the 5 highest-ranked topwords
ztest = ZTest(target_docs=target, comparison_docs=comparison, topn=5)

# Access topwords directly
print(ztest.topwords)
# [('shakespeare', 2.45), ('unique', 1.89), ...]
```

By default, the topwords will be a list of tuples containing the terms and their z-scores. You can view the results in other formats using `to_dict()`, `to_list_of_dicts()`, or `to_df()` (returns a pandas DataFrame).

By default, terms are counted in a case-sensitive manner. If you wish to count both lower and upper case forms as the same term, set `case_sensitive=False`.

The class counts only single terms by default, but you can count ngrams by setting `ngrams=2` (or 3, 4, etc.). You can also count multiple ngrams with a tuple. For instance, `ngrams=(1, 2)` will count both single terms and bigrams.

In the example above, we have used strings for documents, but you can also use spaCy `Doc` objects. Internally, strings are converted to spaCy `Docs` using the `xx_sent_ud_sm` model. You can change this to another model with the `model` keyword (e.g. `model=en_core_web_sm`). Note that converting strings to `Docs` can take some time, so you may want to do that in advance and then pass your `Docs` directly to `ZTest`.

The use of spaCy `Docs` provides access to token attributes, it is possible to filter the tokens before they are counted by setting the `remove_stopwords`, `remove_punct`, and `remove_digits` keywords to `True`.

---

### MannWhitney

The Mann-Whitney U test (also called the Wilcoxon rank-sum test) is a statistical method that compares two groups to determine if they differ significantly. Unlike the z-test, it doesn't make assumptions about how the data is distributed.

Instead of comparing proportions directly (like the z-test), the Mann-Whitney test:

1. Ranks all term frequencies across both document sets
2. Compares the ranks between target and comparison documents
3. Calculates a U-statistic that measures how different the rankings are
4. Provides a p-value indicating the probability the difference occurred by chance

The `MannWhitney` class takes as its input a pandas DataFrame of term frequencies with docs in rows and terms in columns. Any filtering of your terms must be done in advance. The easiest way to produce the input DataFrames is with the Lexos DTM module. Here is an example:

```python
from lexos.dtm import DTM
from lexos.tokenizer import Tokenizer
from lexos.topwords.mann_whitney import MannWhitney

texts = [
    "This is a sample text for testing.",
    "Here is another example of a text to analyze.",
    "This text is different from the others.",
    "Yet another sample text for comparison.",
    "This text is similar to the first one.",
    "A completely different text for the analysis.",
]

# Process the sample texts with spaCy to create documents
docs = tokenizer.make_docs(texts)
labels = [f"Doc{i + 1}" for i in range(len(docs))]

# Create a Document-Term Matrix (DTM) using the sample documents
# Limit to terms occurring in at least 2 documents
dtm = DTM()
dtm(docs=docs, labels=labels, min_df=2)
df = dtm.to_df(transpose=True)
```

In the example above, we'll split the DataFrame into target and comparison data based on whether the label has an even or odd number.

```python
even_df = df[df.index.isin(["Doc2", "Doc4", "Doc6"])]
odd_df = df[df.index.isin(["Doc1", "Doc3", "Doc5"])]

mw = MannWhitney(target=even_df, comparison=odd_df)
mw.to_df()
```

The output will show the terms ranked by their distinctiveness, along with their U statistic and p-value.

The p-value is the probability that a test statistic is extreme or more extreme than the one observed, assuming that the two samples come from the same distribution. A small p-value (typically less than 0.05) indicates that the observed difference between the two samples is statistically significant, and we conclude that the two samples do not come from the same distribution.

By default, the table displays the average frequency of terms in the control group along with the increase in frequency in the comparison group. This provides us with another view of how important the word is to the sample and its relative over- or under-usage in comparison to the other sample. You can suppress the average frequency and difference columns with `add_freq=False`.

The following points provide a useful guide to interpreting the results:

- **u_statistic**: Higher values indicate the term appears more in target documents
- **p_value**: Lower values (< 0.05) indicate statistically significant differences
  - p < 0.05: Statistically significant (95% confident)
  - p < 0.01: Highly significant (99% confident)
  - p < 0.001: Very highly significant (99.9% confident)

### When to Use Mann-Whitney vs. Z-Test

There are a number of key differences between the two types of tests:

- Z-test assumes terms are normally distributed (that is, data has a rough bell curve shape with frequencies decreasing evenly on both sides of the central mean). Mann-Whitney makes no such assumption.
- Z-test is more powerful with large, well-behaved data; Mann-Whitney works better with small or irregular data.
- Z-test gives a z-score (can be positive or negative); Mann-Whitney gives a U-statistic and p-value. Lower p-values (< 0.05) indicate more significant differences.

Here are some rules of thumb for choosing a method:

- **Large corpus (100+ documents)** → Use `ZTest` for faster, more powerful results
- **Small corpus (< 30 documents)** → Use `MannWhitney` for more reliable results
- **Unsure about your data** → Use `MannWhitney` to be safe
- **Need fast computation** → Use `ZTest` (it's computationally simpler)

---

### spaCy `Doc` Extensions

When spaCy `Doc` objects are used as input, the `ZTest` and `MannWhitney` classes automatically register a custom `._.topwords` attribute on each `Doc` (both target and comparison documents). The topwords are assign to this attribute and can be accessed directly form the `Doc`. For example:

```python
ztest = ZTest(
    target_docs=target_docs,
    comparison_docs=background_docs
)

# Access results via the extension
print(target_docs[0]._.topwords)
# [('distinctive', 2.5), ('unique', 2.1), ...]
```

---

### Comparison

The `Compare` class provides powerful methods for analyzing and comparing documents using statistical measures. This class wraps around `ZTest` or `MannWhitney` to enable three comparison strategies:

1. **`document_to_corpus()`** - Compare each document to all other documents. Use this method when you want to find what terms make each document unique.
2. **`documents_to_classes()`** - Compare each document to documents in other classes. Use this method when you want to find outliers or representative terms within classes.
3. **`classes_to_classes()`** - Compare entire classes to each other. Use this method when you want to find the signature vocabulary of particular categories.

All methods support three output formats: `dict`, `dataframe`, and `list_of_dicts`.

As a basic example, we will take four short texts. Although you can perform experiments with raw strings, they will generally be converted to spaCy `Doc` objects internally. So, for efficiency, we will preprocess the texts into spaCy `Docs`. We'll then create an instance of the `ZTest` class for our example. We provide it with no docs because these will be passed to it when we choose what we want to compare. The `ZTest` instance is our calculator. We can swap it out for other classes in the `topwords` module or with our own custom classes. Finally, we create an instance of the `Compare` class and pass it our calculator.

```python
from lexos.tokenizer import Tokenizer
from lexos.topwords.ztest import ZTest
from lexos.topwords.compare import Compare

# Load spaCy model
tokenizer = Tokenizer(model="en_core_web_sm")

# Prepare sample documents
docs = tokenizer.make_docs([
    "Dracula was a vampire who lived in Transylvania. He had sharp fangs and drank blood.",
    "Frankenstein created a monster in his laboratory. The creature was terrifying and misunderstood.",
    "Alice fell down the rabbit hole into Wonderland. She met the Cheshire Cat and Mad Hatter.",
    "Peter Pan could fly and never wanted to grow up. He lived in Neverland with the Lost Boys."
])

# Create a calculator instance (ZTest in this example)
calculator = ZTest(target_docs=[], comparison_docs=[])

# Create Compare instance
compare = Compare(calculator=calculator)
```

!!! Note
    By creating the calculator class first, we are also able to configure it with any parameters relevant to the class. For instance, we might want to set the `case_sensitive` parameter with `calculator = ZTest(target_docs=[], comparison_docs=[], case_sensitive=False)`.

We'll now go through the three methods of comparison.

#### `document_to_corpus()`

The `document_to_corpus()` method is used to find what terms make each document unique compared to all other documents in your corpus or collection.

```python
# Compare each document to the rest of the corpus
results = compare.document_to_corpus(docs)

# Results structure (dict format):
# {
#     'Doc 1': [('vampire', 3.845), ('Transylvania', 3.845), ...],
#     'Doc 2': [('monster', 3.621), ('laboratory', 3.621), ...],
#     'Doc 3': [('Alice', 3.512), ('Wonderland', 3.512), ...],
#     'Doc 4': [('fly', 3.412), ('Neverland', 3.412), ...]
# }

print(results['Doc 1'])
# [('vampire', 3.845), ('Transylvania', 3.845), ('blood', 3.845), ...]
```

By default, your docs will be named "Doc 1", "Doc 2", "Doc 3", etc. However, you can also provide your own custom labels in a separate `doc_labels` list (with the labels in the same order as the docs).

```python
# Provide custom document labels
doc_labels = ["Dracula", "Frankenstein", "Alice", "Peter Pan"]

results = compare.document_to_corpus(
    corpus=docs,
    doc_labels=doc_labels
)

print(results['Dracula'])
# [('vampire', 3.845), ('Transylvania', 3.845), ...]
```

Internally, the method creates a dict like `{"doc_label": label, "doc": doc}` which can be accessed with `compare.data`. Additionally, the results and keywords can be accessed with `compare.keywords` and `compare.keywords`.

The default output format is a dict with your document labels as keys and the topwords as values. The format of the topwords will vary depending on which calculator you use, but it will generally be a list of tuples. You can modify the output to create a pandas DataFrame:

```python
df = compare.document_to_corpus(
    corpus=docs,
    doc_labels=doc_labels,
    output_format="dataframe"
)

print(df)
#              vampire  Transylvania  blood  monster  laboratory  Alice  Wonderland  fly  Neverland
# doc_label
# Dracula        3.845         3.845  3.845      0.0         0.0    0.0         0.0  0.0        0.0
# Frankenstein   0.000         0.000  0.000      3.621       3.621  0.0         0.0  0.0        0.0
# Alice          0.000         0.000  0.000      0.0         0.0    3.512       3.512  0.0        0.0
# Peter Pan      0.000         0.000  0.000      0.0         0.0    0.0         0.0  3.412      3.412
```

Additionally, you can produce a list of dicts:

```python
list_results = compare.document_to_corpus(
    corpus=docs,
    doc_labels=doc_labels,
    output_format="list_of_dicts"
)

print(list_results[:3])
# [
#     {'doc_label': 'Dracula', 'term': 'vampire', 'score': 3.845},
#     {'doc_label': 'Dracula', 'term': 'Transylvania', 'score': 3.845},
#     {'doc_label': 'Dracula', 'term': 'blood', 'score': 3.845}
# ]
```

Once you have generated the results, calling a Class method like `document_to_corpus` with a different output format will use the cached data instead of re-calculating all the scores.

#### `document_to_classes()`

The `documents_to_classes()` method is used to find what makes each document distinctive compared to documents in other categories. In addition to `docs` and `doc_labels`, it accepts a `class_labels` list that supplies categories for each document (class labels indices must correspond to document indices). The result is a dict as shown below.

```python
# Define class labels for each document
doc_labels = ["Dracula", "Frankenstein", "Alice", "Peter Pan"]
class_labels = ["gothic", "gothic", "whimsy", "whimsy"]

results = compare.documents_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels
)

# Results structure (dict format):
# {
#     'Dracula': {
#         'comparison_class': 'whimsy',
#         'topwords': [('vampire', 4.123), ('Transylvania', 4.123), ...]
#     },
#     'Alice': {
#         'comparison_class': 'gothic',
#         'topwords': [('Wonderland', 3.956), ('rabbit', 3.956), ...]
#     }
# }

print(results['Dracula'])
# {
#     'comparison_class': 'whimsy',
#     'topwords': [('vampire', 4.123), ('Transylvania', 4.123), ('blood', 4.012)]
# }
```

Class labels are also added to the `data` attribute.

When you call `documents_to_classes()`, each document is compared to **all documents in other classes**:

- "Dracula" (gothic) is compared to ["Alice", "Peter Pan"] (whimsy)
- "Frankenstein" (gothic) is compared to ["Alice", "Peter Pan"] (whimsy)
- "Alice" (whimsy) is compared to ["Dracula", "Frankenstein"] (gothic)
- "Peter Pan" (whimsy) is compared to ["Dracula", "Frankenstein"] (gothic)

You can output the results in DataFrame format:

```python
df = compare.documents_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels,
    output_format="dataframe"
)

print(df)
#     doc_label comparison_class         term  score
# 0     Dracula           whimsy      vampire  4.123
# 1     Dracula           whimsy Transylvania  4.123
# 2     Dracula           whimsy        blood  4.012
# 3  Frankenstein        whimsy      monster  4.012
# 4       Alice           gothic   Wonderland  3.956
# 5       Alice           gothic       rabbit  3.956
# 6   Peter Pan           gothic          fly  3.889
```

Likewise, you can generate a list of dicts.

```python
list_results = compare.documents_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels,
    output_format="list_of_dicts"
)

print(list_results[:2])
# [
#     {'doc_label': 'Dracula', 'comparison_class': 'whimsy', 'term': 'vampire', 'score': 4.123},
#     {'doc_label': 'Dracula', 'comparison_class': 'whimsy', 'term': 'Transylvania', 'score': 4.123}
# ]
```

#### `classes_to_classes()`

The `classes_to_classes()` method is used to find what what terms characterize entire categories/genres compared to other categories. It also takes a `class_labels` parameter.

```python
results = compare.classes_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels
)

# Results structure (dict format):
# {
#     'gothic': {
#         'comparison_class': 'whimsy',
#         'topwords': [('vampire', 5.234), ('monster', 5.123), ...]
#     },
#     'whimsy': {
#         'comparison_class': 'gothic',
#         'topwords': [('Wonderland', 5.112), ('fly', 4.998), ...]
#     }
# }

print(results['gothic'])
# {
#     'comparison_class': 'whimsy',
#     'topwords': [('vampire', 5.234), ('monster', 5.123), ('dark', 4.987)]
# }
```

Each class is treated as a unified group:

- "gothic" class: ["Dracula", "Frankenstein"] combined vs. ["Alice", "Peter Pan"] combined
- "whimsy" class: ["Alice", "Peter Pan"] combined vs. ["Dracula", "Frankenstein"] combined

This is different from `documents_to_classes()` which compares individual documents.

As with the other classes, you can output the results as a pandas DataFrame or as a list of dicts.

```python
df = compare.classes_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels,
    output_format="dataframe"
)

print(df)
#   class_label comparison_class        term  score
# 0      gothic           whimsy     vampire  5.234
# 1      gothic           whimsy     monster  5.123
# 2      gothic           whimsy        dark  4.987
# 3      whimsy           gothic  Wonderland  5.112
# 4      whimsy           gothic         fly  4.998
# 5      whimsy           gothic   Neverland  4.887

list_results = compare.classes_to_classes(
    docs=docs,
    doc_labels=doc_labels,
    class_labels=class_labels,
    output_format="list_of_dicts"
)

print(list_results[:2])
# [
#     {'class_label': 'gothic', 'comparison_class': 'whimsy', 'term': 'vampire', 'score': 5.234},
#     {'class_label': 'gothic', 'comparison_class': 'whimsy', 'term': 'monster', 'score': 5.123}
# ]
```

#### Using Dictionary Input

Instead of passing separate `doc_label` and `class_label` lists, you can pass documents as a list of dictionaries if you already have your data in the at format. Just pass the dictionary with the `docs` parameter:

```python
docs = [
    {"doc": docs[0], "doc_label": "Dracula", "class_label": "gothic"},
    {"doc": docs[1], "doc_label": "Frankenstein", "class_label": "gothic"},
    {"doc": docs[2], "doc_label": "Alice", "class_label": "whimsy"},
    {"doc": docs[3], "doc_label": "Peter Pan", "class_label": "whimsy"}
]

# No need to specify doc_labels and class_labels separately
results = compare.documents_to_classes(docs=docs)
```

Note that only the `doc` and `doc_label` keys are used for `document_to_corpus()`. The `class_label` key (if present) will be ignored.

If `doc_label` values are not available in the dict, `documents_to_classes()` and `classes_to_classes()` will supply "Doc 1", "Doc 2", "Doc 3", etc.

#### Using spaCy `Doc` Extensions

If given a list of spaCy `Doc` objects, the `Compare` class will attempt to extract class values from custom extensions before trying other methods. For instance, if you supply the class label "author", `Compare` will first try to assign values for each `Doc` from its `._.author` extension. If that fails, the value "author" will be assigned as the class label for the doc.

!!! Note
    The class does not support nested dictionaries like `{"metadata": "author": "Shakespeare", "language": "en"}` If you have metadata in this form, you can convert it to an class instance. Here is a simple way to do this:

    ```python
    from dataclasses import dataclass

    @dataclass
    class Metadata:
        author: str
        language: str

    doc._.metadata = Metadata("Shakespeare", "en")
    ```

    You can now use dot notation for the nested attributes:

    ```python
    results = comparison.documents_to_classes(
          docs=docs,
          class_labels=["_.metadata.author"]
      )
    ```

#### Helper Methods

The `Compare` class provides two helper methods for getting information about your data once it has been populated.

##### `get_class()`

This method takes a document label and returns the name of the class to which the document it belongs (if available).

```python
# First, run a comparison to populate the data
compare.documents_to_classes(docs, doc_labels, class_labels)

# Get the class for a specific document
doc_class = compare.get_class("Dracula")
print(doc_class)  # 'gothic'
```

##### `get_docs_by_class()`

This method returns a dict containing all documents grouped by class (the dictionary key). If you supply a `class_label` the output will be restricted to only documents with that label.

```python
# Get all documents grouped by class
docs_by_class = compare.get_docs_by_class()
print(docs_by_class.keys())  # dict_keys(['gothic', 'whimsy'])

# Get documents for a specific class
gothic_docs = compare.get_docs_by_class(class_label="gothic")
print(gothic_docs)  # {'gothic': ['Dracula']}
```
