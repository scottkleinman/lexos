## About the Document-Term Matrix

A document-term matrix (DTM) is the standard interface for analysis and information of document data. It consists in its raw form of a list of token counts per document in the corpus. Each unique token form is called a term. Thus it is really a list of term counts per document, arranged as matrix.

In the Lexos App, sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank"><code>CountVectorizer</code></a> is used to produce the DTM. In the Lexos API, Textacy's <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/representations.html#textacy.representations.vectorizers.Vectorizer" target="_blank">Vectorizer</a></code> is the default vectorizer. Here is a vanilla implication to get a DTM containing the raw term counts for each document.

```python
from textacy.representations.vectorizers import Vectorizer

vectorizer = Vectorizer(
    tf_type="linear",
    idf_type=None,
    norm=None
)

tokenized_docs = (LexosDoc(doc).get_tokens() for doc in docs)
doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
```

The main advantage of this procedure is that sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank"><code>CountVectorizer</code></a> employs a regular expression pattern to tokenize the text and has very limited functionality to implement the kind of language-specific knowledge available in a document tokenised with a language model.

## Getting Term Counts and Frequencies

The Lexos API provides an easy method of using the `Vectorizer` to retrieve term counts or frequencies from a single document and returning the results in a pandas dataframe.

```python
from lexos.dtm import get_doc_term_counts
df = get_doc_term_counts(docs, as_df=True)
```

Setting `normalize=True` will return relative frequencies instead of raw counts. `dtm.get_doc_term_counts()` has various parameters for limiting and filtering the output based on token labels or regex patterns.

## The `DTM` Class

Lexos also wraps Textacy's `Vectorizer` in the `DTM` class with greater functionality. You can import it with

```python
from lexos.dtm import DTM
```

Most work will leverage the `DTM` class to builds a document-term matrix and provide methods for manipulating the information held therein. The standard method of creating a DTM object is as follows:

```python
labels = ["Pride_and_Prejudice", "Sense_and_Sensibility"]
dtm = DTM(docs, labels)
```

The labels are human-readable names for the documents which would otherwise be referenced by numeric indices.

Instantiating a `DTM` object creates a vectorizer. By default, this is a Textacy `Vectorizer` object with parameters set to produce raw counts. The vectorizer settings can be viewed by calling `lexos.dtm.vectorizer_settings` and they can be adjusted by calling `set_vectorizer()`. The vectorizer is an object, so you can also inspect individual vectorizer settings with calls like `DTM.vectorizer.idf_type`.

!!! important
    After changing the settings of an object, you must call `DTM.build()` to rebuild the document-term matrix.

## Getting a Term Counts Table

The `DTM` class method for getting a table of raw term counts is `DTM.get_table(). You can also call `DTM.table`, which will return a table based on state after the last time `DTM.build()` was called. The options are as follows:

```python
# Get a table of counts with documents as columns and terms as rows
df = dtm.get_table()

# Get a table of counts with terms as columns and documents as rows
df = dtm.get_table(transpose=True)
```

The second option is equivalent to calling `dtm.get_table().T`, using pandas notation. The `dtm.get_table()` output is generally intended to allow you to use the pandas API once you have the data in the form of a pandas dataframe.

If you change vectorizer settings, remember to rebuild the DTM. For instance, you want to use the Lexos app's implementation of TF-IDF, you would use the following:

```python
dtm.set_vectorizer(tf_type="log", idf_type="smooth", norm="l2")
dtm.build()
df = dtm.get_table()
```

!!! important
    Currently, `DTM.build()` resets `dtm.table=None`, so you will need to call `DTM.get_table()` to use the new vectorizer. This is intended to reduce overhead if an app only needs to interact directly with the vectorizer. Perhaps down the line, it might be advisable to give `DTM.build()` a boolean parameter to allow the user to decide whether the table gets regenerated automatically.

!!! note
    The Lexos culling function is now handled by the `min_df` parameter and extended by the `max_df` parameter in the vectorizer. The Lexos most frequent words function is handled by `max_n_terms`. But see the section below.

## Getting Statistics from the DTM

Pandas has methods for calculating the sum, mean, and median of rows in the table. However, to save users from Googling, the DTM class has the `DTM.get_stats_table()` method that calculates these statistics and adds them to the columns in the default DTM table.

```python
stats_table = dtm.get_stats_table(["sum", "mean", "median"])
```

Once the new dataframe is generated, it is easy to extract the data to a list with standard pandas syntax like `stats_table["sum"].values.tolist()`.

The table can also be sorted using pandas to get the most or least frequent terms:

```python
most_frequent = stats_table.sort_values(by="sum", ascending=False).head()
least_frequent = stats_table.sort_values(by="sum", ascending=False).head()
```

## Getting Relative Frequencies

`DTM.get_freq_table()` converts the raw counts in the default DTM table to relative frequencies. Since the resulting values are typically floats, there is an option to set the number of digits used for rounding.

```python
frequency_table = dtm.get_freq_table(rounding=2, as_percent=True)
```

The setting `as_percent=True` multiples the frequencies by 100. The default is `False`.

## Getting Lists of Terms and Term Counts

By default, most of the `DTM` methods return a pandas dataframe. Two methods provide output in the form of lists. `DTM.get_terms()` provides a simple, alphabetised list of terms in the document-term matrix. `DTM.get_counts()` returns a list of tuples with terms as the first element and sums (the total number of occurrences of the term in all documents) as the second element. This method has parameters for sorting by column and direction. By default, terms are sorted by `natsort.ns.LOCALE` (i.e. the computer's locale is used for the sorting algorithm). This can be configured using the options in the <a href="https://natsort.readthedocs.io/en/master/api.html#natsort.ns" target="_blank">Python natsort reference</a>.

## Visualising the DTM

Once a document-term matrix table has been generated as a pandas dataframe, it becomes possible to use any of the <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html" target="_blank"><code>pandas.DataFrame.plot</code></a> methods, or to export the data for use with other tools. However, the Lexos API has two built-in visualisations: word clouds and bubble charts. Word clouds can be generated for the entire DTM or for individual documents. Multiple word clouds arrange for comparison are referred to as multiclouds. For information on generating these and other visualizations, see the [Visualization page](visualization.md).
