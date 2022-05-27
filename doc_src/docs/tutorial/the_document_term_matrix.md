## About the Document-Term Matrix

A document-term matrix (DTM) is the standard interface for analysis and information of document data. It consists in its raw form of a list of token counts per document in the corpus. Each unique token form is called a term. Thus it is really a list of term counts per document, arranged as matrix.

In the Lexos App, sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank"><code>CountVectorizer</code></a> is used to produce the DTM. In the Lexos API, Textacy's <code><a href="https://textacy.readthedocs.io/en/latest/api_reference/representations.html#textacy.representations.vectorizers.Vectorizer" target="_blank">Vectorizer</a></code> is used. Here is a vanilla implication to get a DTM containing the raw term counts for each document.

```python
from textacy.representations.vectorizers import Vectorizer

vectorizer = Vectorizer(
    tf_type="linear",
    idf_type=None,
    norm=None
)

tokenised_docs = (LexosDoc(doc).get_tokens() for doc in docs)
doc_term_matrix = vectorizer.fit_transform(tokenised_docs)
```

The main advantage of this procedures is that sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank"><code>CountVectorizer</code></a> employs a regular expression pattern to tokenise the text and has very limited functionality to implement the kind of language-specific knowledge available in a document tokenised with a language model.

## Getting Term Counts and Frequencies

The Lexos API provides an easy method of using the `Vectorizer` to retrieve term counts or frequencies from a single document and returning the results in a pandas dataframe.

```python
from lexos.dtm import get_doc_term_counts
df = get_doc_term_counts(docs, as_df=True)
```

Setting `normalize=True` will return relative frequencies instead of raw counts. [lexos.dtm.get_doc_term_counts][] has various parameters for limiting and filtering the output based on token labels or regex patterns.

## The `DTM` Class

Lexos also wraps Textacy's `Vectorizer` in the [lexos.dtm.DTM][] class with greater functionality. You can import it with

```python
from lexos.dtm import DTM
```

Most work will leverage the [lexos.dtm.DTM][] class to builds a document-term matrix and provide methods for manipulating the information held therein. The standard method of creating a DTM object is as follows:

```python
labels = ["Pride_and_Prejudice", "Sense_and_Sensibility"]
dtm = DTM(docs, labels)
```

The labels are human-readable names for the documents which would otherwise be referenced by numeric indices.

Instantiating a [lexos.dtm.DTM][] object creates a vectorizer. By default, this is a Textacy `Vectorizer` object with parameters set to produce raw counts. The vectorizer settings can be viewed by calling `lexos.dtm.vectorizer_settings` and they can be adjusted by calling [lexos.dtm.DTM.set_vectorizer][]. The vectorizer is an object, so you can also inspect individual vectorizer settings with calls like `lexos.dtm.vectorizer.idf_type`.

!!! important
    After changing the settings of an object, you must call [lexos.dtm.DTM.build][] to rebuild the document-term matrix.

## Getting a Term Counts Table

The [lexos.dtm.DTM][] class method for getting a table of raw term counts is [lexos.dtm.DTM.get_table][]. You can also call `lexos.dtm.table`, which will return a table based on state after the last time [lexos.dtm.DTM.build][] was called. The options are as follows:

```python
# Get a table of counts with documents as columns and terms as rows
df = dtm.get_table()

# Get a table of counts with terms as columns and documents as rows
df = dtm.get_table(transpose=True)
```

The second option is equivalent to calling `dtm.get_table().T`, using pandas notation. The [lexos.dtm.DTM.get_table][] output is generally intended to allow you to use the pandas API once you have the data in the form of a pandas dataframe.

If you change vectorizer settings, remember to rebuild the DTM. For instance, you want to use the Lexos app's implementation of TF-IDF, you would use the following (I think):

```python
dtm.set_vectorizer(tf_type="log", idf_type="smooth", norm="l2")
dtm.build()
df = dtm.get_table()
```

!!! important
    Currently, [lexos.dtm.DTM.build][] resets `dtm.table=None`, so you will need to call [lexos.dtm.DTM.get_table][] to use the new vectorizer. This is intended to reduce overhead if an app only needs to interact directly with the vectorizer. Perhaps down the line, it might be advisable to give [lexos.dtm.DTM.build][] a boolean parameter to allow the user to decide whether the table gets regenerated automatically.

!!! note
    The Lexos culling function is now handled by the `min_df` parameter and extended by the `max_df` parameter in the vectorizer. The Lexos most frequent words function is handled by `max_n_terms`. But see the section below.

## Getting Most Frequent Terms

In addition to `max_n_terms`, it is possible to use pandas to sort and slice the `DTM.table` after it has been built in order to get the most frequent terms. The [lexos.dtm.DTM.most_frequent][] method is a helper function for this purpose:

```python
most_frequent = dtm.most_frequent(max_n_terms=25)
```

It is possible to take slices (start counting from a particular index) in the `DTM.table`:

```python
most_frequent = dtm.most_frequent(max_n_terms=25, start=25)
```

This will return terms 25 to 50.

There is also an equivalent function [lexos.dtm.DTM.least_frequent][] to get the least frequent terms in the table.

!!! important
    [lexos.dtm.DTM.most_frequent][] and [lexos.dtm.DTM.least_frequent][] should not be used
    if `min_df` or `max_df` are set in the vectorizer, as this will cause the document-term matrix to be reduced twice.

## Getting Statistics from the DTM

Pandas has methods for calculating the sum, mean, and median of rows in the table. However, to save users from Googling, the DTM class has the [lexos.dtm.DTM.get_stats_table][] method that calculates these statistics and adds them to the columns in the default DTM table.

```python
stats_table = dtm.get_stats_table(["sum", "mean", "median"])
```

Once the new dataframe is generated, it is easy to extract the data to a list with standard pandas syntax like `stats_table["sum"].values.tolist()`.

## Getting Relative Frequencies

[lexos.dtm.DTM.get_freq_table][] converts the raw counts in the default DTM table to relative frequencies. Since the resulting values are typically floats, there is an option to set the number of digits used for rounding.

```python
frequency_table = dtm.get_freq_table(rounding=2, as_percent=True)
```

The setting `as_percent=True` multiples the frequencies by 100. The default is `False`.

## Getting Lists of Terms and Term Counts

By default, most of the [lexos.dtm.DTM][] methods return a pandas dataframe. Two methods provide output in the form of lists. [lexos.dtm.DTM.get_terms][] provides a simple, alphabetised list of terms in the document-term matrix. [lexos.dtm.DTM.get_term_counts][] returns a list of tuples with terms as the first element and sums (the total number of occurrences of the term in all documents) as the second element. This method has parameters for sorting by column and direction. By default, terms are sorted by `natsort.ns.LOCALE` (i.e. the computer's locale is used for the sorting algorithm). This can be configured using the options in the <a href="https://natsort.readthedocs.io/en/master/api.html#natsort.ns" target="_blank">Python natsort reference</a>.

## Visualising the DTM

Once a document-term matrix table has been generated as a pandas dataframe, it becomes possible to use any of the <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html" target="_blank"><code>pandas.DataFrame.plot</code></a> methods, or to export the data for use with other tools. However, the Lexos API has two built-in visualisations: word clouds and bubble charts. Word clouds can be generated for the entire DTM or for individual documents. Multiple word clouds arrange for comparison are referred to as multiclouds.

!!! note
    A primitive function exists for generating word clouds in Plotly as proof of concept. However, this function is not documented because Plotly (and other interactive graphing libraries) are really not good for word clouds. It is better to output the DTM as json and parse it in d3.s for a pretty, interactive visualisation.

### Single Word Clouds

It is easiest to make a word cloud from [lexos.dtm.DTM.get_stats_table][]. Start by getting a table with the sums (total term count per document).

```python
data = dtm.get_stats_table("sum")
```

Next rename the columns to the header labels expected by [lexos.dtm.wordcloud.make_wordcloud][].

```python
data = data.rename({"terms": "term", "sum": "count"}, axis=1)
```

Wordclouds are generated by the Python [Wordcloud](https://amueller.github.io/word_cloud/) library. It has various option which can be defined in a dictionary and passed to [lexos.dtm.wordcloud.make_wordcloud][]. Figures are generated using Python's `matplotlib`, and its options can also be passed to [lexos.dtm.wordcloud.make_wordcloud][].

```python
wordcloud_opts = {
    "max_words": 2000,
    "background_color": "white",
    "contour_width": 0,
    "contour_color": "steelblue"
}
matplotlib_opts = {"figsize": (15, 8)}
wordcloud = make_wordcloud(
    data, wordcloud_opts,
    figure_opts=matplotlib_opts
    show=True,
    round=150
)
```

The `round` parameter (normally between 100 and 300) will add various degrees of rounded corners to the word cloud.

By default, `show=True`, and the wordcloud will be plotted to the screen if the environment is appropriate. If `show=False`, the `WordCloud` object will be returned, and it can be saved or further manipulated. For instance, the word cloud above could be saved by calling `wordcloud.to_file(filename)`.

There are various ways to plot a word cloud of a single document. For instance, you could make a copy of the DTM `table` with only the counts for the desired document.

```python
doc1 = dtm.table[["terms", "doc1"]].copy()
data = doc1.rename({"terms": "term", "doc1": "count"}, axis=1)
```

You can now submit the data to [lexos.dtm.wordcloud.make_wordcloud][] as above.

For a subset of documents, you might do the following:

```python
# Copy the table with only the desired document columns
some_docs = dtm.table[["terms", "doc1", "doc2"]].copy()

# Create a new column with the total for each row
some_docs["count"] = some_docs.sum(axis=1)

# Make a copy with only the terms and counts
data = some_docs.table[["terms", "count"]].copy()
```

!!! note
    [lexos.dtm.wordcloud.make_wordcloud][] takes a number of other input formats, including raw text, but a `lexos.dtm.DTM.table` is by far the easiest method to generate data from pre-tokenised texts.

### Multiclouds

Multiclouds are grid-organised word clouds of individual documents, which allow you to compare the document clouds side by side. The method of generating multiclouds is similar to word clouds. The basic input is a `lexos.dtm.DTM.table`, where one word cloud will be generated for each document. If a subset of documents is required, a similar method to that shows above can be used to limit the documents in the table. Once the data is prepared, the multiclouds are generated with a [lexos.dtm.wordcloud.make_multiclouds][].

```python
multiclouds = make_multiclouds(
    data,
    wordcloud_opts,
    matplotlib_opts,
    show=True,
    title="My Title",
    labels=labels,
    ncols=3
    round=150,
)
```

Since multicloud produce multiple subplots, there is a `title` parameter to give the entire figure a title and a `labels` parameter, which includes a list labels to be assigned to each subplot. The `ncols` parameter sets the number of subplots per row.

If `show=False`, [lexos.dtm.wordcloud.make_multiclouds][] returns a list of word clouds. These can be saved individually by calling `to_file()` on them, as shown above. There is not currently a method of saving the entire plot.

As with word clouds, [lexos.dtm.wordcloud.make_multiclouds][] takes a number of different input formats, but pandas dataframes are the easiest to work with.

### Bubble Charts

Bubble charts ("bubbleviz" in the Lexos app) are produce by the following imports:

```python
from lexos.dtm.bubbleviz import BubbleChart, make_bubble_chart
```

To generate the visualisation:

```python
# Get a table with the sums of each row
sums = dtm.get_stats_table("sum")
terms = sums["term"].values.tolist()
area = sums["count"].values.tolist()
colors = ["#5A69AF", "#579E65", "#F9C784", "#FC944A", "#F24C00", "#00B825"]
make_bubble_chart(terms,
                  area,
                  limit=100,
                  title="My Title",
                  bubble_spacing=0.1,
                  colors=colors,
                  figsize=(15, 15),
                  show=True,
                  filename
)
```

[lexos.dtm.bubbleviz.make_bubble_chart][] has a slightly simpler interface than the other visualisations. It takes a list of terms and a list of counts or frequencies (easily obtainable from a dataframe as shown), as well as other parameters that affect the appearance of the image. If a `filename` is provided, the image will be saved automatically.
