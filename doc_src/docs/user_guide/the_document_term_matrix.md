# The Document-Term Matrix

## Overview

## About the Document-Term Matrix

A document-term matrix (DTM) is the standard interface for analysis and information of document data. It consists in its raw form of a list of token counts per document in the corpus. Each unique token form is called a term. Thus it is really a list of term counts per document, arranged as matrix.

Producing a DTM is easy with Lexos. All you need a is list of document tokens and a list of labels for each document (the labels are human-readable names which would otherwise be referenced by numeric indices). In the example below, we will use spaCy docs as the input since we can iterate through their tokens just like a list.

```python
from lexos.dtm import DTM
from lexos.tokenizer import Tokenizer

# Define some texts and their labels
texts = [
    "Our first text.",
    "Our second text.",
    "Out third text."
]
labels = ["Doc1", "Doc2", "Doc3"]

# Tokenize the texts
tokenizer = Tokenizer()
docs = list(tokenizer.make_docs(texts=texts))

# Create a Document-Term Matrix (DTM)
dtm = DTM()
dtm(docs=docs, labels=labels)
```

If we did not want to use spaCy docs, we would need to have a list containing lists of tokens like this:

```python
docs = [
    ["Our", "first", "text"],
    ["Our", "second", "text"],
    ["Our", "third", "text"]
]
```

!!! note "Developer's Note"
    Lexos uses Textacy's <a href="https://textacy.readthedocs.io/en/latest/api_reference/representations.html#textacy.representations.vectorizers.Vectorizer" target="_blank">Vectorizer</a> as the default vectorizer. It is possible to use Textacy directly to produce a DTM. For instance, the following method will produce a a DTM containing the raw term counts for each document.

    ```python
    from textacy.representations.vectorizers import Vectorizer
    vectorizer = Vectorizer(tf_type="linear", idf_type=None, norm=None)
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(token.text for token in doc)
    vectorizer.fit_transform(tokenized_docs)
    ```

    Using the Lexos `DTM` class allows you to swap in your own custom vectorizer and gives access to additional helper methods such as `to_df()` to output the DTM as a pandas DataFrame.

## Understanding the Vectorizer

When you create an instance of the `DTM` class, you automatically assign it a vectorizer. By default, this is Textacy's `Vectorizer` class. Here's how it works:

- The `Vectorizer` scans all documents to build a vocabulary of unique **terms** (token forms).
- It then counts the occurrences of each term in each document, resulting in a sparse matrix where rows represent documents and columns represent terms.

!!! note "About Sparse Matrixes"
    Since each document only contains a small subset of all terms in the corpus, a document-term matrix can be very large and mostly filled with zeros. A sparse matrix is useful for storage, especially with large corpora, because it only stores nonzero values. Lexos uses data structures from the `scipy.sparse` library to store the DTM as a sparse matrix to make computations faster, which allows you to work with large corpora without running into memory issues. You can learn more about the <code><a href="https://docs.scipy.org/doc/scipy/reference/sparse.html" target="_blank">scipy.sparse</a></code> library.

    If you need a dense (regular) matrix for certain operations or compatibility with other libraries, you can convert the sparse DataFrame to a dense one by calling:

    ```python
    dense_df = dtm.to_df().sparse.to_dense()
    ```

    Be aware that this may use a lot of memory for large corpora.

The default vectorizer can be configured to perform additional culling or normalization functions. **Culling** refers to reducing the size of the matrix to include only part of the data. **Normalization** refers to performing additional calculations on the raw term counts in order to better represent the sigificance of those counts within the broader corpus (e.g. to take into account document varying lengths). Each of these categories is discussed below.

### Culling the DTM

In many cases, you will want to cull terms from your DTM in order to reduce the size of the data or to remove terms which you think might not be meaningful for your research. A common form of culling is to restrict the data to the *n* most-frequently occurring terms. You can do this with the `max_n_terms` parameter. You can also restrict your data to terms occurring in a minimum number of documents with `min_df` or a maximum number of documents with `max_df`. Here is an example using all three:

```python
dtm = DTM(max_n_terms=100, min_df=2, max_df=5)
```

Depending on your workflow, you can also configure the vectorizer directly or when you call the `DTM` instance. Here are some examples show the three alternative ways to do it:

```python
# Configure the DTM instance
dtm = DTM(max_n_terms=100)

# Configure the DTM tokenizer directly
dtm.vectorizer.min_df=2

# Set the parameters when calling the DTM instance
dtm(docs=docs, labels=labels, max_df=5)
```

Feel free to use whichever approach you find most comfortable.

!!! note
    A further method of limiting the vocabulary is to provide a list of specific terms to be included in the matrix using the `vocabulary_terms` parameter.

### Normalizing the Values

The vectorizer is configured by default to generate a matrix of raw counts. However, it can often be beneficial to normalize the values in some way such as by calculating the term's frequency in proportion to all terms in the corpus. Or, if your documents vary in length, it can be beneficial to apply a weighting function. The vectorizer allows you to do that with the `tf_type`, `idf_type`, `dl_type`, and `norm` parameters:

- `tf_type` controls how term frequencies are calculated (e.g., raw counts, log-scaled, binary presence/absence). Options: "linear", "sqrt", "log", "binary". Default is "linear".
- `idf_type` controls inverse document frequency type, how document frequency scaling is applied (for TF-IDF weighting). Options: None, "linear", "sqrt", "log". Default is None.
- `dl_type`: Controls normalization based on document length. Options: None, "linear", "sqrt", "log". Default is None.
- `norm`: Applies vector normalization. Options: None, "l1", "l2". Normalizes the resulting vectors (rows) to unit length. L1 normalization scales the term frequencies in each document so that the sum of the absolute values equals 1. This means each document vector is divided by the sum of its term frequencies, turning the values into proportions that sum to 1. L2 normalization scales the term frequencies so that the sum of the squares of the values equals 1 (i.e., the Euclidean norm is 1). This is useful for algorithms that are sensitive to the length of the document vectors, such as cosine similarity. Default is None.

!!! note "What Settings Should I Choose?

    If you only want raw counts, just use the default settings. Changing an of the other settings will implement various types of weighting functions. The <a href="https://github.com/chartbeat-labs/textacy/blob/f08ecbc46020f514b8cbb024778ec4f80456291f/src/textacy/representations/vectorizers.py#L163" target="_blank">Textacy source code</a> provides helpful advice, which is partially reproduced here.

    > In general, weights may consist of a local component (term frequency),
    a global component (inverse document frequency), and a normalization
    component (document length). Individual components may be modified:
    they may have different scaling (e.g. tf vs. sqrt(tf)) or different behaviors
    (e.g. "standard" idf vs bm25's version). There are *many* possible weightings,
    and some may be better for particular use cases than others. When in doubt,
    though, just go with something standard.

    One of the most commonly-used weighting settings is <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank">term frequency-inverse document frequency (TF-IDF)</a>, which measures the importance of a term to a document in a collection of documents, adjusted for the fact that some terms appear more frequently than others. The Lexos web app's implementation of TF-IDF has the following settings:

    - `tf_type="log"`
    - `idf_type="smooth"`
    - `norm="l2"`

## Getting Term Counts and Frequencies

Once you have generated your DTM, you can extract useful information from its properties:

- `DTM.shape`: returns a tuple with the width and height of the matrix.
- `DTM.sorted_terms_list`: Returns a sorted list of terms in the DTM.
- `sorted_term_counts`: Returns a sorted dictionary of terms and their total counts across all documents in the DTM.

!!! note
    By default, terms are sorted according to the rules of language used by your operating system. You can set the DTM to use a different sorting algorithm with the `alg` keyword, which takes a `natsorted.ns.LOCALE` object. For further information, see the <a href="https://natsort.readthedocs.io/en/5.1.0/ns_class.html" target="_blank">natsort</a> documentation.

Perhaps the most useful method of the `DTM` class is `to_df()`, which converts the matrix to a pandas DataFrame for display or for further manipulation. As a DataFrame, the output can be modified using the full range of options available in the pandas API. However, `to_df()` provides parameters that can ease the process:

- `by`: The term or terms to sort by.
- `ascending`: Whether to sort by ascending values.
- `as_percent`: Whether to convert counts to percentages.
- `rounding`: The number of digits after the decimal point to include.
- `transpose`: Whether to pivot the rows and columns in the matrix.
- `sum`: Add a column showing the sum of each row.
- `mean`: Add a column showing the mean of each row.
- `median`: Add a column showing the median of each row.

!!! important
    The `transpose` parameter is applied before the other parameters and may not work in tandem with them. If you need to transpose a DataFrame after applying `sum`, `mean`, or `median`, it is better to use the pandas `transpose()` method:

    ```python
    # Transpose the DataFrame after creation
    dtm.df(sum=True).transpose()

    # Or use the alternative `T` property
    dtm.df(sum=True).T
    ```

## Visualising the DTM

Once a document-term matrix table has been generated as a pandas dataframe, it becomes possible to use any of the Pandas plotting methods to visualise the data. Here is a short example of a bar chart containing the top 20 terms in the DTM:

```python
# Get the first 20 rows of the DTM as a DataFrame sorted by sum
df = dtm.to_df(sum=True, by="Total", ascending=False)[0:20]

# Plot the DataFrame
df.Total.plot(
    kind="bar",
    title="Top 20 Most Frequent Terms",
    xlabel="Terms",
    ylabel="Frequency"
)
```

See the Pandas <code><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html" target="_blank">DataFrame.plot</a></code> documentation for the complete range of keywords.

!!! note

    By default, `pandas.DataFrame.plot` uses the `matplotlib` plotting library. If you are familiar with `matplotlib`, you can use it directly by using the data from the DataFrame. The same goes for any other plotting library. In Lexos, interactive plots are frequently made using the Plotly library, for which Pandas has a backend. To use it, simply call the following code:

    ```python
    import pandas as pd
    pd.options.plotting.backend = "plotly"
    df.Total.plot(
            kind="bar",
            title="Top 20 Most Frequent Terms",
            labels={"index": "Terms", "value": "Frequency"}
    )
    ```

    Note that the keywords described in Pandas documentation apply only to the `matplotlib` backend. For Plotly, you will beed to consult the equivalent Plotly documentation to find the appropriate keywords (`labels`, in the example above).

Lexos word clouds and bubble charts are also ideal for visualising DTMs. Word clouds can be generated for the entire DTM or for individual documents. Multiple word clouds arrange for comparison are referred to as multiclouds. For information on generating these and other visualizations, see the [Visualization page](visualization.md).

## Advanced Usage with `scikit-learn` Vectorizers

The popular machine-learning library `scikit-learn` (`sklearn`) provides its own vectorizer classes, such as <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank">CountVectorizer</a></code> and <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html" target="_blank">TfidfVectorizer</a></code>, which often form components of machine-learning pipelines. However, these tokenizers use simple regex patterns, rather than language models, to tokenize documents. In the example below, we'll show how you can use the scikit-learn's `CountVectorizer` as part of a pipeline for training a scikit-learn logistic regression model whilst still leveraging language-specific knowledge available in a document tokenised with Lexos.

```python
# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. We also need the Lexos `tokenizer` module
from lexos.tokenizer import Tokenizer

# 2. Create a Lexos Tokenizer instance
lexos_tokenizer = Tokenizer(model="en_core_web_sm")

# 3. Define a custom tokenizer function to return lists of tokens
# We'll get token lemmas instead of text just to show that we have
# access to NLP capabilities.
def my_tokenizer(text, tokenizer=lexos_tokenizer):
    return [token.lemma_ for token in tokenizer.make_doc(text)]

# 4. Create a `CountVectorizer` instance
# Notice that we supply `CountVectorizer`, passing it our own tokenizer.
# `CountVectorizer` expects its own pattern for tokenizing, so we have to set
# `token_pattern=None`, or it'll give us a warning.
count_vectorizer = CountVectorizer(tokenizer=my_tokenizer, token_pattern=None)

# Define the steps of the scikit-learn pipeline
pipeline_steps = [
    ('vectorizer', count_vectorizer),
    ('logistic_regression', LogisticRegression())
]

# Create the pipeline
model_pipeline = Pipeline(pipeline_steps)

# Now, model_pipeline can be used like any other scikit-learn estimator
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
```

Only the steps numbered 1-4 are really important for showing how to construct a document-term-matrix (the rest is all specific to running the logistic regression pipeline).

If you need to construct a Lexos `DTM` from documents tokenized with `scikit-learn`, you can follow the example below:

```python
# Set our custom tokenizer in CountVectorizer
# Note that `CountVectorizer` expects `token_pattern=None`
# when using a custom tokenizer.
vectorizer = CountVectorizer(tokenizer=my_tokenizer, token_pattern=None)

# Use CountVectorizer to fit and transform the documents
document_term_matrix = vectorizer.fit_transform(docs)

# Get a list of terms from the vectorizer
terms_list = vectorizer.get_feature_names_out()

# Create a new `DTM` instance
new_dtm = DTM(docs=docs, labels=labels)

# Assign our scikit-learn matrix as a numpy array to the new instance
new_dtm.doc_term_matrix=document_term_matrix.toarray()

# Assign the vocabulary terms to the `DTM`'s vectorizer
# Note that the `_validate_vocabulary()` method returns a tuple,
# so we take the first element.
new_dtm.vectorizer.vocabulary_terms = new_dtm.vectorizer._validate_vocabulary(
    terms_list)[0]
```

This should enable you to access all properties and methods of the `DTM` class.
