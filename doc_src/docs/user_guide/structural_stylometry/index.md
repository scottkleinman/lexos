# Analyzing Structural Patterns

Texts are composed not just of tokens but of a variety of structural elements, including punctuation, spacing, and formatting. These elements can be used to identify stylistic patterns in a corpus. For example, some authors may use more exclamation points than others, or they may have a preference for two spaces at the end of a sentence. By analyzing these patterns as linguistic features, we can gain insights into the writing style of different authors.

Whilst markup languages like XML can be used to explicitly encode structural information, most corpora are not marked up in this way. Plain written texts typically use punctuation and spacing to convey structure. The Lexos `StructuralAnalyzer` class is designed to provide a convenient workflow for extracting and analyzing the significance of these structural features from a corpus.

To get started, import the `StructuralAnalyzer` class from the `lexos.structural_stylometry` module. You can also import the `Loader` and `Corpus` classes if you want to load your data from files or a Lexos corpus.

```python
# Import the StructuralAnalyzer class
from lexos.structural_stylometry import StructuralAnalyzer

# Optionally, import the Loader and Corpus classes if you want to load your data from files or a Lexos corpus
from lexos.io.loader import Loader
from lexos.corpus import Corpus
```

## Get the Corpus

We'll illustrate the various ways of getting a corpus into the `StructuralAnalyzer` class using a small dictionary of labels and raw text strings. You can pass this dictionary directly to the `StructuralAnalyzer` class. In the examples below, we will use a small sample corpus in this format. however, you can also load files with the Lexos `loader` module and pass the Lexos `Loader` object. If you have a `Corpus`, you can also pass it directly to the `StructuralAnalyzer` class. Here are some code samples:

```python
sample_corpus = {
    "AuthorA_Doc1": "Thus, it came to pass; the empire fell,  not by sword, but by time. \n",
    "AuthorA_Doc2": "Therefore, the king wept;  no crown could save him, nor gold. \n",
    "AuthorB_Doc1": "The empire fell! \n\n Time conquered all! Swords were useless!",
    "AuthorB_Doc2": "Victory was lost! \n\n All hope vanished! Shields shattered!",
}
```

!!! note

    If the documents in your corpus are raw text strings, they will be converted internally to spaCy `Doc` objects using the default "xx_sent_ud_sm" model unless you specify the model when you instantiate the `StructuralAnalyzer` object.

    Notice that the sample documents above contain a variety of whitespace combinations. These can optionally be recorded as tokens as a pre-processing step before the texts are converted to `Doc` objects. This can be useful if you are submitting texts with irregularities or unusual formatting such as might be produced by OCR.

If you wish to load files with a Lexos `Loader` object, you can do so by instantiating the `Loader` class and calling the `load()` method with a path to a directory of files.

```python
from lexos.io.loader import Loader

loader = Loader()
loader.load("path/to/directory_of_files")
```

If you wish to use a Lexos `Corpus`, it might look like this:

```python
from lexos.corpus import Corpus

corpus_directory = "path/to/directory_of_files"
lexos_corpus = Corpus(
    name="My Research Corpus",
    corpus_dir=corpus_directory
)
```

Whichever method you use, you are now ready to create a `StructuralAnalyzer` object with your corpus data.

```python
# Using a dict
analyzer = StructuralAnalyzer(corpus=sample_corpus, ...)

# Using a Lexos Loader object
analyzer = StructuralAnalyzer(corpus=loader, ...)

# Using a Lexos Corpus
analyzer = StructuralAnalyzer(corpus=lexos_corpus, ...)
```

We'll discuss the creation of the `StructuralAnalyzer` object in more detail below, but for now, you can see that the `corpus` argument can accept a variety of data types.

## Create the Analyzer Object

The `StructuralAnalyzer` object is instantiated with your corpus data and a variety of settings:

- `min_punctuation_threshold`: The minumum number of punctuation marks required per document. The default is 20.
- `action_on_low_count`: The action taken if a document falls below the `min_punctuation_threshold`. The default "warn" sends a warning to the console. Set it to "drop" if you want the document to be dropped from the analysis.
- `max_features`: The maximum number of features to track in the vocabulary. The default is 100.
- `include_whitespace`: Whether to include whitespace markers as features. The default is `True`.
- `feature_mode`: The default "all" tracks all tokens in the corpus. Set to "structural_only" to include only punctuation and spacing or to "punctuation_only" to include only punctuation.

Here is a full example:

```python
analyzer = StructuralAnalyzer(
    corpus=lexos_corpus,
    model="xx_sent_ud_sm",
    min_punctuation_threshold=1,
    action_on_low_count="warn",
    max_features=15,
    include_whitespace=True,
    feature_mode="structural_only" # This mode includes punctuation and spacing, but excludes words
)
```

### View the Vocabulary

If your input data consists of raw text strings and you set `include_whitespace` to `True`, your documents are pre-processed to convert double spaces, multiple newlines, and trailing whitespace to `[WS_DOUBLE_SPACE]`, `[WS_MULTIPLE_NEWLINE]`, and `[WS_TRAILING_SPACE]`. This allows them to be counted as tokens when the documents are converted to spaCy `Doc` objects.

In many cases, you will not want these features to be counted because they may be indicators of OCR errors or other formatting irregularities, rather than stylistically significant features. It is thus a good idea to inspect the vocabulary in the the `StructuralAnalyzer` object before proceeding.

```python
print(analyzer.vocabulary)
```

## Build the Matrix

You are now ready to build a frequency matrix to represent our corpus data. By default, the `get_feature_matrix` method uses Term Frequency-Inverse Document Frequency (TF-IDF), which we specify explicitly in the code below.

```python
matrix = analyzer.get_feature_matrix(method="tfidf")
print(matrix)
```

We can also supply `method="raw"` to get raw counts or `method="burrows-z"` to get a matrix of z-scores.

The output is a `numpy.ndarray` of shape (`number_of_docs`, `number_of_features`). The rows represent the documents in the corpus, and the columns represent the features in the vocabulary. The values in the matrix are the frequencies of each feature in each document. However, you can also view the matrix as a pandas DataFrame, which is often more convenient for analysis and visualization.

```python
df = analyzer.to_df(method="tfidf")
print(df)
```

### Choosing a Representation Method

Using raw counts can be statistically misleading if your documents vary in length. For this reason, the default method is "tfidf". However, because punctuation occurs at vastly higher frequencies than individual words, standard TF-IDF can occasionally over-penalize punctuation marks across a small corpus. To fix this, we use raw instead of log-scaled term frequencies (TF), combined with standard Inverse Document Frequency (IDF).[^1](https://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/)

For an alternative approach, we can use the Burrows' Z method. This method normalizes the raw relative frequencies into standard deviations away from the corpus mean (Z-scores). The output is a matrix of shape (`number_of_docs`, `number_of_features`).

## Get Delta Distances

A common way to use such frequency data is to transform the matrix into a distance matrix to analyse vector similarities between documents.

There are many metrics to calculate distances between features of a frequency matrix. We'll start with the Classic Burrows' Delta, which calculates the Manhattan distance (mean absolute difference) between the Z-scores of two documents. The output is a pairwise distance matrix of shape (`number_of_docs`, `number_of_docs`).

You can view the distance between two documents by indexing into the matrix. For example, `delta_distances[0, 1]` gives the distance between the first and second documents in the corpus.

```python
delta_distances = analyzer.get_distance_matrix(method="classic")

print("\n--- Burrows' Delta Distance Matrix ---")
print(f"Distance between Text 1 & Text 2 (Same Author): {delta_distances[0, 1]:.4f}")
print(f"Distance between Text 1 & Text 3 (Diff Author): {delta_distances[0, 2]:.4f}")
```

A smaller Delta value implies closer stylistic alignment. In the example output, `delta_distances[0, 1]` yields a much lower numerical distance than `delta_distances[0, 2]`. The engine relies heavily on punctuation layout choices to recognize that Author A wrote the first two documents, separating them from the exclamation-heavy habits of Author B.

You can also implement the Argamon quadratic variant, which handles the high standard deviations introduced by mixed token types much better than the original Z-score cityblock (Manhattan) distance.[^2](https://rpubs.com/Shevek/deltas). To generate a distance matrix, use `method="quadratic"`

```python
quad_delta_distances = analyzer.get_distance_matrix(method="quadratic")

print("\n--- Argamon’s Quadratic Delta Distance Matrix ---")
print(f"Distance between Text 1 & Text 2 (Same Author): {quad_delta_distances[0, 1]:.4f}")
print(f"Distance between Text 1 & Text 3 (Diff Author): {quad_delta_distances[0, 2]:.4f}")
```

The output of `get_distance_matrix` can also be viewed as a pandas DataFrame with `as_df=True`:

```python
quad_delta_distances_df = analyzer.get_distance_matrix(method="quadratic", as_df=True)
```

## Plot and Analyze

Distance matrices can be difficult for humans to read, so it is common to plot visualizations which capture their qualities. This can allow us to view exactly which punctuation marks, words, or whitespace behaviors are discriminating features in our documents.

The `visualize` method generates a dendrogram and a Principal Component Analysis (PCA) plot showing document similarity. The layout of the PCA plot is driven by PCA components (loadings) that represent the linear combination of the original features. A high positive or negative loading means that feature heavily influences where a text lands along that specific PCA axis.[^3](http://www2.imm.dtu.dk/courses/02450/DemoInterpretPCA.html)

You can build these plots with the `visualize` function. The function takes the method and the top number of features to show for each principal component. Optionally, you can hide the plots or the loadings with `show_plots` and `show_loadings` (by default, set to `True`).

By default, the function will use Euclidean calculations with "ward" linkage. If you specify `method="burrows_z"`, it will use Manhattan distance with average linkage.

```python
analyzer.visualize(method="burrows_z", top_n=3, show_plots=True, show_loadings=True)
```

Should you need to access the loadings directly, you can call `get_loadings`, which returns a dictionary with "PC1" and "PC2" as the keys, and pandas DataFrames as the values.

```python
loadings = analyzer.get_loadings(method="burrows_z")
print(loadings["PC1"])
```

## Export the Results

The `to_csv` function allows you to export your matrix to a CSV file in a format compatible with R's `stylo` package: The generated layout places variables as columns and observations as rows. If you are feeding this into standard R workflows, this aligns directly with standard `read.csv(file, row.names=1)` matrix construction templates.

```python
analyzer.to_csv(filepath="burrows_z_loadings.csv", method="burrows_z")
```
