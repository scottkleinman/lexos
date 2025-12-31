# Hierarchical Agglomerative Clustering

## Overview

Hierarchical cluster analysis is a method grouping similar data points into a hierarchy of nested clusters. It builds a tree-like structure that shows the relationships between clusters, where closer clusters are more similar. This approach helps reveal patterns and relationships within datasets, especially complex ones, by visualizing data groupings at multiple levels of similarity.

Hierarchical clustering may be agglomerative or divisive. Agglomerative clustering starts with each data point (typically a single document in Lexos) as a separate cluster and then merges them into larger clusters called **clades**. Divisive clustering takes the opposite approach, starting with all the data points in one cluster and splitting it. Currently, Lexos only provides an agglomerative algorithm.

The the results of hierarchical cluster analysis produced by this approach are typically represented visually as a **dendrogram**, which shows the hierarchical relationshops between clusters and the distance (similarity) between them.

In order to group documents into clusters, the algorithm relies on two important settings, a distance metric and a linkage criterion.

The **distance metric** metric is a measure used to quantify the similarity or dissimilarity between data points or clusters. A simple way to understand this is that a term occurring once in document A and twice in document B will have a distance of 1 between the two documents (based on that term alone). Of course, there are more numerous different ways in which we could calculate difference, and these will be discussed in more detail below.

The **linkage criterion** determines how the distance between clusters is calculated when merging. A simple way to think of this is to imagine two circles with dots in them representing the terms. The dots closest to the outer edge of the first circle will be closest to the dots closest to the outer edge of the second circle (in the direction where the circles are closest). We could use the short distance of these circles to select both to be merged into a single cluster at the next level of the hierarchy. But, equally, we could base our decision whether the merge them on the position of the dots in the centre. We have several other option to choose from, and these will be discussed in more detail below.

With this knowledge, we can describe the clustering algorithm.

1. We start with each document as its own cluster, or "leaf".
2. We identify the two closest clusters based on our chosen distance metric and linkage criterion.
3. We merge the two closest clusters into a single cluster. We then repeat steps 2 and 3 until all documen are in one cluster (the root of the tree).
4. We plot a dendrogram to represent this hierarchical structure.

An advantage of hierarchical clustering is that we do not need to choose the number of clusters in advance, and we can explore our clusters at different levels of granularity. However, our results can be sensitive to the choice of distance metric and linkage criterion. Further, the diagram represents a single cluster at the root level and a number of clusters equal to the number of documents at the leaf level. There may be more meaningful clusters on between hierarchy between these two extremes, but there is no clear method of determining a cut-off point (known as "cutting the dendrogram"). The discussion below will provide some guidance in dealing with these issues.

## How to Perform Hierarchical Agglomerative Clustering

To perform cluster analysis and generate a dendrogram, you will need document-term matrix produced by the DTM module. Then you simply import the Dendrogram class and feed it your DTM. You will also need a list of labels for the documents in your DTM object. In the example below, we will use the default settings for the distance metric and linkage criterion.

```python
# Import the Dendrogram class
from lexos.cluster import Dendrogram

# Create an instance of the Dendrogram object (feel free to change the parameters)
dendrogram = Dendrogram(
    dtm=dtm,
    labels=labels,
    metric="euclidean",
    method="average",
    orientation="top",
    # color_threshold=1.5, # Uncomment to color branches
    title="My First Dendrogram",
    figsize=(10, 8),
    show=True
)

# Show the dendrogram
dendrogram.show()
```

<figure>
  <img src="../../cluster/dendrogram.png" alt="Sample dendrogram">
  <figcaption>Sample dendrogram</figcaption>
</figure>

### Dendrogram Settings

When we create the `Dendrogram`, we need to tell it how to measure document similarity and how to connect those similarities into a tree. Here are the key parameters you can adjust:

- `dtm`: This is our "linguistic spreadsheet" (`dtm`). See the notes below for the possible formats.
- `metric`: This sets the distance metric, which tells the dendrogram how to measure the "distance" or dissimilarity between your documents. Shorter distances mean more similar documents. Options include `euclidean` (the default), `cosine`, and `cityblock`. For other options, see [Choosing a Distance Metric](#choosing-a-distance-metric) below.
- `method`: This sets the linkage criterion, which determines how individual documents (or existing clusters of documents) are joined together to form larger branches and clusters in the tree. Option `average` (the default), `single`, `complete`, and `ward`. For further information, see [Choosing a Linkage Method](#choosing-a-linkage-method) below.
  - `labels`: This is simply the list of descriptive names for your documents (e.g., "Poe", "Lippard") that we defined earlier. These will appear as the leaves (endpoints) on your tree.
- `orientation`: Controls the direction of the dendrogram. The default `"top"` orients the branches so that they extend downwards from root at the top. Other options are `"bottom"`, `"left"`, and `"right"`.
- `color_threshold`: If set, branches with a distance below this threshold will be colored differently from those above it. This helps visualize clusters at a certain distance level. You can try a number like `1.0` or `1.5` to see its effect.
- `show`: Controls whether the generated tree figure is displayed automatically. If `False`, the tree will not be displayed, but you display it later by calling `dendrogram.showfig()`. There are also methods that enable you to save it to a variable or file.
- `title`: Adds a title to your dendrogram plot.
- `figsize`: A tuple `(width, height)` in inches to set the size of the overall figure. For example, `(12, 8)` for a wider and taller plot.

The easiest way to format your data for plotting is to generate a Lexos `DTM` instance and pass it to the `Dendrogram` class. However, `Dendrogram` also accepts two other formats:

1. A Pandas DataFrame with document labels as row indexes and terms as column indexes (this is the equivalent of `DTM.to_df(transpose=True)`).
2. A list of documents in which each document is a sublist containing the term counts. You can also pass equivalent numpy arrays. If you use data in this format, you will probably want to include a list of document labels using the `labels` keyword.

### Plotting Dendrograms with Plotly

The `Dendrogram` class uses Python's matplotlib library to produce static images. However, in very large dendrograms, there is a danger of leaf labels overlapping, making the plot unreadable. In this case, you can use the Plotly plotter, which provides the ability to pan and zoom around the dendrogram, making it more readable. The Plotly plotter is also ideal if you are including the dendrogram in a web app.

To use the Plotly plotter, import the `PlotlyDendrogram` class, create an instance, and use it as above.

```python
# Import the PlotlyDendrogram class
from lexos.cluster import PlotlyDendrogram

# Create an instance of the PlotlyDendrogram object
dendrogram = PlotlyDendrogram(
    dtm=dtm,
    labels=labels,
    metric="euclidean",
    method="average",
    orientation="bottom",
    title="Document Similarity Dendrogram",
)

# Show the dendrogram using Plotly
dendrogram.show()
```

<figure>
  <img src="../../cluster/plotly_dendrogram.png" alt="Sample Plotlydendrogram">
  <figcaption>Sample Plotly dendrogram</figcaption>
</figure>

Note that the image above is a static representation and does not demonstrate Plotly's interactive features.

Data should be passed to `PlotlyDendrogram` either as a Lexos `DTM` instance or using one of the other datatypes described for the `Dendrogram` class above.

## Choosing a Distance Metric

One of the most important (and least well-documented) aspects of the hierarchical clustering method is the distance metric. Since we are representing texts as document vectors, it makes sense to define document similarity by comparing the vectors. One way to do this is to measure the distance between each pair of vectors. For example, if two vectors are visualized as lines in a triangle, the hypotenuse between these lines can be used as a measure of the distance between the two documents. This method of measuring how far apart two documents are is known as **Euclidean distance**. This is the default setting used by Lexos. It is good for general comparisons but can be sensitive to the overall length of documents (longer documents might naturally have larger term counts, increasing their "distance").

Another common metric is **cosine similarity**. Imagine each document as an arrow pointing in a specific linguistic "direction." Cosine similarity measures how much these arrows point in the same direction. If the angle at which they point is almost identical, the documents are very similar, even if one document is much longer than another. This is often an excellent choice for text analysis as it focuses on stylistic or thematic *direction* rather than raw word counts.

**Cityblock distance** also called Manhattan distance is another common metric. Imagine moving on a city grid where you can only go along streets (no diagonal shortcuts). This distance is the sum of the absolute differences for each term between two documents. This metric is useful when the individual differences in term counts are important.

Many other metrics are available (e.g., "jaccard", "chebyshev") from the Python scipy package, which Lexos runs under the hood. You can find a full list in the <code><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html" target="_blank">SciPy documentation</a></code>.

The table below provides some additional guidance on how to choose a distance metric.

|                      | Small Number of terms per segment                              | Large Number of terms per segment                                     |
|----------------------|----------------------------------------------------------------|-----------------------------------------------------------------------|
| **Small Vocabulary** | Bray-Curtis, Hamming    (e.g. character dialogue)              | Euclidean, Chebyshev, Standardized Euclidean (e.g. chapters of books) |
| **Large Vocabulary** | Correlation, Jaccard, Squared Euclidean (e.g. non-epic poetry) | Cosine, Manhattan, Canberra (e.g. comparing entire corpora)           |

## Choosing a Linkage Method

At each stage of the clustering process, a choice must be made about whether two clusters should be joined (and recall that a single document itself forms a cluster at the lowest level of the hierarchy). "Average" is the default, but you may choose other linkage methods by clicking the button.

- Average: This method is a compromise between single and complete linkage. It takes the average distance of the points in each cluster and uses the shortest average distance for deciding which cluster should be joined to the current one. When combining two clusters, this method considers the average distance between *all* pairs of documents in the two clusters. It tends to produce well-balanced clusters.
- Single: An intuitive means for doing this is to join the cluster containing a point (e.g. a term frequency) closest to the current cluster. This is known as single linkage, which joins clusters based on only a single point. In other words, clusters are joined based on the *closest* pair of documents between them. Single linkage does not take into account the rest of the points in the cluster, and the resulting dendrograms tend to have spread out clusters. This process is called "chaining". When this happens, where documents connect one after another, forming long, straggly branches.
- Complete: Complete linkage uses the opposite approach. It takes the two points furthest apart between the current cluster and the others. The cluster with the shortest distance to the current cluster is joined to it. Complete linkage thus takes into account all the points on the vector that come before the one with the maximum distance. It tends to produce compact, evenly distributed clusters, ensuring all documents within a cluster are relatively similar to each other.
- Weighted: The weighted average linkage performs the average linkage calculation but weights the distances based on the number of terms in the cluster. It, therefore, may be a good option when there is significant variation in the size of the documents under examination.
- Ward: This method aims to minimize the increase in "variance" (or spread) within clusters when they are merged. It tries to make clusters that are as "tight" and internally similar as possible. It often produces intuitive and well-structured clusters.

Which linkage criterion you choose depends greatly on the variability of your data and your expectations of its likely cluster structure. The fact that it is very difficult to predict this in advance may explain why the "compromise" of average linkage is the best default.

## Intepreting Dendrograms

### Choosing Where to Cut the Dendrogram

Hierarchical clustering is an exploratory technique, so it's often helpful to try different cut-off points and evaluate the resulting clusters. The height of the cut determines the number of clusters. A higher cut will result in fewer, larger clusters, while a lower cut will result in more, smaller clusters.

The best way to cut a dendrogram will always depend on the specific dataset and the goals of the analysis. There's no single "right" way to do it. To determine where to cut a dendrogram for clustering, you can use visual cues like the longest vertical distance between nodes, or consider numerical criteria like the Silhouette score or Dunn's index, or even trial and error. The choice of cut-off point depends on how many clusters you want and the desired level of similarity within each cluster.

Here is a procedure to use as a starting point:

1. Identify the longest vertical distance: Look for the longest vertical line (distance) between merging nodes on the dendrogram. Cutting at this point often reveals a natural separation between clusters.
2. Consider the overall structure: Observe how the data points are grouped at different height levels. You might choose a cut that separates well-defined, compact clusters or one that creates a few large clusters.

Lexos does not offer any numerical criteria for evaluating the quality of hierarchical clusters. However, you can count the number of leaves at your cutoff point and use that as the *k* value in a k-means clustering analysis to provide comparative evidence.

In addition, Lexos does not offer a method of drawing the dendrogram showing the cut. SciPy provides the `fcluster` method for cutting dendrograms, and we will hopefully implement it in the future. This <a href="https://stackoverflow.com/questions/70801281/how-can-i-plot-a-truncated-dendrogram-plot-using-plotly" target="_blank">Stack Overflow discussion</a> provides information on how to add truncate mode.

### Cluster Robustness

By cutting trying different distance metrics and linkage methods, as well as by cutting the dendrogram at different heights, you can evaluate the **robustness** of individual clusters. A "robust" cluster is one that persists, regardless of the setup criteria. If the cluster is sensitive to changes in the setup criteria, it is more likely to be a statistical artefact of those criteria, rather than a meaningful pattern. This, however, is a guideline, and its usefulness will depend on your data.

!!! note "Measuring Robustness with Bootstrap Consensus Trees"
    One way to automate the process of assessing cluster robustness is to implement Bootstrap Consensus Trees, which perform clustering with multiple settings and record the most-consistent clusters. See the section on [Bootstrap Consensus Trees](#bootstrap-consensus-trees) below.

## Clustermaps

A clustermap is a dendrogram attached to a heatmap, showing the relative similarity of documents using a colour scale. A clustermap combines the best of two worlds: hierarchical clustering (dendrograms) and pairwise similarity representation (heatmap).

The dendrogram on the top shows the hierarchical clustering of your documents based on their content (the rows of your DTM). The dendrogram on the left shows the same clustering, but rotated. As with standalone dendrograms, shorter branches mean documents (or clusters) are more similar. The order of documents along the heatmap axes is determined by these dendrograms, grouping similar documents together.

The heatmap grid visually represents the **pairwise distances** between your documents. Each cell at the intersection of a row and a column represents the distance between two documents. The color intensity on the heatmap will represent the distance between documents: typically, darker/different colors show greater distance (less similarity), while lighter/similar colors show shorter distance (more similarity). The diagonal of the heatmap will always be the same color, usually representing zero distance, as a document has zero distance from itself.

Clustermaps can be useful for observing the following:

- **Stylistic Groupings:** Does the heatmap show a strong block of similarity among authors from the same literary period or movement?
- **Thematic Cohesion:** If your DTM focused on specific themes, do documents discussing similar themes cluster together?
- **Influence or Divergence:** You might see how a text aligns with or diverges from others, giving insights into authorship, genre, or evolution of style.

Lexos can generate static clustermap images using the Python Seaborn library or dynamic images using Plotly.

### Using Seaborn¤

The Python Seaborn visualization library has a clustermap function, which has somewhat limited functionality. Lexos wraps the Seaborn function in the `Clustermap` class to provide additional convenience features, such as the inclusion of titles. To generate a clustermap with Seaborn, use the following code:

```python
# Import the ClusterMap class
from lexos.cluster import Clustermap

# Create a ClusterMap object
cm = Clustermap(dtm=dtm, title="My Clustermap")
cm.show()
```

<figure>
  <img src="../../cluster/clustermap_example.png" alt="Sample clustermap">
  <figcaption>Sample clustermap</figcaption>
</figure>

The `dtm` can be a Lexos `DTM` instance, a compatible Pandas DataFrame, or a list of lists of tokens. For the clustering parameters, see the advice in [Choosing a Distance Metric](#choosing-a-distance-metric) and [Choosing a Linkage Method](#choosing-a-linkage-method) above.

!!! important
    Unlike `Dendrogram` and `PlotlyDendrogram`, `Clustermap` accepts a Pandas DataFrame formatted with documents as columns and terms as rows. This is the equivalent of `DTM.to_df()`.

In addition to the `title`, `metric`, and `method` keywords, `Clustermap` takes the following parameters:

- `labels`: A list of descriptive names for your documents. These will appear as the leaves (endpoints) on your tree. If not supplied, the labels from your Lexos `DTM` or Pandas DataFrame will be used.
- `z_score`: Standardizes the values within each row (documents) or column (terms). If the value is set to `None`, the heatmap shows raw frequencies (or whatever your DTM contains). The setting `0` standardizes each row (document) by subtracting its mean and dividing by its standard deviation. This highlights how *terms vary within a single document* relative to that document's average term frequency. Useful for comparing patterns across documents regardless of their length. The setting `1` standardizes each column (term) by subtracting its mean and dividing by its standard deviation. This highlights how *a single term's frequency varies across different documents* relative to that term's average frequency. Useful for seeing which documents use a term more or less than average.
- `standard_scale`: Similar to `z_score`, but scales to a specific range (usually 0 to 1). The setting `0` scales each row (document) so its minimum value is 0 and its maximum is 1. The setting `1` scales each column (term) so its minimum is 0 and its maximum is 1.
- `cmap`: Sets the color scheme (colormap) for the heatmap. It determines which colors represent low values and which represent high values. The default setting "vlag" is a diverging colormap (red/blue), which is good for showing values around a center point (especially after `z_score` scaling). Other good general-purpose colormaps are "viridis" and "coolwarm". You can find listings of other <code><a href="https://matplotlib.org/stable/gallery/color/colormap_reference.html" target="_blank">matplotlib</a></code> and <code><a href="https://seaborn.pydata.org/tutorial/color_palettes.html" target="_blank">seaborn</a></code> colormaps online.
- `hide_upper`: Setting the value to `True` removes the dendrogram above the heatmap. Useful if you are not interested in the clustering of columns/terms.
- `hide_side`: Setting the value to `True` removes the dendrogram to the left of the heatmap. Useful if you are not interested in the clustering of rows/documents.
- `row_cluster`: Perform clustering on rows (documents). Default is `True`. Along with `col_cluster`, this setting is useful if you have a specific ordering in mind for comparison, or if you've pre-computed a linkage.
- `col_cluster`: Perform clustering on columns (terms). Default is `False`. If `False`, items will be displayed in their original order.
- `row_colors`: Allows you to add colored strips alongside the rows, which can be used to visually group or categorize your documents. Provide a list of colors (e.g., `['red', 'blue', 'green']`). The list should match the number of documents/terms. You can also use a named `seaborn` palette (e.g., `"husl"`). Setting the value to "default" will use `seaborn.husl_palette(8, s=0.45)`. This setting is great for adding metadata! For example, if you have two categories of documents (e.g., "male authors" vs. "female authors"), you could assign a color to each category to see if your clustering aligns with these external factors.
- `col_colors`:Allows you to add colored strips alongside the columns, which can be used to visually group or categorize your terms. Setting values are the same as for `row_colors`.
- `title`: Adds a title to your dendrogram plot.
- `figsize`: A tuple `(width, height)` in inches to set the size of the overall figure. For example, `(12, 8)` for a wider and taller plot.

The `Clustermap` instance can be further customized with any  <code><a href="https://seaborn.pydata.org/generated/seaborn.clustermap.html" target="_blank">Seaborn.clustermap</a></code> parameter.

After you've generated your clustermap, you'll likely want to save it as an image for reports or presentations. The `save()` method lets you do this easily. Just provide a file path, and it'll save the image. You can specify different file formats by changing the extension (e.g., `.png`, `.jpg`, `.pdf`, `.svg`). The `save()` method wraps the `matplotlib` <code><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html" target="_blank">savefig</a></code> function and accepts any of its keywords.

### Using Plotly¤

Plotly clustermaps are somewhat experimental and may or may not render plots that are as informative as Seaborn clustermaps. One advantage they have is that, instead of providing labels for each document at the bottom of the graph, they provide the document labels on the x- and y-axes, as well as the z (distance) score in the hovertext. This allows you to mouse over individual sections of the heatmap to see which documents are represented by that particular section, as well as the exact distance values.

Plotly clustermaps are constructed in the same manner to Seaborn clustermaps with the same settings, so far as is possible in the Plotly library:

```python
# Import the PlotlyClustermap class
from lexos.cluster import PlotlyClustermap

# Create a PlotlyClustermap object
cm = PlotlyClustermap(dtm=dtm, title="My Clustermap")
```

<figure>
  <img src="../../cluster/plotly_clustermap_example.png" alt="Sample Plotly clustermap">
  <figcaption>Sample Plotly clustermap</figcaption>
</figure>

!!! note
    Note that the image above is a static representation and does not demonstrate Plotly's hover effects.

All the options for Plotly dendrograms are available with the following differences:

- `figsize` is measured in pixels.
- `colorscale` is the name of a built-in <a href="https://plotly.com/python/builtin-colorscales/" target="_blank">Plotly colorscale</a>. This is applied to the heatmap and converted internally to a list of colours to apply to the dendrograms.

Two additional parameters, `hide_upper` and `hide_side` allow you to hide the individual dendrograms.

!!! warning
    Note that panning and zooming can cause the heatmap and dendrograms to become unsynced. There is currently no way to maintain the syncing in pure Python. If you need to zoom in on particular sections of the plot, you may be able to achieve the effect you are looking for by saving the plot as an HTML file with the *experimental* `include_sync` parameter:

  ```python
  html = cm.to_html(include_sync=True)
  with open("filename.html", "w") as f:
      f.write(html)
  ```

  Open the HTML file in a web browser, and you may get the behaviour you need. See below for other options for saving your Plotly clustermaps.

You have several option for saving your Plotly clustermaps. In the Plotly toolbar, there is a "Download plot as png" option to save the plot as a static `.png` file. You can also save the the image to a static file programmatically by calling `PlotlyClustermap.write_image()`. Just provide a file name (including the extension), and it will save the image. You can choose different file formats by changing the extension (e.g., `.png`, `.jpg`, `.pdf`, `.svg`). This is a wrapper around Plotly's <code><a href="https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html" target="_blank">write_image()</a></code> function and accepts all the same arguments.

Plotly figures are highly interactive when saved as HTML, allowing you to zoom, pan, and hover over data points in your saved file. If you wish to save your diagram as an HTML file, call `PlotlyClustermap.write_html()`. This is a wrapper around Plotly's <code><a href="https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html" target="_blank">write_html()</a></code> function and accepts all the same arguments.

Note that `write_image()` and `write_html()` have parallel `to_image()` and `to_html()` methods that allow you to assign the results to a variable, rather than saving to a file.

## Bootstrap Consensus Trees

A **Bootstrap Consensus Tree** is particularly robust because it doesn't just build one tree. Instead, it builds many, many trees by randomly sampling portions of your DTM. It then finds the "consensus": the most consistently appearing relationships across all those individual trees.

Generating bootrap consensus dendrograms involves submitting the same distance metric and linkage method parameters as regular dendrogram. However, there are a few additional parameters to set:

- `dtm`: Unlike the other clustering modules, the `BCT` class accepts only an instance of a Lexos `DTM`.
- `cutoff`: This is a confidence threshold. As mentioned, the BCT is built from many individual "bootstrap" trees. A `cutoff` of `0.5` (which means 50%) means that a specific grouping of documents (a branch on the tree) must appear in at least 50% of all the trees generated during the `iterations` to be considered reliable enough to show up in the final consensus tree. Higher `cutoff` values (e.g., 0.7 or 0.8) will result in a "sparser" tree, showing only the most robust and consistent relationships. Lower `cutoff` values (e.g., 0.3) will show more relationships, but some of these might be less statistically reliable.
- `iterations`: This is the number of "bootstrap resampling" rounds. In each round, Lexos takes a random 80% sample of the terms (columns) from your DTM and builds a tree from that sample. More iterations (e.g., 100, 1000) makes the consensus tree more statistically reliable and representative of the underlying relationships in your texts, as it averages out more variations. However, it will take longer to compute. Fewer iterations (e.g., 10, 20) are good for quick testing or initial explorations. For final research results, `100` (the default in the `BCT` class) or higher is often recommended if computation time allows.
- `replace`: This relates to how the terms are sampled during each iteration. Setting the value to "with" means a term column can be selected multiple times within a single 80% sample (allows for more randomness). The value "without" means each term column can only be selected once per 80% sample (more stable). This setting is generally suitable for DTMs as it ensures each unique term contributes uniquely within a sample.
- `doc_labels`: This is simply the list of descriptive names for your documents (e.g., "Poe", "Lippard") that we defined earlier. These will appear as the leaves (endpoints) on your tree.
- `text_color`: Sets the color for all text on the plot (axis labels, branch lengths, and document labels). You can use "rgb(R, G, B)" format. For example: `"rgb(0, 0, 0)"` (black) or `"rgb(255, 0, 0)"` (red).
- `layout`: Sets the layout of the dendrogram, either "rectangular" (the default) or "fan".

### Plotting Bootstrap Consensus Trees

To create a bootstrap consensus tree with rectangular layout, use the following code, setting the parameters describe above as required:

```python
# Import the BCT class for Bootstrap Consensus Tree
from lexos.cluster import BCT

# Create an instance of the BCT object (feel free to adjust parameters)
bct = BCT(
    dtm=dtm,
    metric="euclidean",
    method="average",
    cutoff=0.5,
    iterations=10,
    replace="without",
    labels=labels,
    text_color="rgb(0, 0, 0)",
    layout="rectangular",
    title="Bootstrap Consensus Tree (Rectangular Layout)"
)

# Show the figure
bct.show()
```

<figure>
  <img src="../../cluster/bootstrap_consensus_rectangular.png" alt="Sample Bootstrap Consensus Tree rectangular layout">
  <figcaption>Sample Bootstrap Consensus Tree rectangular layout</figcaption>
</figure>

To generate a diagram with a fan layout, set `layout="fan"` (and adjust the `title` set above).

<figure>
  <img src="../../cluster/bootstrap_consensus_fan.png" alt="Sample Bootstrap Consensus Tree fan layout">
  <figcaption>Sample Bootstrap Consensus Tree fan layout</figcaption>
</figure>
