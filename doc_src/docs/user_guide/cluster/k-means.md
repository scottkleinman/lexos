# K-Means Cluster Analysis

## Overview

K-Means clustering partitions a set of documents into a number of groups or clusters in a way that minimizes the variation within clusters. The _k_ refers to the number of partitions, so for example, if you wish to see how your documents might cluster into three (3) groups, you would set `k=3`. In fact, k-means clustering differs from [hierarchical agglomerative clustering](../cluster/hierarchical-agglomerative-clustering.md) because you must begin by choosing the number of clusters into which you group your documents.

The k-means algorithm works something like this:

1. You decide on the number of clusters you wish to form.
2. The algorithm computes a **centroid** for each cluster. The centroid is the center (mean point) of a cluster. The procedure for creating centroids at the very start can be varied and is discussed below.
3. Assign each of your documents to the cluster with the nearest centroid.
4. Repeat steps 2 and 3, thereby re-calculating the locations of centroids for the documents in each cluster and reassigning documents to the cluster with the closest center. The algorithm continues until no documents are reassigned to different clusters.

## How to Perform K-Means Clustering

Lexos requires that you choose in advance a value for _k_, that is, how many groups you want to cluster your documents into.

To perform a simple k-means analysis with the default settings, start by constructing a Document-Term Matrix (DTM) as discussed in [The Document-Term Matrix](../the_document_term_matrix.md). The import the `KMeans` class and instantiate it with the DTM. You can then run the analysis with your chosen _k_ value:

```python
# Import KMeans
from lexos.cluster import KMeans

# Assuming you have your DTM saved to the dtm variable
kmeans = KMeans(dtm=dtm, k=4)
```

Pre-configuring your k-means settings can be valuable in helping you to produce meaningful results. Lexos provides a number of options for configuring the k-means procedure.

- `k`: The number of clusters to create.
- `init`: This is the initialization strategy, which can be "k-means++" or "random". "k-means++" selects initial cluster centers using a weighted probability distribution to speed up convergence. This can help can help to constrain the initial placement of the centroids. The "random" option chooses _k_ observations at random from the data to serve as the initial centroids. The default is "k-means++".
- `max_iter`: The maximum number of iterations of the k-means algorithm for a single run. The default is 300.
- `n_init`: The number of times (N) the k-means algorithm will be run with different centroid seeds (the tolerance for convergence). The final results will be the best output of those N consecutive runs. The default is 10.
- `tol` The relative tolerance with respect to inertia to declare convergence. The default is 0.0001.
- `random_state`: A number to use as the initial seed to insure that the results are reproducible. The default is 42.

The easiest way to format your data for plotting is to generate a Lexos `DTM` instance and pass it to the `Dendrogram` class. However, `Dendrogram` also accepts two other formats:

1. A Pandas DataFrame with document labels as column indexes and terms as row indexes (this is the equivalent of `DTM.to_df()`).
2. A list of documents in which each document is a sublist containing the term counts. You can also pass equivalent numpy arrays. If you use data in this format, you will probably want to include a list of document labels using the `labels` keyword.

There is no obvious way to choose the number of clusters, but some strategies will be discussed below. The k-means procedure can be very sensitive to how you have constructed your DTM, for instance, whether you have performed normalization or restricted it to only the most frequesnt terms. The procedure is also very sensitive to the position of the initial centroid seeds, although employing the "k-means++" setting of the `init` parameter helps to constrain this placement.

You can play with these settings to determine which one provide you with the best results.

## How to Choose the Number of Clusters (_k_)?

A mathematical metric known as the elbow method is commonly used to decide the optimal value of _k_. This method involves trying numerous settings of _k_, running k-means on each, and the sum of squared distances of data points to their cluster centroids. These within-cluster sum of squares (WCSS) values are plotted and the "elbow" is the point where the rate of decrease between _k_ settings slows down. Lexos can produce an elbow plot like the one below to allow you to identify the "elbow".

We can generate an elbow plot with the `elbow_plot()` method, submitting a range between 1 and 10 clusters to evaluate.

```python
kmeans.elbow_plot(k_range=range(1, 10))
```

<figure>
  <img src="../../cluster/elbow.png" alt="Sample elbow plot">
  <figcaption>Sample elbow plot</figcaption>
</figure>

The following points can help you to interpret the elbow plot.

- The x-axis shows the number of clusters (`k`) we tried.
- The y-axis shows the inertia (or within-cluster sum of squares), which measures how compact the clusters are.
- Lower values of inertia mean tighter, more defined clusters.
- The "elbow" is where the curve sharply changes direction — it’s the point beyond which adding more clusters doesn't significantly reduce inertia. This is the point of diminishing returns in decreasing WCSS.

In this example, the elbow occurs at `k=4`, meaning that 4 clusters is a good balance between under- and over-clustering. It is thus a good candidate for the optimal number of clusters.

!!! import "Limitations of the Elbow Method"

    Although the elbow method has a mathematical basis, the elbow point may not always be clear, and interpreting it is still subjective. For some datasets, it may not work well. It can be helpful to perform [hierarchical agglomerative clustering](../cluster/hierarchical-agglomerative-clustering.md) before performing k-means clustering, as the resulting dendrogram may suggest a certain number of clusters that is likely to produce meaningful results. Mathematical metrics like the elbow method are not a substitute for human knowledge about the texts being considered.

## Visualizing K-Means Clusters

Lexos provides three methods of visualizing the results of a k-means cluster analysis. In each case, Lexos first applies PCA (Principal Component Analysis) to reduce the dimensions of the data so it can be viewed in a 2D or 3D graph.

- 2D-Scatter: k-means viewed as a traditional 2D scatter plot with each cluster as a data point
- 3D-Scatter: k-means viewed as a traditional 3D scatter plot with each cluster as a data point
- Voronoi: This is the default method of visualization which identifies a centroid in each cluster (a black X) and draws a trapezoidal polygon around it. Your documents are plotted as colored dots based on which cluster they belong to. This may be helpful in allowing you to see which points fall into which cluster and how close they are to the centroid.

To generate a plot, use the `scatter()` or `voronoi()` methods as shown below:

### Generate a 2D-Scatter Plot

```python
kmeans.scatter(dim=2, title="KMeans Clustering 2D Plot", show=True)
```

<figure>
  <img src="../../cluster/2D_scatter.png" alt="Sample KMeans Clustering 2D Plot">
  <figcaption>Sample KMeans Clustering 2D Plot</figcaption>
</figure>

The `show=True` automatically displays the plot. In some cases, you may wish to save it to a variable or file for display later. In that case, set `show=False`.

### Generate a 3D-Scatter Plot

```python
kmeans.scatter(dim=3, title="KMeans Clustering 3D Plot", show=True)
```

<figure>
  <img src="../../cluster/3D_scatter.png" alt="Sample KMeans Clustering 3D Plot">
  <figcaption>Sample KMeans Clustering 3D Plot</figcaption>
</figure>

### Generate a Voronoi Diagram

```python
kmeans.voronoi(title="KMeans Clustering Voronoi Diagram" show=True)
```

<figure>
  <img src="../../cluster/voronoi.png" alt="Sample KMeans Clustering Voronoi diagram">
  <figcaption>Sample KMeans Clustering Voronoi diagram</figcaption>
</figure>

When considering visualizations of k-means clusters, we recommend that you think of each of your documents as represented by a single (x, y) point on a two-dimensional coordinate plane. In this view, a cluster is a collection of documents (points) that are close to one another and together form a group. Assigning documents to a specific cluster amounts to determining which cluster "center" is closest to your document.

All three types of visualizations are generated using the Python Plotly library. If you pan over the graph, you'll notice that a menu appears in the top right corner. This menu provides the following options:

- Download Plot as a PNG: This button allows you to download the image as a .png file. See below for methods of downloading in other formats.
- Zoom: This option allows you to click and drag to zoom in to a specific part of the graph.
- Pan: This option will change the click and drag function to panning across the graph.
- Zoom in and Zoom out: These will automatically zoom to the center of the graph.
- Auto-scale and Reset Axis: These options will zoom all the way out with the axis reset to fit the window
- Show closest data on hover: If you hover over a data point, this option will show you the value of the data point.
- Compare data on hover: If you hover over a data point, this option will show you the value of the data point and it's corresponding x-axis value.
- When you enable the 3D scatter plot, there are other options that function essentially the same as for a 2D graph that will allow you to view the 3D plot at different angles.

You can also save the images programmatically using `kmeans.save()`. The image type will be detected automatically by the file extension in the `path` parameter. You can also set `html=True` to save the image as an HTML file. Here are some examples:

```python
kmeans.save(path="myimage.png") # Saves as a .png file
kmeans.save(path="myimage.html", html=True) # Saves as an HTML file
```

Under the hood, `save()` calls Plotly's <code><a href="https://plotly.github.io/plotly.py-docs/generated/plotly.io.image_html.html" target="_blank">write_image()</a></code> and <code><a href="https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html" target="_blank">write_html()</a></code>, and it will accept any keywords taken by those methods.

You can also call `to_csv()` to export your data to a CSV file of PCA coordinates and cluster labels. `Kmeans.to_csv()` accepts any parameter taken by the Pandas<code><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html" target="_blank">to_csv()</a></code> method.
