# Visualization

## Overview

The Lexos `visualization` module provides a set of modular tools for visualizing the frequency of terms in textual data. Currently, the primary tool is the "word cloud", a cover term for a number of variant types of charts that display terms scaled according to their frequency. There are two basic types: traditional word clouds and packed circle charts, or bubble charts.

There are two methods of generating word cloud variants: pure Python approaches that produce static images and Javascript versions that generate charts in the web browser with interactive features. Each of these will be discussed below.

## Word Clouds

Word clouds display each submitted term in your text(s), scaled according to its frequency and laid out in a compact display so that you can easily "eyeball" which terms are most frequent. To produce a basic word cloud, import the `WordCloud` class and submit a simple text.

```python
from lexos.visualization.cloud import WordCloud

text = "This is a sample text to demonstrate how to produce a word cloud."
wc = WordCloud(data=text, title="My Word Cloud")
wc.show()
```

<figure>
  <img src="basic_wordcloud.png" alt="Basic word cloud">
  <figcaption>Basic word cloud</figcaption>
</figure>

Notice that you can optionally supply a title with the `title` keyword. The last line (`wc.show()`) will display the word cloud.

Unlike many other word cloud generators, Lexos prefers for you to pre-tokenise your data *before* you pass it to the `WordCloud` class, such as by converting it to a spaCy `Doc`. You *can* pass a spaCy `Doc` object directly, but it is better to convert it to a list of token strings. This allows you to perform other types of pre-processing, such as removing punctuation and stop words (you could also generate a list of token strings by some other means). In the example below, we convert our text to a spaCy `Doc`, filter out punctuation, stop words, and white space, then generate our word cloud.

```python
# Import the Lexos Tokenizer class
from lexos.tokenizer import Tokenizer

# Create an instance of the Tokenizer class and make a spaCy doc
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc(text)

# Generate a filtered list of tokens
tokens = [
    token.text for token in doc
    if not token.is_punct
    and not token.is_stop
    and not token.is_space
]

# Create a new word cloud
wc = WordCloud(data=tokens, title="Word Cloud from Tokenized Doc")
wc.show()
```

<figure>
  <img src="tokenized_wordcloud.png" alt="Word cloud from a pre-tokenized doc">
  <figcaption>Word cloud from a pre-tokenized doc</figcaption>
</figure>

That's much nicer! You can also pass a list of documents. By default, they will be merged into one document before the terms are counted. In the example below, we'll combine the document we created above with another document consisting of three addtional tokens. By using only the first 10 tokens of the first document, you'll be able to see whether the second document is being added.

```python
# Add some more tokens
multiple_docs = [
    tokens[:10],
    ["Some", "additional", "tokens"]
]

# Create and display a word cloud
wc = WordCloud(
    data=multiple_docs,
    title="Additional Parameters",
    height=200,
    width=200,
    round=100,
    limit=10
)
wc.show()
```

<figure>
  <img src="wordcloud_additional_parameters.png" alt="Word cloud with additional parameters">
  <figcaption>Word cloud with additional parameters</figcaption>
</figure>

!!! important
    The `limit` parameter will select the ten *most common* terms, not the *first* ten tokens in the token list.

Under the hood, Lexos uses the Python <code><a href="https://amueller.github.io/word_cloud/" target="_blank">WordCloud</a></code> and <code><a href="https://matplotlib.org/" target="_blank">matplotlib</a></code> to create the word cloud. You can pass options to `WordCloud` with the `opts` keyword, the value of which should be a dictionary of options and their values. You can pass options to `matplotlib` with the `figure_opts` keyword, which also takes a dictionary. A full discussion of the available options is beyond the scope of this tutorial, and you are encouraged to consult the `WordCloud` and `matplotlib` documentation for ways to customise your word cloud. Here, we'll just provide a simple example showing how to change the background colour.

```python
# Define an options dictionary
opts = {"background_color": "lightblue"}

# Create the word cloud
wc = WordCloud(data=text, title="My Blue Word Cloud", opts=opts)
wc.show()
```

<figure>
  <img src="wordcloud_light_blue.png" alt="Word cloud with light blue background">
  <figcaption>Word cloud with light blue background</figcaption>
</figure>

!!! note
    Advanced users can manipulate the plot directly using `matplotlib`, as shown below:

    ```python
    # Create a pyplot figure object with the specified options
    fig = plt.figure(**wc.figure_opts)

    # Modify the figure
    fig.set_facecolor("lightgreen")
    fig.suptitle("My Light Green Word Cloud")

    # Hide the axis lines and labels
    plt.axis("off")

    # Create the image
    # The semicolon prevents display of the object in Jupyter notebooks, or you can add plt.show()
    plt.imshow(wc.cloud, interpolation="bilinear");
    ```

To save your image, use the save method. The file type will be determined by your file suffix (`.png` or `.jpg`).

```python
# Change the file path to the location where you wish to save the image file
wc.save("wordcloud.png")

# Or save as a jpg file
wc.save("wordcloud.jpg")
```

The `save` method accepts any arguments allowed by `matplotlib`'s <code><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html" target="_blank">savefig</a></code> method:

```python
plt.savefig("wordcloud.png", dpi = 300)
```

## Bubble Charts

Bubble charts (also known as packed circle charts or bubble visualisations) arrange terms into labelled circles, which can sometimes be easier to read than traditional word clouds. They are produced in a similar manner.

```python
from lexos.visualization.cloud import WordCloud

text = "This is a sample text to demonstrate how to produce a bubble chart."
bc = BubbleChart(data=text, title="My Bubble Chart")
bc.show()
```

<figure>
  <img src="basic_bubblechart.png" alt="Basic bubble chart">
  <figcaption>Basic bubble chart</figcaption>
</figure>

Bubble charts must have the same height and width, so the figure dimensions are controlled with the `figsize` keyword with the value in inches.

```python
bc = BubbleChart(data=text, figsize=6.5, title="My Bubble Chart")
```

<figure>
  <img src="bubblechart_parameters.png" alt="Bubble chart with parameters">
  <figcaption>Basic bubble chart with parameters</figcaption>
</figure>

Bubble charts do not have the `opts` and `figure_opts` keywords you can access in the `WordCloud` class. The two configuration options available are `bubble_spacing`, the spacing between bubbles (the default is 0.1) and `colors`, a list of hexidecimal codes designating the available bubble colours (the default is `["#5A69AF", "#579E65", "#F9C784", "#FC944A", "#F24C00", "#00B825"]`).

To save your image, use the save method. The file type will be determined by your file suffix (`.png` or `.jpg`).

```python
# Change the file path to the location where you wish to save the image file
wc.save("bubble_chart.png", dpi=300)

# Or save as a jpg file
wc.save("bubble_chart.jpg")
```

As with word clouds, you can pass additional arguments to `matplotlib`'s `savefig` method.

!!! note
    You can also manipulate the image in `matplotlib` after it has been generated. Here's an example showing how to change the title and plot size:

    ```python
    # Import matplotlib
    import matplotlib.pyplot as plt

    # Get the term counts from the BubbleChart instance
    data = list(bc.counts.keys())

    # Create a new figure and axis, setting the figure size
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=(10, 10))

    # Call the internal _plot method to create the bubble chart
    bc._plot(ax, data)

    # Turn off the axis lines and labels
    plt.axis("off")

    # Recompute the data limits
    ax.relim()

    # Autoscale the view limits using the data limits
    ax.autoscale_view()

    # Add a new title
    ax.set_title("A different title")

    # Display the plot
    plt.show()
    ```

## Multiclouds

Multiclouds are grids of word clouds that allow you to compare the term counts in multiple documents. They are produced using the `MultiCloud` class. Each cloud in the grid is an individual instance of a `WordCloud`, so you can use the customisation parameters available for that class. Here is a short example:

```python
# Import the MultiCloud class
from lexos.visualization.cloud import MultiCloud

texts = [
    "Natural language processing is a fascinating field that combines linguistics, computer science, and artificial intelligence.",
    "Text analysis, sentiment analysis, and language modeling are key components of modern NLP systems.",
    "Machine learning algorithms help computers understand and process human language effectively.",
    "Natural language processing is a fascinating field that combines linguistics, computer science, and artificial intelligence.",
    "Text analysis, sentiment analysis, and language modeling are key components of modern NLP systems.",
    "Machine learning algorithms help computers understand and process human language effectively."
]

# Create and display a MultiCloud chart
mc = MultiCloud(data=texts, ncols=3, round=150, title="Sample Multiclouds")
mc.show()
```

<figure>
  <img src="multiclouds.png" alt="Sample three-column multicloud chart with round=150">
  <figcaption>Sample three-column multicloud chart with <code>round=150</code></figcaption>
</figure>

You can change the number of columns in the layout with the `ncols` parameter. The default is 3. The `height` and `width` parameters will be applied to each individual word cloud.

The input is a list of texts (it can also be a list of token sublists), where each list item represents a document to be rendered as a word cloud.

You will notice that by default the documents are labelled "Doc 1", "Doc 2", "Doc 3", etc. If you wish to use different titles for each word cloud, you can supply a list using the `labels` keyword.

If you want to change the labels ("Doc 1", "Doc 2", etc.), you can pass a list to the `labels` parameter. For example:

```python
labels = ["A", "B", "C", "D", "E", "F"]
mc = MultiCloud(data=texts, labels=labels, round=150, title="Sample Multiclouds")
```

!!! note
    If you want to use `matplotlib` to modify the multicloud after it has been created, you can do it like this:

    ```python
    # Create the MultiCloud chart
    mc = MultiCloud(data=texts, title="Starting Title", round=150)

    # Draw the canvas
    mc.fig.canvas.draw()

    # Get the figure data as an image
    im = mc.fig.canvas.buffer_rgba()

    # Add the figure data to matplotlib pyplot instance
    plt.imshow(im, interpolation="bilinear")

    # Change the title and size of the figure
    plt.suptitle("A Different Title")

    # Turn off the axis lines and labels
    plt.axis("off")

    # Save the figure
    plt.savefig("multiclouds.png", dpi=300)

    # Display the plot
    plt.show()
    ```

## Types of Data

In the examples above, we have shown how `WordCloud`, `BubbleChart`, and `MultiCloud` first tokenise the text on whitespace if you use raw text strings. We have also seen that `WordCloud` and `BubbleChart` accept lists of token strings. For `MultiCloud`, you need to submit a list of lists where each list item at the top level is a document and each sublist is a list of tokens. For instance, here is a shortened version of the data we used to produce our multicloud above:

```python
tokens = [
    ["Natural", "language", "processing", "is", "a", "fascinating", "field", "."],
    ["Text", "analysis", "sentiment", "analysis", "," "and", "language", "modeling"]
]
```

Each sublist represents a separate document. By default, `WordCloud` and `BubbleChart` will merge these into a single document, whilst `MultiCloud` will generate an individual cloud for each document int he list. We will see below how you can modify this behaviour.

If your data has already been tokenised into spaCy `Doc` objects, you can pass them directly to the visualisation classes (likewise, you can pass spaCy `Span` objects or lists of `Token` objects). However, a better approach is to use spaCy to pre-process your documents, such as by filtering punctuation and stop words, and then pass the tokens as lists of strings. Here is an example where we additionally convert the tokens to lower case.

```python
tokens = [
    token.lower_ for token in doc
    if not token.is_punct and not token.is_stop
]
```

This allows you to take advantage of spaCy's natural language processing functionality.

Another scenario is where you might have pre-tokenised texts is if you have already generated a document-term matrix with the Lexos `dtm` module. The `WordCloud` and `BubbleChart` classes accept a Lexos `DTM` object, as well as a pandas DataFrame produced by the `DTM.to_df()` method.

Finally, you may have a pre-generated list of term counts, such as is produced by the Python `collections.Counter` class:

```python
from collections import Counter

tokens = ["this", "is", "a", "sample", "text", "to", "demonstrate", "how", "to", "produce", "a", "bubble", "chart"]
counter = dict(Counter(tokens))
print(counter)
# {'this': 1, 'is': 1, 'a': 2, 'sample': 1, 'text': 1, 'to': 2, 'demonstrate': 1, 'how': 1, 'produce': 1, 'bubble': 1, 'chart': 1}
```

You can pass this dictionary directly to the `data` parameter in the `WordCloud` and `BubbleChart` classes.

## Limiting the Number of Documents

If you pass a list of documents, a `DTM` object, or a pandas DataFrame, you may want to limit the chart to data from individual documents. You can do this by passing a list of document indexes (beginning with 0) to the `docs` keyword:

```python
bc = BubbleChart(data=dtm, docs=[0, 2])
```

Only terms from the first and third documents in the document-term matrix will appear in the chart. The `docs` keyword is available in all three classes.

## Making Interactive Word Clouds with D3.js

## Generating Dynamic Images

The static images produced by `WordCloud`, `BubbleChart`, and `MultiCloud` are very good for presentations, but they have their limitations, especially for more cluttered data. Because of this, Lexos offers alternative versions that use the Javascript <a href="https://d3js.org/" target="_blank">D3.js</a> library. This allows you produce interactive features such as the ability to hover over the terms in your word cloud to see their counts. D3 visualisations are beautiful and useful for exploring data when static images are hard to read. They are also ideal for embedding in web applications.

The cells below demonstrate how to generate D3 versions of word clouds and bubble charts.

!!! important
    Because D3 is a Javascript library, it processes data into charts in the web browser. As a result, the charts will probably not display in a Jupyter notebook. Instead, you have to save your chart as an HTML page and open it separately in the web browser.

    For most system configurations, the Lexos D3 visualisations will automatically open a web browser when you generate your chart. If you do not wish to open the browser automatically, set `auto_open=False`.

    If your system does not have a default web browser set or saves temporary files in an unexpected location, the web browser may not open automatically or may open a blank page. In this circumstance, set the `auto_open` parameter to `False` and save the file. Then search for the file using your operating system and launch the file manually.

To generate a D3 word cloud, follow the procedure below (noting the import):

```python
# Import the D3WordCloud class
from lexos.visualization.d3_wordcloud import D3WordCloud

# Generate and save a word cloud
wc = D3WordCloud(data=text, width=300, height=300)
wc.save("d3_wordcloud.html")
```

<iframe src="d3_wordcloud.html" width="100%" height="400" frameborder="0" style="padding:0;"></iframe>

Hover over the words to see further information.

The `width` and `height` parameters are measured in pixels, and, as with the static image classes, you can set the `limit` and `docs` keywords.

```python
wc = D3WordCloud(
    data=docs,
    docs=1,
    title="30 Most Common Words in Doc 2",
    width=300,
    height=300,
    limit=30
)
```

<iframe src="d3_wordcloud_limit.html" width="100%" height="420" frameborder="0" style="padding:0;"></iframe>

!!! note
    If you wish to generate a chart from a single doc in a list of docs, you can pass the doc index directly; it does not have to be in a list.

    Remember that, because docs are zero-indexed, the index 1 refers to the second document in the list.

The `D3WordCloud` class provides a number of other parameters for customising the appearance of the word cloud:

- `font`: The name of the font to use.
- `spiral`: The spiral type to use for the word cloud, "archimedean" (the default) or "rectangular".
- `scale`: The scale type to use for the word cloud, "log", "sqrt", or "linear".
- `angle_count`: The number of angles to use for the word cloud.
- `angle_from`: The starting angle for the word cloud. The default is -60°.
- `angle_to`: The ending angle for the word cloud. The default is 60°.
- `background_color`: The background color of the word cloud. The default is white.
- `colorscale`: The name of a categorical d3 scale to use for the word cloud. The default is d3.scale.category20b". For other colorscales, see the <a href="https://d3js.org/d3-scale" target="_blank">d3-scale</a> documentation.

!!! note
    The available options are based on the exceptional <a href="https://www.jasondavies.com/wordcloud/" target="_blank">word cloud generator</a> produced by Jason Davies.

You can also generate a multicloud in D3:

```python
from lexos.visualization.d3_wordcloud import D3MultiCloud

# Create multi-cloud
mc = D3MultiCloud(
    data_sources=texts,
    title="D3 Multiclouds",
    labels=None,
    cloud_width=250,
    cloud_height=250,
    columns=2
)
mc.save("d3_multiclouds.html")
```

<iframe src="d3_multiclouds.html" width="100%" height="400" frameborder="0" style="padding:0;"></iframe>

All the customisation parameters listed above for `D3WordCloud` are available. Notice, however, that few minor differences. You input your data using the `data_sources` keyword. Since each source document can have its own title, you can supply these titles as a list with the `labels` parameter (if you do not provide this, generic titles "Doc 1", "Doc 2", etc. will be used). Likewise, you can specify the dimensions of individual clouds (in pixels) with the `cloud_width` and `cloud_height` parameters. Finally, you can set the number of columns in the layout.

To generate a D3 bubble chart, you use the `D3BubbleChart` class:

```python
from lexos.visualization.d3_bubbleviz import D3BubbleChart

bc = D3BubbleChart(data=text, title="D3 Bubble Chart")
bc.save("d3_bubblechart.html)
```

<iframe src="d3_bubblechart.html" width="100%" height="640" frameborder="0" style="padding:0;"></iframe>

Apart from the standard parameters, `D3BubbleChart` has two extra keywords for styling the chart.

- `margin`: A dictionary with the keys "top", "right", "bottom", and "left", used to configure the margin around the chart in pixels.
- `color`: The colour scheme for the chart, either the name D3 colour scheme or a list of custom colours. The default is "schemeCategory10". For other colour schemes, see the <a href="https://d3js.org/d3-scale" target="_blank">d3-scale</a> documentation.

### Customising D3 Visualisations

D3 visualisations are standalone web pages, so they must be viewed in the browser. There are additional parameters for all three classes that allow you to choose whether to include the D3 Javascript in the web page (leading to a bigger file) or download it from the internet (which means it will only work if you have an internet connection). See the API documentation for usage. In most cases, it is safe to leave the default setting and include the D3 Javascript in the web page.

The actual logic used to produce the visualisation is not loaded from the internet, and it is not minimised. This allows you to open the HTML file and modify the Javascript, as well as the CSS styling, if you are comfortable doing so.

!!! note "Developer's Note"
    The visualisations are designed for display as web pages. However, if you are planning to incorporate them in an application, you may want to make more extensive changes to incorporate the charts into your layout. Each visualisation is produced from an HTML template, which is populated with variables passed from Python. You can design your own template appropriate for your application and specify the path to your template with the `template` parameter.
