# The Rolling Window Tool

Rolling window analysis is a method for tracing the frequency of terms within a designated window of tokens over the course of a document. It can be used to identify small- and large-scale patterns of individual features or to compare these patterns for multiple features. Rolling window analysis tabulates term frequency as part of a continuously moving metric, rather than in discrete segments. Beginning with the selection of a window, say 100 tokens, rolling window analysis traces the frequency of a term's occurrence first within tokens 1-100, then 2 to 101, then 3, 102, and so on until the end of the document is reached. The result can be plotted as a line graph so that it is possible to observe gradual changes in a token’s frequency as the text progresses. Plotting different tokens on the same graph allows us to compare their frequencies over the progression of a document.

!!! note
    Rolling Windows only works on a single document. If you would like to perform an analysis on a sequence of documents, you must first merge them into a single document.

## Basic Terminology

In a Rolling Window analysis, you search for specific *units* within *windows* of *n* units. A unit can be a character, a token (span of characters), or a span of tokens. If you are using a spaCy language model, spans can correspond to semantic units, such as words or sentences. Lexos can perform complex searches of units within different types of windows, so it is important to have in mind what kind of search you want to perform.

The first step is to generate a set of windows with the `Windows` class based on your desired units. You will then want to calculate how often your desired search terms occur in these windows. "How often" can be understood mathematically in a number of ways. The most obvious is a rolling count of the number of times a search term occurs in each window. But you can also understand this as the average number of times it occurs relative to the overall vocabulary, or the ratio of averages between two search terms. To obtain this information, you apply a calculator class. Lexos has three built-in calculator classes `Counts`, `Averages`, and `Ratios`, but you can also add custom calculator classes if the built-in ones do not provide the information you need.

The results of the calculations can be viewed in a Pandas DataFrame or CSV file, but it can be useful to visualize them on a line graph so that you can see peaks and valleys in the occurrence of your search terms. Lexos provides built-in plotter classes to do this. The `SimplePlotter` class produces static images, and the `PlotlyPlotter` class produces interactive ones. The latter may be better for presentations and the latter can sometimes be easier to read.

## Generating Windows

When creating windows, you can use data as `input` in a number of formats:

- A string or list of strings (the latter might correspond words)
- A spaCy Doc or Span object
- A list of spaCy Token or Span objects

You will need to specify the size of your window (`n`) and the type of unit used (`window_type`): "characters", "spans", or "tokens". In other words, if you choose `n=1000`, that will produce windows of 1000 characters, spans, or tokens, dependingo on what you choose for `window` type. Finally, you must choose an `output` type, which can be "strings" or "tokens".

Here is a sample setup.

```python
# Import the Windows class
from lexos.rolling_windows import Windows

# Assume that you have a raw text string
windows = Windows(input=text, n=1000, window_type="characters", output="strings")
```

This will produce windows where each window is 1000 characters long. Here's another example:

```python
# Assume that you have a spaCy Doc
windows = Windows(input=doc, n=1000, window_type="tokens", output="tokens")
```

This will produce windows where each window is 1000 tokens long. If you chance `output` to strings, each window will be the character length of a window of 1000 tokens. In other words, since not all tokens are the same length in chararacters, the length of your windows will not be consistent.

!!! note
    Some combinations of `input`, `window type`, and `output` are not possible. If Lexos cannot combine your choices, it will raise an error.

You will then need to generate the instance with `windows()`. Alternatively, you can configure your `Windows` object and generate the windows at the same time like this:

```python
# Assume that you have a spaCy Doc
windows = Windows()
windows(input=doc, n=1000, window_type="tokens", output="tokens")
```

You can view the windows with `list(windows)`.

!!! important
    The `Windows` class creates a generator, not a list of windows. If you try to access it in any way, the generator will be exhausted, and you will not be able to access them again. In this case, you will have to call `windows()` again to re-create the windows.

### Choosing Window Type and Size

The are no hard and fast rules to how to select window types and sizes. To some degree, it will depend on your data type and research question. It is always best to try several options to see what is most revealing. The following general guidelines may be helpful.

| Doc  Type | Window Type | Suggested Size | Reasoning |
|-----------|-------------|----------------|-----------|
| Short story | characters | 200-500 | May capture local patterns |
| Novel/Book | characters | 500-2000 | May balance detail and trends |
| Short text | tokens | 20-50 | Enough words for patterns |
| Novel/Book | tokens | 50-200 | May captures thematic shifts |
| Poetry | tokens (lines) | 5-20 | May respects verse structure |

### Aligning Window Boundaries to Token Boundaries

Imagine you had a spaCy Doc object from which you wanted to generate windows of characters. Since windows simply slide one unit at a time, in most cases window boundaries will not correspond to token boundaries. To make this clearer, consider a simple document with five tokens: "The", "cat", "in", "the", "hat". If we wanted windows of three tokens, the first window would be "The", but the second would be "he c". Perhaps this does not matter for your analysis, but, if it does, Lexos provides the possibility of aligning your windows on token boundaries with the `alignment_mode` keyword. "Strict" alignment (the default) behaves exactly as we have just observed, but, if you set `alignment_mode` to `None`, Lexos will attempt to snap the windows to token boundaries. It is recommended that you play with different alignment modes and examine the results to make sure they are satisfactory before you continue your analysis.

## Using Calculators

We can now import a calculator and perform a search on our windows. Each search term is considered a "pattern", which we can pass as a list using the `patterns` keyword.

```python
from lexos.rolling_windows.calculators.counts import Counts

# Using the `windows` object previously created
calculator = Counts(
    windows=windows,
    patterns=["a", "e"],
    case_sensitive=False,
    mode="exact"
)
```

Here we are searching for the patterns "a" and "e" in our windows. Set `case_sensitive` to `True` if you want the search to distinguish between "A" and "a".

Lexos can search for patterns using five different modes. The default "exact" mode will search for the exact character pattern in each unit in the window. The "regex" mode will interpret your pattern as a regular expression. In other words, if your window contains "The end", a search for "e." will not find a match in "exact" mode. In "regex" mode, it will find that the pattern occurs twice, once followed by a space and once followed by "n". The "spacy_rule" setting employs spaCy's `Matcher` class, which allows you to set up powerful rules to do all sorts of complex token matching. The "multi_token" mode allows you to search for sequences of tokens without matching the exact token boundaries (for this, use "multi_token_exact"). It is recommended that you play with these settings to determine what best suits your research question.

!!! "Technical Note"

    - `exact` mode uses Python's string `count()` method.
    - `regex` mode uses `re.findall()` with appropriate flags.
    - `spacy_rule` mode requires spaCy Token objects, not strings.

If you wish to see the numerical data produced by the calculator, use the `to_df()` method to display it as a Pandas DataFrame. You can also use Pandas to save it to a CSV file (or other format), as shown below:

```python
calculator = Counts(
    windows=windows,
    patterns=["a", "e"],
    case_sensitive=False,
    mode="exact"
)
df = calculator.to_df()
df.to_csv("filename.csv")
```

The `Averages` and `Ratios` calculators work in a similar fashion. Rolling averages provides a useful statistic if you have windows of different sizes or you need a normalized frequency statistic.

Rolling ratios compares the frequencies of exactly **two** patterns. A statistic of `0.0` means that only the second pattern appears in a window. `0.5` means that both patterns appear equally. `1.0` means that only the first pattern appears in the window. Values closer 0 favor the second pattern. Values closer to 1 favor the first pattern.

## Plotting the Results

Once you have a pandas DataFrame, you can also use the pandas interface to save it, for instance, as a CSV file: `calculator.to_df().to_csv("filename")`. You can also the built-in pandas plotting function to generate charts based on the results. See <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html" target="_blanket">pandas.DataFrame.plot</a> for further information.

### Plotter Classes

Lexos has its own built-in plotting classes, `SimplePlotter` and `PlotlyPlotter`, which are specifically designed for the type of data produced by Rolling Windows. Each of the plotter classes has numerous parameters you can use to change the appearance of the plot, including the size of the image, title, and access label. Only some of the options are shown below. See the API documentation for explanations of all of the available parameters.

!!! note "Developer's Note"
    If one of these plotters does not suit your need, you can also write your own custom calculator class that inherits from `BasePlotter`.

### `SimplePlotter`

The `SimplePlotter` class generates high-quality static plots suitable for publications using Python's `matplotlib` library. This well-suited for reports, publications, presentations.

```python
from lexos.rolling_windows.plotters import SimplePlotter

plotter = SimplePlotter(title="Word Frequencies Over Time")
plotter.plot(averages.to_df())
```

To save the plot use the `save()` method:

```python
plotter.save("filename.png")
```

Files can be saved in `.jpg`, `.svg`, and `.pdf` formats by changing the file extension.

### `PlotlyPlotter`

The `PlotlyPlotter` class generates interactive web-based visualizations with hover tooltips and zoom capabilities. `PlotlyPlotter` is best for exploration, web presentation, detailed analysis with hover information.

```python
from lexos.rolling_windows.plotters import PlotlyPlotter

interactive_plotter = PlotlyPlotter()
interactive_plotter.plot(averages.to_df(), show_plot=True)
```

Hovering over the top of the diagram displays a menu bar with the following interactive features:

- **Hover** over points to see exact values
- **Zoom** in/out with mouse wheel or zoom controls
- **Pan** by clicking and dragging
- **Toggle** lines on/off by clicking legend items
- **Download** plot as PNG using the camera icon

### Using Milestones

When tracing rolling windows patterns over the course of a document, it can be useful to see how the patterns occur with respect to structural divisions in the text. For instance, if you are analyzing a novel, you may wish to know the changes in the rolling average over a feature in each chapter. The point of transition between one section of the text and the next is known as a "milestone". Both rolling windows plotters allow you to display milestones using `show_milestones=True`, which places a vertical line on the graph at the milestone location. If `show_milestone_labels=True`, the graph will additionally display labels such as "chapter", "scene", "line", etc. above each line.

You can manually create milestones dictionary consisting of labels and token locations (which will be character locations, if your windows are composed of characters). An example is shown below:

```python
milestones = {"Chapter 1": 0, "Chapter 2": 500, "Chapter 3": 1000}
plotter = SimplePlotter(
    milestone_labels=milestones
    show_milestones=True,
    show_milestone_labels=True,
    milestone_label_rotation=45
)
```

!!! note
    The Lexos `milestones` module allows you to detect milestone locations and create a milestone dictionary. This can be much easier than constructing a milestone dictionary yourself.
