# Visualization

The Lexos `visualization` module is a container for assorted submodules which typically generate visualizations of document-term matrices. The following submodules are included:

## Bubble Visualizations

The [`bubbleviz`](bubbleviz.md/#static-bubble-visualizations) module produces bubble visualizations, also known as bubble charts using `matplotlib`. The [`d3_bubbleviz`](bubbleviz.md/#d3-bubble-visualizations) module renders interactive D3 bubble charts as HTML documents.

## Word Clouds

The [`cloud`](cloud.md/#static-word-clouds) module produces traditional word clouds using the Python `WordCloud` package and `matplotlib`. It also produces "multclouds" -- word clouds of multiple documents laid out in a grid. The [`d3_wordcloud`](cloud.md/#d3-word-clouds) module renders interactive D3-based word clouds and multi-cloud layouts as HTML documents.

## The `processors` Module

The [`processors`](processors.md) module contains helper functions for processing input data in various formats. Its functions are shared by the other two modules.
