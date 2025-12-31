# Visualization

The Lexos `visualization` module is a container for assorted submodules which typically generate visualizations of document-term matrices. The following submodules are included:

## `bubbleviz`

The [`bubbleviz`](bubbleviz.md) module produces bubble visualizations, also known as bubble charts using `matplotlib`.

## `cloud`

The [`cloud`](cloud.md) module produces traditional word clouds using the Python `WordCloud` package and `matplotlib`. It also produces "multclouds" -- word clouds of multiple documents laid out in a grid.

## `processors`

The [`processors`](processors.md) module contains helper functions for processing input data in various formats. Its functions are shared by the other two modules.
