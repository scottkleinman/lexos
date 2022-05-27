Begin by importing in some modules in the Lexos API.

```python
from lexos.io.basic import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import scrubber_components, load_components
```

!!! warning
    If you are working in a Jupyter notebook and you cannot import the Lexos API modules, see the advice on the [installation][installation] page.

Here are some explanations what these modules do:

1. The `io` module contains IO functions. Right now, there is a "basic" Loader class that takes a file path, url, list of file paths or urls, or a directory name indicating where the source data is. More sophisticated loaders can be created later.
2. The `scrubber` module consists of thematic "components": `normalize`, `remove`, `replace`, and so on. Each component has a number of functions, such as converting to lower case, removing digits, stripping tags, etc.
3. Component functions are registered in a registry. They can be loaded into memory as needed and applied to texts in any order.

Loading and scrubbing comprise the main part of the "text preparation" portion of the Lexos workflow. Parsing and analytical functions have separate modules which will be dealt with later in this tutorial.
