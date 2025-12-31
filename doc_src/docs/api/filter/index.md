# Filter

The [`filter`](filter.md) module provides a base class for applying filters to a document and returning a new document, as well as extracting tokens or ids form filtered docs. Filters are applied by passing a spaCy matcher to identify matches to the filter criteria. Since the doc's language model may not supply the required token attributes, you can create a custom filter to add and set the attributes as custom extensions. Examples of custom filter classes are `IsRomanFilter` and `IsWordFilter`.

The module also provides a useful `StopwordFilter` class to add or remove stop words from a spaCy `Doc` without retokenising. Note, however, that it works by changing the model's defaults, so they will apply to any `Doc` created with that model unless the model is reloaded.
