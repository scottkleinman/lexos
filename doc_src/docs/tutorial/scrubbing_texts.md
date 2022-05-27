## About Scrubber

Scrubber can be defined as a _destructive_ preprocessor. In other words, it changes the text as loaded in ways that potentially make mapping the results onto the original text potentially impossible. It is therefore best used before other procedures so that the scrubbed text is essentially treated as the "original" text. The importance of this will be seen below when we see the implementation of the tokeniser. But, to be short, the Lexos API differs from the web app in that Scrubber does not play a role in tokenisation by separating tokens by whitespace.

Scrubbing works by applying a single function or a pipeline of functions to the text. As a reminder, we need to load the scrubber components registry with `from lexos.scrubber.registry import scrubber_components, load_components`.

## Loading Scrubber Components
We can now load the components we want. We can load them individually, as in the first example below, or we can specify multiple components in a tuple, as in the second example. In both cases, the returned variable is a function, which we can then feed to a scrubbing pipeline.

```python
# Load a component from the registry
lower_case = scrubber_components.get("lower_case")

# Or, if you want to do several at once...
title_case, remove_digits = load_components(("title_case", "remove_digits"))
```

In the first example, a component is loaded using the registry's built-in `get` method. It is also possible to load a single component with the the [lexos.scrubber.registry.load_component][] helper method. This parallels [lexos.scrubber.registry.load_components][] for multiple components and is possibly easier to remember.

## Making a Pipeline

Now let's make the pipeline. We simply feed our component function names into the [make_pipeline()][lexos.scrubber.pipeline.make_pipeline] function in the order we want them to be implemented. Notice that `remove_digits` has to be passed through the [pipe()][lexos.scrubber.pipeline.pipe] function. This is because [lexos.scrubber.remove.digits][] requires extra arguments, and [pipe()][lexos.scrubber.pipeline.pipe] allows those arguments to be passed to the main pipeline function.

```python
# Make the pipeline
scrub = make_pipeline(
    lower_case,
    title_case,
    pipe(remove_digits, only=["1"])
)
```

The value returned is a function that implements the full pipeline when called on a text, as shown below.

```python
# Scrub the text
scrubbed_text = scrub("Lexos is the number 12 text analysis tool.")
```

This will return "Lexos Is The Number 2 Text Analysis Tool".

!!! Note
    You can also call component functions without a pipeline. For instance,`scrubbed_text = remove_digits("Lexos123", only=["2", "3"])` will return "Lexos1".

## Custom Scrubbing Components

The `title_case` function in the example above will not work because `title_case` is a custom component. To use it, we need to add it to the registry.

```python
# Define the custom function
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title()

# Register the custom function
scrubber_components.register("title_case", func=title_case)
```

Users can add whatever scrubbing functions they want. For development purposes, we can start by creating custom functions, and, if we use them a lot, migrate them to the permanent registry.

!!! important
    To use a custom scrubbing function, you must register it _before_ you call [lexos.scrubber.registry.load_component][] or [lexos.scrubber.registry.load_components][].
