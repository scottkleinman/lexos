## About Scrubber

Scrubber can be defined as a _destructive_ preprocessor. In other words, it changes the text as loaded in ways that potentially make mapping the results onto the original text potentially impossible. It is therefore best used before other procedures so that the scrubbed text is essentially treated as the "original" text. The importance of this will be seen below when we see the implementation of the tokeniser. But, to be short, the Lexos API differs from the web app in that Scrubber does not play a role in tokenisation by separating tokens by whitespace.

Scrubbing works by applying a single function or a pipeline of functions to the text. As a reminder, we need to load the scrubber components registry with `from lexos.scrubber.registry import scrubber_components, load_components`.

## Scrubber Components
Scrubber components are divided into three categories:   
1. [Normalize](https://scottkleinman.github.io/lexos/api/scrubber/normalize/) components are used to manipulate text into a standardized form.   
2. [Remove](https://scottkleinman.github.io/lexos/api/scrubber/remove/) components are used to remove strings and patterns from text.   
3. [Replace](https://scottkleinman.github.io/lexos/api/scrubber/replace/) components are used to replace strings and patterns in text.   

Follow these links to view all of the default scrubber components.

## Loading Scrubber Components
Components must be loaded before they can be used. We can load them individually, as in the first example below, or we can specify multiple components in a tuple, as in the second example. In both cases, the returned variable is a function, which we can then feed to a scrubbing pipeline.

```python
# Load a single component from the registry
lower_case = scrubber_components.get("lower_case")
#or
#lower_case = load_component("lower_Case")

# Or, if you want to do several at once...
punctuation, remove_digits = load_components(("punctuation", "remove_digits"))
```

In the first example, a component is loaded using the registry's built-in `get` method. It is also possible to load a single component with the the [lexos.scrubber.registry.load_component][] helper method. This parallels [lexos.scrubber.registry.load_components][] for multiple components and is possibly easier to remember.

## Using Components
Loaded component functions can be called like any normal function. For example:
`scrubbed_text = remove_digits("Lexos123", only=["2", "3"])` will return "Lexos1".

If you are intending to apply multiple components to a single piece of text, the more efficient method is to use a pipeline.

## Making a Pipeline

Now let's make the pipeline. We simply feed our component function names into the [make_pipeline()][lexos.scrubber.pipeline.make_pipeline] function in the order we want them to be implemented. Notice that `remove_digits` has to be passed through the [pipe()][lexos.scrubber.pipeline.pipe] function. This is because [lexos.scrubber.remove.digits][] requires extra arguments, and [pipe()][lexos.scrubber.pipeline.pipe] allows those arguments to be passed to the main pipeline function.

```python
# Make the pipeline
scrub = make_pipeline(
    lower_case,
    punctuation,
    pipe(remove_digits, only=["1"])
)
```

The value returned is a function that implements the full pipeline when called on a text, as shown below.

```python
# Scrub the text
scrubbed_text = scrub("Lexos is the number 12 text analysis tool!!")
```

This will return "lexos is the number 2 text analysis tool"

## Custom Scrubbing Components

Users can write and use custom scrubbing functions. The function is written like a normal function, and to use it like a scrubber component it must be added to the registry. Below is an example with a custom `title_case` function.

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
