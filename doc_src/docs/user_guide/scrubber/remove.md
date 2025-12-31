# Remove

## Overview

The `remove` component of Scrubber is a submodule containing functions for components removing strings and string patterns from text. This includes functions for removing diacritics, punctuation marks, digits, HTML and XML markup tags, and other items that are typically removed from texts prior to analysis. This page offers an overview of the component functions. For a fuller description of their usage, see the [API documentation](../../api/scrubber/remove.md).

!!! note
    The functions in the `remove` component can often be used to clean up texts prior to some other use, frequently because the content to be removed is undersirable for a downstream task. Note that this approach can be heavy handed because it depends on identifying string patterns and replacing them without regard to context. An alternative approach is to apply a language model to tokenize the text and then filter the tokens. This approach is discussed further in [Tokenizing Texts](../tokenizing_texts.md).

Here is a list of the functions available in the `remove` component:

- `accents`: Removes accented characters or replaces them with ASCII equivalents (similar to `normalize.unicode`).
- `brackets`: Removes square, curly, or round brackets (parentheses) from the text.
- `digits`: Removes digits from the text.
- `new_lines`: Removes all line-breaking spaces from the text.
- `pattern`: Truncates repeating characters to a specified maximum length.
- `project_gutenberg_headers`: Attempts to remove Project Gutenberg headers and footers from the text.
- `punctuation`: Removes all punctuation marks from textRemoves tabs from the text.
- `tabs`: Removes all punctuation marks from text (if you want to replace tabs with a single space, use `normalize.whitespace` instead).
- `tags`: Removes all HTML and XML markup tags from text. See the [`tags`](tags.md) component for a more nuanced approach to removing tags.

Many of these functions have parameters that allow you to specify how the function should behave. For example, the `accents` function has a `replace` parameter that allows you to specify whether accented characters should be replaced with their ASCII equivalents or simply removed. The `punctuation` function has a `keep` parameter that allows you to specify which punctuation marks should be retained in the text. See the [API documentation](../../api/scrubber/remove.md) for more details on the parameters available for each function.

## Example Using Direct Import

```python
from lexos.scrubber.remove import punctuation, digits

text = "Hello, world! 1234"

scrubbed_text = punctuation(digits(text))
print(scrubbed_text)
# "Hello world "
```

## Example Using in a Pipeline

The example below shows how the components can used in a Scrubber pipeline.

```python
from lexos.scrubber import Scrubber
from lexos.scrubber.remove import punctuation, digits

scrubber = Scrubber()
scrubber.add_pipe([punctuation, digits])

text = "Hello, world! 1234"

scrubbed_text = scrubber.scrub(text)
print(scrubbed_text)
# "Hello world "
```
