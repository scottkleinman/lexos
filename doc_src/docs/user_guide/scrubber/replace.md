# Replace

## Overview

The `replace` component of Scrubber is a submodule containing functions for components replacing strings and string patterns from text. This includes functions for replacing diacritics, punctuation marks, digits, HTML and XML markup tags, and other items that are typically removed from texts prior to analysis. This page offers an overview of the component functions. For a fuller description of their usage, see the [API documentation](../../api/scrubber/replace.md).

Here is a list of the functions available in the `replace` component:

- `currency_symbols`: Replaces all currency symbols with a common symbol.
- `digits`: Replaces all digits with a common symbol.
- `emails`: Replaces all emails with a common symbol.
- `emojis`: Replaces all emojis with a common symbol.
- `hashtags`: Replaces all hashtags with a common symbol.
- `pattern`: Replaces all examples of a string pattern or patterns  with a common symbol.
- `phone_numbers`: Replaces all phone numbers with a common symbol.
- `punctuation`: Replaces punctuation marks with whitespace.
- `special_characters`: Replaces special characters following a dictionary of rules with an option to handle HTML and XML entities.
- `urls`: Replaces all urls with a common symbol.
- `user_handles`: Replaces all Twitter-style user handles with a common symbol.

Each function has parameters that allow you to specify how the function should behave. For example, the `pattern` function allows you to specify a regular expression pattern to match and replace in the text. The `punctuation` function allows you to specify which punctuation marks should be replaced with whitespace. The functions that replace patterns with a common symbol allow you to specify the common symbol, such as "_EMAIL_". See the [API documentation](../../api/scrubber/replace.md) for more details on the parameters available for each function.

!!! note
    The `replace` component can often perform the same or similar actions on text, so it is up to the user to decide which is appropriate. For instance, both components have `digits`, `pattern`, and `punctuation` functions. In order to disambiguate the function calls, it can be helpful to import these functions with aliases:

    ```python
    from lexos.scrubber.remove import punctuation as remove_punctuation
    from lexos.scrubber.replace import punctuation as replace_punctuation
    ```

## Example Using Direct Import

```python
from lexos.scrubber.replace import punctuation, digits

text = "Hello, world! 12"
scrubbed_text = punctuation(digits(text))

print(scrubbed_text)
# Hello  world  _DIGIT__DIGIT_
```

## Example Using in a Pipeline

The example below shows how the components can used in a Scrubber pipeline.

```python
from lexos.scrubber import Scrubber
from lexos.scrubber.replace import punctuation, re-digits

scrubber = Scrubber()
scrubber.add_pipe([punctuation, digits])

text = "Hello, world! 12"
scrubbed_text = scrubber.scrub(text)

print(scrubbed_text)
# Hello  world  _DIGIT__DIGIT_
```
