# Normalize

## Overview

The `normalize` component of Scrubber is a submodule containing functions for manipulating text into a standardized form. This includes functions for converting text to lower case, removing whitespace, and more. This page offers an overview of the component functions. For a fuller description of their usage, see the [API documentation](../../api/scrubber/normalize.md).

A classic problem in text processing is that the same text string can be represented in many different ways. For example, the single quotation mark, or apostrophe, may be straight (') or curly (‘ or ’), depending on the surrounding characters. A text in the wild may contain either one or a mixture of the two, which may not be desirable for a target publication. However, it may also have significant downstream consequences such as breaking XML code or influencing token counts used in computational analysis. The `normalize` component contains a functions that allow you to convert variants like these into a single desired (normalized) form.

Here is a list of the functions available in the `normalize` component:

- `bullet_points`: Converts all types of bullet points to a hyphen (-).
- `hyphenated_words`: Joins words split by a hyphen across line breaks.
- `lower_case`: Converts all text to lower case.
- `quotation_marks`: Converts all types of single and double quotation marks to straight quotes.
- `repeating_chars`: Truncates repeating characters to a specified maximum length.
- `unicode`: Converts Unicode characters in `text` into canonical forms (for instance, by changing "é" to plain ASCII "e").
- `whitespace`: Strips leading and trailing whitespace from text and replaces multiple spaces with a single space.

## Example Using Direct Import

```python
from lexos.scrubber.normalize import lower_case, whitespace

text = "  This is a Sample Text with   Irregular Whitespace.  "

normalized_text = whitespace(lower_case(text))
print(normalized_text)
# "this is a sample text with irregular whitespace.
```

## Example Using in a Pipeline

The example below shows how the components can used in a Scrubber pipeline.

```python
from lexos.scrubber import Scrubber
from lexos.scrubber.normalize import lower_case, whitespace

scrubber = Scrubber()
scrubber.add_pipe([lower_case, whitespace])

text = "  This is a Sample Text with   Irregular Whitespace.  "

normalized_text = scrubber.scrub(text)
print(normalized_text)
# this is a sample text with irregular whitespace.
```
