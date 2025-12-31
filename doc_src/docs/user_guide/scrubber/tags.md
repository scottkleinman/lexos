# Tags

## Overview

The `tag` component of Scrubber is a submodule containing functions for transforming HTML and XML content (if you simply wish to remove all tags, using `remove.tags` is more efficient). These markup languages wrap content into **elements** indicated by angular brackets. Each element can further contain **attributes**, the **value** of which is contained within quotation marks. For example, the markup `<span id="1">John Smith</span>` indicates that the content "John Smith" is a `span` element with an `id` attribute, the value of which is "1". The functions in Scrubber's tag component allow you to manipulate the elements, attributes, and content by choosing a **selector** (usually the name of an element) and providing filters if only certain occurrences of this selector should be changed. This allows for some fairly nuanced transformations to be applied to the markup. For instance, in many cases Scrubber could be used to transform XML content into HTML markup for presentation on the web. This page offers an overview of the component functions. For a fuller description of their usage, see the [API documentation](../../api/scrubber/tags.md).

Here is a list of the functions available in the `replace` component:

- `remove_attribute`: Removes all instances of an attribute in a specified element.
- `remove_comments`: Removes all comments from the text.
- `remove_doctype`: Removes the HTML document type declaration or XML declaration from the text.
- `remove_element`: Removes a specified HTML or XML element from the text, including the tag's content.
- `remove_tag`: Remove a tag from the text but retain the tag's content.
- `replace_attribute`: Replaces the value of an attribute with another value.
- `replace_tag`: Replaces a tag with anther tag.

Each function has parameters that allow you to specify how the function should behave. For example, you can specify which elements to target, whether to include or exclude certain attributes, and how to handle the content of the elements. The functions can be used individually or in combination to achieve complex transformations on your text. See the [API documentation](../../api/scrubber/tags.md) for more details on the parameters available for each function.

In order to keep the functions simple, each function call can perform only a single transformation. To perform multiple transformations, it is necessary to call functions multiple times (or construct a pipeline that calls the functions iteratively). You may have to get to know your text's markup fairly well or inspect the results after each transformation in order to achieve the desired effects. When calling `tags` functions multiple times, the order in which the functions are called can make a considerable difference in the output.

!!! note
    Under the hood, Lexos uses the Python <a href="https://beautiful-soup-4.readthedocs.io/en/latest/" target="_blank">BeautifulSoup</a> library to parse and transform the string before returning the output as a new string.

## Removing Elements and Replacing Tag Names

```python
from lexos.scrubber.tags import remove_element, replace_tag

text = "<p>Hello World</p><span>Hello again!</span>"

# Remove <p>
scrubbed_text = remove_element(text, selector="p")

# Replace <span> with <p>
scrubbed_text = replace_tag(scrubbed_text, selector="span", replacement="p")

print(scrubbed_text)
# <p>Hello again!</p>
```

!!! note
    Setting `selector=None` will target all elements in the document.

The `remove_element` function removes the element and all of its content. The `replace_tags` function replaces an element's tag with a different tag. Both functions have a `mode` parameter set to "html" by default. To parse the text as "xml", set `mode="xml"`. You can also set the `matcher_type`. By default, it is "exact", which means that the selector "p" will match all `p` elements. However, setting `matcher_type=regex` uses regular expressions to perform the matching. In this example, using the selector "p" will also target `span` elements (or any other elements with tag names containing the letter *p*).

Elements can also be targeted based on their attributes by setting the `attribute` and `attribute_value` parameters:

```python
text = "<p id="1">Hello World</p><p id="2">Hello again!</p>"

scrubbed_text = remove_element(text, selector="p", attribute="id", attribute_value="1)

print(scrubbed_text)
# <p id="2">Hello again!</p>
```

The `replace_tag` function also has a boolean `preserve_attributes` parameter, which allows you to choose whether or not to keep an elements attributes when changing the tag name.

## Removing and Replacing Attributes

To remove or change the attribute values of an element, you can use the `remove_attribute` and `replace_attribute` functions.

```python
from lexos.scrubber.tags import remove_attribute, replace_attribute

text = '<div class="main">Text</div>'

scrubbed_text = remove_attribute(text, selector="div", attribute="class")

print(scrubbed_text)
# <div>Text</div>
```

Additionally, you can filter which attributes to target with the `attribute_filter` and `attribute_value` parameters. In the example below, we remove the `class` attribute only when it has the value "remove":

```python
text = '<div class="keep">Text</div><div class="remove">Text</div>'

scrubbed_text = remove_attribute(text, selector="div", attribute_filter="class", attribute_value="remove")

print(scrubbed_text)
# <div class="keep">Text</div><div>Text</div>
```

The `replace_attribute` function is somewhat more complex. It has the following parameters:

- `old_attribute`: The name of the attribute to replace
- `new_attribute`: The name of the new attribute (or same name if only changing the value)
- `attribute_value`: Only replace attributes with this specific value
- `replace_value`: The new value to use (keeps original value if `None`)
- `attribute_filter`: Optional attribute name to filter elements
- `filter_value`: Optional value for the attribute filter

Here are some examples of how to use these parameters:

```python
# Replace class attribute with data-type, keeping the value
text = '<div class="main">Text</div>'
replace_attribute(text, "div", "class", "data-type")
# <div data-type="main">Text</div>

# Replace class="info" with class="highlight"
text = '<p class="info">Text</p><p class="data">More</p>'
replace_attribute(text, "p", "class", "class", filter_value="info", replace_value="highlight")
# <p class="highlight">Text</p><p class="data">More</p>

# Only replace attributes on elements with a specific attribute value
text = '<div class="main" id="content">Text</div><div class="sidebar">Side</div>'
replace_attribute(text, "div", "class", "role", attribute_filter="id", filter_value="content")
# <div role="main" id="content">Text</div><div class="sidebar">Side</div>
```

As with `remove_element` and `replace_tag`, both functions allow you can use the `mode` and `matcher_type` parameters to perform more complex parsing of your document

## Using Tags Functions in a Scrubber Pipeline

The example below shows how the components can used in a Scrubber pipeline. The partial functions are created for clarity so that you can see what each function does. In practice, you could also use the functions directly in the pipeline without creating partials.

```python
from functools import partial
from lexos.scrubber import Scrubber
from lexos.scrubber.replace import remove_element, replace_tag

# Create partial functions for specific transformations
remove_p_by_class_value = partial(remove_element, attribute="class", value="remove")
replace_span = partial("span", "p")

# Create Scrubber pipeline and add the partial functions
scrubber = Scrubber()
scrubber.add_pipe([remove_p_by_class_value, replace_span])

# Pass the text through the pipeline and print the scrubbed result
text = "<p>Hello world</p><p class="remove">Hello World</p><p class="change">Hello, world!</p>"
scrubbed_text = scrubber.scrub(text)
print(scrubbed_text)
# <span>Hello world</span><span class="change">Hello, world!</span>
```
