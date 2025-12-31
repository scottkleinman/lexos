# Scrubber

## Overview

The Scrubber module provides a flexible, pipeline-based system for text cleaning and normalization as part of the Lexos project. It enables users to preprocess text by applying a customizable sequence of "scrubber components" (pipes) to remove, replace, or normalize elements such as punctuation, digits, whitespace, and more.

## Features

- Modular pipeline for text scrubbing
- Built-in registry of reusable scrubber components
- Easy addition and removal of pipeline components
- Support for custom components and configuration
- Integration with other Lexos modules
- Batch processing of texts via generator interface
- Robust error handling

## Submodules

### [Normalize](normalize.md)

The **normalize** submodule contains functions to normalize all [bullet points](normalize.md/#lexos.scrubber.normalize.bullet_points), [hyphenated words](normalize.md/#lexos.scrubber.normalize.hyphenated_words), [letters](normalize.md/#lexos.scrubber.normalize.lower_case) (to lowercase), [quotation marks](normalize.md/#lexos.scrubber.normalize.quotation_marks), [repeating characters](normalize.md/#lexos.scrubber.normalize.repeating_chars), [unicode](normalize.md/#lexos.scrubber.normalize.unicode), and [whitespace](normalize.md/#lexos.scrubber.normalize.whitespace) by replacing them with more standardized characters.

### [Pipeline](pipeline.md)

The **pipeline** submodule allows the user to create a pipeline which calls functions from the other submodules in a specific order.

### [Registry](registry.md)

The **registry** submodule contains functions [`get_component`](registry.md/#lexos.scrubber.registry.get_component) and [`get_components`](registry.md/#lexos.scrubber.registry.get_component) to get one component from a string, or multiple from a tuple, respectively.

### [Remove](remove.md)

The **remove** submodule contains functions to remove [accents](remove.md/#lexos.scrubber.remove.accents), all [brackets](remove.md/#lexos.scrubber.remove.brackets) ( ) [ ] { } and the text within them, [digits](remove.md/#lexos.scrubber.remove.digits), [new_lines](remove.md/#lexos.scrubber.remove.new_lines), given regex a [pattern](remove.md/#lexos.scrubber.remove.pattern), [Project Gutenberg headers](remove.md/#lexos.scrubber.remove.project_gutenberg_headers), [punctuation](remove.md/#lexos.scrubber.remove.punctuation), [tabs](remove.md/#lexos.scrubber.remove.tabs), and [tags](remove.md/#lexos.scrubber.remove.tags).

### [Replace](replace.md)

The **replace** submodule contains functions which replace [currency symbols](replace.md/#lexos.scrubber.replace.currency_symbols), [digits](replace.md/#lexos.scrubber.replace.digits), [emails](replace.md/#lexos.scrubber.replace.emails), [emojis](replace.md/#lexos.scrubber.replace.emojis), [hashtags](replace.md/#lexos.scrubber.replace.hashtags), given a regex [pattern](replace.md/#lexos.scrubber.replace.pattern), [phone numbers](replace.md/#lexos.scrubber.replace.phone_numbers), [punctuation](replace.md/#lexos.scrubber.replace.punctuation), [special characters](replace.md/#lexos.scrubber.replace.special_characters), [urls](replace.md/#lexos.scrubber.replace.urls), and [user handles](replace.md/#lexos.scrubber.replace.user_handles) with a string of the form `_TYPE_`.

### [Resources](resources.md)

The **resources** submodule contains the [HTMLTextExtractor](resources.md/#lexos.scrubber.resources.HTMLTextExtractor) class, a subclass of <a href="https://docs.python.org/3/library/html.parser.html" target="_blank">html.parser.HTMLParser</a>.

### [Scrubber](scrubber.md)

The **scrubber** submodule contains the main logic for the Scrubber module. It contains the [`Pipe`](scrubber.md/#lexos.scrubber.scrubber.Pipe) dataclass and the [`Scrubber`](scrubber.md/#lexos.scrubber.scrubber.Scrubber) class. The Pipe class contains only a call method and the Scrubber class contains an initialization method along with methods [`add_pipe`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.add_pipe), [`pipe`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.pipe), [`remove_pipe`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.remove_pipe), [`reset`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.reset), and [`scrub`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.scrub). The Scrubber class also contains the attribute [`pipes`](scrubber.md/#lexos.scrubber.scrubber.Scrubber.pipes) which returns a list of the pipeline components. The submodule also includes the function [`scrub`](scrubber.md/#lexos.scrubber.scrubber.scrub) which takes in the text to scrub, the pipeline, and the optional factory and returns the scrubbed text

### [Tags](tags.md)

The **tags** submodule uses <a href="https://www.crummy.com/software/BeautifulSoup/" target="_blank">Beautiful Soup</a> for several functions to [remove attributes](tags.md/#lexos.scrubber.tags.remove_attribute), [remove comments](tags.md/#lexos.scrubber.tags.remove_comments), [remove doctypes](tags.md/#lexos.scrubber.tags.remove_doctype), [remove elements](tags.md/#lexos.scrubber.tags.remove_element), [remove tags](tags.md/#lexos.scrubber.tags.remove_tag), [replace attributes](tags.md/#lexos.scrubber.tags.replace_attribute), and [replace tags](tags.md/#lexos.scrubber.tags.replace_tag) in HTML and XML files.

### [Utils](utils.md)

The **utils** submodule contains the function [`get_tags`](utils.md/#lexos.scrubber.utils.get_tags).
