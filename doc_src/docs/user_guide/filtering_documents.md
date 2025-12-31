# Filtering Tokens

## Overview

The `filter` module provides a set of tools for filtering and identifying tokens within spaCy `Doc` objects. Filters allow you to identify specific types of tokens (such as words, Roman numerals, or stop words) and work with the matched results. This is useful for preprocessing, analysis, and text transformation tasks.

### Basic Concepts

Lexos filters are built around the concept of matching tokens based on specific criteria. Each filter accepts a spaCy `Doc` object as input and outputs either a modified `Doc` or lists of matched token IDs.

### Basic Usage

The basic procedure for using filters is as follows:

```python
# Import the filter class
from lexos.filter import IsWordFilter

# Create a spaCy doc from your text
from lexos.tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")
text = "Hello, world! This is a test."
doc = tokenizer.make_doc(text)

# Create an instance of a filter
word_filter = IsWordFilter()

# Apply the filter to the doc
filtered_doc = word_filter(doc)
```

## Filter Classes

### BaseFilter

The `BaseFilter` class is the foundation for all filters. It provides common functionality for matching tokens using spaCy matchers.

#### Key Properties

- `matched_token_ids`: Returns a set of token IDs that matched the filter criteria
- `matched_tokens`: Returns a list of tokens that matched the filter criteria
- `filtered_token_ids`: Returns a set of token IDs that did NOT match the filter criteria
- `filtered_tokens`: Returns a list of tokens that did NOT match the filter criteria

#### Key Methods

- `get_matched_doc()`: Creates a new spaCy `Doc` containing only the matched tokens
- `get_filtered_doc()`: Creates a new spaCy `Doc` containing only the filtered (non-matched) tokens

Filtering tokens with `BaseFilter` methods can cause the neighbouring tokens to run together in the new document. You can use the `add_spaces` parameter in the above methods to insert spaces between tokens in the new document to prevent this.

### IsWordFilter

The `IsWordFilter` class identifies tokens that are words (as opposed to punctuation, whitespace, or other non-word tokens).

#### Parameters

- `attr` (optional): The name of the custom token attribute to add (default: "is_word")
- `default` (optional): The default value for the custom attribute (default: False)
- `exclude`: A string pattern or list of string patterns to exclude from being considered words (default: [" ", "\n"])
- `exclude_digits`: If True, numeric tokens will not be treated as words (default: False)
- `exclude_roman_numerals`: If True, Roman numerals (capital letters only) will not be treated as words (default: False)
- `exclude_pattern`: Additional regex pattern or list of regex patterns to exclude

#### Example

```python
# Python imports
from lexos.filter import IsWordFilter
from lexos.tokenizer import Tokenizer

# Create a spaCy doc from your text
tokenizer = Tokenizer(model="en_core_web_sm")
text = "Hello, world! 123 and IV are not words."
doc = tokenizer.make_doc(text)

# Create a word filter that excludes digits
word_filter = IsWordFilter(exclude_digits=True, exclude_roman_numerals=True)
filtered_doc = word_filter(doc)

# Access matched words
matched_words = word_filter.matched_tokens
print([token.text for token in matched_words])
# Output: ['Hello', 'world', 'and', 'are', 'not', 'words']

# Get a new doc with only words
words_only_doc = word_filter.get_matched_doc()
```

### IsRomanFilter

The `IsRomanFilter` class identifies tokens that are Roman numerals (capital letters only).

The class has the following attributes:

- `attr` (str, optional): The name of the custom token attribute to add
- `default` (any, optional): The default value for the custom attribute

For example:

```python
# Python imports
from lexos.filter import IsRomanFilter
from lexos.tokenizer import Tokenizer

# Create a spaCy doc from your text
tokenizer = Tokenizer(model="en_core_web_sm")
text = "Chapter IV begins here. Not iv, but IV."
doc = tokenizer.make_doc(text)

# Create a Roman numeral filter
roman_filter = IsRomanFilter(attr="is_roman")
result_doc = roman_filter(doc)

# Access matched Roman numerals
roman_numerals = roman_filter.matched_tokens
print([token.text for token in roman_numerals])
# Output: ['IV', 'IV']
```

### IsStopwordFilter

The `IsStopwordFilter` class manages stop words in a spaCy model. Stop words are common words that are often filtered out during text processing (such as "the", "a", "is", etc.).

!!! important
    This filter modifies the model's default stop words. Changes will apply to any document created with that model unless the model is reloaded.

The class has the following attributes:

- `stopwords` (list[str] | str, optional): A list or string containing the stop word(s) to add or remove
- `remove` (bool, optional): If True, the specified stop words will be removed from the model. If False, they will be added (default: False)
- `case_sensitive` (bool, optional): If False (default), stop word changes apply to all case variations (lowercase, original, and capitalized). If True, only the exact case provided is modified (default: False)

Here are some examples:

**Adding Stop Words:**

```python
# Python imports
from lexos.filter import IsStopwordFilter
from lexos.tokenizer import Tokenizer

# Create tokenizer and doc
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("The quick brown fox jumps over the lazy dog.")

# Add custom stop words
stopword_filter = IsStopwordFilter()
result_doc = stopword_filter(doc, stopwords=["quick", "brown"], remove=False)

# Now "quick" and "brown" are marked as stop words in any docs created with this tokenizer
doc2 = tokenizer.make_doc("The quick brown fox")
for token in doc2:
    if token.is_stop:
        print(f"'{token.text}' is a stop word")
```

**Removing Stop Words:**

```python
# Python imports
from lexos.filter import IsStopwordFilter
from lexos.tokenizer import Tokenizer

# Create tokenizer
tokenizer = Tokenizer(model="en_core_web_sm")

# Create a doc
doc = tokenizer.make_doc("The quick brown fox jumps over the lazy dog.")

# Remove "the" from stop words (case-insensitive by default)
stopword_filter = IsStopwordFilter()
result_doc = stopword_filter(doc, stopwords="the", remove=True)

# Now "the" and "The" are no longer marked as stop words in any docs created with this tokenizer
doc2 = tokenizer.make_doc("The quick brown fox")
for token in doc2:
    if token.text.lower() == "the":
        print(f"'{token.text}' is_stop: {token.is_stop}")  # Output: is_stop: False for both
```

**Case-Sensitive Stop Word Removal:**

```python
# Remove only lowercase "the" (case-sensitive)
stopword_filter = IsStopwordFilter()
result_doc = stopword_filter(doc, stopwords="the", remove=True, case_sensitive=True)

# Now lowercase "the" is not a stop word, but capitalized "The" still is
doc2 = tokenizer.make_doc("The quick brown fox. See the dog.")
for token in doc2:
    if token.text.lower() == "the":
        print(f"'{token.text}' is_stop: {token.is_stop}")
# Output: 'The' is_stop: True, 'the' is_stop: False
```

## Working with Matched Tokens

After applying a filter, you can access and work with the results in several ways:

### Accessing Token IDs

```python
word_filter = IsWordFilter()
word_filter(doc)

# Get the IDs of matched tokens
matched_ids = word_filter.matched_token_ids
print(f"Matched token IDs: {matched_ids}")

# Get the IDs of filtered-out tokens
filtered_ids = word_filter.filtered_token_ids
print(f"Filtered token IDs: {filtered_ids}")
```

### Creating New Documents

```python
# Create a document with only matched tokens
matched_doc = word_filter.get_matched_doc()
print(matched_doc.text)

# Create a document with only filtered-out tokens
filtered_doc = word_filter.get_filtered_doc()
print(filtered_doc.text)
```

### Accessing Token Objects

```python
# Get the actual token objects
matched_tokens = word_filter.matched_tokens
for token in matched_tokens:
    print(f"{token.text}: POS={token.pos_}, LEMMA={token.lemma_}")

# Get tokens that were filtered out
filtered_tokens = word_filter.filtered_tokens
for token in filtered_tokens:
    print(f"Excluded: {token.text}")
```

## Custom Token Attributes

When you apply a filter with a custom attribute name, the filter adds a custom extension to each token. This allows you to programmatically check whether a token matches the filter criteria:

```python
from lexos.filter import IsWordFilter
from lexos.tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("Hello, world!")

word_filter = IsWordFilter(attr="is_word")
filtered_doc = word_filter(doc)

# Check the custom attribute on each token
for token in filtered_doc:
    print(f"{token.text}: is_word={token._.is_word}")
# Output:
# Hello: is_word=True
# ,: is_word=False
# world: is_word=True
# !: is_word=False
```

## Advanced Patterns

### Excluding Specific Token Types

```python
word_filter = IsWordFilter(
    exclude_digits=True,           # Exclude numeric tokens
    exclude_roman_numerals=True,   # Exclude Roman numerals
    exclude_pattern=r"[^\w\s]"     # Exclude special characters
)
filtered_doc = word_filter(doc)
```

### Combining Multiple Filters

You can apply multiple filters sequentially to progressively refine your results:

```python
from lexos.filter import IsWordFilter, IsRomanFilter
from lexos.tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("Chapter IV: The quick brown fox (123) jumps.")

# First, filter out non-words
word_filter = IsWordFilter(exclude_digits=True, exclude_roman_numerals=True)
doc = word_filter(doc)
words_only = word_filter.get_matched_doc()

# Then, filter for Roman numerals (on the original doc)
roman_filter = IsRomanFilter()
doc = tokenizer.make_doc("Chapter IV: The quick brown fox (123) jumps.")  # Reset to original
roman_filter(doc)
roman_numerals = roman_filter.matched_tokens
print([token.text for token in roman_numerals])
```

## Common Use Cases

### Extract Only Words (No Punctuation)

```python
from lexos.filter import IsWordFilter
from tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("Hello, world! How are you?")

word_filter = IsWordFilter()
words_doc = word_filter.get_matched_doc()
print(words_doc.text)
# Output: "Hello world How are you"
```

### Identify Roman Numerals in Text

```python
from lexos.filter import IsRomanFilter
from tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc("Book I begins with Chapter III and ends with Chapter VII.")

roman_filter = IsRomanFilter(attr="is_roman")
roman_filter(doc)

# Extract Roman numerals
for token in doc:
    if hasattr(token._, "is_roman") and token._.is_roman:
        print(f"Roman numeral found: {token.text}")
```

### Manage Custom Stop Words

```python
from lexos.filter import IsStopwordFilter
from lexos.tokenizer import Tokenizer

tokenizer = Tokenizer(model="en_core_web_sm")

# Add domain-specific stop words (case-insensitive by default)
domain_stopwords = ["moreover", "furthermore", "therefore"]
stopword_filter = IsStopwordFilter()
doc = tokenizer.make_doc("dummy text")
stopword_filter(doc, stopwords=domain_stopwords, remove=False)

# Now these are treated as stop words (including "Moreover", "FURTHERMORE", etc.)
text = "Moreover, the data shows something. Therefore, we conclude."
doc = tokenizer.make_doc(text)
meaningful_words = [token.text for token in doc if not token.is_stop]
print(meaningful_words)
# Output: ['data', 'shows', 'conclude']
```
