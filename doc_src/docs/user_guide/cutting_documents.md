# Cutting Documents

The Lexos `cutter` module is used to split documents into smaller, more manageable pieces, variously called "segments" or "chunks." This is particularly useful for processing large texts, enabling more efficient analysis and manipulation. If your documents are raw text files or strings, you can use the `TextCutter` class. If your documents are tokenized spaCy `Doc` objects, you can use the `TokenCutter` class. Documents can be cut based on byte size, number of tokens, number of sentences, line breaks, or custom-defined spans called milestones. The different cutting methods are described below.

## Cutting Text Strings with `TextCutter`

Let's say that you have a  long text string that you wanted to break into smaller chunks every *n* characters. The cell below demonstrates a simple way you can do this with the `TextCutter` class. To begin, we'll use a very short sample text for demonstration purposes, but you can replace the `text` variable with any long string of your choice. You can also change the `chunksize` parameter to specify how many characters you want in each chunk.

```python
# Import the TextCutter class
from lexos.cutter.text_cutter import TextCutter

# Create a sample text
text = (
    "It is a truth universally acknowledged, that a single  man in possession of a good fortune, "
    "must be in want of a wife."
)

# Initialize TextCutter
cutter = TextCutter()

# Split text into chunks of 50 characters each
cutter.split(text, chunksize=5)
```

The first parameter (for which you can use the keyword `docs`) is the text string to be cut. You can also supply a list of text strings (e.g., multiple documents) to the `docs` parameter, and each document will be cut separately.

!!! note
    Occasionally, you may want to cut text based on byte size rather than character count. By default, the `TextCutter` counts characters, but you can set the `by_bytes` attribute to `True` if you want to cut based on byte size instead. However, be aware that cutting by bytes may split multi-byte characters in the middle, which can lead to encoding issues.

Once the text has been split, its chunks are stored in a list of lists (one list per document). This list can be accessed three different ways:

- By calling `cutter.chunks`.
- By returning a value with `chunks = cutter.split(docs=text, chunksize=50)`.
- By iterating through the `Cutter` object:

```python
for chunk in cutter:
    print(chunk)
```

You can check how many documents are in your `TextCutter` instance by using the `len()` function:

```python
print(len(cutter))
```

If you have multiple documents, you can get a dictionary of the chunks for each document using the `to_dict()` method. The dictionary keys are the document names and the values are lists of string chunks.

```python
chunks_dict = cutter.to_dict()
print(chunks_dict)
# {
#     "doc001": ["First 50 chars...", "Next 50 chars...", ...]
#     "doc002": ["First 50 chars...", "Next 50 chars...", ...], etc.
# }
```

By default, each doc is named with the prefix "doc" followed by the doc number (starting from 001). Doc names can be accessed calling the `names` attribute. You can provide a custom list of names using the `names` parameter. You can also adjust zero padding with the `pad` parameter. For instance,

```python
cutter.split(text, chunksize=50, pad=2, names=["Doc"])
```

This will produce doc names like "Doc01", "Doc02", "Doc03", etc.

Cutting texts can often leave small dangling pieces at the end. To address this, you can use the `merge_threshold` and `merge_final` parameters to control whether the last two chunks should be merged based on their size.

- `merge_threshold`: The threshold for merging the last two chunks. The default is 0.5 (50%).
- `merge_final`: Whether to merge the last two chunks. The default is `False`.

!!! important
    Always inspect your chunks to see if the merging behaviour meets your expectations.

You can also generate chunks that overlap with each other by using the `overlap` parameter. This parameter specifies the number of characters that should be repeated at the start of each chunk (except the first one). For example, if you set `chunksize=100` and `overlap=20`, each chunk will contain 20 characters from the end of the previous chunk.

### Cutting Text Files with `TextCutter`

You can also cut texts directly from files, and the resulting chunks can be saved to disk. The `split()` method accepts the following parameters for file-based cutting:

- `docs`: The a file path or buffer, or a list of file paths or buffers.
- `file`: If `True`, treat each doc as a file path.

By default, the `docs` parameter is assumed to be a list of text strings. If you set `file=True`, each doc will be treated as a file path, and the contents of the files will be read and cut accordingly. Each chunk will be named using the original file name (unless you provide custom names through the `names` parameter), followed by the chunk number, separated by the specified `delimiter` and padded to the specified length.

!!! note
    You can also assign the docs when creating the `TextCutter` instance by using the `docs` parameter in the constructor. (e.g., `cutter = TextCutter(docs=["file1.txt", "file2.txt"], file=True)`).

### Cutting Documents into a Fixed Number of Chunks

The `n` parameter allows you to specify the number of chunks to split the text into. When you provide a value for `n`, the text will be divided into that many approximately equal parts. This is useful when you want to ensure that the text is split into a specific number of segments, regardless of their size. The `n` parameter overrides the `chunksize` parameter if both are provided.

```python
cutter.split(text, n=5)
print(len(cutter.chunks))  # 5
```

### Cutting Documents by Line

The `newline` parameter allows you to split the text based on line breaks instead of byte size. When `newline=True`, the `chunksize` or `n` parameters will refer to the number of lines per chunk or the number of chunks based on lines, respectively.

```python
cutter.split(text, n=5, newline=True)
print(len(cutter.chunks))  # 5
```

This will return chunks each containing 5 lines, except the last chunk, the length of which will depend on your merge settings.

### Setting `TextCutter` Attributes

Most of the parameters described above can also be set as attributes of the `TextCutter` instance. For example:

```python
cutter = TextCutter()
cutter.chunksize = 100
cutter.delimiter = "-"
cutter.pad = 2
cutter.split(docs=text)
```

This allows you to configure the cutter once and then use it multiple times with the same settings.

### Saving Chunks to Disk

The save your chunks to disk, call the `save()` method. Each chunk will be saved as a separate `.txt` file in the specified with the `output_dir` parameter.

```python
cutter.save(output_dir="output_chunks")
```

This will save files like `doc001_001.txt`, `doc001_002.txt`, `doc002_001.txt`, `doc002_002.txt` in the output directory.

You can customize the file naming convention by providing your own list of document names, changing the delimiter, and adjusting the padding. For example:

```python
cutter.save(output_dir="output_chunks", names=["A","B"], delimiter="-", pad=2)
```

This will save files like `A-01.txt`, `A-02.txt`, `B-01.txt`, `B-02.txt`.

!!! note
    In some cases, your chunks may have leading or trailing whitespace. By default, Lexos will strip this whitespace, but you can control this by setting `strip_chunks= False`.

## Merging Text Chunks

You can also merge a list of string chunks back into a single string using the `merge()` method. This is useful if you want to recombine the chunks after processing.

```python
chunks = ["This is chunk one.", "This is chunk two.", "This is chunk three."]
merged_text = cutter.merge(chunks, sep=" ")
print(merged_text)
# "This is chunk one. This is chunk two. This is chunk three."
```

By default, chunks are separated by spaces, but you can specify a different separator using the `sep` parameter.

## Cutting spaCy `Doc` Objects with `TokenCutter`

If you have tokenized documents represented as spaCy `Doc` objects, you can use the `TokenCutter` class to split them into smaller segments. `TokenCutter` has the same attributes as `TextCutter` and works in a similar way.

However, one important difference is that the resulting chunks will be spaCy `Doc` objects rather than plain text strings. This allows you to preserve token-level information and annotations. If you wish to access the chunk string, you can do so using the `.text` attribute of each `Doc` chunk.

If your spaCy `Doc` objects are stored in files, you can load them and cut them using them by setting `file=True` and the name of the spaCy model with the `model` parameter:

```python
cutter.split(docs=["doc1.spacy", "doc2.spacy"], file=True, model="en_core_web_sm", chunksize=100)
```

The files must be in spaCy's binary format, which can be created using the `Doc.to_bytes()` or `Doc.to_disk()` methods. You must also specify the spaCy model used to create the `Doc` objects so that they can be deserialised correctly.

### Cutting on Sentences Breaks

Some spaCy language models include sentence boundary detection. If your `Doc` objects have sentence boundaries defined, you can use the `split_on_sentences()` method to cut the documents into chunks based on a specified number of sentences. For instances, assume that your spaCy `Doc` object has ten sentences. You can split it into chunks of 5 sentences each as follows:

```python
cutter.split_on_sentences(doc, n=5)
```

If a `Doc` does not have sentence boundaries defined, Lexos will raise an error.

As with other `TokenCutter` methods, the resulting chunks will be spaCy `Doc` objects.

### Setting `TokenCutter` Attributes

Just like in `TextCutter`, you can set attributes in the `TokenCutter` instance for re-use:

```python
cutter = TokenCutter()
cutter.chunksize = 100
cutter.delimiter = "-"
cutter.pad = 2
cutter.split(docs=text)
```

### Saving spaCy `Doc` Files

By default, the `save()` method saves the chunk text strings, rather than the spaCy `Doc` objects. If you would like to store the spaCy `Doc` objects themselves, set the `as_text` parameter to `False`. This is the equivalent of calling spaCy's `Doc.to_bytes()` method on each chunk and saving the resulting bytes to disk.

## Merging Token Chunks

You can also merge a list of string chunks back into a single string using the `merge()` method. This is useful if you want to recombine the chunks after processing. We can demonstrate its usage by starting with the chunks produced in the example above.

```python
# Split the doc on the milestone "quick" (into two chunks)
cutter = TokenCutter()
chunks = cutter.split_on_milestones(docs=doc, milestones=spans)
print(chunks[0]) # The
print(chunks[1]) # brown fox jumps over the lazy dog.

merged_doc = cutter.merge(chunks)
print(merged_doc.text)
# "The brown fox jumps over the lazy dog."
```

Unlike `TextCutter`, the `TokenCutter` `merge()` does not require a `sep` parameter. The start token for each chunk is appended to the end token of the previous chunk, and spacing is handled according to the language model used to create the original `Doc` objects.

## Splitting on Milestones

Milestones are specified locations in the text that designate structural or sectional divisions. A milestone can be either a designated unit *within* the text or a placemarker inserted between sections of text. The Lexos [`milestones` module](using_milestones.md) provides methods for identifying milestone locations by searching for patterns you designate. You can use the `StringMilestones` class in the `milestones` module to generate a list of `StringSpan` objects that mark the locations of milestones in your text. The `TextCutter.split_on_milestones()` method can then use these spans to split the text into chunks at the specified locations. Here is a quick example of how to do it.

```python
# Import the StringMilestones class
from lexos.milestones.string_milestones import StringMilestones

# A sample text
text = "The quick brown fox jumps over the lazy dog."

# Create a String Milestones instance and search for the pattern "quick"
milestones = StringMilestones(doc=text, patterns="quick")

# Create a TextCutter instance and split on the found milestones
cutter = TextCutter()

# Split the text on the milestone "quick"
chunks = cutter.split_on_milestones(
    docs=text,
    milestones=milestones.spans, # The list of StringSpan objects
)
print(chunks[0]) # The
print(chunks[1]) # brown fox jumps over the lazy dog.
```

You will notice that the milestone itself ("quick") is not included in either chunk. By default, the milestone text is removed during the split. You can control this behaviour by setting the `keep_spans` parameter to either `'preceding'` or `'following'`, which will keep the milestone text in the preceding or following chunk, respectively.

```python
chunks = cutter.split_on_milestones(
    docs=text,
    milestones=milestones.spans,
    keep_spans='preceding'  # or 'following' to keep in the next chunk
)
```

If you do not want to use the `milestones` module to find milestones, you can also create your own list of `StringSpan` objects manually and pass them to the `split_on_milestones()` method.

Cutting on milestones works similarly for `TokenCutter` objects, except that the milestones are specified as lists of spaCy `Span` objects rather than `StringSpan` objects.

```python
# Import the TokenMilestones class
from lexos.milestones.token_milestones import TokenMilestones

# Assume this is a spaCy `Doc` object with the text shown
doc = "The quick brown fox jumps over the lazy dog."

# Create a Token Milestones instance and search for the pattern "quick"
milestones = TokenMilestones(doc=doc)
spans = milestones.get_matches(patterns="quick") # The list of Span objects

# Create a TokenCutter instance and split on the found milestones
cutter = TokenCutter()
chunks = cutter.split_on_milestones(docs=doc, milestones=spans)
print(chunks[0]) # The
print(chunks[1]) # brown fox jumps over the lazy dog.
```

In the case above, the *token* "quick" is used as the milestone, rather than the string "quick".

The `milestones` module provides additional methods for finding milestones based on patterns, regular expressions, or custom logic. See the [Using Milestones](using_milestones.md) guide for more information.
