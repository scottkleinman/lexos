# Working with Individual Records

## Overview

The `record` module provides the `Record` class, which is the building block for every document in your corpus. Each `Record` wraps your text (or a parsed spaCy Doc) and metadata and offers a suite of methods for serialization, statistics, and manipulation.

### Creating a Record

You can create a record from plain text or a spaCy Doc, and you can attach any metadata you like:

```python
from lexos.corpus import Record

record = Record(
    name="example_doc",
    content="The quick brown fox jumps over the lazy dog.",
    meta={"author": "Jane", "year": 2025}
)
```

You can pass a `data_source` pointing to the original location (filepath or url) where the data came from. The `meta` parameter is for passing a dictionary of more arbitrary metadata items.

If you already have a spaCy `Doc` object, just pass it to the record using the `content` parameter — the record's `is_parsed` property will be set automatically. Using the `model` parameter to identify the spaCy language model used is useful for serialization and deserialization.

```python
record = Record(
    name="parsed_doc",
    content=doc,
    meta={"author": "Jane"},
    model="en_core_web_sm"
)
```

If your `Doc` object has spaCy custom extensions, provide a list of those extensions with the `extensions` parameter.

When you create a `Record` object, it is automatically assigned an `id`, and its `is_active` attribute is set to `True`. The `id` is a UUID value (by default) and you can override it by passing your own `id` value. When `is_active=True`, the record is assumed to be available for analysis. Some applications may want to keep records in a corpus, enabling or disabling them as needed for specific purposes. The `is_active` attribute does not affect the record on its own.

The following properties can be inspected from a `Record` object:

- **is_parsed**: Returns `True` if the content is a spaCy Doc.
- **preview**: Returns a short preview of the record’s text.
- **terms**: Returns a Counter of terms (requires parsed content).
- **tokens**: Returns a list of tokens (requires parsed content).
- **text**: Returns the text of the record (works for both string and parsed Doc).

For instance, for the "parsed_doc" record above, calling `record.is_parsed` will return `True`.

The `set()` method is a helper method to set attributes of the `Record`. For instance, `record.set(is_active=False)` will set its `is_active` attribute to `False`.

---

#### Serialization & Disk Operations

The following methods are used to serialize and deserialize `Record` objects to bytes or to disk:

- **to_bytes(extensions=[], include_hash=True)**: Serialize the record to bytes (for saving or transmitting). If you have a spaCy `Doc` with custom extensions, you need to add the names of the extensions in a list. By default, a data integrity hash is included in the serialization, but you can remove it with `include_hash=False`.
- **from_bytes(bytestring, model=None, model_cache=None, verify_hash=True)**: Load a record from bytes, optionally verifying data integrity. If you have a spaCy `Doc`, you should specify the language model used. If you have multiple docs
- **to_disk(path, extensions=None)**: Save the record to disk.
- **from_disk(path, model=None, model_cache=None)**: Load the record from disk.

When deserializing a `Record` that contains a parsed spaCy `Doc`, you need the correct spaCy model to reconstruct the `Doc` object. `LexosModelCache` provides a way to retrieve and reuse these models. If you pass a `model_cache` to methods like `from_bytes` or `from_disk`, Lexos will use it to get the correct spaCy vocabulary for reconstructing the `Doc`. This is especially useful when working with many records or large corpora parsed with different models by making the process faster and more memory-efficient. Here is an example:

```python
from lexos.corpus.utils import LexosModelCache

# Create a cache and load models as needed
model_cache = LexosModelCache()
doc_model = model_cache.get_model("en_core_web_sm")  # Loads and caches the model

# When loading a Record from bytes or disk:
record.from_bytes(bytestring, model="en_core_web_sm", model_cache=model_cache)
```

If you use spaCy Docs, any custom token attributes and extensions are preserved during serialization

---

#### Statistical Data

`Record` objects provide methods for exposing a number of useful types of statistical information about the documents they contain:

- **num_terms()**: Returns the number of unique terms (words) in the record (requires parsed content).
- **num_tokens()**: Returns the total number of tokens (words) in the record (requires parsed content).
- **vocab_density()**: Returns the ratio of unique terms to total tokens (requires parsed content).
- **most_common_terms(n=None)**: Returns the n most common terms as a list of (term, count) tuples.
- **least_common_terms(n=None)**: Returns the n least common terms as a list of (term, count) tuples.

---

### Example Workflow

```python
import spacy
from lexos.corpus import Record

nlp = spacy.load("en_core_web_sm")
doc = nlp("Lexos makes text analysis easy.")

record = Record(name="demo", content=doc, meta={"topic": "NLP"})

print("Preview:", record.preview)
print("Tokens:", record.tokens)
print("Most common terms:", record.most_common_terms(3))

# Save to disk
record.to_disk("demo_record.lexos")

# Load from disk
loaded = Record()
loaded.from_disk("demo_record.lexos", model="en_core_web_sm")
print("Loaded text:", loaded.text)
```

---

### Technical Notes

- **UUIDs:** Each record gets a unique ID by default. You can use integers if you prefer.
- **spaCy Integration:** If you use spaCy Docs, custom token attributes and extensions are preserved during serialization.
- **Data Integrity:** When serializing, a hash is included to help detect corruption or incomplete writes.
- **Metadata:** Any metadata you attach is sanitized for JSON compatibility.
