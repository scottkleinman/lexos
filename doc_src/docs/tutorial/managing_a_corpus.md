# Overview

The `Corpus` module consists of a `Corpus` class that helps you manage assets in your workflow and serialize them to disk for later use. It is strictly optional; you may find it sufficient to load your documents into memory with a `Loader` or to manage your corpus assets independently through a different application.

It is important to realise that a Lexos `Corpus` is primarily a manager for project assets; it is not used for acquiring those assets and is not used for analysing them, apart from the generation of a few statistics. In general, using a `Corpus` will require a workflow like the following:

1. Use `Loader` to acquire texts.
2. Optionally, use `Scrubber` to perform any required preprocessing on the texts in `Loader.texts`.
3. Optionally, use `Tokenizer` to convert the original or scrubbed texts to spaCy `Doc` objects.
4. Add the documents (texts or spaCy `Doc` objects) to the `Corpus`.

If you wished to analyse the documents, you would get them from the `Corpus` and run them through `Tokenizer` if you did not do so before adding them.

From this workflow, you should be able to see that you can skip the `Corpus` entirely. The `Corpus` simply allows you to attach metadata to the documents, such as a name, description, or classification label, and to save them to and retrieve them from disk easily. One of the important metadata categories is whether or not a document is active. A `Corpus` allows you to retrieve subsets of your documents based on this and other metadata categories.

!!! Note
    The `Corpus` module is designed to work without a database and without necessarily storing all documents in memory. This provides maximum flexibility and ease of use at the expense of performance and efficiency. If using Lexos in a production environment, you may wish to use it as a jumping off point for designing your own corpus manager, or replace it entirely, especially if you are using a database. It should be emphasised again that none of the other Lexos modules require you to use `Corpus`.

## Creating a Corpus

Begin by importing the `Corpus` module:

```python
import lexos.corpus as cp
```

We use the `cp` alias so that we can use `corpus` as a variable below:

```python
corpus = cp.Corpus(
    name="My Corpus",
    description="My test corpus",
    corpus_dir="corpus"
)
```

The `name`, `description`, and `corpus_dir` arguments are all optional. `corpus_dir` is the directory where the corpus will be stored, and the default is "corpus" in the current directory. You can use additional keywords to instantiate the corpus with other metadata such as author or creation date. When you run the code above, the corpus directory will be created if it does not already exist.

You can view your corpus metadata in a number of ways:

`Corpus.meta_table()` will return an overview of your corpus as a pandas dataframe. You can also view this information as a dict with `corpus.meta`. The following individual properties can also be viewed:

- `ids`: A list of document ids in the Corpus.
- `names`: A list of document names in the Corpus.
- `docs`: A dict of document ids and docs in the Corpus if you have opted to cache them to RAM.
- `num_docs`: The number of docs in the Corpus.
- `num_active_docs`: The number of active docs in the Corpus.
- `num_tokens`: The number of tokens in the Corpus.
- `num_terms`: The number of terms in the Corpus.
- `terms`: A set of unique terms in the Corpus.

Call these functions with code like `Corpus.num_tokens`. You can also get a Python collections `Counter` object containing the corpus term counts with `Corpus.get_term_counts()`.

These attributes should all be empty or 0 when the corpus is first created.

!!! note
    The `Corpus` class is constructed using <a href="https://pydantic-docs.helpmanual.io/usage/models/" target="_blank">Pydantic's `BaseModel` class</a>. This means that it has access to any of Pydantic's attributes and methods, such as `model_dump()` and `model_dump_json()`.

## Corpus Records

The basic unit of storage in a `Corpus` is a `Record` object. This is a Python object that provides access to the record's content and its metadata. Constructing a `Record` is simple. You just have to feed it some `content` and, in most cases, give it a name:

```python
record = Record(content=mydoc, name="greeting")
```

Behind the scenes, the `Record` class will give the record a default `id` of 1 (unless you specify a different integer) and set the `is_active` property to `True` (unless you set instantiate the object with it set to `False`). See [lexos.corpus.Record][lexos.corpus.Record] for other arguments that can be passed to the `Record` class.

You can also create a `Record` from a dict using Pydantic's `model_validate()` method:

```python
record = Record.model_validate({"content": mydoc, "name": "greeting"})
```

See the <a href="https://pydantic-docs.helpmanual.io/usage/models/#helper-functions" target="_blank">Pydantic documentation</a> for helper functions for parsing json or file content into objects.

Once instantiated, a record provides access to the following information:

- `preview`: A preview of the first 50 characters of the record's text.
- `terms`: A set of unique terms in the record's content.
- `text`: The full text of the record's content.
- `tokens`: A list of tokens in the record's content.
- `num_terms`: The number of unique terms in the record's content.
- `num_tokens`: The number of tokens in the record's content.

!!! important
    Term counts do not collapse upper- and lower-case words, so, if this is important, you must get the tokens, convert to lower case, and then generate the list of terms yourself. Alternatively, you may use `Scrubber` to preprocess your data before creating the `Record` object.

`Record.set()` allows you to set arbitrary `Record` attributes (such as author or date), and `Record.save()` allows you to save the file to disk.

The `Record` class accepts content only in the form of a pre-tokenized spaCy doc. However, it is possible to store an untokenized text by creating a blank spaCy language model and feeding it the [lexos.corpus.NullTokenizer][lexos.corpus.NullTokenizer] class. This simply returns a spaCy doc with the text as a single token.

```python
nlp = spacy.blank("xx")

nlp.tokenizer = NullTokenizer(nlp.vocab)

content = nlp(content)

record = Record(content)
```

Note that the entire text will be counted as a single token and a single term, so it is preferable to tokenize the text first or to plan to do so later.

If the content is already a tokenized document, it is necessary to label it as such in the metadata. Here is an example of how you would do it:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Hi there!")

record = Record(content=doc, name="greeting", is_parsed=True)
```

The `is_parsed` attribute allows `Corpus` to know that it is dealing with a tokenized document. You can still access the full text by calling `record.text`, but you can also access individual tokens by calling `record.content[0]` (to get the first token).

### Serialising Records

Since a `Record` is a Pydantic object, it can theoretically be serialised to json format with Pydantic's `model_dump_json()` method. However, this is unreliable because the `content` attribute contains a spaCy doc. If all you need is access to the record's metadata, perhaps the easiest method is `Record.model_dump_json(exclude="content")` (you can also use `model_dump()` if you want a dictionary). To serialise the content, you must invoke spaCy's <code><a href="https://spacy.io/api/doc#to_json" target="_blank">Doc.to_json()</a></code>. For inconvenience, the `Record` class wraps this in its own `to_json()` method. Use it as follows:

```python
json_str = record.to_json()
```

You can pass it any of the keywords accepted by Pydantic's <code><a href="https://docs.pydantic.dev/latest/api/base_model/#pydantic.main.BaseModel.model_dump_json" target="_blank">model_dump_json()</a></code> if you wish to format the results.

To deserialise from json, instantiate an empty record and use `Record.from_json(json_str)`.

The `Record` class also has a `save()` method to serialise records to disk, if provided with a filename:

```python
record.save(filename="my_record")
```

By default, the file will be saved as a binary pickle file. However, you can use `format="json"` to serialise the record in json format. The `Record.save()` will call `Record.to_json()`, so you can pass it any formatting arguments you would use with the latter method.

When a `Record` object is saved to disk as a binary pickle file, it will not human readable. To restore it, you can use a normal Python method of reading a binary file:

```python
with open(filename, "rb") as f:
    record = pickle.load(f)
```

You can also instantiate an empty record and use the `load()` method:

```python
record = Record()
record.load(filename="my_file", format="pickle")
```

!!! important
    The pickle format is not considered secure, so never unpickle a file you do not trust.

If you have serialised the file to json, use

```python
record = Record()
record.load(filename="my_file.json", format="json")
```

## Adding Records to a Corpus

Adding records to a corpus is simple with `Corpus.add_record()`:

```python
record = Record(content=doc, name="greeting", is_parsed=True)

corpus.add_record(record, cache=True)
```

There is also a `Corpus.add_records()`, which takes a list of records.

By default, the record's content is not cached in memory; instead, the entire record is serialized to disk. If you want to keep it in memory, you can set `cache=True` (as above). This will allow you to access the record from `corpus.docs` without having to fetch the record from disk.

!!! note
    At present the `docs` property in the `Corpus` class is the only place where a clear distinction between a "record" and a "document" is made.

!!! important
    Currently all records in a corpus are serialised to disk as binary pickle files. There are some concerns about the security and speed of this method, which will be addressed in the future.

## Adding Documents to a Corpus

It is not necessary to pre-generate records from documents before adding them to a corpus. You can also use `Corpus.add()` to add a document directly:

```python
# Use a text string
unparsed_doc = "Hi there!"
corpus.add_record(unparsed_doc, name="greeting")

# Create
parsed_doc = nlp("This is a full sentence.")
corpus.add(content=parsed_doc, name="sentence", is_parsed=True)
```

By default, the `is_active` attribute is `True`.

You can set additional metadata properties by supplying a `metadata` dict:

```python
metadata = {"author": "John Smith", "date": "2011"}

corpus.add(
    content=parsed_doc,
    name="sentence",
    is_parsed=True,
    metadata=metadata,
    cache=True
)
```

The corresponding `Corpus.add_docs()` allows you to insert multiple documents. The format is a little more complicated. It takes a list of dicts with the document as the value of the `content` property:

```python
docs = [
    {"content": doc1, "name": "Doc 1"},
    {"content": doc2, "name": "Doc 2"},
]

corpus.add_docs(docs, cache=True)
```

All the arguments accepted by `Corpus.add()` can be set as keys in the `docs` dictionary.

!!! important
    Whether you are adding documents or records to a corpus, a check is made to ensure that the records stored have unique `id`, `name`, and `filename` attributes. If you do not specify a `name` for a document or record, a <a href="https://docs.python.org/3/library/uuid.html" target="_blank">UUID</a> will be used instead and will be used to generate a corresponding filename. The results of this can be unwieldy. In the future, some other method of ensuring uniqueness will be explored.

## Getting Records from the Corpus

Individual records can be fetched using `Corpus.get()` with a record `id`:

```python
record = corpus.get(1)

doc = record.content
```

The second line above extracts the spaCy doc from the records, and it can be treated like any spaCy doc.

You can also supply a list of ids to `Corpus.get_records()`. If you pass nothing to the method, all the records in the corpus will be retrieved.

If you do not know the id(s) of the document(s) you want, you can provide a query for `Corpus.get_records()`:

```python
records = corpus.get_records(query="id < 10")
for record in records:
    print(record.name)
```

This will yield a generator with each of the records with an `id` less than 10.

!!! note
    On the back end, `Corpus.get()` and `Corpus.get_records()` call `Corpus.meta`, which contains a subset of the metadata for each record. A pandas dataframe is constructed from this metadata. The `query` can therefore be anything acceptable to <code><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html" target="_blank">pandas.DataFrame.query()</a></code>. This allows complex queries to be performed on the corpus.

If you want just the metadata for a record, probably the easiest method is `metadata = corpus.get(1).dict().remove("content")`.

## Viewing the Records Table

`Corpus.records_table()` generates a pandas dataframe with each record in a separate row. By default, the `id`, `name`, `filename`, `num_tokens`, `num_terms`, `is_active`, and `is_parsed` attributes are displayed columns. You can supply your own list of columns with the `columns` argument, or you can exclude specific columns with the `exclude` argument.

## Setting Record Properties

After a corpus is instantiated, you can set the properties of individual records with `Corpus.set()`:

```python
corpus.set(1, {"name": "John Smith"})
```

## Removing Records

`Corpus.remove()` and `Corpus.remove_records()` can be used to remove records from a corpus. The former takes an id number and the latter takes a list of ids.

## Using Records

Typically, you would retrieve records using `Corpus.get_records()` and then pass their content to another Lexos module. For example, here is how you would create a document-term matrix:

```python
# Get the records
records = corpus.get_records()

# Extract the documents and labels
docs = [record.content for record in records]
labels = [record.name for record in records]

# Import the dtm module and generate a document-term matrix
from lexos.dtm import DTM

# Build the DTM
dtm = DTM(docs, labels)
```
