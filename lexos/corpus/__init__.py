"""__init__.py.

This module contains the Corpus class, which is a container for a collection of
records stored as Record objects.

To Do:

- Improve type annotations.
- Look into a better serialisation format than pickle.
- Add more `Corpus` methods. Since the Corpus is just a folder with a metadata file
  and a set of binary resources, it should be easy to archive as a zip file, or, say,
  a Frictionless Data data package. So we could add something like a `Corpus.to_zip()`
  method or a `Corpus.to_datapackage()` method. Adding an ORM layer for database
  storage would also be possible.
"""
import os
import pickle
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
import spacy
import srsly
from pydantic import BaseModel
from spacy.tokens import Doc


class NullTokenizer:
    """Pass the text back as a spaCy Doc with the text as single token."""

    def __init__(self, vocab):
        """Initialise the tokeniser."""
        self.vocab = vocab

    def __call__(self, text):
        """Return the text as a single token."""
        words = [text]
        spaces = [False]
        return Doc(self.vocab, words=words, spaces=spaces)


class Record(BaseModel):
    """The main Record model."""

    content: spacy.tokens.Doc = None
    filename: str = None
    id: int = 1
    is_active: bool = True
    is_parsed: bool = False
    model: str = None
    name: str = None

    class Config:
        """Config for the Record class."""

        arbitrary_types_allowed = True
        extra = "allow"

    def __repr__(self):
        """Return a string representation of the record."""
        fields = {
            field: getattr(self, field)
            for field in self.__fields_set__
            if field != "content"
        }
        properties = {k: getattr(self, k) for k in self.get_properties()}
        fields = {**fields, **properties}
        fields["terms"] = "Counter()"
        fields["text"] = self.preview
        fields["tokens"] = f'[{", ".join([t.text for t in self.content[0:5]])}...]'
        field_list = [f"{k}={v}" for k, v in fields.items()] + [
            f"content={self.preview}"
        ]
        return f"Record({', '.join(sorted(field_list))})"

    @classmethod
    def get_properties(cls):
        """Return a list of the properties of the Record class."""
        return [
            prop for prop in cls.__dict__ if isinstance(cls.__dict__[prop], property)
        ]

    @property
    def preview(self):
        """Return a preview of the record text."""
        return f"{self.content.text[0:50]}..."

    @property
    def terms(self):
        """Return the terms in the record."""
        return Counter([t.text for t in self.content])

    @property
    def text(self):
        """Return the text of the record."""
        return [t.text for t in self.content]

    @property
    def tokens(self):
        """Return the tokens in the record."""
        return self.content

    def num_terms(self):
        """Return the number of terms."""
        return len(self.terms)

    def num_tokens(self):
        """Return the number of tokens."""
        return len(self.content)

    def save(self):
        """Serialise the record to disk."""
        with open(self.filename, "wb") as f:
            pickle.dump(self, f)

    def set(self, k, v):
        """Set a record property."""
        setattr(self, k, v)


class Corpus(BaseModel):
    """The main Corpus model."""

    corpus_dir: str = "corpus"
    description: str = None
    docs: Dict[int, object] = {}
    ids: List[int] = []
    meta: List[dict] = []
    name: str = None
    names: List[int] = []
    num_active_docs: int = 0
    num_docs: int = 0
    num_terms: int = 0
    num_tokens: int = 0
    terms: set = set()

    class Config:
        """Config for the Corpus class."""

        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        """Initialise the Corpus."""
        super().__init__(**data)
        # for field in self.__fields_set__:
        #     self.meta[field] = getattr(self, field)
        Path(f"{self.corpus_dir}/docs").mkdir(parents=True, exist_ok=True)
        srsly.write_json(f"{self.corpus_dir}/corpus_meta.json", self.json())
        print("Corpus created.")

    def __repr__(self):
        """Return a string representation of the Corpus."""
        fields = {field: getattr(self, field) for field in self.__fields_set__}
        field_list = [f"{k}={v}" for k, v in fields.items()]
        rep = f"Corpus({', '.join(sorted(field_list))})"
        return rep

    def add(
        self,
        content: Union[object, str],
        name: str = None,
        is_parsed: bool = False,
        is_active: bool = True,
        model: str = None,
        metadata: dict = None,
        cache: bool = False,
    ):
        """Add a document the Corpus.

        Args:
            content (Union[object, str]): A text string or a spaCy document.
            name (str): A name for the document.
            is_parsed (bool): Whether or not the document is parsed.
            is_active (bool): Whether or not the document is active.
            model (str): The name of the language model used to parse the document (optional).
            metadata (dict): A dict containing any metadata.
            cache (bool): Whether or not to cache the record.
        """
        # If the doc is a string, make a spaCy doc with untokenised text
        if not is_parsed:
            nlp = spacy.blank("xx")
            nlp.tokenizer = NullTokenizer(nlp.vocab)
            content = nlp(content)
        # Create a Record
        unique_name = self._ensure_unique_name(name)
        unique_filename = self._ensure_unique_filename(unique_name)
        record = Record(
            content=content,
            name=unique_name,
            filename=unique_filename,
            id=self._get_new_id(),
            is_active=is_active,
            is_parsed=is_parsed,
            model=model,
        )
        # Add arbitrary metadata properties
        if metadata:
            for k, v in metadata.items():
                record.set(k, v)
        # Add the record to the Corpus
        self._add_to_corpus(record, cache=cache)

    def add_docs(
        self,
        docs: List[dict],
        name: str = None,
        is_parsed: bool = False,
        is_active: bool = True,
        model: str = None,
        cache: bool = False,
    ):
        """Add multiple docs to the corpus.

        Args:
            docs (List[dict]): A list of dicts containing texts or docs to add, plus metadata.
            name (str): The name of the record.
            is_parsed (bool): Whether the record has been parsed.
            is_active (bool): Whether the record is active.
            model (str): The name of the model used to parse the record.
            cache (bool): Whether to cache the record.

        Note:
            Each doc in `docs` should have a `content` field containing doc text or spaCy doc and
            may have any number of additional metadata fields. If "name", "is_parsed", "is_active",
            and "model" are not specified, the defaults will be used.
        """
        for doc in docs:
            content = doc["content"]
            if "name" in doc:
                name = doc["name"]
            if "is_parsed" in doc:
                is_parsed = doc["is_parsed"]
            if "is_active" in doc:
                is_active = doc["is_active"]
            if "model" in doc:
                model = doc["model"]
            for item in ["content", "name", "is_parsed", "is_active", "model"]:
                if item in doc.keys():
                    del doc[item]
            self.add(
                content=content,
                name=name,
                is_parsed=is_parsed,
                is_active=is_active,
                model=model,
                metadata=doc,
                cache=cache,
            )

    def add_record(self, record: object, cache: bool = False):
        """Add a Record the Corpus.

        Args:
            record (object): A Record object.
            cache (bool): Whether or not to cache the record.
        """
        record.name = self._ensure_unique_name(record.name)
        if record.filename in os.listdir(f"{self.corpus_dir}/docs"):
            record.filename = self._ensure_unique_filename(record.name)
        self._add_to_corpus(record, cache=cache)

    def add_records(self, records: List[object], cache: bool = False):
        """Add multiple docs to the corpus.

        Args:
            records (List[objct]): A list of Record objects.
            cache (bool): Whether or not to cache the record.
        """
        for record in records:
            self.add_record(record, cache=cache)

    def get(self, id):
        """Get a record from the Corpus by ID.

        Tries to get the record from memory; otherwise loads it from file.

        Args:
            id (int): A document id from the Corpus records.
        """
        # If the id is in the Corpus cache, return the record
        if id in self.docs.keys():
            return self.docs[id]
        # Otherwise, load the record from file
        else:
            id_idx = self.ids.index(id)
            filename = self.meta[id_idx]["filename"]
            return self._from_disk(filename=filename)

    def get_records(self, ids: List[int] = None, query: "str" = None) -> Callable:
        """Get multiple records using a list of ids or a pandas query.

        Args:
            ids (List[int]): A list of record ids to retrieve
            query (str): A query string parsable by pandas.DataFrame.query

        Yields:
            Callable: A generator containing the docs matching the ids or query
        """
        if not ids and not query:
            ids = self.ids
        if query:
            # Use pandas to query the dataframe metadata
            table = self.records_table().query(query)
            ids = table.id.values.tolist()
        for id in ids:
            yield self.get(id)

    def get_term_counts(self) -> Callable:
        """Get a Counter with the Corpus term counts.

        Returns:
            A collections.Counter
        """
        return sum([self.get(id).terms for id in self.ids], Counter())

    def meta_table(self, drop: List[str] = ["docs", "meta", "terms"]) -> pd.DataFrame:
        """Display Corpus metadata, one attribute per row, in a dataframe.

        Args:
            drop (List[str]): A list of of rows to drop from the table.

        Returns:
            pd.DataFrame: A pandas dataframe containing the table.
        """
        reduced = {k: v for k, v in self.dict().items() if k not in drop}
        df = pd.DataFrame.from_dict(reduced, orient="index", columns=["Corpus"])
        df.sort_index(axis=0, inplace=True)
        return df

    def records_table(
        self,
        columns: List[str] = [
            "id",
            "name",
            "filename",
            "num_tokens",
            "num_terms",
            "is_active",
            "is_parsed",
        ],
        exclude: List[str] = None,
    ) -> pd.DataFrame:
        """Display each document, one per row, in a dataframe.

        Args:
            columns (List[str]): A list of of columns to include in the table.
            exclude (List[str]): A list of columns to exclude from the table.

        Returns:
            pd.DataFrame: A pandas dataframe containing the table.
        """
        df = pd.DataFrame.from_records(self.meta, columns=columns, exclude=exclude)
        df.fillna("", inplace=True)
        return df

    def remove(self, id: int):
        """Remove a Corpus record by id.

        Args:
            id (int): A record id.
        """
        if not id:
            raise ValueError("Please supply a record ID.")
        else:
            id_idx = self.ids.index(id)
        # Get the record entry in Corpus.meta
        record = next(item for item in self.meta if item["id"] == id)
        # Remove the record file
        os.remove(record.filename)
        # Remove the id and docs entries
        del self.ids[id_idx]
        if id in self.docs.keys():
            del self.docs[id]
        del self.meta[id_idx]
        # Decrement the token/term/record counts
        self.num_tokens = self.num_tokens - record.num_tokens()
        self.num_terms = self.num_terms - record.num_terms()
        # Save the Corpus metadata
        srsly.write_json(f"{self.corpus_dir}/corpus_meta.json", self.dict())

    def remove_records(self, ids: List[int]):
        """Remove multiple records from the corpus.

        Args:
            ids (List[int]): A list of record ids to remove.
        """
        for id in ids:
            self.remove(id)

    def set(self, id: int, **props):
        """Set a property or properties of a record in the Corpus.

        Args:
            id (int): A document id.
            **props (dict): The dict containing any other properties to set.
        """
        record = self.get(id)
        old_filename = record.filename
        for k, v in props.items():
            record.set(k, v)
        if record.filename not in os.listdir(f"{self.corpus_dir}/docs"):
            self.save(record, record.filename)
            if os.path.isfile(old_filename):
                os.remove(old_filename)
            # Update corpus table data here
            update = {
                "id": record.id,
                "name": record.name,
                "filename": record.filename,
                "num_terms": record.num_terms(),
                "num_tokens": record.num_tokens(),
                "is_active": record.is_active,
                "is_parsed": record.is_parsed,
            }
            self.meta = [{**d, **update} for d in self.meta]
        else:
            raise ValueError(
                f"A file with the name `{record.filename}` already exists."
            )

    def _add_to_corpus(self, record: object, cache: bool = False):
        """Add a record to the Corpus.

        Args:
            record (object): A Record doc.
        """
        # Update corpus records table
        meta = record.dict()
        del meta["content"]
        meta["num_tokens"] = record.num_tokens()
        meta["num_terms"] = record.num_terms()
        self.meta.append(meta)
        # Save the record to disk
        self._to_disk(record, record.filename)
        # Update the Corpus ids and names
        self.ids.append(record.id)
        self.names.append(record.name)
        # Update the Corpus cache
        if cache:
            self.docs[record.id] = record
        # Update the Corpus statistics
        self.num_docs += 1
        if record.is_active:
            self.num_active_docs += 1
        self.num_tokens += record.num_tokens()
        for term in list(record.terms):
            self.terms.add(term)
        self.num_terms = len(self.terms)

    def _ensure_unique_name(self, name: str = None) -> str:
        """Ensure that no names are duplicated in the Corpus.

        Args:
            name (str): The record name.

        Returns:
            A string.
        """
        if not name:
            name = str(uuid.uuid1())
        if name in self.names:
            name = f"{name}_{uuid.uuid1()}"
        return name

    def _ensure_unique_filename(self, name: str = None) -> str:
        """Ensure that no filenames are duplicated in the Corpus.

        Args:
            name (str): The record name (on which the filename will be based).

        Returns:
            A filepath.
        """
        if not name:
            name = str(uuid.uuid1())
        if name in self.names:
            name = f"{name}_{self.id}"
        docs_dir = f"{self.corpus_dir}/docs"
        filepath = f"{docs_dir}/{name}.pkl"
        try:
            assert filepath not in os.listdir(docs_dir)
            return filepath
        except AssertionError:
            raise AssertionError(
                "Could not make a unique filepath. Try changing the record name."
            )

    def _from_disk(self, filename) -> object:
        """Deserialise a record file from disk.

        Args:
            filename: The full path to the record file.

        Returns:
            Record: A Corpus record
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def _get_new_id(self) -> int:
        """Get the highest id in the ids list.

        Returns:
            int: An id.
        """
        if self.ids == []:
            return 1
        else:
            return max(self.ids) + 1

    def _to_disk(self, record, filename):
        """Serialise a record file to disk.

        Args:
            filename: The full path to the record file.
        """
        with open(filename, "wb") as f:
            pickle.dump(record, f)
