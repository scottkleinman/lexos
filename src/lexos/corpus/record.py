"""record.py.

Last updated: December 4, 2025
Last tested: November 20, 2025


Wrapping texts and spaCy Docs in a Pydantic model provides a lot of extra functionality, particularly through the model_dump() and model_dump_json() methods. See the Pydantic documentation for more information.

Other than that, the Record class provides methods for serializing and deserializing the record to and from bytes, saving and loading the record to and from disk, and calculating various statistics about the record, such as the number of terms, tokens, vocabulary density, and most/least common terms.

The Record class handles the difficult task of keeping track of whether the content is a spaCy Doc or a string, as well as the tricky job of preserving custom Token attributes when spaCy Docs are serialised and deserialised.

This code is designed to work by default with UUID4 for the ID field, which is a universally unique identifier. UUID7 is a better choice but does not yet have full support in the Python standard library and Pydantic. Once that takes place, it can be easily changed in the Record model. Alternatively, the ID can be set to an incrementing integer with `id_type="integer"`.
"""

import hashlib
import uuid
from collections import Counter
from datetime import date, datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import msgpack
import spacy
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    validate_call,
)
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from lexos.corpus.utils import LexosModelCache
from lexos.exceptions import LexosException


class Record(BaseModel):
    """The main Record model."""

    id: int | UUID4 = uuid.uuid4()
    name: Optional[str] = None
    is_active: Optional[bool] = True
    content: Optional[Doc | str] = None
    model: Optional[str] = None
    extensions: list[str] = Field(default_factory=list)
    data_source: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        json_schema_extra=DocJSONSchema.schema(),
    )

    @field_serializer("content")
    def serialize_content(self, content: Doc | str) -> bytes | str:
        """Serialize the content to bytes if it is a Doc object.

        Args:
            content (Doc | str): The content to serialize.

        Returns:
            bytes | str: The serialized content as bytes if it is a Doc, otherwise the original string.
        """
        if isinstance(content, Doc):
            content.user_data["extensions"] = {}
            for ext in self.extensions:
                content.user_data["extensions"][ext] = [
                    token._.get(ext) for token in content
                ]
            return content.to_bytes()
        return content

    @field_serializer("id")
    def serialize_id(self, id, _info) -> str:
        """Always serialize ID as string for JSON compatibility.

        Args:
            id (UUID|int|str): The ID value being serialized.
            _info (Any): Encoder info (pydantic serializer internals).

        Returns:
            str: The serialized ID as a string.
        """
        return str(id)

    @field_serializer("meta")
    def serialize_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata is JSON-serializable by converting special types to strings."""
        return self._sanitize_metadata(meta)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert non-JSON-serializable types to strings.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Sanitized metadata dictionary with JSON-serializable values
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, UUID):
                sanitized[key] = str(value)
            elif isinstance(value, (datetime, date)):
                sanitized[key] = value.isoformat()
            elif isinstance(value, Path):
                sanitized[key] = str(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_metadata(value)  # Recursive
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_metadata({"item": item})["item"]
                    if isinstance(item, dict)
                    else str(item)
                    if isinstance(item, (UUID, datetime, date, Path))
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def __repr__(self):
        """Return a string representation of the record."""
        # We exclude `terms`, `text`, and `tokens` here because these are
        # computed / cached fields that can rely on the record being parsed.
        # For unparsed records, evaluating these computed properties will
        # raise a LexosException. `__repr__` should be lightweight and safe
        # to call in debugging contexts, so we exclude these computed fields
        # intentionally.
        fields = self.model_dump(exclude=["terms", "text", "tokens"])
        fields["is_parsed"] = str(self.is_parsed)
        if self.content and self.is_parsed:
            fields["content"] = f"{self.content.text[:25]}..."
        elif self.content and not self.is_parsed:
            fields["content"] = f"{self.content[:25]}..."
        else:
            fields["content"] = "None"
        field_list = [f"{k}={v}" if v else f"{k}=None" for k, v in fields.items()]
        return f"Record({', '.join(field_list)})"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the record for printing."""
        active = "True" if self.is_active else "False"
        parsed = "True" if self.is_parsed else "False"

        # Get a preview of content
        if self.content is None:
            content_preview = "None"
        elif self.is_parsed:
            content_preview = f"'{self.content.text[:40]}...'"
        else:
            content_preview = f"'{self.content[:40]}...'"

        return f"Record(id={self.id}, name={self.name!r}, active={active}, parsed={parsed}, content={content_preview})"

    @computed_field
    @cached_property
    def is_parsed(self) -> bool:
        """Return whether the record is parsed.

        Returns:
            bool: True if the record content is a spaCy Doc, False otherwise.
        """
        if isinstance(self.content, Doc):
            return True
        return False

    @computed_field
    @cached_property
    def preview(self) -> str:
        """Return a preview of the record text.

        Returns:
            str | None: A shortened preview of the record content, or None if content is None.
        """
        if self.content is None:
            return None

        if self.is_parsed:
            return f"{self.content.text[0:50]}..."
        return f"{self.content[0:500]}..."

    @computed_field
    @cached_property
    def terms(self) -> Counter:
        """Return the terms in the record.

        Returns:
            Counter: Collection mapping term -> count for the record.
        """
        if self.is_parsed:
            return Counter([t.text for t in self.content])
        else:
            raise LexosException("Record is not parsed.")

    @property
    def text(self) -> str:
        """Return the text of the record.

        Returns:
            str | None: The record text as string or None if no content is present.
        """
        if self.is_parsed:
            return self.content.text
        return self.content

    @cached_property
    def tokens(self) -> list[str]:
        """Return the tokens in the record.

        Returns:
            list[str]: A list of token strings extracted from the parsed content.
        """
        if self.is_parsed:
            return [t.text for t in self.content]
        else:
            raise LexosException("Record is not parsed.")

    def _doc_from_bytes(
        self,
        content: bytes,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> Doc:
        """Convert bytes to a Doc object.

        Args:
            content (bytes): The bytes to convert.
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.

        Returns:
            Doc: The content as a Doc object.
        """
        # Create a Doc from the bytes
        vocab = self._get_vocab(model, model_cache)
        doc = Doc(vocab).from_bytes(content)

        # Restore extension values
        for ext, values in doc.user_data["extensions"].items():
            Token.set_extension(ext, default=None, force=True)
            for i in range(len(doc)):
                doc[i]._.set(ext, values[i])

        # Clean up user_data
        doc.user_data["extensions"] = list(doc.user_data["extensions"].keys())

        return doc

    # WARNING: This method is deprecated in favour of field serializer.
    def _doc_to_bytes(self) -> bytes:
        """Convert the content to bytes if it is a Doc object.

        Returns:
            bytes: The content as bytes.
        """
        if not isinstance(self.content, Doc):
            raise LexosException("Content is not a Doc object.")

        doc = self.content

        doc.user_data["extensions"] = {}
        for ext in self.extensions:
            doc.user_data["extensions"][ext] = [token._.get(ext) for token in doc]

        return doc.to_bytes()

    def _get_vocab(
        self, model: Optional[str] = None, model_cache: Optional[LexosModelCache] = None
    ) -> Vocab:
        """Get the vocabulary from the model or model cache.

        Args:
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.

        Returns:
            Vocab: The vocabulary of the model.
        """
        if model_cache and not model:
            raise LexosException("Model cache provided but no model specified.")

        if model_cache:
            return model_cache.get_model(model).vocab
        elif model:
            return spacy.load(model).vocab
        elif self.model:
            return spacy.load(self.model).vocab
        else:
            raise LexosException(
                "No model specified for loading the Doc. Please provide a model name or a model cache."
            )

    @validate_call(config=model_config)
    def from_bytes(
        self,
        bytestring: bytes,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
        verify_hash: bool = True,
    ) -> None:
        """Deserialise the record from bytes.

        Args:
            bytestring (bytes): The bytes to load the record from.
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.
            verify_hash (bool): Whether to verify data integrity hash. Defaults to True.
        """
        try:
            data = msgpack.unpackb(bytestring)
        except Exception as e:
            raise LexosException(
                f"Failed to deserialize record: Invalid or corrupted data format. "
                f"Suggestion: Check if the file was completely written and not corrupted."
            ) from e

        # Verify data integrity if hash is present
        if verify_hash and "data_integrity_hash" in data:
            stored_hash = data["data_integrity_hash"]
            # Recreate hash from core data (excluding the hash itself)
            core_data = {k: v for k, v in data.items() if k != "data_integrity_hash"}
            core_bytes = msgpack.dumps(core_data)
            computed_hash = hashlib.sha256(core_bytes).hexdigest()

            if stored_hash != computed_hash:
                raise LexosException(
                    f"Data integrity check failed: Hash mismatch detected. "
                    f"Expected: {stored_hash[:16]}..., Got: {computed_hash[:16]}... "
                    f"Suggestion: The data may be corrupted during storage or transmission. "
                    f"Try re-serializing the original document."
                )

        # Update the record with the loaded data
        for k, v in data.items():
            if k in self.model_fields:
                if k != "content":
                    setattr(self, k, v)

        # If content is bytes, convert it back to a Doc object
        if data["is_parsed"] and isinstance(data["content"], bytes):
            if not model:
                model = data.get("model")
            try:
                self.content = self._doc_from_bytes(data["content"], model, model_cache)
            except OSError as e:
                raise LexosException(
                    f"Failed to load spaCy model '{model}': {str(e)}. "
                    f"Suggestion: Install the model with 'python -m spacy download {model}' "
                    f"or use a different model available in your environment."
                ) from e
            except Exception as e:
                raise LexosException(
                    f"Failed to deserialize spaCy document with model '{model}': {str(e)}. "
                    f"Suggestion: Check model compatibility - document may have been "
                    f"serialized with a different spaCy or model version."
                ) from e

    @validate_call(config=model_config)
    def from_disk(
        self,
        path: Path | str,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> None:
        """Load the record from disk.

        Args:
            path (Path | str): The path to load the record from.
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.
        """
        if not path:
            raise LexosException("No path specified for loading the record.")

        # Load the data from disk
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError as e:
            raise LexosException(
                f"Record file not found: {path}. "
                f"Suggestion: Check if the file path is correct and the file exists."
            ) from e
        except PermissionError as e:
            raise LexosException(
                f"Permission denied accessing record file: {path}. "
                f"Suggestion: Check file permissions or run with appropriate privileges."
            ) from e
        except IOError as e:
            raise LexosException(
                f"Failed to read record file: {path}. Error: {str(e)}. "
                f"Suggestion: Check disk space, file system health, or network connectivity."
            ) from e

        # Get the record content from the bytestring
        self.from_bytes(data, model=model, model_cache=model_cache)

    def least_common_terms(self, n: Optional[int] = None) -> list[tuple[str, int]]:
        """Return the least common terms.

        Args:
            n (Optional[int]): The number of least common terms to return. If None, return all terms.

        Returns:
            list[tuple[str, int]]: A list of (term, count) pairs sorted by least frequent.
        """
        if self.is_parsed:
            return (
                sorted(self.terms.items(), key=lambda x: x[1])[:n]
                if n
                else sorted(self.terms.items(), key=lambda x: x[1])
            )
        else:
            raise LexosException("Record is not parsed.")

    def most_common_terms(self, n: Optional[int] = None) -> list[tuple[str, int]]:
        """Return the most common terms.

        Args:
            n (Optional[int]): The number of most common terms to return. If None, return all terms.

        Returns:
            list[tuple[str, int]]: A list of (term, count) pairs sorted by most frequent.
        """
        if self.is_parsed:
            return self.terms.most_common(n)
        else:
            raise LexosException("Record is not parsed.")

    def num_terms(self) -> int:
        """Return the number of terms.

        Returns:
            int: The count of unique terms in this record.
        """
        if self.is_parsed:
            return len(self.terms)
        else:
            raise LexosException("Record is not parsed.")

    def num_tokens(self) -> int:
        """Return the number of tokens.

        Returns:
            int: The count of token elements in this record.
        """
        if self.is_parsed:
            return len(self.tokens)
        else:
            raise LexosException("Record is not parsed.")

    @validate_call(config=model_config)
    def set(self, **props: Any) -> None:
        """Set a record property.

        Args:
            **props (Any): A dict containing the properties to set on the record.

        Returns:
            None
        """
        for k, v in props.items():
            setattr(self, k, v)

    @validate_call(config=model_config)
    def to_bytes(
        self, extensions: Optional[list[str]] = [], include_hash: bool = True
    ) -> bytes:
        """Serialize the record to a dictionary.

        Args:
            extensions (list[str]): A list of extension names to include in the serialization.
            include_hash (bool): Whether to include data integrity hash. Defaults to True.

        Returns:
            bytes: The serialized record.
        """
        # Handle extensions
        if extensions:
            self.extensions = list(set(self.extensions + extensions))

        # Convert record to a dictionary
        # model_dump is used to create a serializable dict representation.
        # We exclude the computed fields (`terms`, `text`, `tokens`) because
        # they might trigger evaluation and raise `LexosException` for
        # unparsed `Record` objects. The saved content is handled below,
        # and `id` is stringified to ensure JSON compatibility.
        data = self.model_dump(exclude=["terms", "text", "tokens"])

        # Make UUID serialisable
        data["id"] = str(data["id"])

        # WARNING: This code is deprecated in favour of field serializer.
        # Convert the content to bytes if it is a Doc object
        if self.is_parsed:
            data["content"] = self._doc_to_bytes()

        # Add data integrity hash if requested
        if include_hash:
            # Create hash of the core data (excluding the hash itself)
            core_data = {k: v for k, v in data.items() if k != "data_integrity_hash"}
            core_bytes = msgpack.dumps(core_data)
            data["data_integrity_hash"] = hashlib.sha256(core_bytes).hexdigest()

        return msgpack.dumps(data)

    @validate_call(config=model_config)
    def to_disk(self, path: Path | str, extensions: Optional[list[str]] = None) -> None:
        """Save the record to disk.

        Args:
            path (Path | str): The path to save the record to.
            extensions (list[str]): A list of extension names to include in the serialization.
        """
        if not path:
            raise LexosException("No path specified for saving the record.")

        if not extensions:
            extensions = self.extensions

        # Serialize and save the record
        data = self.to_bytes(extensions)

        try:
            with open(path, "wb") as f:
                f.write(data)
        except PermissionError as e:
            raise LexosException(
                f"Permission denied writing to: {path}. "
                f"Suggestion: Check file/directory permissions or run with appropriate privileges."
            ) from e
        except OSError as e:
            if "No space left on device" in str(e):
                raise LexosException(
                    f"Insufficient disk space to save record: {path}. "
                    f"Suggestion: Free up disk space or choose a different location."
                ) from e
            else:
                raise LexosException(
                    f"Failed to write record to disk: {path}. Error: {str(e)}. "
                    f"Suggestion: Check disk space, file system health, or network connectivity."
                ) from e

    def vocab_density(self) -> float:
        """Return the vocabulary density.

        Returns:
            float: The vocabulary density of the record.
        """
        if self.is_parsed:
            return self.num_terms() / self.num_tokens()
        else:
            raise LexosException("Record is not parsed.")
