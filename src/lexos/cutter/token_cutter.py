"""token_cutter.py.

This class assumes that the docs consist of spaCy Doc objects.

Last Updated: 23 December, 2025
Tested: 23 December, 2025
"""

from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import spacy
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span

from lexos.cutter.spacy_attrs import ENTITY_HEADER, SPACY_ATTRS
from lexos.exceptions import LexosException
from lexos.util import ensure_list, strip_doc

# Configuration validation for the DocCutter class
validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)


class TokenCutter(BaseModel, validate_assignment=True):
    """TokenCutter class for chunking spaCy Doc objects into smaller segments.

    based on token count, line breaks, sentences, or custom milestones.
    Supports overlapping, merging, and export to disk.
    """

    chunks: list[list[Doc]] = Field(default=[], description="The list of chunks.")

    docs: Optional[Doc | list[Doc] | Path | str | list[Path | str]] = Field(
        default=None,
        description="The documents to be split.",
    )
    chunksize: Optional[int] = Field(
        default=1000, gt=0, description="The desired chunk size in tokens."
    )
    n: Optional[int] = Field(
        default=None,
        # gt=0, Removed to allow runtime validation via LexosException instead of Pydantic pre-validation for testing coverage.
        description="The number of chunks or the number of lines or sentences per chunk.",
    )
    names: Optional[list[str]] = Field(
        default=[], description="A list of names for the source docs."
    )
    newline: Optional[bool] = Field(
        default=False, description="Whether to chunk by lines."
    )
    merge_threshold: Optional[float] = Field(
        default=0.5, ge=0, le=1, description="The threshold to merge the last segment."
    )
    overlap: Optional[int] = Field(
        default=None, gt=0, description="The number of tokens to overlap."
    )
    output_dir: Optional[Path | str] = Field(
        default=None, description="The output directory to save the chunks to."
    )
    delimiter: str = Field(
        default="_", description="The delimiter to use for the chunk names."
    )
    pad: int = Field(default=3, gt=0, description="The padding for the chunk names.")
    strip_chunks: bool = Field(
        default=True,
        description="Whether to strip leading and trailing whitespace in the chunks.",
    )

    model_config = validation_config

    def __iter__(self) -> Iterator:
        """Iterate over the object's chunks.

        Returns:
            Iterator: An iterator containing the object's chunks.
        """
        return iter(self.chunks)

    def __len__(self):
        """Return the number of source docs in the instance."""
        if not self.docs:
            return 0
        return len(self.docs)

    @staticmethod
    def list_start_end_indexes(arrays: list[np.ndarray]) -> list[tuple[int, int]]:
        """List start and end indexes for a list of numpy arrays.

        Args:
            arrays (list[np.ndarray]): List of numpy arrays.

        Returns:
            list[tuple[int, int]]: List of tuples with start and end indexes.
        """
        indexes = []
        start = 0

        for array in arrays:
            end = start + len(array)
            indexes.append((start, end))
            start = end

        return indexes

    def _apply_merge_threshold(
        self, chunks: list[Doc], force: bool = False
    ) -> list[Doc]:
        """Apply the merge threshold to the last chunk.

        Args:
            chunks (list[Doc]): The list of chunks.
            force (bool, optional): Whether to force the merge. Defaults to False.

        Returns:
            list[Doc]: The list of chunks with the last chunk merged if necessary.

        Notes:
          - Whitespace is supplied between merged chunks.
          - Length of final chunk is measured in number tokens or number of sentences.
        """
        if len(chunks) == 1:
            return chunks
        merge_threshold = (
            self.merge_threshold if self.merge_threshold is not None else 0.5
        )
        if isinstance(self.n, int):
            threshold = max([len(chunk) for chunk in chunks]) * merge_threshold
        else:
            threshold = (
                self.chunksize if self.chunksize is not None else 1
            ) * merge_threshold
        # If the length of the last chunk < threshold, merge it with the previous chunk
        if force is True or len(chunks[-1]) < threshold:
            # Get rid of the last chunk
            last_chunk = chunks.pop(-1)
            # Combine the last two segments into a single doc
            chunks[-1] = Doc.from_docs([chunks[-1], last_chunk])
        return chunks

    def _apply_overlap(
        self,
        chunks: list[Doc],
    ) -> list[Doc]:
        """Create overlapping chunks.

        Args:
            chunks (list[Doc]): A list of spaCy docs.

        Returns:
            list[Doc]: A list of spaCy docs.
        """
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                overlap_doc = chunks[i + 1][: self.overlap].as_doc()
                overlapped_doc = Doc.from_docs([chunk, overlap_doc])
                overlapped_chunks.append(overlapped_doc)
            elif i == len(chunks) - 1:
                overlapped_chunks.append(chunk)
        return overlapped_chunks

    def _chunk_doc(
        self,
        doc: Doc,
        attrs: "Sequence[int | str]" = SPACY_ATTRS,
        header: Sequence[int | str] = ENTITY_HEADER,
    ) -> list[Doc]:
        """Split a Doc into chunks.

        Args:
            doc: The Doc to split.
            attrs: The attributes to include in the chunks.
            header: The NER attributes to include in the chunks.

        Returns:
            list[Doc]: List of Doc chunks.
        """
        # Check that the document is not empty
        if len(doc) == 0:
            raise LexosException("Document is empty.")

        # Return the whole doc if it is less than the chunksize
        if self.n is None and self.chunksize is not None and len(doc) <= self.chunksize:
            return [doc]

        # Get the names of the custom extensions
        extension_names = [name for name in doc[0]._.__dict__["_extensions"].keys()]

        # Split the doc into n chunks
        if isinstance(self.n, int):
            chunks_arr = np.array_split(doc.to_array(list(attrs)), self.n)
            # If there is only one chunk, skip the rest of the function
            if len(chunks_arr) == 1:
                return [doc]
        else:
            chunks_arr = np.array_split(
                doc.to_array(list(attrs)),
                np.arange(self.chunksize, len(attrs), self.chunksize),
            )
            # Remove empty elements
            chunks_arr = [x for x in chunks_arr if x.size > 0]

        # Create a list to hold the chunks and get the chunk indexes
        chunks = []
        chunk_indexes = TokenCutter.list_start_end_indexes(chunks_arr)

        # Iterate over the chunks
        for i, chunk in enumerate(chunks_arr):
            # Get chunk start and end indexes
            start = chunk_indexes[i][0]
            end = chunk_indexes[i][1]
            span = doc[start:end]
            words = [token.text for token in span]

            # Make a new doc for the chunk
            new_doc = Doc(doc.vocab, words=words)

            # Add the attributes to the new chunk doc
            new_doc.from_array(list(attrs), chunk)

            # Add entities to the new chunk doc
            if doc.ents and len(doc.ents) > 0:
                ent_array = np.empty((len(chunk), len(header)), dtype="uint64")
                for i, token in enumerate(span):
                    ent_array[i, 0] = token.ent_iob
                    ent_array[i, 1] = token.ent_type
                new_doc.from_array(list(header), ent_array)

            # Add custom attributes to doc
            if len(extension_names) > 0:
                for i, token in enumerate(span):
                    for ext in extension_names:
                        new_doc[i]._.set(ext, token._.get(ext))

            # Add the chunk to the chunks list
            chunks.append(new_doc)

        # Return the list of chunks
        return chunks

    def _keep_milestones_bool(
        self, doc: Doc, milestones: list[Span], keep_spans: bool = False
    ) -> list[Doc]:
        """Split a spaCy Doc into chunks on milestones, optionally keeping milestones.

        Args:
            doc (Doc): The spaCy Doc to split.
            milestones (list[Span]): The milestones to split on.
            keep_spans (bool): Whether to keep the spans in the split strings.

        Returns:
            list[Doc]: A list of spaCy Docs.
        """
        chunks = []
        start = 0
        for span in milestones:
            if span.start == 0 or span.end == doc[-1].i:
                if keep_spans:
                    chunks.append(span)
            elif start < span.start:
                chunks.append(doc[start : span.start])
                if keep_spans:
                    chunks.append(span)
            start = span.end
        if start < len(doc):
            chunks.append(doc[start:])
        return chunks

    def _keep_milestones_following(self, doc: Doc, milestones: list[Span]) -> list[Doc]:
        """Split a spaCy Doc into chunks on milestones preserving milestones in the following chunk.

        Args:
            doc (Doc): The spaCy Doc to split.
            milestones (list[Span]): The milestones to split on.

        Returns:
            list[Doc]: A list of spaCy Docs.
        """
        chunks = []
        start = 0
        for index, span in enumerate(milestones):
            # Text before milestone
            if start < span.start:
                chunks.append(doc[start : span.start])

            # Find end of chunk (next milestone or doc end)
            end = (
                milestones[index + 1].start if index < len(milestones) - 1 else len(doc)
            )

            # Milestone + following text as one chunk
            chunks.append(doc[span.start : end])
            start = end
        return chunks

    def _keep_milestones_preceding(self, doc: Doc, milestones: list[Span]) -> list[Doc]:
        """Split a spaCy Doc into chunks on milestones preserving milestones in the preceding chunk.

        Args:
            doc (Doc): The spaCy Doc to split.
            milestones (list[Span]): The milestones to split on.

        Returns:
            list[Doc]: A list of spaCy Docs.
        """
        # Check that the document is not empty
        if len(doc) == 0:
            raise LexosException("Document is empty.")
        if len(milestones) == 0:
            return [doc]
        chunks = []
        start = 0
        for span in milestones:
            index = span.start
            if index != -1:
                chunks.append(doc[start : index + len(span)])
                start = index + len(span)
        if start < len(doc):
            chunks.append(doc[start:])
        if milestones[0].start == 0:
            _ = chunks.pop(0)
            chunks[0] = doc[: chunks[0].end]
        return chunks

    def _set_attributes(self, **data) -> None:
        """Set attributes after initialization."""
        for key, value in data.items():
            setattr(self, key, value)

    def _split_doc(
        self,
        doc: Doc,
        attrs: Optional[Sequence[int | str]] = SPACY_ATTRS,
        merge_final: Optional[bool] = False,
    ) -> list[Doc]:
        """Split a spaCy doc into chunks by a fixed number of tokens.

        Args:
            doc (Doc): A spaCy doc.
            attrs (Optional[int | str]): The spaCy attributes to include in the chunks.
            merge_final (Optional[bool]): Whether to merge the final segment.

        Returns:
            list[Doc]: A list of spaCy docs.
        """
        if len(doc) == 0:
            raise LexosException("Document is empty.")

        attrs = attrs if attrs is not None else SPACY_ATTRS
        chunks = self._chunk_doc(doc, attrs)
        chunks = self._apply_merge_threshold(
            chunks, force=merge_final if merge_final is not None else False
        )
        if self.overlap:
            chunks = self._apply_overlap(chunks)
        if self.strip_chunks:
            return [strip_doc(chunk) for chunk in chunks]
        # Ensure that all chunks are spaCy docs
        else:
            return [
                chunk.as_doc() if isinstance(chunk, Span) else chunk for chunk in chunks
            ]

    def _split_doc_by_lines(
        self, doc: Doc, merge_final: Optional[bool] = False
    ) -> list[Doc]:
        """Split a spaCy Doc into chunks of n lines.

        Args:
            doc: spaCy Doc to split.
            merge_final: Whether to merge the final segment.

        Returns:
            list[Doc]: Chunks of the doc split by lines.
        """
        if len(doc) == 0:
            raise LexosException("Document is empty.")

        indices = []  # The indices immediately following the newline tokens
        count = 0
        chunks = []
        for token in doc:
            if "\n" in token.text:
                count += 1
                if (
                    self.n is not None and count % self.n == 0
                ):  # Check if it's the nth occurrence
                    indices.append(token.i + 1)
        if len(indices) == 0:
            chunks.append(doc)
        else:
            prev_index = 0
            for index in indices:
                chunks.append(doc[prev_index:index].as_doc())
                prev_index = index
            chunks.append(doc[prev_index:].as_doc())  # Append the remaining elements

        # Ensure there are no empty docs
        chunks = [chunk for chunk in chunks if len(chunk) > 0]

        # Apply the merge threshold and overlap
        chunks = self._apply_merge_threshold(
            chunks, force=merge_final if merge_final is not None else False
        )
        if self.overlap:
            chunks = self._apply_overlap(chunks)

        if self.strip_chunks:
            return [strip_doc(chunk) for chunk in chunks]

        return chunks

    def _split_doc_by_sentences(
        self, doc: Doc, merge_final: Optional[bool] = False
    ) -> list[Doc]:
        """Split a spaCy Doc into chunks of n sentences.

        Args:
            doc: A spaCy Doc object.
            merge_final: Whether to merge the final segment.

        Returns:
            Doc: Chunks containing n sentences each (last chunk may have fewer).
        """
        if len(doc) == 0:
            raise LexosException("Document is empty.")

        try:
            next(doc.sents)
        except (StopIteration, ValueError):
            raise LexosException("The document has no assigned sentences.")

        # Split the doc into chunks of n sentences
        sents = list(doc.sents)
        chunks = []
        n = self.n if self.n is not None else 1
        for i in range(0, len(sents), n):
            chunk_sents = sents[i : i + n]
            start_idx = chunk_sents[0].start
            end_idx = chunk_sents[-1].end
            chunks.append(doc[start_idx:end_idx].as_doc())
        # No need to append doc[end_idx:] since all sentences are already included in the chunks

        # Ensure there are no empty docs
        chunks = [chunk for chunk in chunks if len(chunk) > 0]

        # Apply the merge threshold and overlap
        chunks = self._apply_merge_threshold(
            chunks, force=merge_final if merge_final is not None else False
        )
        if self.overlap:
            chunks = self._apply_overlap(chunks)

        if self.strip_chunks:
            return [strip_doc(chunk) for chunk in chunks]

        return chunks

    def _split_doc_on_milestones(
        self,
        doc: Doc,
        milestones: Span | list[Span],
        keep_spans: Optional[bool | str] = False,
        merge_final: Optional[bool] = False,
    ) -> list[Doc]:
        """Split document on a milestone.

        Args:
            doc (Doc): The document to be split.
            milestones (Span | list[Span]): A Span or list of Spans to be matched.
            keep_spans (Optional[bool | str]): Whether to keep the spans in the split strings. Defaults to False.
            merge_final (Optional[bool]): Whether to force the merge of the last segment. Defaults to False.

        Returns:
            list[Doc]: A list of chunked spaCy Doc objects.
        """
        if len(doc) == 0:
            raise LexosException("Document is empty.")

        milestones = ensure_list(milestones)
        if keep_spans == "following":
            chunks = self._keep_milestones_following(doc, milestones)
        elif keep_spans == "preceding":
            chunks = self._keep_milestones_preceding(doc, milestones)
        else:
            # Only pass a boolean to keep_spans
            chunks = self._keep_milestones_bool(
                doc, milestones, keep_spans=bool(keep_spans)
            )

        # Ensure that all chunks are spaCy docs
        chunks = [
            chunk.as_doc() if isinstance(chunk, Span) else chunk for chunk in chunks
        ]

        # Apply the merge threshold and overlap
        chunks = self._apply_merge_threshold(
            chunks, force=merge_final if merge_final is not None else False
        )
        if self.overlap:
            chunks = self._apply_overlap(chunks)

        if self.strip_chunks:
            return [strip_doc(chunk) for chunk in chunks]

        return chunks

    def _write_chunk(
        self, path: str, n: int, chunk: Doc, output_dir: Path, as_text: bool = True
    ) -> None:
        """Write chunk text to file with formatted name.

        Args:
            path (str): The path of the original file.
            n (int): The number of the chunk.
            chunk (Doc): The chunk to save.
            output_dir (Path): The output directory for the chunk.
            as_text (bool): Whether to save the chunk as a text file or a spaCy Doc object.
        """
        output_file = f"{path}{self.delimiter}{str(n).zfill(self.pad)}.txt"
        output_path = output_dir / output_file
        if as_text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chunk.text)
        else:
            chunk.to_disk(output_path)

    def merge(self, chunks: list[Doc]) -> Doc:
        """Merge a list of chunks into a single Doc.

        Args:
            chunks (list[Doc]): The list of chunks to merge.

        Returns:
            Doc: The merged doc.

        Note:
            - The user_data dict of the docs will be ignored. If they contain information
              that needs to be preserved, it should be stored in the doc extensions.
              See https://github.com/explosion/spaCy/discussions/9106.
        """
        if len(chunks) == 0:
            raise LexosException("No chunks to merge.")
        return Doc.from_docs(chunks)

    @validate_call(config=validation_config)
    def save(
        self,
        output_dir: Path | str,
        names: Optional[str | list[str]] = None,
        delimiter: Optional[str] = "_",
        pad: Optional[int] = 3,
        strip_chunks: Optional[bool] = True,
        as_text: Optional[bool] = True,
    ) -> None:
        """Save the chunks to disk.

        Args:
            output_dir (Path | str): The output directory to save the chunks to.
            names (Optional[str | list[str]]): The doc names.
            delimiter (str): The delimiter to use for the chunk names.
            pad (int): The padding for the chunk names.
            strip_chunks (bool): Whether to strip leading and trailing whitespace in the chunks.
            as_text (Optional[bool]): Whether to save the chunks as text files or spaCy Doc objects (bytes).
        """
        self._set_attributes(
            output_dir=output_dir,
            delimiter=delimiter,
            names=names,
            pad=pad,
            strip_chunks=strip_chunks,
        )
        if not self.chunks or self.chunks == []:
            raise LexosException("No chunks to save.")
        if self.names:
            if len(self.names) != len(self.chunks):
                raise LexosException(
                    f"The number of docs in `names` ({len(self.names)}) must equal the number of docs in `chunks` ({len(self.chunks)})."
                )
        elif self.names == [] or self.names is None:
            self.names = [
                f"doc{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        for i, doc in enumerate(self.chunks):
            for num, chunk in enumerate(doc):
                if strip_chunks:
                    chunk = strip_doc(chunk)
                self._write_chunk(
                    self.names[i], num + 1, chunk, Path(output_dir), as_text
                )

    @validate_call(config=validation_config)
    def split(
        self,
        docs: Optional[Doc | list[Doc] | Path | str | list[Path | str]] = None,
        chunksize: Optional[int] = None,
        n: Optional[int] = None,
        merge_threshold: Optional[float] = 0.5,
        overlap: Optional[int] = None,
        names: Optional[str | list[str]] = None,
        newline: Optional[bool] = None,
        strip_chunks: Optional[bool] = True,
        file: Optional[bool] = False,
        model: Optional[str] = None,
        merge_final: Optional[bool] = False,
    ) -> list[list[Doc]]:
        """Split spaCy docs into chunks by a fixed number of tokens.

        Args:
            docs (Optional[Doc | list[Doc] | Path | str | list[Path | str]]): A spaCy doc, list of spaCy docs, or file paths to spaCy docs saved with Doc.to_disk().
            chunksize (Optional[int]): The number of tokens to split on.
            n (Optional[int]): The number of chunks to produce.
            merge_threshold (Optional[float]): The threshold to merge the last segment.
            overlap (Optional[int]): The number of tokens to overlap.
            names (Optional[str | list[str]]): The doc names.
            newline (Optional[bool]): Whether to chunk by lines.
            strip_chunks (Optional[bool]): Whether to strip leading and trailing whitespace in the chunks.
            file (Optional[bool]): Whether to load docs from files using Doc.from_disk().
            model (Optional[str]): The name of the spaCy model to use when loading docs from files. Required when file=True.
            merge_final (Optional[bool]): Whether to force the merge of the last segment.

        Returns:
            list[list[Doc]]: A list of spaCy docs (chunks).
        """
        if docs:
            self.docs = ensure_list(docs)
        if not self.docs:
            raise LexosException("No documents provided for splitting.")
        self._set_attributes(
            chunksize=chunksize,
            n=n,
            merge_threshold=merge_threshold,
            overlap=overlap,
            names=names,
            newline=newline,
            strip_chunks=strip_chunks,
        )

        # Load docs from files if file=True
        if file:
            if model is None:
                raise LexosException("model parameter is required when file=True")
            nlp = spacy.load(model)
            loaded_docs = []
            for doc in ensure_list(docs):
                try:
                    doc = Doc(nlp.vocab).from_disk(doc)
                except ValueError:
                    raise LexosException(
                        f"Error loading doc from disk. Doc file must be in a valid spaCy serialization format: see https://spacy.io/api/doc#to_disk"
                    )
                loaded_docs.append(doc)
            docs = loaded_docs

        if self.newline:
            if not self.n:
                self.n = self.chunksize
            if not self.n or self.n < 1:
                raise LexosException("n must be greater than 0.")
            for doc in ensure_list(docs):
                self.chunks.append(
                    self._split_doc_by_lines(doc, merge_final=merge_final)
                )
        else:
            for doc in ensure_list(docs):
                self.chunks.append(self._split_doc(doc, merge_final=merge_final))

        return self.chunks

    @validate_call(config=validation_config)
    def split_on_milestones(
        self,
        milestones: Span | list[Span],
        docs: Optional[Doc | list[Doc] | Path | str | list[Path | str]] = None,
        merge_threshold: Optional[float] = 0.5,
        merge_final: Optional[bool] = False,
        overlap: Optional[int] = None,
        keep_spans: Optional[bool | str] = False,
        strip_chunks: Optional[bool] = True,
        names: Optional[str | list[str]] = None,
        file: Optional[bool] = False,
        model: Optional[str] = None,
    ) -> list[list[Doc]]:
        """Split document on a milestone.

        Args:
            milestones (Span | list[Span]): A milestone span or list of milestone spans to be matched.
            docs (Optional[Doc | list[Doc] | Path | str | list[Path | str]]): The document(s) to be split, or file paths to spaCy docs saved with Doc.to_disk().
            merge_threshold (Optional[float]): The threshold to merge the last segment.
            merge_final (Optional[bool]): Whether to force the merge of the last segment.
            overlap (Optional[int]): The number of tokens to overlap.
            keep_spans (Optional[bool | str]): Whether to keep the spans in the split strings. Defaults to False.
            strip_chunks (Optional[bool]): Whether to strip leading and trailing whitespace in the chunks.
            names (Optional[str | list[str]]): The doc names.
            file (Optional[bool]): Whether to load docs from files using Doc.from_disk().
            model (Optional[str]): The name of the spaCy model to use when loading docs from files. Required when file=True.

        Returns:
            list[list[Doc]]: A list of spaCy docs (chunks).
        """
        if docs:
            self.docs = ensure_list(docs)
        if not self.docs:
            raise LexosException("No documents provided for splitting.")
        self._set_attributes(
            merge_threshold=merge_threshold,
            overlap=overlap,
            strip_chunks=strip_chunks,
            names=names,
        )

        # Load docs from files if file=True
        if file:
            if model is None:
                raise LexosException("model parameter is required when file=True")
            nlp = spacy.load(model)
            loaded_docs = []
            for doc in ensure_list(docs):
                doc = Doc(nlp.vocab).from_disk(doc)
                loaded_docs.append(doc)
            docs = loaded_docs

        for doc in ensure_list(docs):
            chunks = self._split_doc_on_milestones(
                doc, milestones, keep_spans=keep_spans, merge_final=merge_final
            )
            self.chunks.append(chunks)
        return self.chunks

    @validate_call(config=validation_config)
    def split_on_sentences(
        self,
        docs: Doc | list[Doc] | Path | str | list[Path | str],
        n: Optional[int] = None,
        merge_final: Optional[bool] = False,
        overlap: Optional[int] = None,
        strip_chunks: Optional[bool] = True,
        names: Optional[str | list[str]] = None,
        file: Optional[bool] = False,
        model: Optional[str] = None,
    ) -> list[list[Doc]]:
        """Split spaCy docs into chunks by a fixed number of sentences.

        Args:
            docs (Doc | list[Doc] | Path | str | list[Path | str]): A spaCy doc, list of spaCy docs, or file paths to spaCy docs saved with Doc.to_disk().
            n (Optional[int]): The number of sentences per chunk.
            merge_final (Optional[bool]): Whether to merge the last segment.
            overlap (Optional[int]): The number of tokens to overlap.
            strip_chunks (Optional[bool]): Whether to strip leading and trailing whitespace in the chunks.
            names (Optional[str | list[str]]): The doc names.
            file (Optional[bool]): Whether to load docs from files using Doc.from_disk().
            model (Optional[str]): The name of the spaCy model to use when loading docs from files. Required when file=True.

        Returns:
            list[list[Doc]]: A list of spaCy docs (chunks).

        Raises:
            ValueError: If n is less than or equal to 0.
            ValueError: If the model has no sentences.
        """
        self._set_attributes(
            n=n,
            overlap=overlap,
            strip_chunks=strip_chunks,
            names=names,
        )

        # Load docs from files if file=True
        if file:
            if model is None:
                raise LexosException("model parameter is required when file=True")
            nlp = spacy.load(model)
            loaded_docs = []
            for doc in ensure_list(docs):
                doc = Doc(nlp.vocab).from_disk(doc)
                loaded_docs.append(doc)
            docs = loaded_docs

        if not self.n:
            self.n = self.chunksize
        if not self.n or self.n < 1:
            raise LexosException("n must be greater than 0.")
        for i, doc in enumerate(ensure_list(docs)):
            if not doc.has_annotation("SENT_START"):
                raise LexosException(
                    f"The spaCy model used to create the Doc {i} does not have sentence boundary detection. Please use a model that includes the 'senter' or 'parser' pipeline component."
                )
            else:
                next(doc.sents)
            self.chunks.append(
                self._split_doc_by_sentences(doc, merge_final=merge_final)
            )
        return self.chunks

    @validate_call(config=validation_config)
    def to_dict(self, names: Optional[list[str]] = None) -> dict[str, list[str]]:
        """Return the chunks as a dictionary.

        Args:
            names (Optional[list[str]]): A list of names for the doc Docs.

        Returns:
            dict[str, list[str]]: The chunks as a dictionary.
        """
        if names:
            self.names = names
        if not self.names:
            self.names = [
                f"doc{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        return {
            str(name): [chunk.text for chunk in chunks]
            for name, chunks in zip(self.names, self.chunks)
        }
