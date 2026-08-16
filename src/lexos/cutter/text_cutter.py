"""text_cutter.py.

Last Updated: 15 August, 2026
Last Tested: 15 August, 2026
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Generator, Iterator, Optional

from pydantic import BaseModel, Field, validate_call

from lexos.exceptions import LexosException
from lexos.milestones.string_milestones import StringSpan
from lexos.util import ensure_list


class TextCutter(BaseModel, validate_assignment=True):
    """TextCutter class for chunking files and strings containing untokenised text."""

    chunks: list[list[str]] = Field(default_factory=list)

    docs: Optional[Path | str | list[Path | str]] = Field(
        default=None,
        description="The documents to be split.",
    )
    chunksize: Optional[int] = Field(
        default=1_000_000,
        description="The desired chunk size in characters (or bytes if by_bytes=True). When newline=True, this refers to the number of lines per chunk.",
    )
    n: Optional[int] = Field(
        default=None,
        description="When newline=False: the number of chunks to split into. When newline=True: the number of lines per chunk (equivalent to chunksize).",
    )
    names: Optional[list[str | None]] = Field(
        default_factory=list,
        description="A list of names for the doc files/strings.",
    )
    newline: Optional[bool] = Field(
        default=False, description="Whether to chunk by lines."
    )
    overlap: Optional[int] = Field(
        default=None, description="The number of characters to overlap between chunks."
    )
    by_bytes: Optional[bool] = Field(
        default=False, description="Whether to chunk by bytes instead of characters."
    )
    output_dir: Optional[Path | str] = Field(
        default=None, description="The output directory to save the chunks to."
    )
    merge_threshold: Optional[float] = Field(
        default=0.5, description="The threshold for merging the last two chunks."
    )
    merge_final: Optional[bool] = Field(
        default=False,
        description="Whether to merge the last two chunks.",
    )
    delimiter: str = Field(
        default="_",
        description="The delimiter to use for the chunk names.",
    )
    pad: int = Field(default=3, description="The padding for the chunk names.")
    strip_chunks: bool = Field(
        default=True,
        description="Whether to strip leading and trailing whitespace in the chunks.",
    )

    def __iter__(self) -> Iterator:
        """Make the class iterable.

        Returns:
            Iterator: An iterator containing the object's chunks.
        """
        return iter(self.chunks)

    def __len__(self):
        """Return the number of docs in the instance."""
        if not self.docs:
            return 0
        return len(self.docs)

    def _calculate_chunk_size(self, size: int, n: int) -> tuple[int, int]:
        """Calculate chunk size and remainder for n chunks.

        Args:
            size (int): Total size of file in bytes.
            n (int): Number of chunks to create.

        Returns:
            tuple [int, int]: (chunk_size, remainder)
        """
        chunk_size = size // n
        remainder = size % n
        return chunk_size, remainder

    def _normalize_text(self, text: str) -> str:
        """Normalize line endings for text content."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _get_name(self, doc: Path | str, index: int) -> str:
        """Generate a filename based on doc or fallback rules.

        Args:
            doc (Path | str): Original file path or doc label.
            index (int): Index of the doc being processed.

        Returns:
            str: A formatted name for saving the chunked output.
        """
        if len(self.names) > 0:
            return self.names[index]
        elif isinstance(doc, Path):
            return Path(doc).stem
        else:
            return f"doc{str(index).zfill(self.pad)}"

    def _merge_final_chunks(
        self, chunks: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Merge the last two chunks if the final one is below the merge threshold.

        Args:
            chunks (Generator[str]): Chunks of text to evaluate.

        Yields:
            str: Finalized chunks after merging (if needed).
        """
        buffer = []
        for item in chunks:
            buffer.append(item)
            if len(buffer) > 2:
                yield buffer.pop(0)
        if len(buffer) == 2:
            yield "".join([buffer[0], buffer[1]])
        elif len(buffer) == 1:
            yield buffer[0]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap to chunks by adding trailing characters from previous chunk.

        Args:
            chunks (list[str]): The list of chunks to apply overlap to.

        Returns:
            list[str]: The list of chunks with overlap applied.
        """
        if not self.overlap or len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            # Get the last `overlap` characters from the previous chunk
            overlap_text = (
                prev_chunk[-self.overlap :]
                if len(prev_chunk) >= self.overlap
                else prev_chunk
            )
            # Prepend the overlap text to the current chunk
            overlapped_chunks.append(overlap_text + current_chunk)

        return overlapped_chunks

    def _process_buffer(
        self,
        doc: bytes | str,
        n: bool = False,
    ) -> list[str]:
        """Process single buffer in chunks.

        Args:
            doc (bytes | str): The string or bytes doc.
            n (bool): Whether to chunk by n.

        Returns:
            list[str]: The chunks.
        """
        if self.by_bytes:
            return self._process_buffer_bytes(doc, n)

        if isinstance(doc, bytes):
            doc = doc.decode("utf-8")
        return self._process_buffer_text(self._normalize_text(doc), n)

    def _process_buffer_bytes(self, doc: bytes | str, n: bool = False) -> list[str]:
        if isinstance(doc, str):
            doc = doc.encode()

        with BytesIO(doc) as buffer:
            return self._process_binary_stream(buffer, n=n)

    def _get_binary_size(self, file_or_buf: BinaryIO) -> int:
        if isinstance(file_or_buf, BytesIO):
            return file_or_buf.getbuffer().nbytes
        return os.fstat(file_or_buf.fileno()).st_size

    def _read_binary_lines(
        self, file_or_buf: BinaryIO, lines_per_chunk: int
    ) -> list[str]:
        chunks: list[str] = []
        while True:
            chunk_lines: list[bytes] = []
            for _ in range(lines_per_chunk):
                line = file_or_buf.readline()
                if not line:
                    break
                chunk_lines.append(line)
            if not chunk_lines:
                break
            chunks.append(self._normalize_text(b"".join(chunk_lines).decode("utf-8")))
        return chunks

    def _process_binary_chunk(
        self,
        file_or_buf: BinaryIO,
        size: int,
        extend_to_newline: bool = False,
        is_last_chunk: bool = False,
    ) -> str:
        chunk = file_or_buf.read(size)
        if not chunk:
            return ""

        text_chunk = self._normalize_text(chunk.decode("utf-8"))
        if (
            extend_to_newline
            and not is_last_chunk
            and text_chunk
            and not text_chunk.endswith("\n")
        ):
            rest_of_line = file_or_buf.readline()
            if rest_of_line:
                rest_of_line = rest_of_line.decode("utf-8")
                if rest_of_line.endswith("\n"):
                    rest_of_line = rest_of_line[:-1]
                    file_or_buf.seek(file_or_buf.tell() - 1)
                text_chunk += rest_of_line

        return text_chunk

    def _process_binary_fixed_chunks(
        self,
        file_or_buf: BinaryIO,
        size: int,
        total_chunks: Optional[int] = None,
        remainder: int = 0,
        extend_to_newline: bool = False,
    ) -> list[str]:
        chunks: list[str] = []
        if total_chunks is not None:
            for i in range(total_chunks):
                chunk_size = size + remainder if i == total_chunks - 1 else size
                text_chunk = self._process_binary_chunk(
                    file_or_buf,
                    chunk_size,
                    extend_to_newline=extend_to_newline,
                    is_last_chunk=i == total_chunks - 1,
                )
                if not text_chunk:
                    break
                chunks.append(text_chunk)
            return chunks

        while True:
            text_chunk = self._process_binary_chunk(file_or_buf, size)
            if not text_chunk:
                break
            chunks.append(text_chunk)
        return chunks

    def _process_binary_stream(
        self,
        file_or_buf: BinaryIO,
        n: bool = False,
        extend_to_newline: bool = False,
    ) -> list[str]:
        if n is True:
            if self.newline:
                return self._read_binary_lines(file_or_buf, self.n)
            total_size = self._get_binary_size(file_or_buf)
            chunk_size, remainder = self._calculate_chunk_size(total_size, self.n)
            return self._process_binary_fixed_chunks(
                file_or_buf,
                chunk_size,
                total_chunks=self.n,
                remainder=remainder,
                extend_to_newline=extend_to_newline,
            )
        if self.newline:
            return self._read_binary_lines(file_or_buf, self.chunksize)
        return self._process_binary_fixed_chunks(file_or_buf, self.chunksize)

    def _split_text_by_newlines(self, doc: str, n: bool = False) -> list[str]:
        lines = doc.splitlines(keepends=True)
        lines_per_chunk = self.n if n is True else self.chunksize
        return [
            "".join(lines[i : i + lines_per_chunk])
            for i in range(0, len(lines), lines_per_chunk)
            if lines[i : i + lines_per_chunk]
        ]

    def _split_text_into_n_chunks(self, doc: str) -> list[str]:
        total_len = len(doc)
        chunk_size, remainder = self._calculate_chunk_size(total_len, self.n)
        chunks: list[str] = []
        start = 0
        for i in range(self.n):
            size = chunk_size + remainder if i == self.n - 1 else chunk_size
            end = start + size
            chunk = doc[start:end]
            if i < self.n - 1 and chunk and not chunk.endswith("\n"):
                next_newline = doc.find("\n", end)
                if next_newline != -1:
                    chunk = doc[start:next_newline]
                    start = next_newline
                else:
                    start = end
            else:
                start = end
            if not chunk:
                break
            chunks.append(chunk)
        return chunks

    def _split_text_by_size(self, doc: str) -> list[str]:
        return [
            chunk
            for i in range(0, len(doc), self.chunksize)
            if (chunk := doc[i : i + self.chunksize])
        ]

    def _process_buffer_text(self, doc: str, n: bool = False) -> list[str]:
        if self.newline:
            return self._split_text_by_newlines(doc, n)
        if n is True:
            return self._split_text_into_n_chunks(doc)
        return self._split_text_by_size(doc)

    def _process_file_bytes(self, path: Path | str, n: bool = False) -> list[str]:
        """Process a binary file path into chunks."""
        with open(path, "rb") as f:
            extend_to_newline = n is True and not self.newline
            return self._process_binary_stream(
                f,
                n=n,
                extend_to_newline=extend_to_newline,
            )

    def _process_file(
        self,
        path: Path | str,
        n: bool = False,
    ) -> list[str]:
        """Split the contents of a file into chunks.

        Args:
            path (Path | str): Path to the input file.
            n (bool): Whether to split into a fixed number of parts.

        Returns:
            list[str]: List of chunked text segments.
        """
        if self.by_bytes:
            return self._process_file_bytes(path, n)

        # Character-based chunking (default) - read entire file as text
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = self._normalize_text(text)
        return self._process_buffer(text, n=n)

    def _set_attributes(self, **data: Any) -> None:
        """Update multiple attributes on the TextCutter instance.

        Args:
            **data (Any): Arbitrary keyword arguments matching attribute names.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _write_chunk(
        self, path: Path | str, n: int, chunk: str, output_dir: Path
    ) -> None:
        """Write chunk to file with formatted name.

        Args:
            path (Path | str): The path of the original file.
            n (int): The number of the chunk.
            chunk (str): The chunk to save.
            output_dir (Path): The output directory for the chunk.
        """
        path = Path(path)
        output_file = f"{path.stem}{self.delimiter}{str(n).zfill(self.pad)}.txt"
        output_path = output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(chunk)

    @validate_call
    def merge(self, chunks: list[str], sep: str = " ") -> str:
        """Merge a list of chunks into a single str.

        Args:
            chunks (list[str]): The list of chunks to merge.
            sep (str): The separator to use.

        Returns:
            str: The merged string.
        """
        if len(chunks) == 0:
            raise LexosException("No chunks to merge.")
        return f"{sep}".join(string for string in chunks)

    @validate_call
    def save(
        self,
        output_dir: Path | str,
        names: Optional[list[str]] = None,
        delimiter: Optional[str] = "_",
        pad: Optional[int] = 3,
        strip_chunks: Optional[bool] = True,
    ) -> None:
        """Save the chunks to disk.

        Args:
            output_dir (Path | str): The output directory to save the chunks to.
            names (Optional[list[str]]): The doc names.
            delimiter (str): The delimiter to use for the chunk names.
            pad (int): The padding for the chunk names.
            strip_chunks (bool): Whether to strip leading and trailing whitespace in the chunks.
        """
        output_dir = Path(output_dir)
        self._set_attributes(
            output_dir=output_dir,
            delimiter=delimiter,
            names=names,
            pad=pad,
            strip_chunks=strip_chunks,
        )
        if not self.chunks:
            raise LexosException("No chunks to save.")
        if self.names:
            if len(self.names) != len(self.chunks):
                raise LexosException(
                    f"The number of docs in `names` ({len(self.names)}) must equal the number of docs in `chunks` ({len(self.chunks)})."
                )
        else:
            self.names = [
                f"doc{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        for i, doc in enumerate(self.chunks):
            for num, chunk in enumerate(doc):
                if strip_chunks:
                    chunk = chunk.strip()
                self._write_chunk(self.names[i], num + 1, chunk, output_dir)

    def _prepare_split(
        self,
        docs: Optional[Path | str | list[Path | str]],
        chunksize: Optional[int],
        names: Optional[str | list[str]],
        delimiter: Optional[str],
        pad: Optional[int],
        n: Optional[int],
        newline: Optional[bool],
        overlap: Optional[int],
        by_bytes: Optional[bool],
        file: Optional[bool],
        merge_threshold: Optional[float],
        merge_final: Optional[bool],
    ) -> bool:
        if docs:
            self.docs = ensure_list(docs)
        if not self.docs:
            raise LexosException("No documents provided for splitting.")

        self.chunks = []
        self._set_attributes(
            n=n,
            newline=newline,
            overlap=overlap,
            by_bytes=by_bytes if by_bytes is not None else self.by_bytes,
            merge_threshold=merge_threshold,
            merge_final=merge_final,
            delimiter=delimiter,
            pad=pad,
        )

        if chunksize:
            self.chunksize = chunksize

        if names:
            self.names = ensure_list(names)
        elif file:
            self.names = [Path(name).stem for name in ensure_list(docs)]
        else:
            self.names = [
                f"doc{str(i).zfill(self.pad)}" for i in range(1, len(self.docs) + 1)
            ]

        return isinstance(self.n, int)

    def _split_doc(self, doc: Path | str, split_by_num: bool, file: bool) -> list[str]:
        return (
            self._process_file(doc, n=split_by_num)
            if file
            else self._process_buffer(doc, n=split_by_num)
        )

    def _get_milestone_text(self, doc: Path | str, file: bool) -> str:
        if not file:
            return doc
        try:
            with open(doc, "r", newline="") as f:
                return f.read()
        except BaseException as e:
            raise LexosException(e)

    def _build_milestone_chunks(
        self,
        text: str,
        milestones: list[StringSpan],
        keep_spans: Optional[bool | str],
    ) -> list[str]:
        chunks: list[str] = []
        start = 0
        for span in milestones:
            end = span.start
            chunk = text[start:end]
            if keep_spans == "preceding":
                chunk = chunk + text[span.start : span.end + 1]
            chunks.append(chunk)
            start = span.end + 1

        last_chunk = text[start:]
        if keep_spans == "following" and milestones:
            for idx in range(1, len(chunks)):
                milestone_text = text[
                    milestones[idx - 1].start : milestones[idx - 1].end + 1
                ]
                chunks[idx] = milestone_text + chunks[idx]
            last_span = milestones[-1]
            last_chunk = text[last_span.start : last_span.end + 1] + last_chunk

        chunks.append(last_chunk)
        return chunks

    @validate_call
    def split(
        self,
        docs: Optional[Path | str | list[Path | str]] = None,
        chunksize: Optional[int] = None,
        names: Optional[str | list[str]] = None,
        delimiter: Optional[str] = "_",
        pad: Optional[int] = 3,
        n: Optional[int] = None,
        newline: Optional[bool] = None,
        overlap: Optional[int] = None,
        by_bytes: Optional[bool] = None,
        file: Optional[bool] = False,
        merge_threshold: Optional[float] = 0.5,
        merge_final: Optional[bool] = False,
    ) -> list[list[str]]:
        """Chunk the file or buffer.

        Args:
            docs (Optional[Path | str | list[Path | str]]): The file path or buffer.
            chunksize (Optional[int]): The size of the chunks in characters (or bytes if by_bytes=True).
            names (Optional[str | list[str | None]]): The doc names.
            delimiter (Optional[str]): The delimiter to use for the chunk names.
            pad (Optional[int]): The padding for the chunk names.
            n (Optional[int]): The number of chunks.
            newline (Optional[bool]): Whether to chunk by lines.
            overlap (Optional[int]): The number of characters to overlap between chunks.
            by_bytes (Optional[bool]): Whether to chunk by bytes instead of characters.
            file (Optional[bool]): Whether to chunk the file or buffer.
            merge_threshold (Optional[float]): The threshold for merging the last two chunks.
            merge_final (Optional[bool]): Whether to merge the last two chunks.

        Returns:
            list[list[str]]: A list of chunked strings.
        """
        split_by_num = self._prepare_split(
            docs=docs,
            chunksize=chunksize,
            names=names,
            delimiter=delimiter,
            pad=pad,
            n=n,
            newline=newline,
            overlap=overlap,
            by_bytes=by_bytes,
            file=file,
            merge_threshold=merge_threshold,
            merge_final=merge_final,
        )

        threshold = self.chunksize * self.merge_threshold
        for doc in self.docs:
            chunks = self._split_doc(doc, split_by_num, file)
            if chunks and (self.merge_final is True or len(chunks[-1]) < threshold):
                chunks = list(self._merge_final_chunks(chunks))
            if self.overlap and chunks:
                chunks = self._apply_overlap(chunks)
            self.chunks.append(chunks)

        return self.chunks

    @validate_call
    def split_on_milestones(
        self,
        milestones: list[StringSpan],
        docs: Optional[Path | str | list[Path | str]] = None,
        names: Optional[Path | str | list[Path | str]] = None,
        delimiter: Optional[str] = "_",
        pad: Optional[int] = 3,
        keep_spans: Optional[bool | str] = False,
        strip: Optional[bool] = True,
        file: Optional[bool] = False,
    ) -> list[list[str]]:
        """Split text at each milestone span, optionally retaining the milestone text.

        Args:
            milestones (list[StringSpan]): A list of milestone StringSpans to split the text at.
            docs (Optional[Path | str | list[Path | str]]): List of file paths or buffers.
            names (Optional[Path | str | list[Path | str]]): List of doc names.
            delimiter (Optional[str]): The delimiter to use for the chunk names.
            pad (Optional[int]): The padding for the chunk names.
            keep_spans (Optional[bool | str]): Whether to keep the spans in the split strings. Defaults to False.
            strip (Optional[bool]): Whether to strip the text. Defaults to True.
            file (Optional[bool]): Set to True if reading from file(s), False for strings.

        Returns:
            list[list[str]]: A list of chunked strings.
        """
        if docs:
            self.docs = ensure_list(docs)
        if not self.docs:
            raise LexosException("No documents provided for splitting.")
        self.chunks = []
        self._set_attributes(
            delimiter=delimiter,
            pad=pad,
        )
        if names:
            self.names = ensure_list(names)
        elif file:
            self.names = [Path(name).stem for name in ensure_list(docs)]
        else:
            self.names = [
                f"doc{str(i).zfill(self.pad)}" for i in range(1, len(self.docs) + 1)
            ]
        for doc in self.docs:
            text = self._get_milestone_text(doc, file)
            chunks = self._build_milestone_chunks(text, milestones, keep_spans)
            if strip:
                chunks = [chunk.strip() for chunk in chunks]
            self.chunks.append(chunks)

        return self.chunks

    @validate_call
    def to_dict(
        self, names: Optional[Path | str | list[Path | str]] = None
    ) -> dict[str, list[str]]:
        """Return the chunks as a dictionary.

        Args:
            names (Optional[Path | str | list[Path | str]]): The doc names.

        Returns:
            dict[str, list[str]]: The chunks as a dictionary.
        """
        if names:
            self.names = ensure_list(names)
        if self.names == [] or self.names is None:
            self.names = [
                f"doc{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        return {str(doc): chunks for doc, chunks in zip(self.names, self.chunks)}
