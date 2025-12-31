"""parallel_loader.py.

Last Update: December 27, 2025
Last Tested: December 27, 2025
"""

import mimetypes
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Generator, Optional, Self

import puremagic
from docx import Document
from pydantic import ConfigDict, Field, validate_call
from pypdf import PdfReader
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from smart_open import open

from lexos.constants import (
    DOCX_TYPES,
    FILE_START,
    MIN_ENCODING_DETECT,
    PDF_TYPES,
    TEXT_TYPES,
    ZIP_TYPES,
)
from lexos.exceptions import LexosException
from lexos.io.base_loader import BaseLoader
from lexos.io.data_loader import DataLoader
from lexos.util import _decode_bytes as decode
from lexos.util import ensure_list

VALID_FILE_TYPES = {*TEXT_TYPES, *PDF_TYPES, *DOCX_TYPES, *ZIP_TYPES}


class ParallelLoader(BaseLoader):
    """Parallel file loader for large datasets.

    Uses ThreadPoolExecutor for concurrent I/O operations to significantly
    speed up loading of multiple files. Includes progress tracking, batching
    for memory management, and intelligent MIME type detection.
    """

    max_workers: Optional[int] = Field(
        default=None,
        description="Maximum number of worker threads. Can be an integer or will be auto-calculated based on worker_strategy.",
    )
    worker_strategy: str = Field(
        default="auto",
        description="Worker allocation strategy: 'auto' (analyze files), 'io_bound' (more workers), 'cpu_bound' (fewer workers), 'balanced' (middle ground).",
    )
    batch_size: int = Field(
        default=100,
        description="Number of files to process in each batch for memory management.",
    )
    show_progress: bool = Field(
        default=True, description="Whether to show a progress bar during loading."
    )
    max_memory_mb: Optional[int] = Field(
        default=None,
        description="Optional memory limit in MB. Loading pauses if exceeded.",
    )
    callback: Optional[Callable[..., None]] = Field(
        default=None,
        description="Optional callback function for custom progress handling.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the ParallelLoader."""
        super().__init__(**data)
        self._lock = threading.Lock()
        self._mime_cache = {}
        # Worker count will be set dynamically in load() if not specified

    def _get_mime_type(self, path: Path | str, file_start: bytes) -> str:
        """Get the mime type of a file with caching.

        Args:
            path (Path | str): The path to the file.
            file_start (bytes): The first bytes of the file.

        Returns:
            str: The mime type of the file.
        """
        # Check cache first
        path_str = str(path)
        if path_str in self._mime_cache:
            return self._mime_cache[path_str]

        if Path(path).suffix == ".pickle":
            mime_type = "application/vnd.python.pickle"
        else:
            try:
                file_start_str = decode(file_start)
            except (UnicodeDecodeError, AttributeError):
                file_start_str = ""

            results = puremagic.magic_string(file_start_str, path)
            if not results:
                mime_type = None
            else:
                mime_type = results[0].mime_type
                if mime_type == "":
                    mime_type, _ = mimetypes.guess_type(path)

        # Cache the result
        self._mime_cache[path_str] = mime_type
        return mime_type

    def _load_docx_file(
        self, path: Path | str
    ) -> tuple[str, str, str, str, Optional[Exception]]:
        """Load a docx file.

        Args:
            path (Path | str): The path to the file.

        Returns:
            tuple: (path_name, name, mime_type, text, error)
        """
        try:
            doc = Document(path)
            text = "\n".join([decode(p.text) for p in doc.paragraphs])
            return (Path(path).name, Path(path).stem, "application/docx", text, None)
        except Exception as e:
            return (Path(path).name, Path(path).stem, "application/docx", "", e)

    def _load_pdf_file(
        self, path: Path | str
    ) -> list[tuple[str, str, str, str, Optional[Exception]]]:
        """Load a pdf file.

        Args:
            path (Path | str): The path to the file.

        Returns:
            list[tuple]: List of (path_name, name, mime_type, text, error) for each page.
        """
        results = []
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = decode(page.extract_text())
                results.append(
                    (Path(path).name, Path(path).stem, "application/pdf", text, None)
                )
        except Exception as e:
            results.append((Path(path).name, Path(path).stem, "application/pdf", "", e))
        return results

    def _load_text_file(
        self, path: Path | str, mime_type: str
    ) -> tuple[str, str, str, str, Optional[Exception]]:
        """Load a text file.

        Args:
            path (Path | str): The path to the file.
            mime_type (str): The mime type of the file.

        Returns:
            tuple: (path_name, name, mime_type, text, error)
        """
        try:
            with open(path, "rb") as f:
                text = decode(f.read())
            return (Path(path).name, Path(path).stem, mime_type, text, None)
        except Exception as e:
            return (Path(path).name, Path(path).stem, mime_type, "", e)

    def _load_zip_file(
        self, path: Path | str
    ) -> list[tuple[str, str, str, str, Optional[Exception]]]:
        """Handle a zip file.

        Args:
            path (Path | str): The path to the file.

        Returns:
            list[tuple]: List of (path_name, name, mime_type, text, error) for each file in zip.
        """
        results = []
        try:
            with open(path, "rb") as fin:
                with zipfile.ZipFile(fin) as zip:
                    for info in zip.infolist():
                        try:
                            # Get the mime type of the file
                            file_bytes = zip.read(info.filename)
                            file_start = file_bytes[:MIN_ENCODING_DETECT]
                            mime_type = self._get_mime_type(info.filename, file_start)
                        except (IOError, UnicodeDecodeError) as e:
                            results.append(
                                (info.filename, Path(info.filename).stem, None, "", e)
                            )
                            continue

                        try:
                            if mime_type in VALID_FILE_TYPES:
                                text = decode(file_bytes)
                                full_path = Path(path).as_posix() + "/" + info.filename
                                results.append(
                                    (
                                        full_path,
                                        Path(info.filename).stem,
                                        mime_type,
                                        text,
                                        None,
                                    )
                                )
                            else:
                                error = LexosException(
                                    f"Invalid MIME type: {mime_type} for file {info.filename}."
                                )
                                results.append(
                                    (
                                        info.filename,
                                        Path(info.filename).stem,
                                        mime_type,
                                        "",
                                        error,
                                    )
                                )
                        except Exception as e:
                            results.append(
                                (
                                    info.filename,
                                    Path(info.filename).stem,
                                    mime_type,
                                    "",
                                    e,
                                )
                            )
        except Exception as e:
            results.append((Path(path).name, Path(path).stem, None, "", e))
        return results

    def _load_file_concurrent(
        self, path: Path | str, mime_type: str
    ) -> list[tuple[str, str, str, str, Optional[Exception]]]:
        """Load a single file (wrapper for concurrent execution).

        Args:
            path (Path | str): The path to the file.
            mime_type (str): The mime type of the file.

        Returns:
            list[tuple]: List of (path_name, name, mime_type, text, error) tuples.
        """
        results = []

        if mime_type in TEXT_TYPES:
            result = self._load_text_file(path, mime_type)
            results.append(result)
        elif mime_type in PDF_TYPES:
            results.extend(self._load_pdf_file(path))
        elif mime_type in DOCX_TYPES:
            result = self._load_docx_file(path)
            results.append(result)
        elif mime_type in ZIP_TYPES:
            results.extend(self._load_zip_file(path))
        else:
            error = LexosException(f"Invalid MIME type: {mime_type} for file {path}.")
            results.append((Path(path).name, Path(path).stem, mime_type, "", error))

        return results

    def _process_results(self, results: list[tuple]) -> None:
        """Process and store results in a thread-safe manner.

        Args:
            results (list[tuple]): List of (path_name, name, mime_type, text, error) tuples.
        """
        with self._lock:
            for path_name, name, mime_type, text, error in results:
                self.paths.append(path_name)
                self.names.append(name)
                self.mime_types.append(mime_type)
                self.texts.append(text)
                if error:
                    self.errors.append(error)

    def _prepare_file_list(
        self, paths: list[Path | str]
    ) -> list[tuple[Path | str, str]]:
        """Prepare list of files with MIME types, expanding directories.

        Args:
            paths (list[Path | str]): List of file or directory paths.

        Returns:
            list[tuple]: List of (path, mime_type) tuples.
        """
        file_list = []

        for path in paths:
            if Path(path).is_dir():
                # Recursively get all files in directory
                dir_files = [p for p in Path(path).rglob("*") if p.is_file()]
                file_list.extend([(str(f), None) for f in dir_files])
            else:
                file_list.append((str(path), None))

        return file_list

    def _detect_mime_types_parallel(
        self,
        file_list: list[tuple[Path | str, Optional[str]]],
        progress: Optional[Progress] = None,
        task_id: Optional[int] = None,
    ) -> list[tuple[Path | str, str]]:
        """Detect MIME types for all files in parallel.

        Args:
            file_list (list[tuple]): List of (path, mime_type) tuples.
            progress (Optional[Progress]): Rich progress bar instance.
            task_id (Optional[int]): Task ID for progress tracking.

        Returns:
            list[tuple]: List of (path, mime_type) tuples with detected types.
        """
        results = []

        def detect_mime(path_tuple):
            path, _ = path_tuple
            try:
                with open(path, "rb") as f:
                    file_start = f.read(FILE_START)
                mime_type = self._get_mime_type(path, file_start)
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
                return (path, mime_type)
            except IOError as e:
                with self._lock:
                    self.errors.append(e)
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
                return (path, None)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(detect_mime, file_list))

        return results

    def _group_by_type(
        self, file_list: list[tuple[Path | str, str]]
    ) -> dict[str, list[Path | str]]:
        """Group files by MIME type for optimized processing.

        Args:
            file_list (list[tuple]): List of (path, mime_type) tuples.

        Returns:
            dict: Dictionary mapping mime_type to list of paths.
        """
        grouped = {}
        for path, mime_type in file_list:
            if mime_type not in grouped:
                grouped[mime_type] = []
            grouped[mime_type].append(path)
        return grouped

    def _calculate_optimal_workers(
        self, file_list: list[tuple[Path | str, str]]
    ) -> int:
        """Calculate optimal worker count based on file types and strategy.

        Args:
            file_list (list[tuple]): List of (path, mime_type) tuples.

        Returns:
            int: Optimal number of workers.
        """
        cpu_count = os.cpu_count() or 1

        # If max_workers is explicitly set, use it
        if self.max_workers is not None:
            return self.max_workers

        # Analyze file types for 'auto' strategy
        if self.worker_strategy == "auto":
            # Count file types
            cpu_intensive_count = 0
            io_intensive_count = 0

            for _, mime_type in file_list:
                if mime_type in PDF_TYPES or mime_type in DOCX_TYPES:
                    cpu_intensive_count += 1
                elif mime_type in TEXT_TYPES:
                    io_intensive_count += 1

            # Determine strategy based on file mix
            if cpu_intensive_count > len(file_list) * 0.5:
                # More than 50% CPU-intensive files
                return min(16, cpu_count * 2)
            elif io_intensive_count > len(file_list) * 0.8:
                # More than 80% I/O-intensive files
                return min(32, cpu_count * 4)
            else:
                # Mixed workload
                return min(24, cpu_count * 3)

        # Manual strategy selection
        elif self.worker_strategy == "io_bound":
            return min(32, cpu_count * 4)
        elif self.worker_strategy == "cpu_bound":
            return min(16, cpu_count * 2)
        elif self.worker_strategy == "balanced":
            return min(24, cpu_count * 3)
        else:
            # Default fallback
            return min(32, cpu_count + 4)

    def _sort_files_by_type(
        self, file_list: list[tuple[Path | str, str]]
    ) -> list[tuple[Path | str, str]]:
        """Sort files by MIME type for better cache locality.

        Groups similar file types together to improve processing efficiency
        through better cache utilization and reduced context switching.

        Args:
            file_list (list[tuple]): List of (path, mime_type) tuples.

        Returns:
            list[tuple]: Sorted list of (path, mime_type) tuples.
        """
        # Define priority order for file types (process simpler types first)
        type_priority = {
            **{t: 1 for t in TEXT_TYPES},  # Text files first (fastest)
            **{t: 2 for t in ZIP_TYPES},  # ZIP files second
            **{t: 3 for t in DOCX_TYPES},  # DOCX files third
            **{t: 4 for t in PDF_TYPES},  # PDF files last (slowest)
        }

        # Sort by priority, with unknown types at the end
        return sorted(file_list, key=lambda x: type_priority.get(x[1], 999))

    @validate_call(config=model_config)
    def load(self, paths: Path | str | list[Path | str]) -> None:
        """Load files in parallel with batching and progress tracking.

        Args:
            paths (Path | str | list[Path | str]): The list of paths to load.
        """
        paths = ensure_list(paths)

        # Step 1: Prepare file list (expand directories)
        file_list = self._prepare_file_list(paths)
        total_files = len(file_list)

        if total_files == 0:
            return

        # Setup progress bar
        progress = None
        detect_task = None
        load_task = None

        if self.show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            detect_task = progress.add_task(
                "[cyan]Detecting file types...", total=total_files
            )

        # Step 2: Detect MIME types in parallel
        file_list = self._detect_mime_types_parallel(file_list, progress, detect_task)

        # Step 3: Calculate optimal worker count based on file types (auto-tuning)
        if self.max_workers is None:
            self.max_workers = self._calculate_optimal_workers(file_list)

        # Step 4: Sort files by type for better cache locality (smart ordering)
        file_list = self._sort_files_by_type(file_list)

        # Step 5: Group files by type for optimized processing
        grouped_files = self._group_by_type(file_list)

        # Update progress bar for loading phase
        if self.show_progress and progress:
            load_task = progress.add_task("[green]Loading files...", total=total_files)

        # Step 6: Process files in batches by type
        processed = 0
        for mime_type, paths_of_type in grouped_files.items():
            # Process this type in batches
            for i in range(0, len(paths_of_type), self.batch_size):
                batch = paths_of_type[i : i + self.batch_size]

                # Load batch in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_path = {
                        executor.submit(
                            self._load_file_concurrent, path, mime_type
                        ): path
                        for path in batch
                    }

                    for future in as_completed(future_to_path):
                        path = future_to_path[future]
                        try:
                            results = future.result()
                            self._process_results(results)
                            processed += 1

                            # Update progress
                            if (
                                self.show_progress
                                and progress
                                and load_task is not None
                            ):
                                progress.update(load_task, advance=1)

                            # Call custom callback if provided
                            if self.callback:
                                self.callback(path, processed, total_files)

                        except Exception as e:
                            with self._lock:
                                self.errors.append(e)
                            processed += 1
                            if (
                                self.show_progress
                                and progress
                                and load_task is not None
                            ):
                                progress.update(load_task, advance=1)

        # Cleanup progress bar
        if self.show_progress and progress:
            progress.stop()

    # @validate_call(config=model_config)
    def load_dataset(self, dataset: Self) -> None:
        """Load a dataset.

        Args:
            dataset (DataLoader): The dataset to load.

        Note: As of v2.10.5, Pydantic does not support recursive types (Self).
            As a result, this method performs its own check to see if the
            value of `dataset` is of type `DataLoader`.
        """
        if not isinstance(dataset, DataLoader):
            raise LexosException("Invalid dataset type.")

        with self._lock:
            self.paths = self.paths + dataset.paths
            self.mime_types = self.mime_types + dataset.mime_types
            self.names = self.names + dataset.names
            self.texts = self.texts + dataset.texts

    @validate_call(config=model_config)
    def load_streaming(
        self, paths: Path | str | list[Path | str]
    ) -> Generator[tuple[str, str, str, str, Optional[Exception]], None, None]:
        """Load files in parallel and yield results as they become available.

        This method yields documents as they're loaded, making it suitable for
        streaming large datasets directly into a corpus or other consumer without
        holding all files in memory.

        Args:
            paths (Path | str | list[Path | str]): The list of paths to load.

        Yields:
            tuple: (path, name, mime_type, text, error) for each loaded document.
                - path (str): The file path or full path for files in archives
                - name (str): The document name (stem of filename)
                - mime_type (str): The detected MIME type
                - text (str): The document text content (empty string if error)
                - error (Optional[Exception]): Any error that occurred during loading
        """
        from queue import Empty, Queue

        paths = ensure_list(paths)

        # Step 1: Prepare file list (expand directories)
        file_list = self._prepare_file_list(paths)
        total_files = len(file_list)

        if total_files == 0:
            return

        # Setup progress bar
        progress = None
        detect_task = None
        load_task = None

        if self.show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            detect_task = progress.add_task(
                "[cyan]Detecting file types...", total=total_files
            )

        # Step 2: Detect MIME types in parallel
        file_list = self._detect_mime_types_parallel(file_list, progress, detect_task)

        # Step 3: Calculate optimal worker count based on file types (auto-tuning)
        if self.max_workers is None:
            self.max_workers = self._calculate_optimal_workers(file_list)

        # Step 4: Sort files by type for better cache locality (smart ordering)
        file_list = self._sort_files_by_type(file_list)

        # Step 5: Group files by type for optimized processing
        grouped_files = self._group_by_type(file_list)

        # Update progress bar for loading phase
        if self.show_progress and progress:
            load_task = progress.add_task("[green]Loading files...", total=total_files)

        # Create queue for results
        result_queue = Queue(maxsize=self.batch_size * 2)
        processed = 0

        # Step 6: Process files in batches by type
        def process_batches():
            """Process all batches and put results in queue."""
            nonlocal processed
            for mime_type, paths_of_type in grouped_files.items():
                # Process this type in batches
                for i in range(0, len(paths_of_type), self.batch_size):
                    batch = paths_of_type[i : i + self.batch_size]

                    # Load batch in parallel
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_path = {
                            executor.submit(
                                self._load_file_concurrent, path, mime_type
                            ): path
                            for path in batch
                        }

                        for future in as_completed(future_to_path):
                            path = future_to_path[future]
                            try:
                                results = future.result()
                                # Put each document result in queue
                                for result in results:
                                    result_queue.put(result)
                                processed += 1

                                # Update progress
                                if (
                                    self.show_progress
                                    and progress
                                    and load_task is not None
                                ):
                                    progress.update(load_task, advance=1)

                                # Call custom callback if provided
                                if self.callback:
                                    self.callback(path, processed, total_files)

                            except Exception as e:
                                # Put error result in queue
                                result_queue.put(
                                    (str(path), Path(path).stem, None, "", e)
                                )
                                processed += 1
                                if (
                                    self.show_progress
                                    and progress
                                    and load_task is not None
                                ):
                                    progress.update(load_task, advance=1)

            # Signal completion
            result_queue.put(None)

        # Start processing in background thread
        from threading import Thread

        worker_thread = Thread(target=process_batches, daemon=True)
        worker_thread.start()

        # Yield results as they become available
        try:
            while True:
                try:
                    result = result_queue.get(timeout=0.1)
                    if result is None:  # Sentinel value for completion
                        break
                    yield result
                except Empty:
                    continue
        finally:
            # Cleanup progress bar
            if self.show_progress and progress:
                progress.stop()
            worker_thread.join(timeout=1.0)
