# Parallel Loader

The `ParallelLoader` class is an implementation of the `Loader` class optimized for large data sets using Python's `ThreadPoolExecutor` for parallel I/O operations.

::: lexos.io.parallel_loader.VALID_FILE_TYPES
    rendering:
      show_root_heading: true
      heading_level: 3

## class `ParallelLoader`

!!! note
    Mkdocstrings does not properly render the the class docstrings because `griffe_pydantic` gets confused when trying to render fields inherited from `BaseLoader` combined with new fields in `ParallelLoader`. Attributes are still documented correctly, but the overall class docstring is missing.

::: lexos.io.parallel_loader.ParallelLoader.max_workers
    rendering:
      show_root_heading: true
      heading_level: 3

::: lexos.io.parallel_loader.ParallelLoader.worker_strategy
    rendering:
      show_root_heading: true
      heading_level: 3

::: lexos.io.parallel_loader.ParallelLoader.batch_size
    rendering:
      show_root_heading: true
      heading_level: 3

::: lexos.io.parallel_loader.ParallelLoader.show_progress
    rendering:
      show_root_heading: true
      heading_level: 3

::: lexos.io.parallel_loader.ParallelLoader.callback
    rendering:
      show_root_heading: true
      heading_level: 3

::: lexos.io.parallel_loader.ParallelLoader.__init__
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._calculate_optimal_workers
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._detect_mime_types_parallel
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._get_mime_type
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._group_by_type
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._load_docx_file
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._load_pdf_file
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._load_text_file
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._load_zip_file
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._load_file_concurrent
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._prepare_file_list
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._process_results
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader._sort_files_by_type
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader.load_dataset
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.io.parallel_loader.ParallelLoader.load
    rendering:
      show_root_heading: true
      heading_level: 3
