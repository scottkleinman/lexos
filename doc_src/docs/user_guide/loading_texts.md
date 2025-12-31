# IO

The `IO` module contains the classes and methods useful for loading in texts and text data from various souces and formats into a consistant structure so they can be analyzed within the Lexos enviroment.

This module contains three main components:

1. `Loader`: The main loader used for Lexos. Designed to handle individual files (.txt, .pdf, and docx), directories of files, and zip archives.
2. `ParallelLoader`: An optimized version of `Loader` for loading large numbers of files using concurrent processing.
3. `DataLoader`: A specialized loader for structured data files such as CSVs, JSON, or Excel files.

All loaders inherit from a common `BaseLoader` class that provides a bluepring and common features for other classes. It includes methods for loading files, processing text, and handling errors. If neither of the provided classes can accommodate the content you are trying to load, you can build a custom loader that derives from this class.

All loaders built on `BaseLoader` have the following attributes for storing loaded data:

- `paths`: File paths or other sources of the loaded texts.
- `mime_types`: MIME types of the loaded items.
- `names`: Names assigned to each loaded text.
- `texts`: The text content of the loaded items.
- `errors`: Any errors encountered during loading.

Additionally loaders will have access to the following properties:

- `records`: Returns a list of dictionaries, with each representing a loaded item with keys such as `name`, `path`, and `mime_type`.
- `data`: Returns a single dictionary containing all of the data stored in the loader.
- `df`: Returns the loaded file records in the form of a Pandas DataFrame.

Common methods available to all loaders include:

- `load_dataset`: Abstract method to be implemented by loaders.
- `dedupe`: Removes duplicate entries from the loaded data and returns a DataFrame with the duplicates removed. The fields to be checked for duplication can be specified.
- `show_duplicates`: Returns a DataFrame containing any duplicates found in the data. Can specify which fields to check for duplicates.
- `reset`: Clear all data from a loader instance. Reset to an empty loader.
- `to_csv`: Save the loaded data to a CSV file.
- `to_excel`: Save the loaded data as an Excel file.
- `to_json`: Save the loaded data to a JSON file.

The features inherited from `BaseLoader` will be demonstrated when we look at the `Loader` and `BaseLoader` classes below.

## `Loader`

The `Loader` class is the main loader used in Lexos. It is designed to handle a variety of input formats and sources, including individual text files, directories of files, and zip archives.

The `Loader` class can load files with the following extensions:

- `.txt`: Plain text files.
- `.pdf`: PDF documents.
- `.docx`: Microsoft Word documents.
- `.zip`: Zip archives containing any of the above file types.

The `Loader` class automatically detects the file type based on the file extension and uses the appropriate method to extract the text content. It also handles errors gracefully, logging any issues encountered during loading. The path to the source file can be a local file path or a URL. For multiple files, the path can be a list of file paths or a path to a directory.

Here is an example of how to use `Loader`:

```python
from lexos.io.loader import Loader

# Create a Loader instance
loader = Loader()

# Sample texts from various sources
loader.load("path/to/file1.txt")
loader.load(["path/to/file2.txt", "path/to/file3.txt"])
loader.load("path/to/directory_of_files")
loader.load("url/to/file4.txt")
```

Once texts are loaded, they can be accessed through the `texts` attribute or the `df` property, which returns a pandas DataFrame of the loaded records. If there is a problem loading a file, the error will be logged in the `errors` attribute.

By default, the `Loader` class assigns names to loaded texts based on the file name, minus the extension. However, custom names can be provided using the `names` parameter when loading files.

```python
from lexos.io.loader import Loader

# Create a Loader instance
loader = Loader(names=["Doc1", "Doc2"])

# Sample texts
loader.load(["path/to/file1.txt", "path/to/file2.txt"])

print(loader.names)
# ["Doc1", "Doc2"]
```

!!! note
    Names assigned to documents can be useful as labels, especially when generating tabular representations or visualisations of your data.

## `ParallelLoader`

The `ParallelLoader` class is an optimized version of `Loader` designed for loading large numbers of files efficiently using concurrent processing. It uses Python's `ThreadPoolExecutor` to load multiple files in parallel, which can significantly speed up loading times when working with hundreds or thousands of files.

For large-scale corpus loading (1000+ documents), `ParallelLoader` can provide 5-20x speedup compared to sequential loading, especially when loading from network storage or processing CPU-intensive formats like PDFs.

### When to Use ParallelLoader

`ParallelLoader` is most beneficial for:

- **Large file counts**: Loading 100+ files, especially 1000+
- **Remote files**: Loading files from URLs with network latency
- **Network storage**: Files on network drives or remote filesystems where I/O latency is high
- **CPU-intensive formats**: PDFs and DOCX files that require parsing
- **Mixed file types**: Processing different file types that can be handled independently

For small file counts (<50 files) on fast local storage, the standard `Loader` may be faster due to threading overhead.

### Basic Usage

The `ParallelLoader` API is identical to the standard `Loader`, making it a drop-in replacement:

```python
from lexos.io.parallel_loader import ParallelLoader

# Create a ParallelLoader instance
loader = ParallelLoader()

# Load files just like with Loader
loader.load(["path/to/file1.txt", "path/to/file2.txt"])
loader.load("path/to/directory_of_files")

# Access results the same way
print(loader.texts)
print(loader.names)
print(loader.df)
```

### Configuration Options

`ParallelLoader` provides several options to customize performance:

```python
from lexos.io.parallel_loader import ParallelLoader

# Customize worker threads and batch size
loader = ParallelLoader(
    max_workers=16,          # Number of concurrent threads (default: auto-calculated)
    worker_strategy="auto",  # Worker allocation strategy (default: "auto")
    batch_size=50,           # Files per batch (default: 100)
    show_progress=True,      # Show progress bar (default: True)
)

# Load with custom callback for progress tracking
def my_progress(path, processed, total):
    print(f"Loaded {processed}/{total}: {path}")

loader = ParallelLoader(
    show_progress=False,
    callback=my_progress
)
loader.load(file_list)
```

### Worker Strategy (Auto-tuning)

The `worker_strategy` parameter controls how many worker threads are allocated based on your workload:

- **`"auto"`** (default): Analyzes file types and automatically chooses the optimal strategy
  - Detects CPU-intensive formats (PDF, DOCX) vs I/O-intensive formats (text files)
  - Adjusts worker count accordingly for best performance

- **`"io_bound"`**: Allocates more workers for I/O-intensive operations
  - Best for: Large numbers of text files, network storage, remote URLs
  - Workers: `min(32, cpu_count * 4)`

- **`"cpu_bound"`**: Allocates fewer workers for CPU-intensive operations
  - Best for: Predominantly PDF or DOCX files requiring parsing
  - Workers: `min(16, cpu_count * 2)`

- **`"balanced"`**: Middle ground between I/O and CPU strategies
  - Best for: Mixed file types
  - Workers: `min(24, cpu_count * 3)`

```python
# Let ParallelLoader analyze and choose the best strategy
loader = ParallelLoader(worker_strategy="auto")

# Or explicitly choose a strategy
loader = ParallelLoader(worker_strategy="io_bound")  # For text-heavy workloads
loader = ParallelLoader(worker_strategy="cpu_bound")  # For PDF-heavy workloads

# Override with explicit worker count (ignores strategy)
loader = ParallelLoader(max_workers=8)  # Always use 8 workers
```

### Progress Tracking

By default, `ParallelLoader` displays a <a href="https://github.com/Textualize/rich" target="_blank">Rich progress bar</a> showing the loading progress:

```python
loader = ParallelLoader(show_progress=True)
loader.load(files)  # Shows: Detecting file types... ━━━━ 100%
                    #        Loading files...        ━━━━ 100%
```

You can disable the progress bar or provide a custom callback function:

```python
# No progress bar
loader = ParallelLoader(show_progress=False)

# Custom callback
def track_progress(path, processed, total):
    if processed % 10 == 0:
        print(f"Progress: {processed}/{total} files")

loader = ParallelLoader(show_progress=False, callback=track_progress)
```

### Performance Tuning

For optimal performance, adjust settings based on your use case:

```python
# For maximum speed (disable progress bar)
loader = ParallelLoader(show_progress=False, max_workers=32)

# For memory-constrained environments
loader = ParallelLoader(batch_size=20, max_workers=4)

# For network/remote files (more aggressive parallelization)
loader = ParallelLoader(max_workers=64, batch_size=200)
```

## `DataLoader`

Collections of texts are frequently stored or distributed in a single file, often with one document per line, or in a structured format like JSON. The `DataLoader` class allows you to load these files directly into a Lexos loader.

### Loading Lineated Text Files

The basic method for loading a file with one document per line is as follows:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_lineated_text("path/to/file.txt")
```

Note that each document will be named "text001", "text002", "text003", etc. unless you provide a list of document names with the `names` parameter:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader(names=["author1", "author2", "author3"])
loader.load_lineated_text("path/to/file.txt")
```

### Loading CSV and Excel Files

The procedure is similar for CSV and Excel files. However, you must designate which columns contain the document name and text by indicating their headers with the `name_col` and `text_col` parameters.

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_csv("path/to/file.csv", name_col="name", text_col="content")
loader.load_csv("path/to/file.tsv", sep="\t", name_col="name", text_col="content")
loader.load_excel("path/to/file.xlsx", name_col="name", text_col="content")
```

If you are working with a tab-separated file, just use the `sep`, parameter as shown above.

!!! note
    Currently, your file must have headers. Setting the `name_col` and `text_col` by column index is on the roadmap.

### Loading JSON Files

In a JSON-formatted file, each document is a separate object consisting of fields in which the value is referenced by the field's key (e.g. `{"text": "Some text here"}`). When loading JSON files, it is necessary to specify the key indicating which field contains the text name and which field contains the text content. This is done with the `name_field` and `text_field` parameters, as shown below:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_json("path/to/file.json", name_field="name", text_field="content")
```

In standard JSON format, each document is separated by a comma. However, data is frequently formatted with each document separated by a new line, known as JSONL format. If your data is formatted as JSONL, indicate this with the `lines` parameter:

```python
loader.load_json("path/to/file.json", lines=True, name_field="name", text_field="content")
```

### Merging Data into Standard Loaders

Texts loaded from a dataset can be merged into a standard loader with the `Loader.load_dataset` method:

```python
# Create a Dataset instance and load some data
dataset = DataSet()
dataset.load_json("path/to/file.json", name_field="name", text_field="content")

# Create a Loader instance and load a single file
loader.load("path/to/file.txt")

# Merge the dataset into the loader
loader.load_dataset(dataset)
```

## Working with Other Forms of Data

If your data is not in a format that can be loaded with the `Loader` or `DataLoader` classes, it is generally possible to use the Python standard library or third-party tools to load the data into memory and then assign it to an instance of `Loader`.

However, you may wish to create your own custom loader class (e.g. one that uses an authentication token to access an online service) to introduce the logic required for your particular type of data. Custom loaders that inherit from the `BaseLoader` class are welcome as pull requests. If they seem useful to other users, they will be accepted into the main Lexos library.
