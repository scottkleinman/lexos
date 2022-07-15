A typical workflow would create a `Loader` object and call `loader.load()` to load the data from disk or download it from the internet. You can access all loaded texts by calling `Loader.texts`.

!!! note
    It is more efficient simply to use Python's `open()` to load texts into a list _if_ you know the file's encoding. The advantage of the `Loader` class is that it automatically coerces the data to Unicode and it allows you to use the same method regardless of the file's format or whether it is on your local machine or downloaded from the internet.

When you use a `Loader`, all your data is stored in memory for use in a Lexos workflow. You can save it to disk, but it is largely up to you to keep track of your data folder(s) and file locations. If you wish to have a more sophisticated system for managing your data, look at [Managing a Corpus](managing_a_corpus.md).

Lexos has multiple `Loader` classes found in the `IO` module. The simplest to use is the `smart` loader, described below.

## The Smart Loader

Here is a sample of the code for loading a single text file:

```python
#import Loader
from lexos.io.smart import Loader

# Create the loader and load the data
loader = Loader()
loader.load("myfile.txt)

# Print the first text in the Loader
text = loader.texts[0]
print(text)
```

The `load()` function accepts filepaths, urls, or lists of either. If urls are submitted, the content will be downloaded automatically. Valid formats are `.txt` files, `.docx` files, and `.pdf` files, as well as directories or `.zip` files containing only files of these types.

A `Loader` object has six properties:

- `source`: The filepath or url of the last item added to the `Loader`.
- `names`: A list of the names of all items added to the `Loader`. This will normally be the filenames without the extensions, unless you change them.
- `locations`: A list of the filepaths or urls of all items added to the `Loader`.
- `texts`: A list contain the full text of all the items added to the `Loader`.
- `errors`: A list of filepaths or urls for which loading failed.

As you can see from the example above, each of these properties can be accessed by called `Loader.names`, `Loader.texts`, etc.

You can also iterate through a `Loader` and get the `name`, `location`, and `text` of each item:

```python
for item in loader:
    print(item.name)
    print(item.text)
```

## The Dataset Loader

A "dataset" refers to a collection of documents which are often stored and meant to be accessed from a single file. Lexos has a `DatasetLoader` class designed to work with these data sources. Here is an example in which a single plain text file containing one document per line is loaded.

```python
#import Loader
from lexos.io.dataset import DatasetLoader

# Create the loader and load the data
dataset_loader = Loader()
dataset_loader.load("myfile.txt", labels=["Doc1", "Doc2"])
```

Each line in the file is added to the `dataset.texts` list. Since we cannot use the filename to generate names for our documents, you need to supply a list of names using the `labels` parameter. These values will then be accessible in `dataset.names`.

The `DatasetLoader.load()` method accepts files, urls, and directories of files in `.txt`, `.csv`, `.tsv`, `.xlsx`, `json`, and `jsonl` format, as well zip archives containing files in those formats. As shown above, `.txt` files must be line-delimited, without a header, and must be accompanied by a list of `labels`.

`.csv`, `.tsv`, and `.xlsx` files must have a header line containing the values `title` and `text`. Lexos will use these columns to assign your documents' `name` and `text` values. If your source file has a different header, you can tell Lexos which headers to use, as in the following example:

```python
dataset_loader.load(
    "myfile.tsv",
    title_col="label",
    text_col="content",
    sep="\t"
)
```

The example above also tells Lexos to use a tab as the separator between columns since the file being loaded is a tab-separated value file. Under the hood, Lexos reads the data with the Pandas library's `read_csv`, `read_excel`, and `read_json` file, and you can pass along any keywords accepted by those methods. The `sep` keyword in the example above is an example.

For JSON-formatted files, use `title_field` and `text_field` to assign which columns should be read by Lexos. If your file is in newline-delimited JSON (JSONL) format, add the parameter `lines=True`.

Once loaded, texts and their metadata can be accessed with the `DatasetLoader.data` property. This is a list of dicts where each document dict has keywords for `title` and `text`. To access the first document's title, you would use `Dataset.data[0]["title"]`. When iterating through the dataset, the `data` property is optional:

```python
for item in dataset:
    print(item["title"])
```

produces the same result as

```python
for item in dataset.data:
    print(item["title"])
```

!!! warning
    Notice that iterating through the `DatasetLoader` requires that you reference keywords of a dict (`item["text"]`, where as the `smart` loader yields an object, allowing you to reference `item.text`. We hope to make this behaviour more consistent in the future.

## The `Dataset` Class

Internally, the `DatasetLoader` detects the format of the input data and then calls the appropriate method of the `Dataset` class. For instance, if the file is a CSV file, the `Dataset.parse_csv()` method will be used. In most case, it makes sense to take advantage of the `DatasetLoader`'s format detection so that you can use the same syntax for all inputs, but in some circumstances, it may be useful to call `Dataset` directly. Here is an example of how you would do it:

```python
from lexos.io.dataset import Dataset

dataset = Dataset.parse_csv("myfile.csv")

for item in dataset:
    print(item["title"])
```

`Dataset.parse_csv()` takes the same `text_col` and `title_col` arguments that you would pass to the `DatasetLoader`. Here is a list of the main `Dataset` methods and the arguments they take:

- `parse_string()`: Parses line-delimited text files. Requires `labels`.
- `parse_csv()`: Parses a CSV file. Requires `text_col` and `title_col` if there are no `text` and `title` headers. Requires `sep="\t"` is the file is a tab-separated value file.
- `parse_excel()`: Parses an Excel file. Requires `text_col` and `title_col` if there are no `text` and `title` headers.
- `parse_json()`: Parses a JSON file. Requires `text_field` and `title_field` if there are no `text` and `title` fields.
- `parse_jsonl()`: Parses a JSONL file. Requires `text_field` and `title_field` if there are no `text` and `title` fields.

## Adding Datasets to a Standard Lexos Loader

If you already have a Loader, it is easy to add datasets to it.

```python
# Import the loaders
from lexos.io.smart import Loader
from lexos.io.dataset import Dataset, DatasetLoader

# Create and empty `Loader`
loader = Loader()

# Create a `DatasetLoader` and load a dataset
dataset_loader = DatasetLoader("myfile1.csv")

# Load a dataset with `Dataset`
dataset = Dataset.parse_csv("myfile1.csv")

# Add the text and names for each dataset to the standard loader
for item in [dataset_loader, dataset]:
    loader.names.extend(item.names)
    loader.texts.extend(item.texts)
```

Once you have all your data in a `Loader`, you can manipulate the text. Almost inevitably, some of the text you have loaded will be "dirty" &mdash; meaning that it is not quite in the shape you want it in for further analysis. This may be a moment to do some preprocessing with the `Scrubber` module.
