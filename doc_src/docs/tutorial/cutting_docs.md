## Cutting Documents

`Cutter` is a module that divides files, texts, or documents into segments. At present, it is highly experimental. There are two classes for cutting documents into segments, one for working with spaCy documents (codename `Ginsu`) and one for working with raw texts (codename `Machete`).

### `Ginsu`

The `Ginsu` class is used for splitting spaCy documents (pre-tokenised texts).

<iframe style="width: 560px; height: 315px; margin: auto;" src="https://www.youtube.com/embed/Sv_uL1Ar0oM" title="YouTube video player -- Ginsu knives" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

`Ginsu` is the preferred method for creating segments because it can access information supplied by the language model.

`Ginsu` has the following features:

- Split by number of tokens (i.e. every N token).
- Split by number of segments (i.e. return a predetermined number of segments).
- Split into overlapping segments of N tokens.
- Merge the last segment into the preceding if its length falls under a customisable threshold.
- Split on milestone tokens.

[lexos.cutter.Ginsu.split][lexos.cutter.Ginsu.split], [lexos.cutter.Ginsu.splitn][], and [lexos.cutter.Ginsu.split_on_milestones][lexos.cutter.Ginsu.split_on_milestones] return a list of lists, where each item in the sublist is a spaCy document.

In [lexos.cutter.Ginsu.split_on_milestones][lexos.cutter.Ginsu.split_on_milestones], the user can choose whether or not to preserve the milestone token at the beginning of each segment. A milestone must be a token, and it can be as simple as a string matching the token text. However, there is also an elaborate query language for more fine-grained matching.

There is also a [lexos.cutter.milestones.Milestones][] class, which can be used to set milestones in advance. This method adds the custom extension `token._.is_milestone` (by default `False`) to each token in the document and uses the same query language to allow the user to match tokens where the value should be `True`. If documents are pre-processed in this way, the [lexos.cutter.Ginsu.split_on_milestones][lexos.cutter.Ginsu.split_on_milestones] method can leverage that information.

!!! note
    At present, [lexos.cutter.Ginsu.split_on_milestones][lexos.cutter.Ginsu.split_on_milestones] cannot set `token._.is_milestone` on the fly. Although segments created from the milestones are saved, if the user chooses to preserve the milestone token at the beginning of each segment, it will not have `._.is_milestone` attribute.

### Machete

`Machete` is a cruder method of cutting raw text into segments without the benefit of a language model. It may be particularly valueable as a standalone method of segmenting texts for outside applications.

`Machete` works in a manner similar to `Ginsu` and has all the same functionality, except for cutting on milestones (see below). However, before splitting the text it applies a makeshift tokenizer function, and the class then splits the text based on the resulting list of tokens.

The Lexos API has three tokenizer functions in the `cutter` function registry: "whitespace" (the default), "character", and "linebreaks". A `Machete` object can be initialised with one of the tokenizers or the tokenizer can be passed to the [lexos.cutter.Machete.split][lexos.cutter.Machete.split] or [lexos.cutter.Machete.splitn][lexos.cutter.Machete.splitn] functions with the `tokenizer` parameter.

!!! information "What if I don't like the tokenizer?"
    You can supply a custom function after first adding it to the registry. Here is an example:

    ```python
    def custom_punctuation_tokenizer(text: str) -> str:
        """Split the text on punctuation or whitespace."""
        return re.split(r"(\W+)", text)

    # Register the custom function
    import lexos.cutter.registry
    registry.tokenizers.register("custom_punctuation_tokenizer", func=custom_punctuation_tokenizer)

    # Create a `Machete` object
    machete = Machete()

    # Split the texts into 5 segments
    result = machete.splitn(texts, n=5, tokenizer="custom_punctuation_tokenizer")
    ```

[lexos.cutter.Machete.split][lexos.cutter.Machete.split] and [lexos.cutter.Machete.splitn][lexos.cutter.Machete.split] return a list of lists, where each item in the sublist corresponds to a text. By default, each text contains a list of strings in which the string corresponds to a segment. It is possible to return the segments as lists of tokens by setting the `as_string` parameter to `False`.

!!! note
    [lexos.cutter.Machete][lexos.cutter.Machete] has no method of splitting texts on milestones. This is because the desired effect can be achieved by writing a custom function with a regex pattern containing the milestone. Such a function would probably lack the fine-grained possibilities of the query language implemented in [lexos.cutter.Ginsu][lexos.cutter.Ginsu], but this functionality is probably only suitable for accessing token attributes supplied by a language model.

### Filesplit (codename `Chainsaw`)

The [lexos.cutter.filesplit.Filesplit][] class allows the user to cut binary files into numbered file segments with the format `filename_1.txt`, `filename_2.txt`, etc. The source file is divided by number of bytes. This class would typically be used as a first step in a workflow if a large file needs to be divided into many smaller files.

The class is a fork of Ram Prakash Jayapalan's <a href="https://github.com/ram-jayapalan/filesplit/releases/tag/v3.0.2" target="_blank">filesplit</a> module with a few minor tweaks. The most important is that the `split` function takes a `sep` argument to allow the user to specify the separator between the filename and number in each generated file.

Typical usage is as follows:

```python
from lexos.cutter.filesplit import Filesplit

fs = Filesplit()

fs.split(
    file="/filesplit_test/longfile.txt",
    split_size=30000000,
    output_dir="/filesplit_test/splits/"
)
```

There are several options available, for which see the API documentation.

The class generates a manifest file called `fs_manifest.csv` in the output directory. This can be used to re-merge the files, if desired:

```python
fs.merge("/filesplit_test/splits/", cleanup=True)
```
