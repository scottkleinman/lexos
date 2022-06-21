A typical workflow would create a [lexos.io.smart.Loader][] object and call [lexos.io.smart.Loader.load][] to load the data from disk or download it from the internet. You can access all loaded texts by calling `Loader.texts`.

!!! note
    It is more efficient simply to use Python's `open()` to load texts into a list _if_ you know the file's encoding. Currently, the main advantage of the `Loader` class is that it automatically coerces the data to Unicode.

At this stage of development, the user or application developer is expected to maintain their data folders and know their file locations. More sophisticated project/corpus management methods could be added to the API at a later date.

Here is a sample of the code for loading a single text file:

```python
#import Loader
from lexos.io.smart import Loader

# Data source
data = "tests/test_data/Austen_Pride.txt"

# Create the loader and load the data
loader = Loader()
loader.load(data)

# Print the first text in the Loader
text = loader.texts[0]
print(text)
```

[lexos.io.smart.Loader][] accepts filepaths, urls, or lists of either. If urls are submitted, the content will be downloaded automatically.
   
[lexos.io.smart.Loader][] handles `.txt` files, `.docx` files, and `.pdf` files, as well as directories or `.zip` files containing only files of these types.
