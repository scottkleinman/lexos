"""test_loader.py.

Test the Loader API for the Lexos project.

Notes:

- To run from the command line:

  ```
  poetry run python test_loader.py
  ```

- You may need to adjust the path to the data files for this script to work.
"""
from lexos.io import basic

data = ["tests/test_data/txt/Austen_Pride.txt", "tests/test_data/txt/Austen_Sense.txt"]

loader = basic.Loader()
loader.load(data)

for i, text in enumerate(loader.texts):
    print(f"Text {i} preview:")
    print(text[0:50])
    print("\n")
