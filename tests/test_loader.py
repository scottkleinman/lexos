"""test_loader.py.

Test the Loader API for the Lexos project.

Notes:

- To run from the command line, `cd` to the `tests` folder and run:

  ```
  poetry run python test_loader.py
  ```

- You may need to adjust the path to the data files for this script to work.
"""
from lexos.io import basic

data = [
    "test_data/Austen_Pride.txt",
    "test_data/Austen_Sense.txt"
]

loader = basic.Loader()
loader.load(data)

for i, text in enumerate(loader.texts):
    print(f"Text {i} preview:")
    print(text[0:50])
    print("\n")
