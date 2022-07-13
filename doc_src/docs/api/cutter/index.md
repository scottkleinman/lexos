# Cutter

`Cutter` is a module that divides files, texts, or documents into segments using separate classes (with cute codenames).

- `Ginsu` splits pre-tokenized documents into shorter segments.
- `Machete` splits raw text strings into shorter segments.
- `Filesplit` splits binary files into shorter files.

`Ginsu` acts as a more precise cutter, using language-based tokenization, which provides greater accuracy in exchange for longer processing times. `Machete`, on the other hand, allows for faster processing in exchange for precision. `Filesplit` (aka "Chainsaw") allows files to be split before they are loaded, but potentially in the middle of linguistically significant units.

A separate `Milestones` class is used to populate pre-tokenized texts with milestones for cutting.
