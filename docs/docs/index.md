# Introduction

The Lexos API is a library of methods for programmatically implementating and extending the functionality in the <a href="http://lexos.wheatoncollege.edu/" target="_blank">Lexos</a> web app. Eventually, the web app will be rewritten to use the API directly.

For the moment, much of the thinking behind the API is explained in the [Tutorial][tutorial].

## Current Status (v0.0.1)

I have mostly built out the basic architecture of the API so far. The basic `Loader` will accept any local file, regardless of format, and it will also download text from URLs. Obviously, there are some security features that need to be added. It would also be nice to load from different file formats (json, docx, zip, etc.), which is not currently supported.

All of the functionality of the Lexos app's `scrubber` module has been ported over, and the basic `tokenizer` module works. However, there needs to be some error checking in both modules.

The `cutter` module will need some consideration, as it will probably require a combination of features from `scrubber` and `tokenizer`, depending on whether the user wants to cut based on some pattern or cut by token ranges.
