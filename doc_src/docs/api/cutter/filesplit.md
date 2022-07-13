# Filesplit

The `Filesplit` class allows the user to cut binary files into numbered file segments with the format `filename_1.txt`, `filename_2.txt`, etc. It would typically be used as a first step in a workflow if a large file needs to be divided into many smaller files.

The class is a fork of Ram Prakash Jayapalan's <a href="https://github.com/ram-jayapalan/filesplit/releases/tag/v3.0.2" target="_blank">filesplit</a> module with a few minor tweaks. The most important is that the `split`
function takes a `sep` argument to allow the user to specify the separator between the filename and number in each generated file.

### ::: lexos.cutter.filesplit.Filesplit
    rendering:
      show_root_heading: true
      heading_level: 3
