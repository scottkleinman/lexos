# Lexos Documentation

Documentation for Lexos takes three forms:

1. The **User Guide**: Readable web pages with code samples describing the use of Lexos modules
2. The **API docs**: Technical documentation for Lexos classes and functions auto-generated from their docstrings
3. **Tutorials**: Jupyter notebooks (and accompanying sample data) guiding users through a workflow

Contributions for all three are welcome. If you design a new feature or module, you should submit new documentation to accompany it (or make pull requests for changes to the current documentation, if appropriate).

If you contribute a new feature to Lexos, you do not have to produce a tutorial for that feature, but it will be greatly appreciated.

## The Documentation Website

The documentation website is static website generated with <a href="https://www.mkdocs.org/" target="_blank">MkDocs</a> and <a href="https://squidfunk.github.io/mkdocs-material/" target="_blank">Material for MkDocs</a>. Each page is a Markdown file, which is converted to HTML when the site is built.

To preview changes to the documentation, serve it locally with

```bash
cd lexos/doc_src
uv run mkdocs serve
```

This will start a local server and automatically build a `docs` folder in the project root to contain the built website. If you do not want to serve the site, you can call `uv run mkdocs build`. However, in most case you will want to serve it to observe your changes in a web browser.

If you make a new page, you must add it to the `doc_src/docs/mkdocs.yml` configuration. If the page is under an `overview.md` page, check to see if the `overview.md` page has discussion or a table of contents where you might want to link to the new page. Note that the `mkdocs.yml` file is very easy to corrupt, so **be careful**.

When building the documentation, errors and warnings will be printed to the console. Please check and resolve them before making a pull request. If there are many warnings, it can be helpful to redirect the console output to a file. You can do this with `uv run mkdocs build > build_full.log 2>&1` and then inspect the generated `build_full.log` file, which will be saved in the `doc_src` folder. **Make sure that you don't push this file to the repository.**

Whether you make a change to an existing page or add a new one, your text should follow <a href="https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax" target="_blank">GitHub Markdown conventions</a>, especially for code and code blocks. To see examples, you may find it helpful to review the current documentation files in the `doc_src/docs` folder or via `mkdocs serve`.

Before you make a pull request, check that the site builds properly in your local environment and make sure that your content does not contain any Markdown linting errors Lexos uses the default Markdown linting rules of of <a href="https://github.com/DavidAnson/markdownlint" target="_blank">Markdownlint</a>, except as specified in the `doc_src/.markdownlint.json` file.

!!! note
    It is recommended that you install the Markdownlint extension in VS Code for linting Markdown files when producing documentation. The Markdownlint extension will show you any errors.

## The User Guide

The User Guide is intended to provide an entry-level introduction to the major features of Lexos. Pages in the User Guide are primarily intended to provide user-friendly overviews of the Lexos modules without being too exhaustive or too technical. It is acceptable to provide more technical explanations or notes for developers in admonitions (see below), but these should be relatively infrequent. Whether you are considering contributing to existing User Guide pages or adding a new one, use the existing pages as guides for the appropriate content, tone, and technical depth. For instance, you do not necessarily need to give an account of every parameter available in a given function, just those most likely to be used by an entry-level user. You can assume that the user has some familiarity with the Python programming language, but it may be worthwhile to define some terms or explain certain concepts.

Where possible, provide code samples in code blocks. Sample code should follow the conventions described on the [Code Conventions](code-conventions.md) page. If your code generates visualizations, provide links to static images in `.png` format. Typically, your page would be in a folder along with accompanying images.

User Guide pages should follow the Markdown principles noted above under **The Documentation Website**. Since User Guide pages are mostly written description, they should be well-edited and follow established standards for published writing. The Lexos documentation does not follow a specific style guide, but we recommend <a href="https://www.chicagomanualofstyle.org/" target="_blank">The Chicago Manual of Style, 17th Edition</a> if you are in need of guidance but what written convention to adopt. This obviously only applies to documents in English. At present, the Lexos documentation does not have any pages in other languages, but we can imagine adding sections in other languages if users contribute them.

## The API Documentation

Each API documentation is meant primarily for developers, as it is highly technical, but it is also the only portion of the documentation that describes the full functionality of all Lexos features. For instance, a User Guide page or a Tutorial may describe only the major parameters of a function or method &mdash; those most likely to be used or most relevant to the workflow being discussed. If the User Guide or a tutorial does not mention a possible configuration or customization of a function, it is worth checking the API Documentation to see if the function has a parameter to do what you want.

Unlike the User Guide, the API documentation is mostly generated automatically from the type hints and docstrings in the Python source code. This information is converted to HTML with <code><a href="https://mkdocstrings.github.io/" target="_blank">mkdocstrings</a></code> when you build the documentation website.

Each module should have its own folder, the name of which should correspond to the name of the module. Inside, there should be an `index.md` page, which is the starting point for the module's API documentation. The `index.md` file should contain a brief Markdown description of the module and a link to any other pages in the module's API documentation (these should be additional Markdown files). API pages should follow the Markdown principles noted above under **The Documentation Website**.

Each Markdown file in the module's API documentation should contain a <code><a href="https://mkdocstrings.github.io/" target="_blank">mkdocstrings</a></code> templates like the following:

```yaml
### ::: lexos.cutter.TextCutter

    rendering:
      show_root_heading: true
      heading_level: 3
```

A separate template should be provided for each member such as

```yaml
### ::: lexos.cutter.TextCutter.__init__

    rendering:
      show_root_heading: true
      heading_level: 3
```

!!! note
  It is possible to do this concisely with mkdocstrings "selection" syntax.

   ```yaml
     ::: lexos.module
         handler: python
         selection:
           members:
             - MyClass
             - MyClass.my_method
             - my_function
   ```

  However, this method is discouraged because the current version of `mkdocstrings` does not provide a way to hide unformattable material that may be in your docstring at the top of the module.

The only API documentation file that does not require any `mkdocstrings` templates is the `index.md` file. This file only needs templates if it is the only Markdown file in the API's documentation.

A short introduction (in Markdown) may be placed above the template, an further explanations can be added below, if necessary. You can also provide further discussion between member templates.

To preview changes to the documentation, serve it locally with

  ```bash
  uv run mkdocs serve
  ```

To create direct links to individual classes, properties, and methods anywhere in the documentation, use syntax like the following:

- To link to `BaseLoader.data`, use `base_loader/#lexos.io.base_loader.BaseLoader.data`
- To link to `BaseLoader.load_dataset`, use `base_loader/#lexos.io.base_loader.BaseLoader.load_dataset`

If you create API documentation for a new module, be sure to add it to the HTML table in `doc_src/docs/api/index.md`. When you add another row, make sure that you edit the `row-even` and `row-odd` class names so that the table striping alternates in the generated output.

## The Tutorials

The User Guide is the beginner's entry-point into using Lexos, but there is no substitute for hands-on experience. So, as part of the "documentation" offer a series of Jupyter notebooks with executable code where the user can try out Lexos features. Notebooks may or may not come with sample datasets. If they do, the dataset should be compressed to a zip file in a subfolder inside the tutorial's folder. This allows the user to download both the tutorial notebook and the data to run locally. If you create a new tutorial, make sure to add it to the table of contents in `docs_src/docs/tutorials/index.md`.

Tutorials should be aimed at entry-level users and their Markdown narrative should follow the same principles as outlined for the User Guide (except that sample code should mostly be in executable Python cells). All code blocks and Python cells should follow the conventions described on the [Code Conventions](code-conventions.md) page.

## Submitting Changes

Start by committing your changes. Make sure you write clear, descriptive commit messages.

An example using the command line would be
     ```bash
     git add .
     git commit -m "New documentation page about a fancy new feature"
     ```

However, you may use the `git` client of your choice.

1. **Push to Your Fork**

   ```bash
   git push origin new-module-doc
   ```

2. **Open a Pull Request**

   - Go to the original repo and open a pull request from your branch.
   - Fill out the pull request form describing the new module.

3. **Review and Collaboration**

   - Respond to feedback from maintainers.
   - Make requested changes and push updates.
   - Once approved, your changes will be merged!
