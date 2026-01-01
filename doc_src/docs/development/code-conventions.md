# Code Conventions

This page provides a general overview of code conventions used in the Lexos project.

!!! note
    Some discussion and examples in this coding guide are reproduced from spaCy's very well-documented <a href="https://github.com/explosion/spaCy/blob/master/extra/DEVELOPER_DOCS/Code%20Conventions.md" target="_blank">code conventions</a>, which we consider an excellent guide to best practices. Adjustments have been made where Lexos differs from or adds to spaCy's guidelines.

## Policy on AI-Assisted Contributions

See the separate [Policy on AI-Assisted Contributions](ai-policy.md) page for information on the use of AI tools in contributing to this repository.

## Code Compatibility and Consistency

Lexos supports **Python 3.12** and above, so all code should be written compatible with 3.12.

Please observe the following conventions:

- Every Python file should begin with a docstring that serves as the [Python header](#python-headers).
- Use [Python type hints](#type-hints) to annotate your methods and functions with expected data types for variables, function parameters, and return values.
- Document your methods and functions with docstrings in the appropriate [Docstring style](#docstring-style).
- Add [comments](#comments) that explain the purpose of code where it might not be clear or that point out areas for further development.

Before submitting code, follow the [Code Style Workflow](#code-style-workflow) to check for errors.

## Python Headers

For the purpose of this guide, a "Python header" is the beginning of a Python file, which must start with a docstring containing specific content:

1. The docstring must be enclosed in three double quotation marks, the first line of which must begin with the name of the file and which must end with a period.
2. The file must contain a line "Last Updated: " followed by a date.
3. Code files in the `doc_src` folder must additionally contain a line reading "Last Tested: ", followed by a date.
4. The date format should be something like "November 5, 2025". If you use a different format, `pre-commit` will try to fix it but will generate an error if it cannot do so (see [pre-commit](#using-pre-commit)).
5. Code files in the `tests` folder must additionally contain a line "Coverage: " followed by a percentage and a period. If coverage is not 100%, the period must be followed by a space and "Missing: ". This should be followed by a comma-separated list of lines in the module being tests that are not covered by the tests. The list may contain ranges like "105-107".

Here are two examples:

```python
# Example of a module file
"""mymodule.py.

Last Updated: November 25, 2025
Last Tested: November 25, 2025
"""

# Example of a test file
"""test_mymodule.py.

Coverage: 97%. Missing: 201, 305-306
Last Updated: November 25, 2025
"""
```

Generally, the coverage line goes before the "Last Updated" line.

!!! warning
    Pay attention to the punctuation, as incorrectly punctuated headers will generate `pre-commit` errors.

Python headers can contain other information such as notes and sample usage, but you should try to keep Python headers short.

## Type Hints

We use <a href="https://docs.python.org/3.12/library/typing.html" target="_bla k">Python type hints</a> across the `.py` files wherever possible. This makes it easy to understand what a function expects and returns, and modern editors will be able to show this information to you when you call an annotated function. Type hints are also used to auto-generate the API documentation files.

If possible, you should always use the more descriptive type hints like `List[str]` or even `List[Any]` instead of only `list`. We also annotate arguments and return types of `Callable` – although, you can simplify this if the type otherwise gets too verbose. Remember that `Callable` takes two values: a **list** of the argument type(s) in order, and the return values.

```diff
- def func(some_arg: dict) -> None:
+ def func(some_arg: Dict[str, Any]) -> None:
    ...
```

```python
def create_callback(some_arg: bool) -> Callable[[str, int], List[str]]:
    def callback(arg1: str, arg2: int) -> List[str]:
        ...

    return callback
```

The only caveat on type hinting is if fully-descriptive type hints become too verbose for human readability. In the case of the example above, describing both the arguments and the return values of the `Callable` type might be more complex than is desirable for a human reader. It is a judgement call when the attempt for full and precise type hinting becomes too complex to be meaningful.

## Docstring Style

All functions and methods you write should be documented with a docstring inline. The docstring provides a simple summary, and an overview of the arguments and their types. Modern editors will show this information to users when they call the function or method in their code, and this information is also used to auto-generate the API documentation.

The Lexos project follows the <a href="https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings" target="_blank">Google Style Python Docstrings</a> for docstrings. This is a widely used style guide that provides a consistent format for writing docstrings in Python code. It is recommended to follow this style guide for all docstrings in the Lexos project.

!!! note
    The Lexos project uses <a href="https://www.mkdocs.org/" target="_blank">MkDocs</a> to generate API documentation from directly docstrings. The API documentation is automatically generated from the docstrings in the codebase, so it is important to keep the docstrings up to date and consistent with the code.

The basic structure of a docstring in the Google Style is as follows:

```python
def function_name(param1: int, param2: str) -> bool:
    """Summary of the function.

    Args:
        param1 (int): Description of parameter 1.
        param2 (str): Description of parameter 2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: Description of the error condition.
    """
    # Function implementation goes here
```

Only the "Args" section is required. Note that type hints should be reproduced in the docstring. The "Returns" section should be provided if the function returns a value, and the "Raises" section should only be included if the function raises exceptions. The "Summary" line should be a short, one-line description of the function's purpose. A "Notes" section can provide additional details about the function's behavior, if necessary.

For test functions, only a "Summary" line is required. The other sections can be provided optionally if they help to explain the function's behaviour.

## Comments

### Inline Code Comments

Code comments do not need to be extensive. However, if your code includes complex logic or aspects that may be unintuitive at first glance (or even included a subtle bug that you ended up fixing), you should leave a comment that provides more context. Comments should preferably begin with a capital letter.

```diff
token_index = indices[value]
+ # Index describes Token.i of last token but Span indices are inclusive
span = doc[prev_token_index:token_index + 1]
```

```diff
+ # To create the components we need to use the final interpolated config
+ # so all values are available (if component configs use variables).
+ # Later we replace the component config with the raw config again.
interpolated = filled.interpolate() if not filled.is_interpolated else filled
```

If your change implements a fix to a specific issue, it can often be helpful to include the issue number in the comment, especially if it's a relatively straightforward adjustment:

```diff
+ # Ensure object is a Span, not a Doc (#1234)
if isinstance(obj, Doc):
    obj = obj[obj.start:obj.end]
```

### Including TODOs

You are encouraged to include comments about future improvements using the `TODO:` prefix.

```diff
+ # TODO: This is currently pretty slow
dir_checksum = hashlib.md5()
for sub_file in sorted(fp for fp in path.rglob("*") if fp.is_file()):
    dir_checksum.update(sub_file.read_bytes())
```

If any of the TODOs you've added are important and should be fixed soon, you should add a GitHub issue that details the task.

## Formatting Strings

Wherever possible, use <a href="https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals" target="_blank">f-strings</a> for any formatting of strings.

## Structuring Logic

### Positional and Keyword Arguments

Try to avoid writing functions and methods with too many arguments, and use keyword-only arguments wherever possible. Python lets you define arguments as keyword-only by separating them with a `, *`. If you are writing functions with additional arguments that customize the behavior, you typically want to make those arguments keyword-only, so their names have to be provided explicitly.

```diff
- def do_something(name: str, validate: bool = False):
+ def do_something(name: str, *, validate: bool = False):
    ...

- do_something("some_name", True)
+ do_something("some_name", validate=True)
```

This makes the function calls easier to read, because it is immediately clear what the additional values mean. It also makes it easier to extend arguments or change their order later on, because you don't end up with any function calls that depend on a specific positional order.

!!! important
    User-facing functions and methods that accept data should be validated with <a href="https://docs.pydantic.dev/latest/" target="_blank">Pydantic</a>. Note that Pydantic enforces the use of keyword arguments instead of positional arguments.

### Avoid Mutable Default Arguments

A common Python gotcha are <a href="https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments" target="_blank">mutable default arguments</a>: if your argument defines a mutable default value like `[]` or `{}` and then goes and mutates it, the default value is created _once_ when the function is created and the same object is then mutated every time the function is called. This can be pretty unintuitive when you first encounter it. We therefore avoid writing logic that does this.

### Don't Use `try`/`except` for Control Flow

We discourage using `try`/`except` blocks for anything that's not third-party error handling or error handling that we otherwise have little control over. There's typically always a way to anticipate the _actual_ problem and **check for it explicitly**, which makes the code easier to follow and understand, and prevents bugs:

```diff
- try:
-     token = doc[i]
- except IndexError:
-     token = doc[-1]

+ if i < len(doc):
+     token = doc[i]
+ else:
+     token = doc[-1]
```

If you have to use `try`/`except`, make sure to only include what's **absolutely necessary** in the `try` block and define the exception(s) explicitly. Otherwise, you may end up masking very different exceptions caused by other bugs.

```diff
- try:
-     value1 = get_some_value()
-     value2 = get_some_other_value()
-     score = external_library.compute_some_score(value1, value2)
- except:
-     score = 0.0

+ value1 = get_some_value()
+ value2 = get_some_other_value()
+ try:
+     score = external_library.compute_some_score(value1, value2)
+ except ValueError:
+     score = 0.0
```

### Avoid Lambda Functions

`lambda` functions can be useful for defining simple anonymous functions in a single line, but they also introduce problems: for instance, they require <a href="https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions" target="_blank">additional logic</a> in order to be pickled and are pretty ugly to type-annotate. So we typically avoid them in the code base and only use them in the serialization handlers and within tests for simplicity. Instead of `lambda`s, check if your code can be refactored to not need them, or use helper functions instead.

```diff
- split_string: Callable[[str], List[str]] = lambda value: [v.strip() for v in value.split(",")]

+ def split_string(value: str) -> List[str]:
+     return [v.strip() for v in value.split(",")]
```

### Iteration and Comprehensions

Wherever possible, use list, dict, or generator comprehension instead of built-in functions like `filter` or `map`.

```diff
- filtered = filter(lambda x: x in ["foo", "bar"], values)
+ filtered = (x for x in values if x in ["foo", "bar"])
- filtered = list(filter(lambda x: x in ["foo", "bar"], values))
+ filtered = [x for x in values if x in ["foo", "bar"]]

- result = map(lambda x: { x: x in ["foo", "bar"]}, values)
+ result = ({x: x in ["foo", "bar"]} for x in values)
- result = list(map(lambda x: { x: x in ["foo", "bar"]}, values))
+ result = [{x: x in ["foo", "bar"]} for x in values]
```

If your logic is more complex, it's often better to write a loop instead, even if it adds more lines of code in total. The result will be much easier to follow and understand.

```diff
- result = [{"key": key, "scores": {f"{i}": score for i, score in enumerate(scores)}} for key, scores in values]

+ result = []
+ for key, scores in values:
+     scores_dict = {f"{i}": score for i, score in enumerate(scores)}
+     result.append({"key": key, "scores": scores_dict})
```

### Don't Use `print`

The core library never `print`s anything. While we encourage using `print` statements for simple debugging (it's the most straightforward way of looking at what's happening), make sure to clean them up once you're ready to submit your pull request. If you want to output warnings or debugging information for users, use the respective dedicated mechanisms for this instead (see sections on warnings and logging for details).

!!! note
    We make occasional exceptions to this guideline. For instance, when the `topic_modeling/mallet` module calls the Java Mallet tool, it uses the <code><a href="https://github.com/explosion/wasabi" target="_blank">wasabi</a></code> and code><a href="https://github.com/Textualize/rich" target="_blank">rich</a></code> libraries to provide aesthetically pleasing console output that tracks the progress of the Java feedback.

## Naming

Naming is hard. The best we can hope for is if everyone follows some basic conventions. Consistent with general Python conventions, we use the following naming formats:

- `CamelCase` for class names including dataclasses
- `snake_case` for methods, functions and variables
- `UPPER_SNAKE_CASE` for constants, typically defined at the top of a module.
- Avoid using variable names that shadow the names of built-in functions, e.g. `input`, `help` or `list`

### Naming Variables

Variable names should always make it clear _what exactly_ the variable is and what it's used for. Choosing short and descriptive names wherever possible and imperative verbs for methods that do something, e.g. `disable_pipes`, `add_patterns` or `get_vector`.

Private methods and functions that are not intended to be part of the user-facing API should be prefixed with an underscore `_`.

!!! note
    In some cases, Pydantic will not let you use an underscore for a class attribute that should be private. In this case, it is acceptable to name the attribute wihtout an underscore.

## I/O and Handling Paths

Code that interacts with the file-system should, if possible accept objects that follow the `pathlib.Path` API. Ideally, user-facing functions and methods should accept `pathlib.Path` objects as input, although in some cases string inputs may be converted to `pathlib.Path` objects early in the function's operation. It is acceptable to convert `pathlib.Path` objects to strings for internal operation.

## Error Handling

We always encourage writing helpful and detailed custom error messages for everything we can anticipate going wrong, and including as much detail as possible. These should be passed to the `LexosException` exception handler.

```python
if something_went_wrong:
    raise LexosException("Something went wrong!")
```

or

```python
try:
    # code that raises a ValueError
except ValueError as e:
    raise LexosException(f"Something went wrong: {e}")
```

The second example exemplifes what we might do if we anticipate possible errors in third-party code that we don't control, or our own code in a very different context, we typically try to provide custom and more specific error messages if possible. This is an example of <a href="https://docs.python.org/3/tutorial/errors.html#exception-chaining" target="_blank">re-raising from</a> the original caught exception so the user sees both the original error, as well as the custom message.

Note that if you are designing an app that uses Lexos in its backend, Python errors are not necessarily what you want to relay to your user interface. Using the `LexosException` class to pass custom errors helps solve this problem.

### Avoid Using Naked `assert`

During development, it can sometimes be helpful to add `assert` statements throughout your code to make sure that the values you are working with are what you expect. However, as you clean up your code, those should either be removed or replaced by more explicit error handling:

```diff
- assert score >= 0.0
+ if score < 0.0:
+     raise ValueError(Errors.789.format(score=score))
```

### Warnings

Instead of raising an error, some parts of the code base can raise warnings to notify the user of a potential problem. This is done using Python's `warnings.warn`. Whether or not warnings are shown can be controlled by the user, including custom filters for disabling specific warnings using a regular expression matching our internal codes, e.g. `W123`.

```diff
- print("Warning: No examples provided for validation")
+ warnings.warn(Warnings.W123)
```

When adding warnings, make sure you're not calling `warnings.warn` repeatedly, e.g. in a loop, which will clog up the terminal output. Instead, you can collect the potential problems first and then raise a single warning. If the problem is critical, consider raising an error instead.

```diff
+ n_empty = 0
for spans in lots_of_annotations:
    if len(spans) == 0:
-       warnings.warn(Warnings.456)
+       n_empty += 1
+ warnings.warn(Warnings.456.format(count=n_empty))
```

## Code Style Workflow

Code prepared for the Lexos project should undergo linting and formatting to detect errors and enforce a consistent style.

Lexos uses two tools for checking for linting and formatting errors:

- <code><a href="https://docs.astral.sh/ruff/" target="_blank">ruff</a></code>: an opinionated linter and formatter
- code><a href="https://pre-commit.com/" target="_blank">pre-commit</a></code>: a tool for running tests and fixing errors before code is committed to the project repository

Code you write should be compatible with our the default `ruff` rules and the Lexos `pre-commit` hooks. It should not cause any errors or warnings.

Depending on your setup, you can perform linting and formatting checks repeatedly at various stages of development, but running `pre-commit` on your code should always be the final stage of the process.

### Using `ruff`

`ruff` should be installed when your set up your development environment. The following examples show how to run `ruff` from the command-line.

#### Lint Your Code

```bash
uv run ruff check .
```

#### Auto-Fix Linting Issues

```bash
uv run ruff check . --fix
```

#### Format Your Code

```bash
uv run ruff format .
```

You can also run `ruff` in your code editor. For example, if you're using <a href="https://code.visualstudio.com" target="_blank">Visual Studio Code</a>, you can install the <a href="https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff" target="_blank">Ruff</a> extension and set up `ruff` as the default formatter. Add the following to your `settings.json` (in the command pallette, type `Preferences: Open Settings (JSON)`):

```json
{
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "ruff.linting.enabled": true,
    "ruff.linting.run": "onType"
}
```

You may wish to comment out `editor.formatOnSave` if you want to manually format your code. You can automatically format your code by running the command palette (`Ctrl+Shift+P`) and selecting "Format Document" or "Format Selection" or by right-clicking and selecting these options.

In some specific cases, e.g. in the tests, it can make sense to disable auto-formatting for a specific block. You can do this by wrapping the code you wish to exclude from formatting in `# fmt: off` and `# fmt: on`. It is also possible to ignore several comma-separated codes at once, e.g. `# noqa: E731,E123`. Pull requests containing disabled linting will be considered on a case by case basis.

Before committing code, you should ensure that it contains no linting and formatting errors. Depending on your code editing setup, most errors may be detected and fixed by `ruff` when you save your code files. If not, you should manually check your files with `ruff` in your editor and the command line.

### Using `pre-commit`

As a further check, you should run <code><a href="https://pre-commit.com/" target="_blank">pre-commit</a></code>. This tool provides various hooks to check your code for stylistic conformity and consistency. It should be installed when you create your development environment.

Start by installing the Lexos `pre-commit` hooks in your environment with

```bash
uv run pre-commit install`
```

Lexos pre-commit hooks include:

- check yaml format (used in some configuration files, especially in the documentation)
- ensure that every file ends in a blank line
- ensures that there is no trailing whitespace at the end of a line
- running `ruff` linting and formatting (in case you forgot)
- enforcing the Python header format

For instance, the following command will run checks on all Python files in the `tokenizer` module.

```bash
uv run pre-commit run --files $(find src/lexos/tokenizer -name "*.py")
```

You can run `pre-commit` on all files in the Lexos project with

```bash
uv run pre-commit run --all-files
```

However, typically, you will want to run `pre-commit` only on the files you have changed. For instance, if you stage a new module with the first line below, the second line will test only those files.

```bash
git add src/lexos/new_module/*.py
uv run pre-commit run
```

Running `pre-commit` fixes any styling errors it can and generates a report of the remaining errors so that you can fix them manually. Running the following command can be very useful:

```bash
uv run pre-commit run > precommit-log.txt 2>&1
```

This will redirect the console output to a text file, which can make it easier to read. **Be careful not to commit the log file to the repo.**

!!! important
    If you contribute new code, don't forget to update and/or add appropriate [documentation](documentation.md) and [tests](tests.md). If your run `pre-commit` on files staged for committing, `pre-commit` will check them for errors as well.

Lexos uses continuous integration to perform the same error checking when you push your code to the repository or make a pull request. However, this runs the full suite of tests on the entire Lexos package, which can take a lot of time. Therefore, it is important to run `pre-commit` **prior** to committing your code (as the name implies) in order to catch any errors earlier in the process.
