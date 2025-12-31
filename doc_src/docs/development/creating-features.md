# Creating Features

New features can be added to existing modules or in the form of new modules if they offer functionality that is different in nature from the existing modules.

## General Principles

Lexos modules begin at the top of the file with a docstring like this:

```python
"""modulename.py.

Last Updated: [date here]
Last Tested: [date here]
```

The docstring may contain additional explanatory material describing the module or providing examples of its usage, but this material should be kept short. Each time you make changes to the module or test the module, update the "Last Updated" and "Last Tested" sections.

When adding new features, keep in mind that the goal of Lexos is to provide an easy interface for something that might otherwise be more complicated when coded from scratch or working with third-party libraries on a more ad hoc basis. In particular, it should help users to put together workflows moving from preprocessing data to statistical analysis and visualization. Lexos functions should be easy to implement in standalone scripts or Jupyter notebooks but also be usable as part of a back end for a more complex tool or application. Above all, the goal of Lexos is to help students and scholars in the Humanities (or those helping them) to make use of computational tools to address the questions that matter to them. To this end, a major goal of Lexos is to be as language-agnostic as possible and to enable as much as possible the processing of historical and under-resourced languages.

## Creating a Module

To create a new module in the Lexos project, you should create a new branch from `main`, check out the new branch, and follow these steps:

1. **Create a New Directory**: Create a new directory for your module inside the Lexos `src/lexos` package. The directory name should be descriptive and follow the naming conventions of the project. If you are creating a submodule of an existing module, simply create the new directory inside the parent module's folder.

2. **Create an `__init__.py` File**: Inside your new module directory, create an `__init__.py` file. This file can be empty or contain initialization code for your module. It is required to make Python treat the directory as a package. The `__init__.py` file should begin with `__init__.py.` (note the period at the end), but it does not need to contain anything else. If you are creating a submodule, you should also create the `__init__.py` file in your submodule's folder.

3. **Create Module Files**: For some simple modules, you can add your code to the `__init__.py` file. You can then import the module with `import lexos.your_module` or `from lexos.your_module import some_function`.

## Handling Exceptions in Your Module

When creating a module, you should handle exceptions properly. If your module raises an exception, it should be a subclass of `lexos.exceptions.LexosException`. This ensures that the exception is consistent with the rest of the Lexos API and can be handled appropriately by users of your module. In general, it is a good idea to add `from lexos.exceptions import LexosException` at the top of your module file to ensure you can raise exceptions correctly.

## Documenting Your Code

Ruff has a large number of built-in rules which will be enforced when you perform linting. For instance, Python modules should be documented with a docstring at the top of the file that contains the name of the file and ends with a period. For further information about formatting docstrings, see the [Code Conventions](code-conventions.md#docstring-style) page.

## Using spaCy Language Models

Lexos aims to be as language-agnostic as possible, so the default spaCy language pipeline should always be `xx_sent_ud_sm` (spaCy's Multi-language model without named entity recognition). Users should be able to designate other models to be used by any function offered as part of the Lexos toolset.

## Adding Pydantic Data Validation

Lexos also tries to be agnostic about the context in which it will be used. One possibility is to build a backend for a text analysis application like the original Lexos web app. For this purpose, it offers its own data validation using the Python <a href="https://docs.pydantic.dev/latest/" target="_blank">Pydantic</a> library. Wherever possible, Lexos classes and functions should offer data validation using Pydantic. This section contains a short primer on how to use it.

!!! note
    Note that Lexos uses Pydantic v2.

In Python, you define a class like this:

```python
class MyPythonClass:
    def __init__(self, value: int):
        self.value = value
```

In Pydantic, the class would like this:

```python
from pydantic import BaseModel

class MyPydanticClass(BaseModel):
    value: int
```

So far, so good. Pydantic looks like a Python dataclass and has much cleaner code. However, compare the following instantiations of our two classes:

```python
python_instance = MyPythonClass("1")
pydantic_instance = MyPydanticClass(value="1")
pydantic_instance = MyPydanticClass(value={"myvalue": 1})
```

The `python_instance` will not raise an error because there is no type checking. The first `pydantic_instance` will also raise an error because, by default, Pydantic attempts to coerce data into the expected data type (you can change this behaviour). By default, it knows to convert strings to integers. However, it will raise a `ValidaError` for the second `pydantic instance` since it doesn't know how to coerce dicts.

!!! important
    Pydantic requires keyword arguments when instantiating a class, and you cannot use positional arguments. This is a <a href="https://github.com/pydantic/pydantic/issues/116" target="_blank">design choice</a> made by Pydantic to avoid ambiguity in the order of arguments. This comes at some cost to libraries like Lexos, where all Pydantic-validated functions (as well as code samples and tutorials) need to supply keywords for every parameter.

### Validating Functions with @validate_call

The @validate_call decorator can be used to apply Pydantic validation to a function.

```python
from pydantic import validate_call

@validate_call
def print_value(value: int) -> None:
    print(value)

print_value({"value": 1})
```

This will raise a `ValidationError` because the `@validate_call` decorator tells, Pydantic to validate the arguments passed to the function based on the type annotation. It works with any function; you don't need to instantiate a class.

```python
from pydantic import validate_call
import spacy
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_sm")

@validate_call
def print_spacy_doc(doc: Doc) -> str:
    print(doc.text)

doc = nlp("This is a test.")
print_spacy_doc(doc)
```

This will return a `ValidationError` because the spaCy `Doc` class is not recognised in the Pydantic `BaseModel`. Luckily, spaCy also uses Pydantic and has a schema available. So you need to remember to import it and add it to the Pydantic class's configuration:

```python
from pydantic import ConfigDict, validate_call
import spacy
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_sm")

config = ConfigDict(json_schema=DocJSONSchema.schema())

@validate_call(config=config)
def print_spacy_doc(doc: Doc) -> str:
    print(doc.text)

doc = nlp("This is a test.")
print_spacy_doc(doc)
```

!!! note
    If working with a class, you simply add `model_config=config` as a class attribute.

However, not all third-party libraries have importable JSON schemas. For instance, I have not found a way to match the `pd.DataFrame` type, so validating that input data is a dataframe involves writing a custom validator (which is also possible in Pydantic but naturally adds to the codebase). Sometimes this requires a procedural re-think such as making input a dict and having the function convert it to a dataframe.

These complications may occasionally slow development, but, since we don't know what kind of applications may be using Lexos, it seems worthwhile to implement Pydantic validation so that Lexos functions fail as early as possible when input data is not as expected.

#### Documenting Pydantic Models Using the `Field` Function

Pydantic models are used to define data structures in Python (see below). When using Pydantic, you can use the `Field` function to provide additional metadata for model fields. This metadata can include descriptions, default values, and validation constraints. Here is an example of how to use the `Field` function to document a Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    name: str = Field(default="John Doe", description="The name of the person.")
    age: Optional[int] = Field(default=20, description="The age of the person.")
```

!!! Important
    Since the Pydantic `Field` function provides the same information as the docstring, the docstring should only contain a summary of the class and any additional information that is not already provided by the `Field` function. For instance, if the `description` parameter is used, there is no need for an `Args` section.

#### Computed Fields and `model_dump()`

Pydantic models can define computed or derived properties that are evaluated on demand. These properties of a Pydantic class can be accessed via its `model_dump()` method. In Lexos, we intentionally use several computed behaviors (for instance, on `Record` objects) to provide convenience accessors such as `terms`, `tokens`, `num_terms` and `num_tokens`.

However, developers should be aware of two important caveats when using `model_dump()`:

- `model_dump()` may evaluate computed fields. If a computed property depends on runtime state (for example, `Record.terms` depends on the `content` being a parsed `Doc`), evaluating it via `model_dump()` may trigger exceptions such as `LexosException("Record is not parsed.")`.
- Computed fields may be expensive. If a computed field does heavy computation (e.g., building a DTM, calculating statistics, or serializing spaCy tokens), calling `model_dump()` on the object could result in unexpected slowdowns.

It is recommended that you guard `model_dump()` calls when the model may not be in a state that supports computed properties. If a model has a boolean state property such as `is_parsed` or `is_ready`, prefer checking it before calling `model_dump()` and constructing a dictionary of required fields manually if the condition is not satisfied. Alternatively, you can call `model_dump()` with the `exclude` and `mode="json"` parameters. For example:

```python
meta = record.model_dump(exclude=["terms", "tokens", "text"], mode="json")
```

## Testing Your Module

New modules should be accompanied by tests functions covering as many lines of your code as possible. To create a test suite for your module, add a new folder for your module in the `lexos/tests` directory, and add your test files there. Tests should be run on your module before you submit a pull request.

For further information on writing tests, see the separate [Tests](tests.md) page.

## Submitting Your Module

Start by committing your changes. Make sure you write clear, descriptive commit messages.

An example using the command line would be
     ```bash
     git add .
     git commit -m "New module offering a fancy new feature"
     ```

However, you may use the `git` client of your choice.

1. **Push to Your Fork**

   ```bash
   git push origin new-module
   ```

2. **Open a Pull Request**

   - Go to the original repo and open a pull request from your branch.
   - Fill out the pull request form describing the new module.

3. **Review and Collaboration**

   - Respond to feedback from maintainers.
   - Make requested changes and push updates.
   - Once approved, your changes will be merged!
