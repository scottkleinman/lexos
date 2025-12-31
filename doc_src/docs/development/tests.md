# Lexos Tests

Lexos uses the <a href="http://doc.pytest.org/" target="_blank">pytest</a> framework for testing. For more info on this, see the <a href="(http://docs.pytest.org/en/latest/contents.html" target="_blank">pytest documentation</a>.

Tests for Lexos modules and classes live in their own directories of the same name. For example, tests for the `Tokenizer` can be found in `/tests/tokenizer`. To be interpreted and run, all test files and test functions need to be prefixed with `test_`.

When adding tests, make sure to use descriptive names, keep the code short and concise and only test for one behavior at a time. Try to `parametrize` test cases wherever possible, use our pre-defined fixtures for spaCy components and avoid unnecessary imports. Extensive tests that take a long time should be marked with `@pytest.mark.slow`.

## Running the Tests

Lexos uses continuous integration to run tests when you push to the public repository or create a pull request. However, this runs the full suite of tests on the entire Lexos package, which can take a lot of time. Therefore, it is important to run all tests locally and make sure they pass before committing. In the header for your test file, indicate the coverage and any uncovered lines as follows:

```markdown
"""test_mymodule.py.

Coverage: 99% Uncovered: 127, 315
Last update: [add date]
"""
```

You can run tests in a specific file or directory, or even only one specific test:

```bash
uv run pytest  # run all tests - this will take a long time
uv run pytest tests/tokenizer  # run all tests in directory
uv run pytest tests/tokenizer/test_exceptions.py # run all tests in file
uv run pytest tests/tokenizer/test_exceptions.py::test_tokenizer_init # run specific test
```

## Writing Tests

Tests for Lexos modules and classes live in their own directories of the same name and all test files should be prefixed with `test_`.

### Test Suite Structure

When adding tests, make sure to use descriptive names and only test for one behavior at a time. Tests should be grouped into modules dedicated to the same type of functionality and some test modules are organized as directories of test files related to the same larger area of the library, e.g. `matcher` or `tokenizer`.

### Fixtures

If multiple tests in a file require a specific configuration, or use the same complex example, it can be helpful to create a separate fixture. This fixture should be added at the top of each file. It is helpful to add comments designating the fixtures and tests sections of the file.

### Parametrizing Tests

If you need to run the same test function over different input examples, you usually want to parametrize the test cases instead of using a loop within your test. This lets you keep a better separation between test cases and test logic, and it'll result in more useful output because `pytest` will be able to tell you which exact test case failed.

The `@pytest.mark.parametrize` decorator takes two arguments: a string defining one or more comma-separated arguments that should be passed to the test function and a list of corresponding test cases (or a list of tuples to provide multiple arguments).

```python
@pytest.mark.parametrize("words", [["hello", "world"], ["this", "is", "a", "test"]])
def test_doc_length(words):
    doc = Doc(Vocab(), words=words)
    assert len(doc) == len(words)
```

```python
@pytest.mark.parametrize("text,expected_len", [("hello world", 2), ("I can't!", 4)])
def test_token_length(en_tokenizer, text, expected_len):  # en_tokenizer is a fixture
    doc = en_tokenizer(text)
    assert len(doc) == expected_len
```

You can also stack `@pytest.mark.parametrize` decorators, although this is not recommended unless it's absolutely needed or required for the test. When stacking decorators, keep in mind that this will run the test with all possible combinations of the respective parametrized values, which is often not what you want and can slow down the test suite.

### Handling Failing Tests

`xfail` means that a test **should pass but currently fails**, i.e. is expected to fail. You can mark a test as currently xfailing by adding the `@pytest.mark.xfail` decorator. This should only be used for tests that don't yet work, not for logic that cause errors we raise on purpose (see the section on testing errors for this). It's often very helpful to implement tests for edge cases that we don't yet cover and mark them as `xfail`. You can also provide a `reason` keyword argument to the decorator with an explanation of why the test currently fails.

```diff
+ @pytest.mark.xfail(reason="Issue #225 - not yet implemented")
def test_en_tokenizer_splits_em_dash_infix(en_tokenizer):
    doc = en_tokenizer("Will this road take me to Puddleton?\u2014No.")
    assert doc[8].text == "\u2014"
```

When you run the test suite, you may come across tests that are reported as `xpass`. This means that they're marked as `xfail` but didn't actually fail. This is worth looking into: sometimes, it can mean that we have since fixed a bug that caused the test to previously fail, so we can remove the decorator. In other cases, especially when it comes to machine learning model implementations, it can also indicate that the **test is flaky**: it sometimes passes and sometimes fails. This can be caused by a bug, or by constraints being too narrowly defined. If a test shows different behavior depending on whether its run in isolation or not, this can indicate that it reacts to global state set in a previous test, which is unideal and should be avoided.

### Writing Slow Tests

If a test is useful but potentially quite slow, you can mark it with the `@pytest.mark.slow` decorator. This is a special marker we introduced and tests decorated with it only run if you run the test suite with `--slow`, but not as part of the main CI process. Before introducing a slow test, double-check that there isn't another and more efficient way to test for the behavior. You should also consider adding a simpler test with maybe only a subset of the test cases that can always run, so we at least have some coverage.

### Skipping Tests

The `@pytest.mark.skip` decorator lets you skip tests entirely. You only want to do this for failing tests that may be slow to run or cause memory errors or segfaults, which would otherwise terminate the entire process and wouldn't be caught by `xfail`. We also sometimes use the `skip` decorator for old and outdated regression tests that we want to keep around but that don't apply anymore. When using the `skip` decorator, make sure to provide the `reason` keyword argument with a quick explanation of why you chose to skip this test.

### Testing Errors and Warnings

`pytest` lets you check whether a given error is raised by using the `pytest.raises` contextmanager. This is very useful when implementing custom error handling, so make sure you're not only testing for the correct behavior but also for errors resulting from incorrect inputs. If you're testing errors, you should always check for `pytest.raises` explicitly and not use `xfail`.

```python
words = ["a", "b", "c", "d", "e"]
ents = ["Q-PERSON", "I-PERSON", "O", "I-PERSON", "I-GPE"]
with pytest.raises(ValueError):
    Doc(Vocab(), words=words, ents=ents)
```

You can also use the `pytest.warns` contextmanager to check that a given warning type is raised. The first argument is the warning type or `None` (which will capture a list of warnings that you can `assert` is empty).

```python
def test_phrase_matcher_validation(en_vocab):
    doc1 = Doc(en_vocab, words=["Test"], deps=["ROOT"])
    doc2 = Doc(en_vocab, words=["Test"])
    matcher = PhraseMatcher(en_vocab, validate=True)
    with pytest.warns(UserWarning):
        # Warn about unnecessarily parsed document
        matcher.add("TEST1", [doc1])
    with pytest.warns(None) as record:
        matcher.add("TEST2", [docs])
        assert not record.list
```

Keep in mind that your tests will fail if you're using the `pytest.warns` contextmanager with a given warning and the warning is _not_ shown. So you should only use it to check that spaCy handles and outputs warnings correctly. If your test outputs a warning that's expected but not relevant to what you're testing, you can use the `@pytest.mark.filterwarnings` decorator and ignore specific warnings starting with a given code:

```python
@pytest.mark.filterwarnings("ignore:\\[W036")
def test_matcher_empty(en_vocab):
    matcher = Matcher(en_vocab)
    matcher(Doc(en_vocab, words=["test"]))
```

### Dos and Don'ts

To keep the behavior of the tests consistent and predictable, we try to follow a few basic conventions:

- **Test names** should follow a pattern of `test_[module]_[tested behaviour]`. For example: `test_tokenizer_keeps_email` or `test_spans_override_sentiment`.
- Only use `@pytest.mark.xfail` for tests that **should pass, but currently fail**.
- Try to keep the tests **readable and concise**. Use clear and descriptive variable names (`doc`, `tokens` and `text` are great), keep it short and only test for one behavior at a time.

## Checking Coverage

We aim to include tests with coverage for 100% or near 100% of the code lines of a Python module. To run tests with coverage, you can use the following command:

```bash
uv run pytest --cov --cov-report=term-missing test_dtm.py
```

This displays a coverage report in your terminal, indicating which lines in the model are not covered by your test functions.

To generate an HTML coverage report, something like use:

```bash
uv run pytest --cov --cov-report=html test_dtm.py
```

After running, open `htmlcov/index.html` in your browser to inspect coverage. As with the terminal report, you can adjust the `--cov` option to target specific modules or directories.

Make sure that you test coverage before submitting a pull request for your test files.
