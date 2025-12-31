"""pipeline.py.

Last Update: 2025-01-16
Tested: 2025-01-16
"""

from functools import partial, update_wrapper
from typing import Callable

from cytoolz import functoolz
from pydantic import validate_call

from lexos.scrubber import normalize, remove, replace


@validate_call
def pipe(func: Callable, *args, **kwargs) -> Callable:
    """Apply functool.partial and add `__name__` to the partial function.

    This allows the function to be passed to the pipeline along with
    keyword arguments.

    Args:
        func (Callable): A callable.

    Returns:
        Callable: A partial function with `__name__` set to the name of the function.
    """
    if not args and not kwargs:
        return func
    else:
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func


@validate_call
def make_pipeline(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """Make a callable pipeline.

    Make a callable pipeline that passes a text through a series of
    functions in sequential order, then outputs a (scrubbed) text string.

    This function is intended as a lightweight convenience for users,
    allowing them to flexibly specify scrubbing options and their order,which (and in which order) preprocessing
    treating the whole thing as a single callable.

    `python -m pip install cytoolz` is required for this function to work.

    Use `pipe` (an alias for `functools.partial`) to pass arguments to preprocessors.

    ```python
    from lexos import scrubber
    scrubber = Scrubber.pipeline.make_pipeline(
        scrubber.replace.hashtags,
        scrubber.replace.emojis,
        pipe(scrubber.remove.punctuation, only=[".", "?", "!"])
    )
    scrubber("@spacy_io is OSS for industrial-strength NLP in Python developed by @explosion_ai 💥")
    '_USER_ is OSS for industrial-strength NLP in Python developed by _USER_ _EMOJI_'
    ```

    Args:
        *funcs (Callable[[str], str): A series of functions to be applied to the text.

    Returns:
        Callable[[str], str]: Pipeline composed of ``*funcs`` that applies each in sequential order.
    """
    return functoolz.compose_left(*funcs)


@validate_call
def make_pipeline_from_tuple(funcs: tuple) -> tuple:
    """Return a pipeline from a tuple.

    Args:
        funcs (tuple): A tuple containing callables or string names of functions.

    Returns:
        tuple: A pipeline composed of the functions in `funcs`.
    """
    return make_pipeline(*[eval(x) if isinstance(x, str) else x for x in funcs])
