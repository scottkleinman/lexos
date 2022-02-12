"""scrubber.py.

    This file contains the main logic for the Scrubber class.
"""
from inspect import getfullargspec
from typing import Callable, List, Union

from lexos import utils
from lexos.exceptions import LexosException

from . import pipeline


class Scrubber:
    """Scrubber class.

    Sample usage:

        scrubber = Scrubber()
        scrubber.to_lower(doc)
    """
    def __init__(self):
        """Initialize the Scrubber class."""
        self.texts = []
        self.pipeline = None

    def add_pipeline(self, *funcs: Callable[[str], str]):
        """Add a pipeline.

        Args:
            *funcs: The functions to add to the pipeline.
        """
        self.pipeline = pipeline.make_pipeline(funcs)

    def get_pipeline(self) -> tuple:
        """Return a tuple representation of the pipeline."""
        pipeline = []
        for f in self.pipeline:
            if getfullargspec(f).kwonlydefaults:
                pipeline.append((f.__name__, getfullargspec(f).kwonlydefaults))
            else:
                pipeline.append(f.__name__)
        return tuple(pipeline)

    def set_pipeline(self, pipeline: tuple):
        """Set the pipeline.

        This is a variant of add_pipeline that takes a tuple of functions.
        The difference is that function names are given as strings and
        keyword arguments as a dictionary. This is useful if you wanted to
        modify the pipeline after initialisation based on the output of
        `get_pipeline()`, rather than passing callables.

        Args:
            pipeline (tuple): A tuple of functions.
        """
        new_pipeline = []
        for x in pipeline:
            if isinstance(x, tuple):
                new_pipeline.append(new_pipeline.pipe(eval(x[0]), **x[1]))
            else:
                new_pipeline.append(eval(x))
        self.pipeline = pipeline.make_pipeline(new_pipeline)

    def scrub(self, data: Union[List[str], str]) -> List[str]:
        """Scrub a text or list of texts.

        Args:
            data (Union[List[str], str]): The text or list of texts to scrub.

        Returns:
            list: A list of scrubbed texts.
        """
        for text in utils.ensure_list(data):
            self.texts.append(self.pipeline[0](text))
        return self.texts

