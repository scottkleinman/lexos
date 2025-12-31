"""scrubber.py.

This file contains the main logic for the Scrubber class.

Last Update: 2025-12-04
Tested: 2025-01-20
"""

from functools import partial
from typing import Any, Callable, Iterable, Optional

import catalogue
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from lexos.exceptions import LexosException
from lexos.scrubber.registry import scrubber_components
from lexos.util import ensure_list

type ScrubberComponent = (
    partial
    | Pipe
    | str
    | tuple[str, dict]
    | Iterable[partial | Pipe | str | tuple[str, dict]]
)
type PipelineComponents = Iterable[Callable | partial | Pipe | str | tuple[str, dict]]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Pipe:
    """A Pydantic dataclass containing a pipeline component.

    Calls:
        The class is callable and returns a function that takes a string and returns a string.
    """

    name: str = Field(..., description="The name of the component.")
    opts: Optional[dict[str, Any]] = Field(
        default={}, description="Options to pass to the component."
    )
    factory: Optional[catalogue.Registry] = Field(
        default_factory=lambda: scrubber_components,
        description="The factory to use to get the component.",
    )

    def __call__(self, text: str) -> Callable:
        """Call the pipeline component on the text.

        Args:
            text (str): The text to scrub.

        Returns:
            Callable: A function that takes a string and returns a string.
        """
        try:
            func = self.factory.get(self.name)
            return func(text, **self.opts)
        except NameError as e:
            raise LexosException(e)
        except catalogue.RegistryError as e:
            raise LexosException(e)


class Scrubber:
    """A class to scrub text using a pipeline of components."""

    def __init__(self) -> None:
        """Initialize the Scrubber object."""
        self._components: list[Pipe] = []

    @property
    def pipes(self) -> list[Pipe]:
        """Return a list of the names of the pipeline components."""
        return [component.name for component in self._components]

    def _get_pipe_index(
        self,
        before: Optional[str | int] = None,
        after: Optional[str | int] = None,
        first: Optional[bool] = None,
        last: Optional[bool] = None,
    ) -> int:
        """Determine where to insert a pipeline component based on the before/after/first/last values.

        Args:
            before (str): Name or index of the component to insert directly before.
            after (str): Name or index of component to insert directly after.
            first (bool): If True, insert component first in the pipeline.
            last (bool): If True, insert component last in the pipeline.

        Returns:
            (int): The index of the new pipeline component.
        """
        if sum(arg is not None for arg in [before, after, first, last]) >= 2:
            raise LexosException("Only one of before, after, first, last can be set.")
        if last or not any(value is not None for value in [first, before, after]):
            return len(self._components)
        elif first:
            return 0
        elif isinstance(before, str):
            if before not in self.pipes:
                raise LexosException(
                    f"The component name {before} is not in the pipeline."
                )
            return self.pipes.index(before)
        elif isinstance(after, str):
            if after not in self.pipes:
                raise LexosException(
                    f"The component name {after} is not in the pipeline."
                )
            return self.pipes.index(after) + 1
        # We only accept indices referring to components that exist.
        # We can't use isinstance here because bools are instance of int.
        elif type(before) is int:
            if before >= len(self._components) or before < 0:
                raise ValueError(f"Index {before} out of range.")
            return before
        elif type(after) is int:
            if after >= len(self._components) or after < 0:
                raise ValueError(f"Index {after} out of range.")
            return after + 1
        raise ValueError("Invalid combination of before, after, first, last.")

    def add_pipe(
        self,
        components: ScrubberComponent,
        *,
        before: Optional[str | int] = None,
        after: Optional[str | int] = None,
        first: Optional[bool] = None,
        last: Optional[bool] = None,
    ) -> None:
        """Add a component to the scrubber pipeline.

        Args:
            components (ScrubberComponent):
                The component to add to the pipeline. If a string is passed, it is assumed
                to be the name of the component. If a tuple is passed, the first element
                is assumed to be the name of the component and the second element is
                assumed to be a dictionary of options to pass to the component. The method
                also accepts Pipe objects.
            before (str | int): Name or index of the component to insert new
                component directly before.
            after (str | int): Name or index of the component to insert new
                component directly after.
            first (bool): If True, insert component first in the pipeline.
            last (bool): If True, insert component last in the pipeline.
        """
        # Ensure a list of tuples
        components = ensure_list(components)
        pipes = []
        for component in components:
            if isinstance(component, partial):
                name = component.func.__name__
                opts = component.keywords
                pipes.append(Pipe(name=name, opts=opts))
            elif isinstance(component, Pipe):
                pipes.append(component)
            elif isinstance(component, str):
                pipes.append(Pipe(name=component, opts={}))
            elif isinstance(component, tuple):
                name, opts = component
                pipes.append(Pipe(name=name, opts=opts))
            else:
                raise LexosException(
                    "Components must be strings, tuples, functools.partial, or Pipe objects."
                )
        # Create and Pipe objects and insert them into the pipeline
        pipe_index = self._get_pipe_index(before, after, first, last)
        for component in pipes:
            # If component exists, merge options
            if component.name in self.pipes:
                # Find the index of the existing component
                idx = self.pipes.index(component.name)
                instance_opts = self._components[idx].opts
                component.opts = {**instance_opts, **component.opts}
            # Insert the component
            self._components.insert(pipe_index, component)
            pipe_index += 1

    def pipe(
        self,
        texts: Iterable[str],
        *,
        disable: Optional[list[str]] = [],
        component_cfg: Optional[dict[str, dict[str, Any]]] = {},
    ) -> Iterable[str]:
        """Scrub a list of texts with the current pipeline.

        Args:
            texts (Iterable[str]): The text(s) to scrub.
            disable	(Optional[list[str]]): Names of pipeline components to disable.
            component_cfg (Optional[dict[str, dict[str, Any]]]): Optional dictionary of keyword arguments for components, keyed by component names. Defaults to None.

        Yields:
            Iterable: An iterator of scrubbed texts.
        """
        pipes = []
        for pipe in self._components:
            if pipe.name in disable:
                continue
            kwargs = component_cfg.get(pipe.name, None)
            # Allow component_cfg to overwrite the pipe options.
            if kwargs is not None:
                pipe.opts = kwargs
            pipes.append(pipe)
        for text in texts:
            for pipe in pipes:
                text = pipe(text)
            yield text

    def remove_pipe(self, components: str | Iterable[str]) -> None:
        """Remove a component from the scrubber pipeline.

        Args:
            components (str | Iterable[str]): The name of the component to remove from the pipeline.
        """
        self._components = [
            pipe
            for pipe in self._components
            if pipe.name not in ensure_list(components)
        ]

    def reset(self) -> None:
        """Remove all components from the pipeline."""
        self._components = []

    def scrub(self, text: str) -> str:
        """Run a text through the scrubber pipeline.

        Args:
            text (str): The text to scrub.

        Returns:
            str: The scrubbed text.
        """
        for pipe in self._components:
            text = pipe(text)
        return text


def scrub(
    text: str,
    pipeline: PipelineComponents,
    factory: Optional[catalogue.Registry] = scrubber_components,
) -> str:
    """Scrub a text with a pipeline of components.

    Args:
        text (str): The text to scrub.
        pipeline (PipelineComponents): An iterable of components. These can be
            functions, partial functions, Pipe objects, tuple, or strings. If a
            string is passed, it is assumed to be the name of the component.
            If a tuple is passed, the first element is assumed to be the name of
            the component and the second element is assumed to be a dictionary
            of options to pass.
        factory (Optional[catalogue.Registry], optional): The factory to use to get the components. Defaults to scrubber_components.

    Returns:
        str: The scrubbed text.
    """
    for pipe in pipeline:
        if isinstance(pipe, (Callable, partial)):
            text = pipe(text)
        elif isinstance(pipe, tuple):
            func, opts = pipe
            text = func(text, **opts)
        else:
            try:
                func = factory.get(pipe)
                text = func(text)
            except AttributeError as e:
                raise LexosException(e)
            except catalogue.RegistryError as e:
                raise LexosException(e)
    return text
