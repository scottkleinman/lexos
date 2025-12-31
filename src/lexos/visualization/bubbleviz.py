"""bubbleviz.py.

Last Update: December 4, 2025
Last Tested: December 5, 2025
"""

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict, Field, field_validator, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization import processors

# Valid input types
single_doc_types = dict[str, int] | Doc | Span | str | list[str] | list[Token]
multi_doc_types = (
    str
    | list[str]
    | list[list[str]]
    | list[Doc]
    | list[Span]
    | list[list[Token]]
    | dict[str, int]
    | pd.DataFrame
    | DTM
)

DEFAULT_COLORS = [
    "#5A69AF",
    "#579E65",
    "#F9C784",
    "#FC944A",
    "#F24C00",
    "#00B825",
]


class BubbleChart(BaseModel):
    """Bubble chart.

    Notes:
    - If the counts are sorted, the results might look weird.
    - If "limit" is raised too high, it will take a long time to generate the plot
    - Based on https://matplotlib.org/stable/gallery/misc/packed_bubbles.html.
    """

    data: Optional[single_doc_types | multi_doc_types | pd.DataFrame] = Field(
        description="The data to plot."
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="The document indices or labels to plot."
    )
    limit: Optional[int] = Field(
        100, description="The maximum number of bubbles to plot."
    )
    title: Optional[str] = Field(None, description="The title of the plot.")
    bubble_spacing: Optional[float | int] = Field(
        0.1, description="The spacing between bubbles."
    )
    colors: Optional[list[str]] = Field(
        DEFAULT_COLORS, description="The colors of the bubbles."
    )
    figsize: Optional[int | float] = Field(
        10, description="The size of the figure in inches."
    )
    font_family: Optional[str] = Field(
        "DejaVu Sans", description="The font family of the plot."
    )
    showfig: Optional[bool] = Field(True, description="Whether to show the plot.")
    bubbles: Optional[np.ndarray] = Field(None, description="The bubbles.")
    maxstep: Optional[int] = Field(None, description="The maximum step.")
    step_dist: Optional[int] = Field(None, description="The step distance.")
    com: Optional[int] = Field(None, description="The center of mass.")
    counts: dict[str, int] = Field({}, description="A dictionary of word counts.")
    fig: Optional[plt.Figure] = Field(
        None, description="The figure for the bubble chart."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    @field_validator("data", mode="after")
    @classmethod
    def is_not_empty(cls, value: Any) -> Any:
        """Check if the value is not empty."""
        if isinstance(value, pd.DataFrame):
            if value.empty:
                raise LexosException("Dataframe is empty.")
            return value

        if value == "" or value == [] or value == {}:
            raise LexosException("Data is an empty list or string.")
        return value

    def __init__(self, **data):
        """Initialize the BubbleChart with the provided data."""
        super().__init__(**data)

        # Process different data types to get individual document data
        self.counts = processors.process_data(self.data, self.docs, self.limit)

        # Set the figure dimensions
        self.figsize = (self.figsize, self.figsize)

        # Reduce the area to the limited number of terms
        area = np.asarray(list(self.counts.values()))
        r = np.sqrt(area / np.pi)

        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # Calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[: len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[: len(self.bubbles)]

        self.com = self._center_of_mass()

        # Create the figure
        self._collapse()
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=self.figsize)
        self._plot(ax, list(self.counts.keys()))
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()

        # Add title
        if self.title:
            ax.set_title(self.title)

        # Save the fig variable
        self.fig = fig

        plt.close()

    def _center_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> np.ndarray:
        """Centre distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            np.ndarray: The centre distance.
        """
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def _center_of_mass(self) -> int:
        """Centre of mass.

        Returns:
            int: The centre of mass.
        """
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def _check_collisions(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Check collisions.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The length of the distance between bubbles.
        """
        distance = self._outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def _collapse(self, n_iterations: int = 50):
        """Move bubbles to the center of mass.

        Args:
            n_iterations (int): Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # Try to move directly towards the center of mass
                # Direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # Shorten direction vector to have length of 1
                # NOTE: Produces invalid value encountered in divide Runtime warnings if dir_vec is zero
                # dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # Shorten direction vector to have length of 1
                # Check if direction vector is non-zero to avoid division by zero
                dir_vec_magnitude = np.sqrt(dir_vec.dot(dir_vec))
                if dir_vec_magnitude > 0:
                    dir_vec = dir_vec / dir_vec_magnitude
                else:
                    # If bubble is already at center of mass, use a small random direction
                    dir_vec = np.array([1.0, 0.0]) * self.step_dist * 0.01

                # Calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # Check whether new bubble collides with other bubbles
                if not self._check_collisions(new_bubble, rest_bub):
                    # NOTE: Produces invalid value encountered in cast Runtime warnings
                    self.bubbles[i, :] = new_bubble
                    self.com = self._center_of_mass()
                    moves += 1
                else:
                    # Try to move around a bubble that you collide with
                    # Find colliding bubble
                    for colliding in self._collides_with(new_bubble, rest_bub):
                        # Calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # Calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self._center_distance(self.com, np.array([new_point1]))
                        dist2 = self._center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self._check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self._center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def _collides_with(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Collide.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The minimum index.
        """
        distance = self._outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) is np.ndarray else [idx_min]

    def _outline_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Outline distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The outline distance.
        """
        center_distance = self._center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def _plot(
        self,
        ax: Axes,
        labels: list[str],
    ):
        """Draw the bubble plot.

        Args:
            ax (Axes): The matplotlib Axes.
            labels (list[str]): The labels of the bubbles.
        """
        plt.rcParams["font.family"] = self.font_family
        color_num = 0
        for i in range(len(self.bubbles)):
            if color_num == len(self.colors) - 1:
                color_num = 0
            else:
                color_num += 1
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=self.colors[color_num]
            )
            ax.add_patch(circ)
            ax.text(
                *self.bubbles[i, :2],
                labels[i],
                horizontalalignment="center",
                verticalalignment="center",
            )

    @validate_call(config=model_config)
    def save(self, path: Path | str, **kwargs: Any):
        """Save the figure as a file.

        Args:
            path (Path | str): The path to the file to save.
            **kwargs (Any): Additional keyword arguments for `plt.savefig`.
        """
        if path == "":
            raise LexosException("You must provide a valid path.")
        if self.fig is None:
            raise LexosException("The figure has not yet been generated.")
        self.fig.savefig(path, **kwargs)

    def show(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure using
        `BubbleChart.fig`. This will generally display in a Jupyter notebook.
        """
        return self.fig
