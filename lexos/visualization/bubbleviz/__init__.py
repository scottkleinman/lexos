"""bubbleviz.py.

This is a very experimental module for making bubble charts.

The easiest way to use it is to import it and call
`bubble_chart(terms, area)` where `terms` is a list of terms
and `area` is a list of corresponding counts or frequencies.
Alternatively, you can use `bubble_chart_from_dtm(dtm)` where
`dtm` is the output of `lexos.dtm.DTM`.
"""
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, validator

from lexos.exceptions import LexosException


class BubbleChartModel(BaseModel):
    """Ensure BubbleChart inputs are valid."""

    terms: list
    area: list
    limit: Optional[int] = 100
    title: Optional[str] = None
    bubble_spacing: Optional[Union[float, int]] = 0.1
    colors: Optional[List[str]] = [
        "#5A69AF",
        "#579E65",
        "#F9C784",
        "#FC944A",
        "#F24C00",
        "#00B825",
    ]
    figsize: Optional[tuple] = (15, 15)
    font_family: Optional[str] = "DejaVu Sans"
    show: Optional[bool] = True
    filename: Optional[str] = None

    @validator("terms")
    def check_terms_not_empty(cls, v):
        """Ensure `terms` is not empty."""
        if v == []:
            raise ValueError("Empty term lists are not allowed.")
        return v

    @validator("area")
    def check_area_not_empty(cls, v):
        """Ensure `area` is not empty."""
        if v == []:
            raise ValueError("Empty area lists are not allowed.")
        return v

    # @validator("area")
    # def num_terms_must_equal_area(cls, v, values):
    #     """Ensure the number of terms equals the number of areas."""
    #     if len(v["terms"]) != len(v["area"]):
    #         raise ValueError(
    #             "The number of terms must equal the number of counts or frequencies in the area."
    #         )
    #     return v


class BubbleChart:
    """Bubble chart."""

    def __init__(self, BubbleChartModel: BubbleChartModel):
        """Instantiate a bubble chart from a BubbleChartModel.

        Args:
            BubbleChartModel (BubbleChartModel): A BubbleChartModel

        Notes:
            - If "area" is sorted, the results might look weird.
            - If "limit" is raised too high, it will take a long time to generate the plot
            - Based on https://matplotlib.org/stable/gallery/misc/packed_bubbles.html.
        """
        self.model = BubbleChartModel
        # Reduce the area to the limited number of terms
        area = np.asarray(self.model.area[: self.model.limit])
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = self.model.bubble_spacing
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

        self.com = self.center_of_mass()

    def center_of_mass(self) -> int:
        """Centre of mass.

        Returns:
            int: The centre of mass.
        """
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> np.ndarray:
        """Centre distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            np.ndarray: The centre distance.
        """
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Outline distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The outline distance.
        """
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Check collisions.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The length of the distance between bubbles.
        """
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Collide.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The minimum index.
        """
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations: int = 50):
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
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # Calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # Check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # Try to move around a bubble that you collide with
                    # Find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # Calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # Calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(
        self,
        ax: object,
        labels: List[str],
        colors: List[str],
        font_family: str = "Arial",
    ):
        """Draw the bubble plot.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes.
            labels (List[str]): The labels of the bubbles.
            colors (List[str]): The colors of the bubbles.
            font_family (str): The font family.
        """
        plt.rcParams["font.family"] = font_family
        color_num = 0
        for i in range(len(self.bubbles)):
            if color_num == len(colors) - 1:
                color_num = 0
            else:
                color_num += 1
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[color_num]
            )
            ax.add_patch(circ)
            ax.text(
                *self.bubbles[i, :2],
                labels[i],
                horizontalalignment="center",
                verticalalignment="center"
            )


def create_bubble_chart(
    terms: List[str],
    area: List[Union[float, int]],
    limit: Optional[int] = 100,
    title: Optional[str] = None,
    bubble_spacing: Optional[Union[float, int]] = 0.1,
    colors: Optional[List[str]] = [
        "#5A69AF",
        "#579E65",
        "#F9C784",
        "#FC944A",
        "#F24C00",
        "#00B825",
    ],
    figsize: Optional[tuple] = (15, 15),
    font_family: Optional[str] = "DejaVu Sans",
    show: Optional[bool] = True,
    filename: Optional[str] = None,
):
    """Create a bubble chart.

    Args:
        terms (List[str]): A list of terms to plot.
        area (List[Union[float, int]]): A list of counts or frequencies corresponding to the terms.
        limit (int): The maximum number of bubbles to plot.
        title (str): The title of the plot.
        bubble_spacing (Union[float, int]): The spacing between bubbles.
        colors (List[str]): The colors of the bubbles.
        figsize (tuple): The size of the figure.
        font_family: (Optional[str]): The font family of the plot.
        show (bool): Whether to show the plot.
        filename (str): The filename to save the plot to.

    Raises:
        ValidationError: If any of the inputs are of the wrong type or if the length of the terms and area lists are not the same.
    """
    # Ensure that the inputs are valid
    try:
        model = BubbleChartModel(
            terms=terms,
            area=area,
            limit=limit,
            title=title,
            bubble_spacing=bubble_spacing,
            colors=colors,
            figsize=figsize,
            font_family=font_family,
            show=show,
            filename=filename,
        )
        bubble_chart = BubbleChart(model)
    except ValidationError as e:
        raise LexosException(e.json())

    # Create the figure
    bubble_chart.collapse()
    _, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=figsize)
    bubble_chart.plot(ax, terms, colors=colors, font_family=font_family)
    ax.axis("off")
    ax.relim()
    ax.autoscale_view()

    # Add title
    if title:
        ax.set_title(title)

    # Show the plot
    if show:
        plt.show()

    # Save the plot
    if filename:
        plt.to_file(filename)


def create_bubble_chart_from_dtm(
    dtm: Union[dict, list, object, pd.DataFrame, str, tuple],
    limit: Optional[int] = 100,
    title: Optional[str] = None,
    bubble_spacing: Optional[Union[float, int]] = 0.1,
    colors: Optional[List[str]] = [
        "#5A69AF",
        "#579E65",
        "#F9C784",
        "#FC944A",
        "#F24C00",
        "#00B825",
    ],
    figsize: Optional[tuple] = (15, 15),
    font_family: Optional[str] = "DejaVu Sans",
    show: Optional[bool] = True,
    filename: Optional[str] = None,
):
    """Create a bubble chart from a DTM.

    Args:
        dtm (Union[dict, list, object, pd.DataFrame, str, tuple]): The output of dtm.DTM.
        limit (int): The maximum number of bubbles to plot.
        title (str): The title of the plot.
        bubble_spacing (Union[float, int]): The spacing between bubbles.
        colors (List[str]): The colors of the bubbles.
        figsize (tuple): The size of the figure.
        font_family: (Optional[str]): The font family of the plot.
        show (bool): Whether to show the plot.
        filename (str): The filename to save the plot to.

    Raises:
        LexosException: If the input is not a valid DTM.
    """
    try:
        df = dtm.get_stats_table()
        assert "terms" in df.columns
        terms = df["terms"].tolist()
        area = df["sum"].tolist()
        assert len(terms) == len(area)
        create_bubble_chart(
            terms=terms,
            area=area,
            limit=limit,
            title=title,
            bubble_spacing=bubble_spacing,
            colors=colors,
            figsize=figsize,
            font_family=font_family,
            show=show,
            filename=filename,
        )
    except Exception:
        raise LexosException("The input is not a valid DTM object.")

