"""bubbleviz.

This is a very experimental module for making bubble charts.

The easiest way to use it is to import it and call
`make_bubble_chart(terms, area)` where `terms` is a list of terms
and `area` is a list of corresponding counts or frequencies.
"""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


class BubbleChart:
    """Bubble chart."""
    def __init__(self,
                 area: list,
                 bubble_spacing: Union[float, int] = 0,
                 limit: int = 100):
        """Setup for bubble collapse.

        Args:
            area (list): List of counts or frequencies
            bubble_spacing: (Union[float, int]): The spacing between bubbles after collapsing.
            limit (int): The maximum number of bubbles to display.

        Notes:
            - If "area" is sorted, the results might look weird.
            - If "limit" is raised too high, it will take a long time to generate the plot
            - Based on https://matplotlib.org/stable/gallery/misc/packed_bubbles.html.
        """
        # Reduce the area to the limited number of terms
        area = np.asarray(area[0:limit])
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # Calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        """Centre of mass.

        Returns:
            int: The centre of mass.
        """
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles) -> np.ndarray:
        """Centre distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            np.ndarray: The centre distance.
        """
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        """Outline distance.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The outline distance.
        """
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        """Check collisions.

        Args:
            bubble (np.ndarray): Bubble array.
            bubbles (np.ndarray): Bubble array.

        Returns:
            int: The length of the distance between bubbles.
        """
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble: np.ndarray, bubbles: np.ndarray):
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
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax: object, labels: List[str], colors: List[str]):
        """Draw the bubble plot.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes.
            labels (List[str]): The labels of the bubbles.
            colors (List[str]): The colors of the bubbles.
        """
        color_num = 0
        for i in range(len(self.bubbles)):
            if color_num == len(colors)-1:
                color_num = 0
            else:
                color_num += 1
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[color_num])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')

def make_bubble_chart(terms: List[str],
                      area: List[Union[float, int]],
                      limit: int = 100,
                      title: str = None,
                      bubble_spacing: Union[float, int] = 0.1,
                      colors: List[str] = [
                          "#5A69AF",
                          "#579E65",
                          "#F9C784",
                          "#FC944A",
                          "#F24C00",
                          "#00B825"
                    ],
                      figsize: tuple = (15,15),
                      show: bool = True,
                      filename: str = None):
    """Make bubble chart.

    Args:
        terms (List[str]): The terms to plot.
        area (List[Union[float, int]]): The area of the bubbles.
        limit (int): The maximum number of bubbles to plot.
        title (str): The title of the plot.
        bubble_spacing (Union[float, int]): The spacing between bubbles.
        colors (List[str]): The colors of the bubbles.
        figsize (tuple): The size of the figure.
        show (bool): Whether to show the plot.
        filename (str): The filename to save the plot to.
    """
    bubble_chart = BubbleChart(area=area, bubble_spacing=bubble_spacing, limit=limit)
    bubble_chart.collapse()
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=figsize)
    bubble_chart.plot(ax, terms, colors=colors)
    ax.axis("off")
    ax.relim()
    ax.autoscale_view()
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    if filename:
        plt.to_file(filename)
