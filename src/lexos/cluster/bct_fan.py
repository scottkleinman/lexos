"""This is a model to produce bootstrap consensus tree of the dtm.

Last update: June 28, 2026
Last tested: June 28, 2026

# TODO:
- Datatype for `dtm` should match those allowable for the `Dendrogram` class.
- Update tests for API changes.
- See https://github.com/koonimaru/omniplot/blob/962310436a153098b671ebd76cdd59f8a7b5e681/omniplot/plot.py#L38 for a method of getting round dendrograms.
"""

import colorsys
import math
from io import StringIO
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.Consensus import majority_consensus
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, field_validator, validate_call
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.util import is_valid_colour

PRECISION = 1  # Precision for branch length formatting in dendrogram labels


class BCT(BaseModel):
    """The Bootstrap Consensus Tree Class."""

    dtm: DTM = Field(None, description="The document term matrix.")
    metric: Optional[str] = Field("euclidean", description="The distance metric.")
    method: Optional[str] = Field("average", description="The linkage method.")
    cutoff: Optional[float] = Field(0.5, description="The cutoff value.")
    iterations: Optional[int] = Field(
        100, description="The number of iterations to run the bootstrap."
    )
    replace: Optional[str] = Field("without", description="The replacement method.")
    labels: Optional[list[int | str] | dict[int, str]] = Field(
        None, description="The document labels."
    )
    text_color: Optional[str] = Field("rgb(0, 0, 0)", description="The text colour.")
    title: Optional[str] = Field(
        "Bootstrap Consensus Tree Result", description="The title of the dendrogram."
    )
    layout: Optional[str] = Field(
        "rectangular", description="Tree visualization layout: 'rectangular' or 'fan'."
    )
    figsize: Optional[tuple[float, float]] = Field(
        (10, 10),
        description="Optional figure size as (width, height) in inches.",
    )
    label_fontsize: Optional[int] = Field(
        12, description="Font size for leaf labels in the tree diagram."
    )
    random_seed: Optional[int] = Field(
        None, description="Optional seed for the random number generator."
    )
    fig: Optional[Figure] = Field(None, description="The figure for the dendrogram.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _doc_term_matrix(self) -> pd.DataFrame:
        """Return a dataframe of the document term matrix.

        Returns:
            pd.DataFrame: The document term matrix with doc labels as the index and terms as the columns.


        Note that the web app uses doc ids as the index.
        """
        if self.dtm is None:
            raise LexosException("No document term matrix found.")
        return self.dtm.to_df().T

    @property
    def _document_label_map(self) -> dict[int, str] | dict:
        """Return a dictionary of document label map.

        Returns:
            list[int | str] | dict[int, str]: A document label map or a list of indices or labels.
        """
        if self.labels is not None and len(self.labels) > 0:
            if isinstance(self.labels, dict):
                return self.labels
            else:
                if isinstance(self.labels[0], int):
                    return {i: f"doc{i + 1}" for i, _ in enumerate(self.labels)}
                else:
                    return {i: label for i, label in enumerate(self.labels)}
        return {}

    def __init__(self, **data) -> None:
        """Construct the BCT instance and render the bootstrap consensus tree."""
        super().__init__(**data)

        # Get the matplotlib figure for bootstrap consensus tree result
        self.fig = self._get_bootstrap_consensus_tree_fig(layout=self.layout)
        plt.close()

    @field_validator("text_color", mode="after")
    @classmethod
    def _validate_text_color(cls, value) -> str:
        """Validate that the text color is a valid color string.

        Args:
            value (str): The text color string.

        Returns:
            str: The validated text color string.

        Raises:
            LexosException: If the text color is not a valid color string.
        """
        if not is_valid_colour(value):
            raise LexosException(
                "Value is not a valid colour: string not recognised as a valid colour."
            )
        return value

    @field_validator("figsize", mode="after")
    @classmethod
    def _validate_figsize(cls, value):
        """Validate that the figure size is a tuple of two positive numbers.

        Args:
            value (tuple[float, float]): The figure size as (width, height).

        Returns:
            tuple[float, float]: The validated figure size.

        Raises:
            LexosException: If the figure size is not valid.
        """
        if value is None:
            return value
        if len(value) != 2:
            raise LexosException("figsize must contain exactly two numeric values.")
        if value[0] <= 0 or value[1] <= 0:
            raise LexosException("figsize values must be greater than 0.")
        return value

    @field_validator("label_fontsize", mode="after")
    @classmethod
    def _validate_label_fontsize(cls, value) -> int:
        """Validate that the label font size is a positive integer.

        Args:
            value (int): The label font size.

        Returns:
            int: The validated label font size.

        Raises:
            LexosException: If the font size is not a positive integer.
        """
        if value is None:
            return 12
        if value <= 0:
            raise LexosException("label_fontsize must be a positive integer.")
        return value

    @staticmethod
    def linkage_to_newick(matrix: np.ndarray, labels: list[str]) -> str:
        """Convert a linkage matrix to a Newick formatted tree.

        Args:
            matrix (np.ndarray): The linkage matrix.
            labels (list[str]): Names of the tree node.

        Returns:
            str: The Newick representation of the linkage matrix.
        """
        # Convert the linkage matrix to a ClusterNode object
        tree = to_tree(matrix, False)

        # Define a recursive function to build the Newick tree
        def _build_newick_tree(
            node: ClusterNode, newick: str, parent_dist: float, leaf_names: list[str]
        ) -> str:
            """Recursively build the Newick tree.

            Args:
                node (ClusterNode): The tree node currently being converted to Newick.
                newick (str): The current Newick representation of the tree.
                parent_dist (float): The distance to parent node.
                leaf_names (list[str]): Names of the tree node.

            Returns:
                str: The Newick representation of the tree.
            """
            # If the node is a leaf, enclose
            if node.is_leaf():
                return f"{leaf_names[node.id]}:{(parent_dist - node.dist) / 2}{newick}"
            else:
                # Write the distance to the parent node
                newick = (
                    f"):{(parent_dist - node.dist) / 2}{newick}"
                    if len(newick) > 0
                    else ");"
                )
                # Recursive call to expand the tree
                newick = _build_newick_tree(
                    newick=newick,
                    node=node.get_left(),
                    parent_dist=node.dist,
                    leaf_names=leaf_names,
                )
                newick = _build_newick_tree(
                    newick=f",{newick}",
                    node=node.get_right(),
                    parent_dist=node.dist,
                    leaf_names=leaf_names,
                )
                # Enclose the tree at the beginning
                return f"({newick}"

        # Trigger the recursive function
        return _build_newick_tree(
            node=tree, newick="", parent_dist=tree.dist, leaf_names=labels
        )

    def _get_newick_tree(self, labels: list[str], sample_dtm: pd.DataFrame) -> str:
        """Get Newick tree based on a subset of the DTM.

        Args:
            labels (list[str]): All file names from the DTM
            sample_dtm (pd.DataFrame): An 80% subset of the complete DTM

        Returns:
            str: A Newick formatted tree representing the DTM subset
        """
        # Get the linkage matrix for the sample doc term matrix
        linkage_matrix = linkage(
            sample_dtm.values, metric=self.metric, method=self.method
        )

        # Get the Newick representation of the tree
        newick = self.linkage_to_newick(matrix=linkage_matrix, labels=labels)

        # Convert linkage matrix to a tree node and return it
        return Phylo.read(StringIO(newick), format="newick")

    def _get_bootstrap_trees(self) -> list[str]:
        """Do bootstrap on the DTM to get a list of Newick trees.

        Returns:
            list[str]: A list of Newick formatted tree where each tree was based on an 80% subset of the complete DTM.
        """
        dtm = self._doc_term_matrix
        labels = dtm.index.tolist()
        random_state = np.random.RandomState(seed=self.random_seed)

        return [
            self._get_newick_tree(
                sample_dtm=dtm.sample(
                    axis=1,
                    frac=0.8,
                    replace=self.replace,
                    random_state=random_state,
                ),
                labels=labels,
            )
            for _ in range(self.iterations)
        ]

    def _get_bootstrap_consensus_tree(self) -> Phylo:
        """Get the consensus tree.

        Returns:
            Phylo: The consensus tree of the list of Newick trees.
        """
        # Find the consensus of all the Newick trees
        return majority_consensus(trees=self._get_bootstrap_trees(), cutoff=self.cutoff)

    def _get_bootstrap_consensus_tree_fig(self, layout: str = "rectangular") -> Figure:
        """Generate a bootstrap consensus tree figure.

        Args:
            layout: Tree visualization style. Options:
                - "rectangular": Traditional rectangular tree layout
                - "fan": Circular fan-style tree layout

        Returns:
            Figure: The matplotlib figure containing the tree visualization.
        """
        # Get the colours
        color = tuple(map(int, self.text_color[4:-1].split(",")))
        normalized_color = tuple(x / 255 for x in color)

        # Get the consensus tree
        tree = self._get_bootstrap_consensus_tree()
        tree.root.color = color

        if layout == "rectangular":
            return self._draw_rectangular_tree(tree, normalized_color)
        elif layout == "fan":
            return self._draw_fan_tree(tree, normalized_color, layout)
        else:
            raise ValueError(f"Unknown layout: {layout}. Use 'rectangular' or 'fan'.")

    def _draw_rectangular_tree(
        self, tree: Phylo, normalized_color: tuple[float, float, float]
    ) -> Figure:
        """Draw traditional rectangular tree layout.

        Args:
            tree (Phylo): The tree to draw.
            normalized_color (tuple[float, float, float]): The base color for the tree in RGB normalized to [0, 1].

        Returns:
            Figure: The matplotlib figure containing the rectangular tree visualization.
        """
        fig, ax = plt.subplots()

        # Remove background
        ax.set_facecolor("none")

        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: (
                f"{clade.branch_length:.{PRECISION}f}\n"
                if clade.branch_length is not None
                else ""
            ),
        )

        # Set labels for the plot
        plt.xlabel("Branch Length", color=normalized_color)
        plt.ylabel("Documents", color=normalized_color)

        # Hide the two unused borders
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # Set the colour of the used borders and labels
        plt.gca().spines["bottom"].set_color(normalized_color)
        plt.gca().spines["left"].set_color(normalized_color)
        plt.gca().tick_params(colors=normalized_color)

        # Extend the x-axis to the right to fit longer labels
        x_left, x_right, y_low, y_high = plt.axis()
        plt.axis((x_left, x_right * 1.25, y_low, y_high))

        # Set the graph size, title, and tight layout
        if self.figsize is not None:
            width, height = self.figsize
        else:
            width = 9.5
            height = len(self._document_label_map) * 0.3 + 1
        plt.gcf().set_size_inches(w=width, h=height)
        plt.title(self.title, color=normalized_color)
        plt.gcf().tight_layout()

        # Change the line spacing
        for text in plt.gca().texts:
            text.set_linespacing(spacing=0.1)
            text.set_color(normalized_color)
            text.set_fontsize(self.label_fontsize)

        return fig

    def _draw_fan_tree(
        self,
        tree: Phylo,
        normalized_color: tuple[float, float, float],
        layout: str = "fan",
    ) -> Figure:
        """Draw fan-style (circular) tree layout with color-coded labels by clade.

        Args:
            tree (Phylo): The tree to draw.
            normalized_color (tuple[float, float, float]): The base color for the tree in RGB normalized to [0, 1].
            layout (str): The tree layout style.

        Returns:
            Figure: The matplotlib figure containing the fan-style tree visualization.
        """
        # Create figure with equal aspect ratio for circular plot
        figsize = self.figsize if self.figsize is not None else (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # Get all terminal nodes and calculate their positions
        terminals = list(tree.get_terminals())
        num_terminals = len(terminals)

        if num_terminals == 0:
            raise ValueError("Tree has no terminal nodes")

        # Calculate tree depth for all nodes
        node_depths = {}
        max_depth = 0

        def calculate_depths(node: ClusterNode, depth: int = 0):
            """Calculate depth from root for each node.

            Args:
                node (ClusterNode): The current node being processed.
                depth (int): The depth of the current node from the root.
            """
            nonlocal max_depth
            node_depths[node] = depth
            max_depth = max(max_depth, depth)

            if not node.is_terminal():
                for child in node.clades:
                    calculate_depths(child, depth + 1)

        calculate_depths(tree.root)

        # Cache terminal counts for proportional sector allocation
        _tc_cache = {}

        def _terminal_count(node: ClusterNode) -> int:
            """Return the number of terminal descendants of a node.

            Args:
                node (ClusterNode): The node to count from.

            Returns:
                int: The number of terminal descendants.
            """
            if node not in _tc_cache:
                _tc_cache[node] = (
                    1
                    if node.is_terminal()
                    else sum(_terminal_count(c) for c in node.clades)
                )
            return _tc_cache[node]

        # Generate color palette
        def generate_colors(n: int) -> list[str]:
            """Generate n distinct colors.

            Args:
                n (int): The number of distinct colors to generate.

            Returns:
                list[str]: A list of n distinct colors.
            """
            if n <= 10:
                # Use predefined colors for small numbers
                colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                return colors[:n]
            else:
                # Generate colors using HSV color space for larger numbers
                colors = []
                for i in range(n):
                    hue = i / n
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    colors.append(mcolors.rgb2hex(rgb))
                return colors

        # Find clade groupings for color assignment at higher levels
        terminal_colors = {}

        def assign_higher_level_colors():
            """Assign colors based on higher-level splits in the tree."""
            color_palette = generate_colors(10)
            color_index = 0

            def find_terminal_descendants(node: ClusterNode) -> list[ClusterNode]:
                """Get all terminal nodes that descend from this node."""
                if node.is_terminal():
                    return [node]

                descendants = []
                for child in node.clades:
                    descendants.extend(find_terminal_descendants(child))
                return descendants

            def assign_colors_recursive(node: ClusterNode, current_depth: int = 0):
                """Recursively assign colors based on tree structure.

                Args:
                    node (ClusterNode): The current node being processed.
                    current_depth (int): The depth of the current node from the root.
                """
                nonlocal color_index

                if node.is_terminal():
                    return

                # For nodes at depth 1 (direct children of root), assign same color to all descendants
                if current_depth == 1:
                    descendants = find_terminal_descendants(node)
                    current_color = color_palette[color_index % len(color_palette)]

                    for terminal in descendants:
                        if terminal not in terminal_colors:
                            terminal_colors[terminal] = current_color

                    color_index += 1

                # Continue recursively for deeper nodes
                for child in node.clades:
                    assign_colors_recursive(child, current_depth + 1)

            # Start from root
            assign_colors_recursive(tree.root)

        # Assign colors based on higher-level clades
        assign_higher_level_colors()

        # Set circumference radius for terminals
        circumference_radius = 1.0

        # Calculate positions using the equal-angle algorithm.
        # Each clade receives an angular sector proportional to its terminal count,
        # so related leaves are naturally grouped close together rather than
        # distributed evenly across the full circle.
        node_positions = {}
        terminal_parent_map = {}

        def assign_positions(
            node: ClusterNode, sector_start: float, sector_end: float
        ) -> None:
            """Recursively assign (x, y, mid_angle) using equal-angle sector allocation.

            Args:
                node (ClusterNode): The current node being processed.
                sector_start (float): Start of this node's angular sector (radians).
                sector_end (float): End of this node's angular sector (radians).
            """
            mid_angle = (sector_start + sector_end) / 2.0
            depth = node_depths[node]
            distance = (
                (circumference_radius * depth / max_depth) if max_depth > 0 else 0.0
            )
            x = distance * math.cos(mid_angle)
            y = distance * math.sin(mid_angle)
            node_positions[node] = (x, y, mid_angle)

            if node.is_terminal():
                return

            total = _terminal_count(node)
            a = sector_start
            for child in node.clades:
                # Add extra space for terminal nodes to prevent label overlap
                frac = _terminal_count(child) / total
                b = (
                    a
                    + frac * (sector_end - sector_start)
                    + (spacer / total_terminals if child.is_terminal() else 0)
                )
                if child.is_terminal():
                    terminal_parent_map[child] = node
                assign_positions(child, a, b)
                a = b

        # Calculate a small spacer (0.2 degrees) between terminals to prevent overlap
        total_terminals = _terminal_count(tree.root)
        spacer = (0.2 * math.pi / 180.0) * total_terminals
        assign_positions(tree.root, 0.0, 2 * math.pi - spacer)

        # Draw the tree with proper branching structure
        def draw_branches(node: ClusterNode):
            """Draw branches connecting nodes with proper tree structure.

            Args:
                node (ClusterNode): The current node being processed.
            """
            if node not in node_positions:
                return

            node_x, node_y, node_angle = node_positions[node]

            # For internal nodes with multiple children, we want to create proper branching
            if not node.is_terminal() and len(node.clades) > 1:
                # First, draw lines from this node to each child
                for child in node.clades:
                    if child in node_positions:
                        child_x, child_y, child_angle = node_positions[child]

                        # Draw the branch
                        if child.is_terminal():
                            linewidth = 2.0
                            alpha = 0.9
                        else:
                            linewidth = 1.8
                            alpha = 0.8

                        ax.plot(
                            [node_x, child_x],
                            [node_y, child_y],
                            color=normalized_color,
                            linewidth=linewidth,
                            alpha=alpha,
                        )

                        # Recursively draw children
                        draw_branches(child)
            elif len(node.clades) == 1:
                # Single child - direct connection
                child = node.clades[0]
                if child in node_positions:
                    child_x, child_y, child_angle = node_positions[child]

                    linewidth = 2.0 if child.is_terminal() else 1.8
                    alpha = 0.9 if child.is_terminal() else 0.8

                    ax.plot(
                        [node_x, child_x],
                        [node_y, child_y],
                        color=normalized_color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )

                    draw_branches(child)

        # Draw all branches starting from root
        draw_branches(tree.root)

        # Draw nodes and labels with perfect alignment and color coding
        for node in node_positions:
            x, y, angle = node_positions[node]

            if node.is_terminal():
                # Terminal node: draw label perfectly aligned by extending the branch vector
                label = str(node.name) if node.name else "Unnamed"

                # Get the color for this terminal node
                label_color = terminal_colors.get(node, normalized_color)

                # Get the parent node position
                if node in terminal_parent_map:
                    parent = terminal_parent_map[node]
                    parent_x, parent_y, parent_angle = node_positions[parent]

                    # Calculate the direction vector from parent to terminal
                    dx = x - parent_x
                    dy = y - parent_y

                    # Normalize the direction vector
                    branch_length = math.sqrt(dx * dx + dy * dy)
                    if branch_length > 0:
                        dx_norm = dx / branch_length
                        dy_norm = dy / branch_length

                        # Extend the label position along the same direction vector
                        label_extension = 0.03
                        label_x = x + dx_norm * label_extension
                        label_y = y + dy_norm * label_extension

                        # Calculate rotation angle from the direction vector
                        branch_angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(branch_angle_rad)

                        # Ensure text is readable (not upside down)
                        if -90 <= angle_deg <= 90:
                            rotation = angle_deg
                            ha = "left"
                            va = "center"
                        else:
                            rotation = angle_deg + 180
                            ha = "right"
                            va = "center"

                        # Draw the label perfectly aligned with branch direction and colored by clade
                        ax.text(
                            label_x,
                            label_y,
                            label,
                            rotation=rotation,
                            rotation_mode="anchor",
                            ha=ha,
                            va=va,
                            color=label_color,
                            fontsize=self.label_fontsize,
                            weight="bold",
                            clip_on=False,
                        )
                    else:
                        # Fallback if branch_length is zero (shouldn't happen)
                        ax.text(
                            x * 1.08,
                            y * 1.08,
                            label,
                            color=label_color,
                            fontsize=self.label_fontsize,
                            weight="bold",
                            clip_on=False,
                        )
                else:
                    # Fallback if no parent found (shouldn't happen)
                    ax.text(
                        x * 1.08,
                        y * 1.08,
                        label,
                        color=label_color,
                        fontsize=self.label_fontsize,
                        weight="bold",
                        clip_on=False,
                    )

                # Don't draw terminal node markers - only show labels

        # Compute plot_limit so labels just reach (but don't exceed) the figure boundary.
        # With adjustable="datalim", the axes fills the full subplot allocation:
        #   axes_h = (top - bottom) * h = 0.96 * h
        #   axes_w = (right - left)  * w = 0.98 * w
        # The uniform scale = min(axes_h, axes_w) / (2 * plot_limit) inches per unit.
        # The radial extent of the farthest label point in figure inches is:
        #   (circumference_radius + 0.08) * scale  +  max_label_chars * char_width_in
        # Setting that equal to figure_half = min(w, h) / 2 and solving for plot_limit:
        #   plot_limit = (circumference_radius + 0.08) * axes_size
        #                / (2 * (figure_half - label_width_in))
        max_label_len = max(
            (len(str(t.name)) if t.name else 0 for t in terminals), default=10
        )
        w, h = figsize
        char_width_in = (
            self.label_fontsize * 0.5 / 72
        )  # 0.5× fontsize height is a reliable avg width
        label_width_in = max_label_len * char_width_in
        axes_size = min(0.97 * h, 0.98 * w)  # matches subplots_adjust below
        figure_half = min(w, h) / 2
        available = figure_half - label_width_in
        # Manual override: set plot_limit slightly beyond circumference to pull labels out
        plot_limit = circumference_radius + 0.1
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add title using suptitle for absolute figure positioning
        fig.suptitle(
            self.title,
            color=normalized_color,
            fontsize=16,
            weight="bold",
            y=0.98,
        )

        # Remove background
        ax.set_facecolor("none")

        # Use tight margins so the axes fills the figure space below the title.
        # We lower the 'top' to 0.93 to leave room for the suptitle while
        # bringing the plot labels closer to the title.
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

        return fig

    @validate_call(config=model_config)
    def save(self, path: Path | str | None) -> None:
        """Save the bootstrap consensus tree result to a file.

        Args:
            path (Path | str | None): The path to save the file.
        """
        if not path or path == "":
            raise LexosException("You must provide a valid path.")
        self.fig.savefig(path)

    def show(self) -> Figure:
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure using `BCT.fig`.
        This will generally display in a Jupyter notebook.

        Returns:
            Figure: The matplotlib figure containing the tree visualization.
        """
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        return self.fig
