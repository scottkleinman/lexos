"""This is a model to produce bootstrap consensus tree of the dtm.

Last update: July 25, 2025
Last tested: December 5, 2025

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
    def _validate_text_color(cls, value):
        if not is_valid_colour(value):
            raise LexosException(
                "Value is not a valid colour: string not recognised as a valid colour."
            )
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
        # Save the DTM to avoid multiple calls
        dtm = self._doc_term_matrix

        # Get doc names, since tree nodes need labels
        labels = [doc for doc in self._doc_term_matrix.index.values.tolist()]

        # The bootstrap process to get all the trees.
        return [
            self._get_newick_tree(
                sample_dtm=dtm.sample(
                    axis=1,
                    frac=0.8,
                    replace=self.replace,
                    random_state=np.random.RandomState(),
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
        elif layout in ["fan"]:
            return self._draw_fan_tree(tree, normalized_color, layout)
        else:
            raise ValueError(f"Unknown layout: {layout}. Use 'rectangular' or 'fan'.")

    def _draw_rectangular_tree(self, tree, normalized_color) -> Figure:
        """Draw traditional rectangular tree layout."""
        fig, ax = plt.subplots()

        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: f"{clade.branch_length:.{PRECISION}f}\n"
            if clade.branch_length is not None
            else "",
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
        plt.gcf().set_size_inches(w=9.5, h=(len(self._document_label_map) * 0.3 + 1))
        plt.title(self.title, color=normalized_color)
        plt.gcf().tight_layout()

        # Change the line spacing
        for text in plt.gca().texts:
            text.set_linespacing(spacing=0.1)
            text.set_color(normalized_color)

        return fig

    def _draw_fan_tree(self, tree, normalized_color, style: str) -> Figure:
        """Draw fan-style (circular) tree layout with color-coded labels by clade."""
        # Create figure with equal aspect ratio for circular plot
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")

        # Get all terminal nodes and calculate their positions
        terminals = list(tree.get_terminals())
        num_terminals = len(terminals)

        if num_terminals == 0:
            raise ValueError("Tree has no terminal nodes")

        # Fan style: use 270 degrees (3/4 circle)
        start_angle = -135  # Start at bottom-left
        total_angle = 270  # 270 degrees total

        # Assign angles to terminal nodes
        terminal_angles = {}
        for i, terminal in enumerate(terminals):
            if num_terminals == 1:
                angle = start_angle
            else:
                angle = start_angle + (i * total_angle / num_terminals)
            terminal_angles[terminal] = math.radians(angle)

        # Calculate tree depth for all nodes
        node_depths = {}
        max_depth = 0

        def calculate_depths(node, depth=0):
            """Calculate depth from root for each node."""
            nonlocal max_depth
            node_depths[node] = depth
            max_depth = max(max_depth, depth)

            if not node.is_terminal():
                for child in node.clades:
                    calculate_depths(child, depth + 1)

        calculate_depths(tree.root)

        # Generate color palette
        def generate_colors(n):
            """Generate n distinct colors."""
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

            def find_terminal_descendants(node):
                """Get all terminal nodes that descend from this node."""
                if node.is_terminal():
                    return [node]

                descendants = []
                for child in node.clades:
                    descendants.extend(find_terminal_descendants(child))
                return descendants

            def assign_colors_recursive(node, current_depth=0):
                """Recursively assign colors based on tree structure."""
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

        # Calculate positions for all nodes
        node_positions = {}
        terminal_parent_map = {}  # Store parent-child relationships for terminals

        def calculate_positions(node):
            """Calculate x, y positions for all nodes."""
            if node.is_terminal():
                # Terminal nodes: place directly on circumference
                angle = terminal_angles[node]
                x = circumference_radius * math.cos(angle)
                y = circumference_radius * math.sin(angle)
                node_positions[node] = (x, y, angle)
                return angle, circumference_radius
            else:
                # Internal nodes: calculate based on children
                child_angles = []
                child_distances = []

                for child in node.clades:
                    child_angle, child_distance = calculate_positions(child)
                    child_angles.append(child_angle)
                    child_distances.append(child_distance)

                    # Store parent-child relationship for terminals
                    if child.is_terminal():
                        terminal_parent_map[child] = node

                # Internal node angle is the average of its children's angles
                avg_angle = sum(child_angles) / len(child_angles)

                # Position internal nodes VERY close to center for maximum clustering
                depth = node_depths[node]

                if depth == 0:  # Root node
                    distance = 0.02  # Almost at center
                else:
                    # Use extremely aggressive scaling - internal nodes very close to root
                    normalized_depth = depth / max_depth
                    distance = circumference_radius * (
                        0.05 + 0.15 * (normalized_depth**3.0)
                    )

                x = distance * math.cos(avg_angle)
                y = distance * math.sin(avg_angle)
                node_positions[node] = (x, y, avg_angle)

                return avg_angle, distance

        # Calculate all positions
        calculate_positions(tree.root)

        # Draw the tree with proper branching structure
        def draw_branches(node):
            """Draw branches connecting nodes with proper tree structure."""
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
                        label_extension = 0.02  # Your preferred distance
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
                            fontsize=12,
                            weight="bold",
                        )
                    else:
                        # Fallback if branch_length is zero (shouldn't happen)
                        ax.text(
                            x * 1.08,
                            y * 1.08,
                            label,
                            color=label_color,
                            fontsize=12,
                            weight="bold",
                        )
                else:
                    # Fallback if no parent found (shouldn't happen)
                    ax.text(
                        x * 1.08,
                        y * 1.08,
                        label,
                        color=label_color,
                        fontsize=12,
                        weight="bold",
                    )

                # Don't draw terminal node markers - only show labels

            elif node == tree.root:
                # Only draw the root node marker - hide all other internal nodes
                ax.plot(x, y, "o", color=normalized_color, markersize=6, alpha=1.0)

        # Set up the plot limits
        plot_limit = circumference_radius * 1.25
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add title
        plt.title(
            self.title,
            color=normalized_color,
            pad=30,
            fontsize=16,
            weight="bold",
        )

        plt.tight_layout()

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
        """
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        return self.fig
