"""bootstrap_consensus.py.

Last update: July 15, 2026
Last tested: July 15, 2026

# NOTE:
- See https://github.com/koonimaru/omniplot/blob/962310436a153098b671ebd76cdd59f8a7b5e681/omniplot/plot.py#L38 for a method of getting radial dendrograms. This might be a third type of layout.
"""

import colorsys
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.Consensus import majority_consensus
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator, validate_call
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.util import is_valid_colour

PRECISION = 1  # Precision for branch length formatting in dendrogram labels


def generate_colors(n: int) -> list[str]:
    """Generate a list of n distinct colors.

    Args:
        n (int): The number of colors to generate.

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


def assign_auto_colors(tree: Tree) -> dict[str, str]:
    """Assign colors to leaf labels based on biopython Clade structure.

    Args:
        tree (Tree): The tree to assign colors for.

    Returns:
        dict[str, str]: A mapping of leaf label names to hex color strings.
    """
    terminal_colors = {}
    color_palette = generate_colors(10)
    color_index = 0

    def find_terminal_descendants(clade: Clade) -> list[Clade]:
        """Get all terminal nodes that descend from this clade."""
        return clade.get_terminals()

    def assign_colors_recursive(clade: Clade, current_depth: int = 0):
        """Recursively assign colors based on tree structure.

        Args:
            clade (Clade): The current clade being processed.
            current_depth (int): The depth of the current clade from the root.
        """
        nonlocal color_index

        if clade.is_terminal():
            return

        # For nodes at depth 1 (direct children of root), assign same color to all descendants
        if current_depth == 1:
            terminals = find_terminal_descendants(clade)
            current_color = color_palette[color_index % len(color_palette)]

            for t in terminals:
                if t.name and t.name not in terminal_colors:
                    terminal_colors[t.name] = current_color

            color_index += 1

        # Continue recursively for deeper nodes
        for child in clade.clades:
            assign_colors_recursive(child, current_depth + 1)

    # Start from root
    assign_colors_recursive(tree.root)
    return terminal_colors


def resolve_label_colors(
    tree: Tree, label_colors: Optional[str | dict[str, list[str]]], base_color: str
) -> dict[str, str]:
    """Resolve leaf label colors based on mapping preference.

    Args:
        tree (Tree): The tree object.
        label_colors (Optional[str | dict[str, list[str]]]): Mapping preference.
            - "auto": Assign colors automatically by clade.
            - dict: Custom mapping of hex colors to lists of labels.
            - None: Use base_color for all.
        base_color (str): The default color to use.

    Returns:
        dict[str, str]: A mapping of label names to hex colors.
    """
    labels = [t.name for t in tree.get_terminals() if t.name]
    resolved = {label: base_color for label in labels}

    if label_colors == "auto":
        auto_colors = assign_auto_colors(tree)
        resolved.update(auto_colors)
    elif isinstance(label_colors, dict):
        for hex_color, label_list in label_colors.items():
            for lbl in label_list:
                if lbl in resolved:
                    resolved[lbl] = hex_color

    return resolved


class FanTree(BaseModel):
    """A class to draw a fan-style tree layout using Biopython's Phylo module."""

    tree_obj: Tree = Field(..., description="The tree to draw.")
    title: Optional[str] = Field(
        "Bootstrap Consensus Tree Result", description="The title of the dendrogram."
    )
    figsize: Optional[tuple[float, float]] = Field(
        (10, 10),
        description="Optional figure size as (width, height) in inches.",
    )
    label_fontsize: Optional[int] = Field(
        12, description="Font size for leaf labels in the tree diagram."
    )
    fontfamily: Optional[str] = Field(
        "sans-serif", description="Font family for leaf labels."
    )
    label_colors: Optional[Union[str, dict[str, list[str]], None]] = Field(
        None,
        description="Optional color mapping: 'auto' or dict of hex colors to labels.",
    )
    linewidth: Optional[float] = Field(1.2, description="The width of the tree lines.")
    min_leaf_len: Optional[float] = Field(
        1.5, description="The minimum length of leaf branches."
    )
    internal_scale: Optional[float] = Field(
        0.6, description="The scaling factor for internal branches."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self) -> Figure:
        """Draw fan-style tree layout.

        Returns:
            Figure: The matplotlib figure containing the fan tree visualization.
        """
        labels = [terminal.name for terminal in self.tree_obj.get_terminals()]

        # 1. Standardize internal tree branch proportions with dynamic stretching
        self._normalize_branches(
            self.tree_obj.root,
            use_proportional_branches=True,
        )
        lines, labels_info = self._compute_equal_angle(
            self.tree_obj.root, min_angle=0, max_angle=2 * np.pi
        )

        # 2. Resolve Label Colors
        resolved_colors = resolve_label_colors(
            self.tree_obj, self.label_colors, "#000000"
        )

        fig, ax = plt.subplots(figsize=self.figsize, dpi=100)

        # Line coordinate unpacking ([X1, X2], [Y1, Y2])
        for p1, p2 in lines:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="black",
                linewidth=self.linewidth,
                zorder=1,
            )

        # Draw text labels with an automatic layout offset
        for x, y, rad, deg, label_text in labels_info:
            color = resolved_colors.get(label_text, "#000000")

            # Determine horizontal alignment based on the hemisphere
            # This matches the 180-degree flip logic in _compute_equal_angle to ensure
            # labels point away from the center but remain readable.
            angle_deg = np.degrees(rad) % 360
            ha = "right" if (90 < angle_deg <= 270) else "left"
            va = "center"

            cos_val, sin_val = np.cos(rad), np.sin(rad)
            # Tiny extra margin past the terminal branch node coordinate tip
            offset = 0.12
            ax.text(
                x + offset * cos_val,
                y + offset * sin_val,
                label_text,
                rotation=deg,
                rotation_mode="anchor",
                ha=ha,
                va=va,
                color=color,
                fontsize=self.label_fontsize,
                fontfamily=self.fontfamily,
                zorder=2,
            )

        # 3. Dynamic Margin Calculations to Prevent Clipping Long Text
        max_char_len = max(len(l) for l in labels) if labels else 1
        estimated_text_width = (max_char_len * (self.label_fontsize * 0.55)) / 72.0

        label_xs = [x for x, y, r, d, lbl in labels_info]
        label_ys = [y for x, y, r, d, lbl in labels_info]
        line_xs = [pt[0] for line in lines for pt in line]
        line_ys = [pt[1] for line in lines for pt in line]

        xs = label_xs + line_xs
        ys = label_ys + line_ys

        dynamic_pad = max(1.0, estimated_text_width * 0.8)
        ax.set_xlim(min(xs) - dynamic_pad, max(xs) + dynamic_pad)
        ax.set_ylim(min(ys) - dynamic_pad, max(ys) + dynamic_pad)

        ax.set_aspect("equal")
        ax.axis("off")

        if self.title:
            fig.suptitle(
                self.title,
                fontsize=self.label_fontsize + 5,
                fontweight="bold",
                fontfamily=self.fontfamily,
                y=0.97,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    def _compute_equal_angle(
        self,
        clade: Clade,
        min_angle: float = 0.0,
        max_angle: float = 2 * np.pi,
        x: float = 0.0,
        y: float = 0.0,
    ) -> tuple[
        list[tuple[tuple[float, float], tuple[float, float]]],
        list[tuple[float, float, float, float, str]],
    ]:
        """Computes (x, y) coordinates for an unrooted tree using the Equal-Angle algorithm.

        Args:
            clade (Clade): The current clade being processed.
            min_angle (float): The minimum angle for the current clade's arc.
            max_angle (float): The maximum angle for the current clade's arc.
            x (float): The x-coordinate of the current clade's position.
            y (float): The y-coordinate of the current clade's position.

        Returns:
            tuple: A tuple containing two lists:
                - lines: A list of line segments represented as tuples of start and end coordinates.
                - labels_info: A list of label information, each represented as a tuple containing (x, y, angle_rad, angle_deg, label).
        """
        lines = []
        labels_info = []

        def layout(
            c: Clade, a_min: float, a_max: float, cur_x: float, cur_y: float
        ) -> tuple[
            list[tuple[tuple[float, float], tuple[float, float]]],
            list[tuple[float, float, float, float, str]],
        ]:
            """Calculate the layout.

            Args:
                c (Clade): The current clade being processed.
                a_min (float): The minimum angle for the current clade's arc.
                a_max (float): The maximum angle for the current clade's arc.
                cur_x (float): The x-coordinate of the current clade's position.
                cur_y (float): The y-coordinate of the current clade's position.

            Returns:
                tuple: A tuple containing two lists:
                    - lines: A list of line segments represented as tuples of start and end coordinates.
                    - labels_info: A list of label information, each represented as a tuple containing (x, y, angle_rad, angle_deg, label).
            """
            if c.is_terminal():
                mid_angle = (a_min + a_max) / 2
                deg = np.degrees(mid_angle)
                if 90 < deg <= 270:
                    deg -= 180
                labels_info.append((cur_x, cur_y, mid_angle, deg, c.name))
                return

            span = a_max - a_min
            total_leaves = self._get_leaf_count(c)

            current_min = a_min
            for child in c.clades:
                child_leaves = self._get_leaf_count(child)
                # Allocate arc space slice proportional to child leaf populations
                child_span = span * (child_leaves / total_leaves)
                child_max = current_min + child_span

                # Dynamic branch angle vector mapping orientation split definitions
                child_angle = (current_min + child_max) / 2
                b_len = child.branch_length if child.branch_length else 0.5

                next_x = cur_x + b_len * np.cos(child_angle)
                next_y = cur_y + b_len * np.sin(child_angle)
                lines.append(((cur_x, cur_y), (next_x, next_y)))

                layout(child, current_min, child_max, next_x, next_y)
                current_min = child_max

        layout(clade, min_angle, max_angle, x, y)
        return lines, labels_info

    def _get_leaf_count(self, c: Clade) -> int:
        """Get leaf count.

        Args:
            c (Clade): The current clade.

        Returns:
            int: The number of leaves in the current clade.
        """
        if c.is_terminal():
            return 1
        return sum(self._get_leaf_count(child) for child in c.clades)

    def _normalize_branches(
        self,
        clade: Clade,
        use_proportional_branches: bool = True,
    ) -> None:
        """Recursively updates branch lengths directly on Biopython Clade objects.

        Args:
            clade (Clade): The current clade being processed.
            use_proportional_branches (bool): Whether to scale internal branches based on child leaf counts to spread dense structures out and prevent overlapping.
        """
        if clade.is_terminal():
            clade.branch_length = self.min_leaf_len
            return

        # Calculate dynamic spacing baseline based on child population density
        if use_proportional_branches:
            num_leaves = self._get_leaf_count(clade)
            # Denser parent clusters get longer internal branches to separate them
            clade.branch_length = max(0.3, num_leaves * 0.15) * self.internal_scale
        else:
            current_len = (
                clade.branch_length
                if (clade.branch_length and clade.branch_length > 0)
                else 1.0
            )
            clade.branch_length = current_len * self.internal_scale

        for child in clade.clades:
            self._normalize_branches(
                child, use_proportional_branches=use_proportional_branches
            )


class RectangularTree(BaseModel):
    """A class to draw a rectangular tree layout using Biopython's Phylo module."""

    tree_obj: Tree = Field(..., description="The tree to draw.")
    labels: Union[list[Union[int, str]], dict[int, str]] = Field(
        ..., description="The document labels."
    )
    label_colors: Optional[Union[str, dict[str, list[str]], None]] = Field(
        None,
        description="Optional color mapping: 'auto' or dict of hex colors to labels.",
    )
    title: Optional[str] = Field(
        "Bootstrap Consensus Tree Result", description="The title of the dendrogram."
    )
    figsize: Optional[tuple[float, float]] = Field(
        (10, 10),
        description="Optional figure size as (width, height) in inches.",
    )
    label_fontsize: Optional[int] = Field(
        12, description="Font size for leaf labels in the tree diagram."
    )
    fontfamily: Optional[str] = Field(
        "sans-serif", description="Font family for leaf labels."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self) -> Figure:
        """Draw traditional rectangular tree layout.

        Returns:
            Figure: The matplotlib figure containing the rectangular tree visualization.
        """
        # Get the labels as a list of strings
        if isinstance(self.labels, dict):
            labels = [str(self.labels[i]) for i in range(len(self.labels))]
        else:
            labels = [str(label) for label in self.labels]

        # Get the colours
        normalized_color = "#000000"  # Default to black

        # Resolve leaf label colors
        resolved_colors = resolve_label_colors(
            self.tree_obj, self.label_colors, normalized_color
        )

        fig, ax = plt.subplots()

        # Remove background
        ax.set_facecolor("none")

        Phylo.draw(
            self.tree_obj,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: (
                f"{clade.branch_length:.{PRECISION}f}\n"
                if clade.branch_length is not None
                else ""
            ),
        )

        # Set labels for the plot
        ax.set_xlabel("Branch Length")
        ax.set_ylabel("Documents")

        # Hide the two unused borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set the colour of the used borders and labels
        ax.spines["bottom"].set_color(normalized_color)
        ax.spines["left"].set_color(normalized_color)
        ax.tick_params(colors=normalized_color)

        # Extend the x-axis to the right to fit longer labels
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_xlim(x_left, x_right * 1.25)
        ax.set_ylim(y_low, y_high)

        # Set the graph size, title, and tight layout
        if self.figsize is not None:
            width, height = self.figsize
        else:
            width = 9.5
            height = len(self.labels) * 0.3 + 1
        fig.set_size_inches(w=width, h=height)
        ax.set_title(self.title)
        fig.tight_layout()

        # Update labels (position and color)
        for text in ax.texts:
            text.set_linespacing(spacing=0.1)
            text.set_fontsize(self.label_fontsize)

            label_text = text.get_text().strip()
            if label_text in resolved_colors:
                text.set_color(resolved_colors[label_text])
            else:
                text.set_color(normalized_color)

        return fig


class BCT(BaseModel):
    """The Bootstrap Consensus Tree Class."""

    dtm: Optional[Union[ArrayLike, DTM, pd.DataFrame]] = Field(
        None, description="The document term matrix."
    )
    metric: Optional[str] = Field("euclidean", description="The distance metric.")
    method: Optional[str] = Field("average", description="The linkage method.")
    cutoff: Optional[float] = Field(0.5, description="The cutoff value.")
    iterations: Optional[int] = Field(
        100, description="The number of iterations to run the bootstrap."
    )
    replace: Optional[str] = Field("without", description="The replacement method.")
    labels: Optional[Union[list[Union[int, str]], dict[int, str]]] = Field(
        None, description="The document labels."
    )
    title: Optional[str] = Field(
        "Bootstrap Consensus Tree Result", description="The title of the dendrogram."
    )
    layout: Optional[Literal["rectangular", "fan"]] = Field(
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
    fontfamily: Optional[str] = Field(
        "sans-serif", description="Font family for leaf labels."
    )
    label_colors: Optional[Union[str, dict[str, list[str]], None]] = Field(
        None,
        description="Optional color mapping: 'auto' or dict of hex colors to labels.",
    )
    linewidth: Optional[float] = Field(1.2, description="The width of the tree lines.")
    min_leaf_len: Optional[float] = Field(
        1.5, description="The minimum length of leaf branches in fan layout."
    )
    internal_scale: Optional[float] = Field(
        0.6, description="The scaling factor for internal branches in fan layout."
    )
    tree: Optional[Tree] = Field(None, description="The consensus tree object.")
    fig: Optional[Figure] = Field(None, description="The figure for the dendrogram.")
    _last_tree_spec: Optional[tuple[Any, ...]] = None
    _last_render_spec: Optional[tuple[Any, ...]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _doc_term_matrix(self) -> pd.DataFrame:
        """Return a dataframe of the document term matrix.

        Returns:
            pd.DataFrame: The document term matrix with doc labels as the index and terms as the columns.
        """
        if self.dtm is None:
            raise LexosException("No document term matrix found.")

        # DTM input
        if isinstance(self.dtm, DTM):
            df = self.dtm.to_df().T
        # DataFrame input
        elif isinstance(self.dtm, pd.DataFrame):
            df = self.dtm
        # Raw array/list input
        else:
            try:
                matrix = np.asarray(self.dtm)
            except (ValueError, TypeError) as e:
                raise LexosException(
                    f"Unsupported document-term matrix type: {e}"
                ) from e

            if matrix.ndim != 2:
                raise LexosException(
                    "The document-term matrix must be two-dimensional."
                )

            df = pd.DataFrame(matrix)

        # Check for numeric values
        if not np.issubdtype(df.values.dtype, np.number):
            raise LexosException(
                "The document-term matrix must contain only numeric values."
            )

        # Apply labels to the index if they exist
        if self.labels:
            if isinstance(self.labels, dict):
                if len(self.labels) != df.shape[0]:
                    raise LexosException(
                        "The number of labels must match the number of documents."
                    )
                df.index = [
                    str(self.labels.get(i, f"Doc{i + 1}")) for i in range(df.shape[0])
                ]
            else:
                if len(self.labels) != df.shape[0]:
                    raise LexosException(
                        "The number of labels must match the number of documents."
                    )
                df.index = [str(label) for label in self.labels]

        return df

    @property
    def _document_label_map(self) -> dict[int, str]:
        """Return a dictionary of document label map.

        Returns:
            dict[int, str]: A document label map.
        """
        if self.labels is not None:
            if isinstance(self.labels, dict):
                return {int(i): str(label) for i, label in self.labels.items()}
            else:
                return {i: str(label) for i, label in enumerate(self.labels)}
        return {}

    def __init__(self, **data) -> None:
        """Construct the BCT instance and render the bootstrap consensus tree."""
        super().__init__(**data)

        # Ensure there is a document-term matrix
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Ensure there are labels
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.index.values.tolist()
            else:
                # Need to convert to array to get length safely
                try:
                    matrix = np.asarray(self.dtm)
                    self.labels = [f"Doc{i + 1}" for i in range(matrix.shape[0])]
                except (ValueError, TypeError):
                    self.labels = []

        # Get the dtm as a dataframe (this also validates shape and types)
        df = self._doc_term_matrix

        # Ensure there is more than one document
        if df.shape[0] < 2:
            raise LexosException(
                "The document-term matrix must contain at least two documents."
            )

        # Build initial tree + figure and snapshot state.
        self._sync_state(force_rebuild=True, force_redraw=True)
        plt.close()

    @staticmethod
    def _serialize_label_colors(
        value: Optional[str | dict[str, list[str]]],
    ) -> tuple[Any, ...]:
        """Create a stable, comparable representation for label color config."""
        if isinstance(value, dict):
            items = []
            for color, labels in sorted(value.items(), key=lambda pair: pair[0]):
                items.append((color, tuple(labels)))
            return ("dict", tuple(items))
        return ("scalar", value)

    @staticmethod
    def _serialize_labels(
        labels: Optional[Union[list[Union[int, str]], dict[int, str]]],
    ) -> tuple[Any, ...]:
        """Create a stable, comparable representation for labels."""
        if labels is None:
            return ("none",)
        if isinstance(labels, dict):
            return (
                "dict",
                tuple(sorted((int(k), str(v)) for k, v in labels.items())),
            )
        return ("list", tuple(str(v) for v in labels))

    def _dtm_signature(self) -> tuple[Any, ...]:
        """Create a lightweight signature for dtm identity tracking."""
        if self.dtm is None:
            return ("none",)
        if isinstance(self.dtm, pd.DataFrame):
            return ("dataframe", id(self.dtm))
        if isinstance(self.dtm, DTM):
            return ("dtm", id(self.dtm))
        return ("arraylike", id(self.dtm))

    def _tree_spec(self) -> tuple[Any, ...]:
        """Spec that determines whether the consensus tree must be rebuilt."""
        return (
            self._dtm_signature(),
            self.metric,
            self.method,
            self.cutoff,
            self.iterations,
            self.replace,
            self.random_seed,
        )

    def _render_spec(self) -> tuple[Any, ...]:
        """Spec that determines whether the figure needs redraw."""
        return (
            self.layout,
            self.title,
            self.figsize,
            self.label_fontsize,
            self.fontfamily,
            self._serialize_label_colors(self.label_colors),
            self.linewidth,
            self.min_leaf_len,
            self.internal_scale,
            self._serialize_labels(self.labels),
        )

    def _sync_state(
        self,
        *,
        layout: Optional[str] = None,
        force_rebuild: bool = False,
        force_redraw: bool = False,
    ) -> bool:
        """Sync tree/figure with current settings.

        Returns:
            bool: True if the figure was regenerated.
        """
        if layout:
            self.layout = layout

        current_tree_spec = self._tree_spec()
        current_render_spec = self._render_spec()

        tree_dirty = self.dtm is not None and (
            force_rebuild
            or self.tree is None
            or self._last_tree_spec != current_tree_spec
        )
        can_render = self.tree is not None or self.dtm is not None
        render_dirty = can_render and (
            force_redraw
            or self.fig is None
            or self._last_render_spec != current_render_spec
        )

        if tree_dirty:
            self.tree = self._get_bootstrap_consensus_tree()
            self._last_tree_spec = current_tree_spec
            render_dirty = True

        regenerated = False
        if render_dirty:
            self.fig = self._get_bootstrap_consensus_tree_fig(layout=self.layout)
            self._last_render_spec = current_render_spec
            regenerated = True

        return regenerated

    @field_validator("label_colors", mode="after")
    @classmethod
    def _validate_label_colors(cls, value) -> str:
        """Validate that the label colors are valid color strings.

        Args:
            value (str): The label color string.

        Returns:
            str: The validated label color string.

        Raises:
            LexosException: If the label color is not a valid color string.
        """
        invalid = set()
        if isinstance(value, dict):
            for color, _ in value.items():
                if not is_valid_colour(color):
                    invalid.add(color)
        elif isinstance(value, str):
            if value != "auto" and not is_valid_colour(value):
                invalid.add(value)
        elif isinstance(value, type(None)):
            pass
        else:
            raise LexosException(
                'label_colors must be a "auto", None, or a dictionary of hex colors to label lists.'
            )
        if invalid:
            raise LexosException(f"Invalid label colors: {', '.join(invalid)}")
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

    def _get_newick_tree(self, labels: list[str], sample_dtm: pd.DataFrame) -> Tree:
        """Get Newick tree based on a subset of the DTM.

        Args:
            labels (list[str]): All file names from the DTM
            sample_dtm (pd.DataFrame): An 80% subset of the complete DTM

        Returns:
            Tree: A Biopython Tree object representing the DTM subset
        """
        # Get the linkage matrix for the sample doc term matrix
        linkage_matrix = linkage(
            sample_dtm.values, metric=self.metric, method=self.method
        )

        # Get the Newick representation of the tree
        newick = self.linkage_to_newick(matrix=linkage_matrix, labels=labels)

        # Convert linkage matrix to a tree node and return it
        return Phylo.read(StringIO(newick), format="newick")

    def _get_bootstrap_trees(self) -> list[Tree]:
        """Do bootstrap on the DTM to get a list of Tree objects.

        Returns:
            list[Tree]: A list of Biopython Tree objects where each tree was based on an 80% subset of the complete DTM.
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

    def _get_bootstrap_consensus_tree(self) -> Tree:
        """Get the consensus tree.

        Returns:
            Tree: The consensus tree of the list of bootstrap trees.
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
        # Get the consensus tree
        if not self.tree:
            self.tree = self._get_bootstrap_consensus_tree()

        if layout == "rectangular":
            return RectangularTree(
                tree_obj=self.tree,
                labels=self._document_label_map,
                label_colors=self.label_colors,
                figsize=self.figsize,
                title=self.title,
                label_fontsize=self.label_fontsize,
                fontfamily=self.fontfamily,
            ).draw()
        elif layout == "fan":
            return FanTree(
                tree_obj=self.tree,
                figsize=self.figsize,
                title=self.title,
                label_fontsize=self.label_fontsize,
                fontfamily=self.fontfamily,
                label_colors=self.label_colors,
                linewidth=self.linewidth,
                min_leaf_len=self.min_leaf_len,
                internal_scale=self.internal_scale,
            ).draw()
        else:
            raise ValueError(f"Unknown layout: {layout}. Use 'rectangular' or 'fan'.")

    @validate_call(config=model_config)
    def save(self, path: Path | str | None) -> None:
        """Save the bootstrap consensus tree result to a file.

        Args:
            path (Path | str | None): The path to save the file.
        """
        if not path or path == "":
            raise LexosException("You must provide a valid path.")
        self._sync_state()
        self.fig.savefig(path)

    def show(self, layout: str = None) -> Figure:
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure using `BCT.fig`.
        This will generally display in a Jupyter notebook.

        Args:
            layout: Optional layout to switch to. Options: "rectangular" or "fan".

        Returns:
            Figure: The matplotlib figure containing the tree visualization.
        """
        regenerated = self._sync_state(layout=layout)
        if regenerated:
            plt.close()

        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        return self.fig
