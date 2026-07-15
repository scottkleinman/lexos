"""test_bootstrap_consensus.py.

Coverage: 100%

Last Updated: 2026-07-15
"""

from io import StringIO
from unittest.mock import Mock, PropertyMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from matplotlib.figure import Figure

from lexos.cluster.bootstrap_consensus import (
    BCT,
    FanTree,
    RectangularTree,
    assign_auto_colors,
    generate_colors,
    resolve_label_colors,
)
from lexos.dtm import DTM
from lexos.exceptions import LexosException


def test_generate_colors_uses_fixed_palette_for_small_n():
    """Small n should return the expected first colors from the fixed palette."""
    colors = generate_colors(3)
    assert colors == ["#1f77b4", "#ff7f0e", "#2ca02c"]


def test_generate_colors_returns_hex_values_for_large_n():
    """Large n should still return n distinct hex colors."""
    colors = generate_colors(12)
    assert len(colors) == 12
    assert len(set(colors)) == 12
    assert all(color.startswith("#") and len(color) == 7 for color in colors)


def test_resolve_label_colors_applies_custom_mapping_to_known_labels_only():
    """Only labels present in the tree should be recolored by explicit mappings."""
    tree = Phylo.read(StringIO("((A:1,B:1):1,C:1);"), "newick")

    resolved = resolve_label_colors(
        tree=tree,
        label_colors={"#ff0000": ["A", "MISSING"], "#0000ff": ["C"]},
        base_color="#111111",
    )

    assert resolved == {"A": "#ff0000", "B": "#111111", "C": "#0000ff"}


def test_resolve_label_colors_auto_uses_generated_assignments():
    """Auto mode should apply assignments returned by assign_auto_colors."""
    tree = Phylo.read(StringIO("((A:1,B:1):1,C:1);"), "newick")

    with patch("lexos.cluster.bootstrap_consensus.assign_auto_colors") as mock_auto:
        mock_auto.return_value = {"A": "#aaaaaa", "B": "#bbbbbb"}
        resolved = resolve_label_colors(
            tree=tree,
            label_colors="auto",
            base_color="#000000",
        )

    assert resolved == {"A": "#aaaaaa", "B": "#bbbbbb", "C": "#000000"}


def test_assign_auto_colors_groups_by_depth_one_clades():
    """All descendants of each root child should share a color."""
    tree = Phylo.read(StringIO("(((A:1,B:1):1,C:1):1,(D:1,E:1):1);"), "newick")

    colors = assign_auto_colors(tree)

    assert set(colors) == {"A", "B", "C", "D", "E"}
    assert colors["A"] == colors["B"] == colors["C"]
    assert colors["D"] == colors["E"]
    assert colors["A"] != colors["D"]


def test_assign_auto_colors_skips_terminals_without_names():
    """Unnamed terminals should be ignored by color assignment."""
    left = Clade(clades=[Clade(name="A"), Clade(name=None)])
    right = Clade(clades=[Clade(name="B"), Clade(name="C")])
    tree = Tree(root=Clade(clades=[left, right]))

    colors = assign_auto_colors(tree)

    assert set(colors) == {"A", "B", "C"}
    assert all(color.startswith("#") and len(color) == 7 for color in colors.values())


def test_bct_accepts_numpy_matrix_and_generates_default_labels():
    """Array input should be accepted and default labels should be generated."""
    dtm = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    with patch.object(BCT, "_get_bootstrap_consensus_tree_fig", return_value=object()):
        bct = BCT(dtm=dtm)

    assert bct.labels == ["Doc1", "Doc2"]
    assert bct._doc_term_matrix.index.tolist() == ["Doc1", "Doc2"]


def test_bct_rejects_non_numeric_document_term_matrix():
    """Non-numeric matrix values should raise a clear LexosException."""
    dtm = pd.DataFrame([["x", "y"], ["a", "b"]], index=["d1", "d2"])

    with patch.object(BCT, "_get_bootstrap_consensus_tree_fig", return_value=object()):
        with pytest.raises(
            LexosException,
            match="must contain only numeric values",
        ):
            BCT(dtm=dtm)


def test_fantree_draw_builds_figure_with_labels_and_title():
    """Draw should render branches/labels and return a matplotlib Figure."""
    tree = Phylo.read(StringIO("((A:1,B:1):1,(C:1,D:1):1);"), "newick")
    fan = FanTree(
        tree_obj=tree,
        title="Coverage Title",
        label_colors={"#ff0000": ["A"], "#00aa00": ["C"]},
        label_fontsize=10,
    )

    fig = fan.draw()

    assert isinstance(fig, Figure)
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Coverage Title"
    assert len(fig.axes) == 1
    assert len(fig.axes[0].texts) == 4


def test_fantree_compute_equal_angle_flips_terminal_label_orientation():
    """Terminal labels in the left hemisphere should rotate by 180 degrees."""
    base_tree = Tree(root=Clade(clades=[Clade(name="A"), Clade(name="B")]))
    fan = FanTree(tree_obj=base_tree)

    lines, labels_info = fan._compute_equal_angle(
        Clade(name="T"), min_angle=np.pi / 2, max_angle=3 * np.pi / 2, x=1.0, y=2.0
    )

    assert lines == []
    assert len(labels_info) == 1
    x, y, angle_rad, angle_deg, label = labels_info[0]
    assert (x, y, label) == (1.0, 2.0, "T")
    assert np.isclose(angle_rad, np.pi)
    assert np.isclose(angle_deg, 0.0)


def test_fantree_compute_equal_angle_internal_clade_uses_default_branch_length():
    """Internal layout should use 0.5 when a child branch length is missing."""
    root = Clade(
        clades=[Clade(name="A", branch_length=None), Clade(name="B", branch_length=1.0)]
    )
    tree = Tree(root=root)
    fan = FanTree(tree_obj=tree)

    lines, labels_info = fan._compute_equal_angle(root)

    assert len(lines) == 2
    assert len(labels_info) == 2
    first_segment = lines[0]
    (x1, y1), (x2, y2) = first_segment
    assert np.isclose(np.hypot(x2 - x1, y2 - y1), 0.5)


def test_fantree_get_leaf_count_terminal_and_recursive():
    """Leaf counting should handle both terminal and nested clades."""
    root = Clade(
        clades=[Clade(name="A"), Clade(clades=[Clade(name="B"), Clade(name="C")])]
    )
    tree = Tree(root=root)
    fan = FanTree(tree_obj=tree)

    assert fan._get_leaf_count(Clade(name="leaf")) == 1
    assert fan._get_leaf_count(root) == 3


def test_fantree_normalize_branches_covers_proportional_and_fixed_modes():
    """Normalization should update internal and terminal branch lengths in both modes."""
    root_prop = Clade(
        clades=[Clade(name="A"), Clade(clades=[Clade(name="B"), Clade(name="C")])]
    )
    fan_prop = FanTree(
        tree_obj=Tree(root=root_prop), min_leaf_len=1.7, internal_scale=0.6
    )

    fan_prop._normalize_branches(root_prop, use_proportional_branches=True)

    expected_root = max(0.3, 3 * 0.15) * 0.6
    assert np.isclose(root_prop.branch_length, expected_root)
    assert np.isclose(root_prop.clades[0].branch_length, 1.7)
    assert np.isclose(root_prop.clades[1].clades[0].branch_length, 1.7)

    root_fixed = Clade(branch_length=None, clades=[Clade(name="X"), Clade(name="Y")])
    fan_fixed = FanTree(
        tree_obj=Tree(root=root_fixed), min_leaf_len=1.2, internal_scale=0.4
    )

    fan_fixed._normalize_branches(root_fixed, use_proportional_branches=False)

    assert np.isclose(root_fixed.branch_length, 0.4)
    assert np.isclose(root_fixed.clades[0].branch_length, 1.2)
    assert np.isclose(root_fixed.clades[1].branch_length, 1.2)


def test_rectangulartree_draw_supports_dict_labels_and_explicit_figsize():
    """Rectangular draw should handle dict labels and return a configured figure."""
    tree = Phylo.read(StringIO("(A:1,B:2);"), "newick")
    rect = RectangularTree(
        tree_obj=tree,
        labels={0: "A", 1: "B"},
        label_colors={"#ff0000": ["A"]},
        figsize=(6.0, 4.0),
        title="Rectangular Dict Labels",
    )

    fig = rect.draw()

    assert isinstance(fig, Figure)
    assert np.allclose(fig.get_size_inches(), [6.0, 4.0])
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Branch Length"
    assert ax.get_ylabel() == "Documents"
    assert ax.get_title() == "Rectangular Dict Labels"
    plt.close(fig)


def test_rectangulartree_draw_uses_default_size_when_figsize_is_none():
    """When figsize is None, width/height should be computed from label count."""
    tree = Phylo.read(StringIO("(A:1,B:2);"), "newick")
    labels = ["A", "B"]
    rect = RectangularTree(tree_obj=tree, labels=labels, figsize=None)

    fig = rect.draw()

    width, height = fig.get_size_inches()
    assert np.isclose(width, 9.5)
    assert np.isclose(height, len(labels) * 0.3 + 1)
    plt.close(fig)


def test_doc_term_matrix_raises_when_dtm_missing():
    """Accessing _doc_term_matrix should fail when dtm is missing."""
    bct = BCT.model_construct(dtm=None, labels=None)

    with pytest.raises(LexosException, match="No document term matrix found"):
        _ = bct._doc_term_matrix


def test_doc_term_matrix_handles_dtm_instance_path_via_to_df_transpose():
    """DTM instance path should use to_df().T before returning the frame."""

    class FakeDTM:
        def __init__(self, frame: pd.DataFrame):
            self._frame = frame

        def to_df(self) -> pd.DataFrame:
            return self._frame

    source = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=["term1", "term2"], columns=["Doc1", "Doc2"]
    )
    fake_dtm = FakeDTM(source)

    with patch("lexos.cluster.bootstrap_consensus.DTM", FakeDTM):
        bct = BCT.model_construct(dtm=fake_dtm, labels=None)
        matrix = bct._doc_term_matrix

    assert matrix.equals(source.T)


def test_doc_term_matrix_raises_for_unconvertible_array_input():
    """Unsupported array-like values should raise a LexosException."""

    class BadArray:
        def __array__(self, dtype=None):
            raise ValueError("cannot convert")

    bct = BCT.model_construct(dtm=BadArray(), labels=None)

    with pytest.raises(LexosException, match="Unsupported document-term matrix type"):
        _ = bct._doc_term_matrix


def test_doc_term_matrix_raises_for_non_2d_array_input():
    """One-dimensional raw array input should be rejected."""
    bct = BCT.model_construct(dtm=[1, 2, 3], labels=None)

    with pytest.raises(LexosException, match="must be two-dimensional"):
        _ = bct._doc_term_matrix


def test_doc_term_matrix_raises_on_dict_label_count_mismatch():
    """Dict label count must match number of documents."""
    dtm = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    bct = BCT.model_construct(dtm=dtm, labels={0: "OnlyOne"})

    with pytest.raises(LexosException, match="number of labels must match"):
        _ = bct._doc_term_matrix


def test_doc_term_matrix_applies_dict_labels_to_index():
    """Dict labels should be applied to dataframe index in document order."""
    dtm = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    bct = BCT.model_construct(dtm=dtm, labels={0: "Alpha", 1: "Beta"})

    matrix = bct._doc_term_matrix

    assert matrix.index.tolist() == ["Alpha", "Beta"]


def test_doc_term_matrix_raises_on_list_label_count_mismatch():
    """List label count must match number of documents."""
    dtm = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    bct = BCT.model_construct(dtm=dtm, labels=["OnlyOne"])

    with pytest.raises(LexosException, match="number of labels must match"):
        _ = bct._doc_term_matrix


def test_document_label_map_handles_dict_list_and_none():
    """Document label map should normalize dict/list labels and support None."""
    from_dict = BCT.model_construct(dtm=None, labels={0: "A", 1: 200})
    from_list = BCT.model_construct(dtm=None, labels=["X", 99])
    from_none = BCT.model_construct(dtm=None, labels=None)

    assert from_dict._document_label_map == {0: "A", 1: "200"}
    assert from_list._document_label_map == {0: "X", 1: "99"}
    assert from_none._document_label_map == {}


def test_bct_init_raises_when_dtm_is_missing():
    """BCT init should reject missing document-term matrices."""
    with pytest.raises(LexosException, match="must provide a document-term matrix"):
        BCT()


def test_bct_init_populates_labels_from_dtm_instance():
    """When labels are absent, init should adopt labels from a DTM-like object."""
    docs_df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=["term1", "term2"], columns=["DocA", "DocB"]
    )
    dtm_obj = DTM.model_construct(labels=["DocA", "DocB"])

    with patch.object(DTM, "to_df", return_value=docs_df):
        with patch.object(
            BCT, "_get_bootstrap_consensus_tree_fig", return_value=object()
        ):
            bct = BCT(dtm=dtm_obj, labels=None)

    assert bct.labels == ["DocA", "DocB"]


def test_bct_init_sets_empty_labels_when_array_conversion_fails():
    """Init should set labels to [] when np.asarray conversion fails."""

    class BadArray:
        def __array__(self, dtype=None):
            raise ValueError("cannot convert")

    mock_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    with patch.object(
        BCT, "_doc_term_matrix", new_callable=PropertyMock
    ) as mock_matrix:
        mock_matrix.return_value = mock_df
        with patch.object(
            BCT, "_get_bootstrap_consensus_tree_fig", return_value=object()
        ):
            bct = BCT(dtm=BadArray(), labels=None)

    assert bct.labels == []


def test_bct_init_raises_for_single_document_matrix():
    """BCT init should require at least two documents."""
    dtm = pd.DataFrame([[1.0, 2.0, 3.0]])

    with pytest.raises(LexosException, match="at least two documents"):
        BCT(dtm=dtm)


def test_validate_label_colors_accepts_valid_values():
    """Validator should accept valid dict, auto string, and None values."""
    assert BCT._validate_label_colors({"#ff0000": ["A"]}) == {"#ff0000": ["A"]}
    assert BCT._validate_label_colors("auto") == "auto"
    assert BCT._validate_label_colors(None) is None


def test_validate_label_colors_rejects_invalid_dict_and_string_values():
    """Validator should reject invalid color literals in dict and string forms."""
    with pytest.raises(LexosException, match="Invalid label colors"):
        BCT._validate_label_colors({"not-a-color": ["A"]})

    with pytest.raises(LexosException, match="Invalid label colors"):
        BCT._validate_label_colors("not-a-color")


def test_validate_label_colors_rejects_unsupported_types():
    """Validator should reject non-dict, non-string, non-None values."""
    with pytest.raises(LexosException, match='label_colors must be a "auto", None'):
        BCT._validate_label_colors(123)


def test_validate_figsize_accepts_none_and_valid_tuple():
    """Figsize validator should accept None and valid positive tuples."""
    assert BCT._validate_figsize(None) is None
    assert BCT._validate_figsize((8.0, 6.0)) == (8.0, 6.0)


def test_validate_figsize_rejects_bad_shapes_and_nonpositive_values():
    """Figsize validator should reject wrong tuple length and nonpositive entries."""
    with pytest.raises(LexosException, match="exactly two numeric values"):
        BCT._validate_figsize((10.0,))

    with pytest.raises(LexosException, match="must be greater than 0"):
        BCT._validate_figsize((10.0, 0.0))


def test_validate_label_fontsize_paths():
    """Font size validator should default None, reject nonpositive, and accept positives."""
    assert BCT._validate_label_fontsize(None) == 12
    assert BCT._validate_label_fontsize(14) == 14

    with pytest.raises(LexosException, match="positive integer"):
        BCT._validate_label_fontsize(0)


def test_linkage_to_newick_builds_recursive_tree_string():
    """Linkage conversion should produce a valid recursive Newick tree."""
    linkage_matrix = np.array([[0, 1, 0.4, 2], [2, 3, 0.8, 3]])
    labels = ["A", "B", "C"]

    newick = BCT.linkage_to_newick(linkage_matrix, labels)

    assert isinstance(newick, str)
    assert newick.endswith(";")
    assert "(" in newick and ")" in newick
    assert "A:" in newick and "B:" in newick and "C:" in newick


def test_get_newick_tree_uses_linkage_and_reads_newick():
    """_get_newick_tree should call linkage, convert to Newick, and parse with Phylo."""
    bct = BCT.model_construct(metric="cityblock", method="average")
    sample_dtm = pd.DataFrame([[1.0, 2.0], [2.0, 3.0]])
    expected_tree = object()

    with patch("lexos.cluster.bootstrap_consensus.linkage") as mock_linkage:
        mock_linkage.return_value = np.array([[0, 1, 1.0, 2]])
        with patch.object(
            BCT, "linkage_to_newick", return_value="(A:0.1,B:0.1);"
        ) as mock_to_newick:
            with patch(
                "lexos.cluster.bootstrap_consensus.Phylo.read",
                return_value=expected_tree,
            ) as mock_read:
                result = bct._get_newick_tree(labels=["A", "B"], sample_dtm=sample_dtm)

    assert result is expected_tree
    mock_linkage.assert_called_once()
    args, kwargs = mock_linkage.call_args
    assert np.array_equal(args[0], sample_dtm.values)
    assert kwargs == {"metric": "cityblock", "method": "average"}
    mock_to_newick.assert_called_once()
    mock_read.assert_called_once()


def test_get_bootstrap_trees_uses_cached_matrix_labels_and_seeded_state():
    """_get_bootstrap_trees should read matrix/labels once and build one tree per iteration."""
    dtm = pd.DataFrame(
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
        index=["Doc1", "Doc2", "Doc3"],
    )
    bct = BCT.model_construct(iterations=3, replace="without", random_seed=42)

    with patch.object(
        BCT, "_doc_term_matrix", new_callable=PropertyMock
    ) as mock_matrix:
        mock_matrix.return_value = dtm
        with patch.object(
            BCT, "_get_newick_tree", side_effect=["t1", "t2", "t3"]
        ) as mock_newick:
            trees = bct._get_bootstrap_trees()

    assert trees == ["t1", "t2", "t3"]
    assert mock_newick.call_count == 3
    for call in mock_newick.call_args_list:
        kwargs = call.kwargs
        assert kwargs["labels"] == ["Doc1", "Doc2", "Doc3"]
        assert isinstance(kwargs["sample_dtm"], pd.DataFrame)
        assert kwargs["sample_dtm"].shape[1] == 4


def test_get_bootstrap_consensus_tree_delegates_to_majority_consensus():
    """Consensus tree helper should pass bootstrap trees and cutoff through."""
    bct = BCT.model_construct(cutoff=0.7)
    tree_list = [object(), object()]
    consensus_tree = object()

    with patch.object(BCT, "_get_bootstrap_trees", return_value=tree_list):
        with patch(
            "lexos.cluster.bootstrap_consensus.majority_consensus",
            return_value=consensus_tree,
        ) as mock_consensus:
            result = bct._get_bootstrap_consensus_tree()

    assert result is consensus_tree
    mock_consensus.assert_called_once_with(trees=tree_list, cutoff=0.7)


def test_get_bootstrap_consensus_tree_fig_rectangular_builds_tree_and_draws():
    """Rectangular layout should lazily build tree and call RectangularTree.draw()."""
    built_tree = object()
    figure = object()
    bct = BCT.model_construct(
        tree=None,
        label_colors={"#ff0000": ["A"]},
        figsize=(9.0, 6.0),
        title="Rect",
        label_fontsize=11,
        fontfamily="serif",
        labels=["A", "B"],
    )

    with patch.object(BCT, "_get_bootstrap_consensus_tree", return_value=built_tree):
        with patch.object(
            BCT,
            "_document_label_map",
            new_callable=PropertyMock,
            return_value={0: "A", 1: "B"},
        ):
            with patch(
                "lexos.cluster.bootstrap_consensus.RectangularTree"
            ) as mock_rect:
                mock_rect.return_value.draw.return_value = figure
                result = bct._get_bootstrap_consensus_tree_fig(layout="rectangular")

    assert result is figure
    assert bct.tree is built_tree
    mock_rect.assert_called_once()
    assert mock_rect.call_args.kwargs["tree_obj"] is built_tree


def test_get_bootstrap_consensus_tree_fig_fan_and_unknown_layout():
    """Fan layout should draw with FanTree and unknown layout should raise."""
    existing_tree = object()
    fan_figure = object()
    bct = BCT.model_construct(
        tree=existing_tree,
        figsize=(8.0, 8.0),
        title="Fan",
        label_fontsize=12,
        fontfamily="sans-serif",
        label_colors=None,
        linewidth=1.5,
        min_leaf_len=1.2,
        internal_scale=0.5,
    )

    with patch("lexos.cluster.bootstrap_consensus.FanTree") as mock_fan:
        mock_fan.return_value.draw.return_value = fan_figure
        result = bct._get_bootstrap_consensus_tree_fig(layout="fan")

    assert result is fan_figure

    with pytest.raises(ValueError, match="Unknown layout"):
        bct._get_bootstrap_consensus_tree_fig(layout="diagonal")


def test_save_requires_nonempty_path_and_calls_figure_savefig():
    """Save should reject empty paths and delegate valid paths to fig.savefig."""
    mock_fig = type("FigStub", (), {"savefig": lambda self, p: None})()
    bct = BCT.model_construct(fig=mock_fig)

    with pytest.raises(LexosException, match="provide a valid path"):
        bct.save("")

    with patch.object(BCT, "_sync_state", return_value=False):
        with patch.object(mock_fig, "savefig") as mock_savefig:
            bct.save("/tmp/test-bootstrap.png")
    mock_savefig.assert_called_once_with("/tmp/test-bootstrap.png")


def test_show_switches_layout_and_returns_regenerated_figure():
    """show() should regenerate figure when layout changes and then return it."""
    dtm = np.array([[1.0, 2.0], [2.0, 3.0]])
    old_tree = object()
    old_fig = object()
    new_fig = object()
    with patch.object(BCT, "_get_bootstrap_consensus_tree", return_value=old_tree):
        with patch.object(
            BCT,
            "_get_bootstrap_consensus_tree_fig",
            side_effect=[old_fig, new_fig],
        ) as mock_get_fig:
            bct = BCT(dtm=dtm, layout="rectangular")

            with patch("lexos.cluster.bootstrap_consensus.plt.close") as mock_close:
                result = bct.show(layout="fan")

    assert bct.layout == "fan"
    assert bct.fig is new_fig
    assert result is new_fig
    assert mock_get_fig.call_count == 2
    assert mock_get_fig.call_args_list[1].kwargs == {"layout": "fan"}
    mock_close.assert_called_once()


def test_show_raises_when_figure_is_missing():
    """show() should raise when no figure is available."""
    bct = BCT.model_construct(layout="rectangular", fig=None)

    with pytest.raises(LexosException, match="before showing the figure"):
        bct.show()


def test_show_redraws_for_visual_setting_change_without_rebuilding_tree():
    """Changing render settings should redraw but not rebuild consensus tree."""
    tree_obj = object()
    fig_obj = object()
    dtm = np.array([[1.0, 2.0], [2.0, 3.0]])

    with patch.object(
        BCT, "_get_bootstrap_consensus_tree", return_value=tree_obj
    ) as mock_tree:
        with patch.object(
            BCT, "_get_bootstrap_consensus_tree_fig", return_value=fig_obj
        ) as mock_fig:
            bct = BCT(dtm=dtm)
            assert mock_tree.call_count == 1
            assert mock_fig.call_count == 1

            bct.title = "Updated Title"
            bct.show()

    assert mock_tree.call_count == 1
    assert mock_fig.call_count == 2


def test_show_rebuilds_tree_when_random_seed_changes():
    """Changing random_seed should invalidate tree_spec and rebuild the tree."""
    dtm = np.array([[1.0, 2.0], [2.0, 3.0]])

    with patch.object(
        BCT, "_get_bootstrap_consensus_tree", side_effect=[object(), object()]
    ) as mock_tree:
        with patch.object(
            BCT, "_get_bootstrap_consensus_tree_fig", return_value=object()
        ) as mock_fig:
            bct = BCT(dtm=dtm, random_seed=1)
            assert mock_tree.call_count == 1
            assert mock_fig.call_count == 1

            bct.random_seed = 99
            bct.show()

    assert mock_tree.call_count == 2
    assert mock_fig.call_count == 2


def test_save_applies_pending_updates_before_saving():
    """save() should synchronize pending render changes before writing output."""
    dtm = np.array([[1.0, 2.0], [2.0, 3.0]])
    saveable_fig = Mock()

    with patch.object(BCT, "_get_bootstrap_consensus_tree", return_value=object()):
        with patch.object(
            BCT,
            "_get_bootstrap_consensus_tree_fig",
            side_effect=[saveable_fig, saveable_fig],
        ) as mock_fig:
            bct = BCT(dtm=dtm)
            assert mock_fig.call_count == 1

            bct.label_fontsize = 18
            bct.save("/tmp/bct-sync.png")

    assert mock_fig.call_count == 2
    saveable_fig.savefig.assert_called_once_with("/tmp/bct-sync.png")


def test_serialize_label_colors_dict_normalizes_order_and_label_sequences():
    """Dict color mappings should be sorted and label lists converted to tuples."""
    serialized = BCT._serialize_label_colors(
        {
            "#00FF00": ["B", "A"],
            "#0000FF": ["C"],
        }
    )

    assert serialized == (
        "dict",
        (("#0000FF", ("C",)), ("#00FF00", ("B", "A"))),
    )


def test_serialize_labels_dict_normalizes_key_and_value_types():
    """Dict label mappings should be normalized to int keys and string values."""
    serialized = BCT._serialize_labels({"2": "Gamma", 1: 99})

    assert serialized == ("dict", ((1, "99"), (2, "Gamma")))


def test_dtm_signature_returns_dataframe_identity_marker():
    """DataFrame dtm should use the dataframe identity marker branch."""
    dtm = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    bct = BCT.model_construct(dtm=dtm)

    assert bct._dtm_signature() == ("dataframe", id(dtm))
