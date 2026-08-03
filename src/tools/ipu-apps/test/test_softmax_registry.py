"""Tests for the softmax app router (``ipu_apps.softmax.lookup``).

The router's whole value is that its answers match reality, so the tests are
mostly consistency checks against the apps themselves rather than assertions
about hardcoded strings:

* every ``supported`` verdict must actually construct its app;
* there are no open coverage gaps left;
* the torch-style front end agrees with the axis/rows/n front end.
"""

from __future__ import annotations

import importlib

import pytest

from ipu_apps.softmax import GAPS, catalog, lookup, lookup_torch

_CTOR_KWARGS = dict(inst_path="x", input_path="y", output_path=None)


def _construct(verdict):
    mod_name, cls_name = verdict.app_class.rsplit(".", 1)
    app_cls = getattr(importlib.import_module(mod_name), cls_name)
    return app_cls(**_CTOR_KWARGS, **verdict.kwargs)


@pytest.mark.parametrize("n", [1, 8, 16, 17, 32, 33, 64, 65, 100, 127, 128, 129, 200, 300, 511])
@pytest.mark.parametrize("rows", [1, 4, 128, 129, 1000])
def test_supported_rows_verdicts_construct(n, rows):
    """A 'supported' answer must never fail at construction time."""
    verdict = lookup(axis="rows", n=n, rows=rows)
    if verdict.supported:
        _construct(verdict)


@pytest.mark.parametrize("width", [1, 16, 32, 64, 65, 100, 127, 128, 129, 256, 384])
@pytest.mark.parametrize("rows", [1, 4, 128])
def test_supported_column_verdicts_construct(width, rows):
    verdict = lookup(axis="columns", width=width, rows=rows)
    if verdict.supported:
        _construct(verdict)


@pytest.mark.parametrize("n,expected", [
    (1, "softmax_rows_partial"),
    (64, "softmax_rows_partial"),
    (127, "softmax_rows_partial"),
    (128, "softmax_rows"),
    (200, "softmax_rows_long"),
    (300, "softmax_rows_long"),
])
def test_rows_axis_routing(n, expected):
    assert lookup(axis="rows", n=n, rows=8).app_name == expected


@pytest.mark.parametrize("width,expected", [
    (1, "softmax_columns_packed"),
    (64, "softmax_columns_packed"),
    (128, "softmax_columns"),
    (384, "softmax_columns"),
])
def test_columns_axis_routing(width, expected):
    assert lookup(axis="columns", width=width, rows=8).app_name == expected


def test_no_verdict_warns_about_packed_output():
    """Every app writes its output in the same row-major layout as its input,
    so no verdict should be telling callers to un-pack anything."""
    for n in (8, 20, 32, 64, 100, 128, 300):
        for rows in (7, 64, 300):
            assert not any(
                "PACKED" in c or "mis-align" in c
                for c in lookup(axis="rows", n=n, rows=rows).caveats
            )


@pytest.mark.parametrize("n", [256, 384, 512, 1024])
def test_rows_multiple_of_128_supported(n):
    """n > 128 and divisible by 128: softmax_rows_long with no tail chunk."""
    verdict = lookup(axis="rows", n=n, rows=8)
    assert verdict.supported
    assert verdict.app_name == "softmax_rows_long"
    app = _construct(verdict)
    assert app.tail == 0
    assert app.chunks_per_row == n // 128   # no phantom tail chunk


@pytest.mark.parametrize("width", [65, 96, 127])
def test_column_width_65_to_127_supported(width):
    """Formerly a gap; softmax_columns handles it (padding lanes are separate
    columns, so they never pollute a real column's reduce)."""
    verdict = lookup(axis="columns", width=width, rows=8)
    assert verdict.supported
    assert verdict.app_name == "softmax_columns"
    _construct(verdict)
    # The honesty requirement: say that lanes go idle.
    assert any("idle" in c for c in verdict.caveats)


def test_no_open_gaps():
    assert GAPS == ()


@pytest.mark.parametrize("rows", [129, 200, 1000])
@pytest.mark.parametrize("n", [8, 32, 128, 300])
def test_rows_over_128_supported(n, rows):
    """Every row app now loops groups of <=128 rows internally, so the former
    row-index overflow (a row index >=128 selecting the unloaded R1) is gone
    and there is no row-count cap on any row-axis path."""
    verdict = lookup(axis="rows", n=n, rows=rows)
    assert verdict.supported
    _construct(verdict)
    # The cap is gone from the advertised limits too, not just the verdict.
    # (Other caveats may mention 128 -- e.g. the packed output layout -- so
    # look for the overflow warning specifically.)
    assert not any("overflow" in c.lower() for c in verdict.caveats)


def test_columns_axis_has_no_row_cap():
    """Column softmax reduces down rows with full-vector scalars -- no cap."""
    verdict = lookup(axis="columns", width=128, rows=5000)
    assert verdict.supported
    _construct(verdict)


def test_verdict_is_truthy_only_when_supported():
    assert lookup(axis="rows", n=128, rows=8)
    assert not lookup(axis="rows", n=0, rows=8)


def test_invalid_axis_and_missing_sizes_raise():
    with pytest.raises(ValueError):
        lookup(axis="diagonal", rows=8, n=128)
    with pytest.raises(ValueError):
        lookup(axis="rows", rows=8)
    with pytest.raises(ValueError):
        lookup(axis="columns", rows=8)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_sizes_unsupported(bad):
    assert not lookup(axis="rows", n=bad, rows=8).supported
    assert not lookup(axis="rows", n=128, rows=bad).supported
    assert not lookup(axis="columns", width=bad, rows=8).supported


def test_describe_reports_both_outcomes():
    assert "NOT SUPPORTED" in lookup(axis="rows", n=-1, rows=8).describe()
    assert "SUPPORTED" in lookup(axis="rows", n=128, rows=8).describe()


# -- torch-style front end --------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(32, 300), (8, 128), (64, 16), (1, 500), (256, 65)])
def test_torch_dim1_matches_rows_axis(rows, cols):
    """dim=1 reduces along each row -> the rows axis, n = row length."""
    assert lookup_torch((rows, cols), dim=1) == lookup(axis="rows", n=cols, rows=rows)


@pytest.mark.parametrize("rows,cols", [(32, 300), (8, 128), (64, 16), (500, 1), (256, 65)])
def test_torch_dim0_matches_columns_axis(rows, cols):
    """dim=0 reduces down each column -> the columns axis, width = #columns."""
    assert lookup_torch((rows, cols), dim=0) == lookup(axis="columns", width=cols, rows=rows)


@pytest.mark.parametrize("dim,equiv", [(-1, 1), (-2, 0)])
def test_torch_negative_dims_count_from_the_end(dim, equiv):
    assert lookup_torch((32, 300), dim=dim) == lookup_torch((32, 300), dim=equiv)


def test_torch_defaults_to_last_dim():
    """torch.softmax defaults to the last dim; so does lookup_torch."""
    assert lookup_torch((32, 300)) == lookup_torch((32, 300), dim=1)


@pytest.mark.parametrize("n", [16, 128, 300])
def test_torch_1d_shape_is_a_single_row(n):
    """A 1-D tensor is one row of n; dim=0 and dim=-1 are the same axis."""
    assert lookup_torch((n,), dim=0) == lookup(axis="rows", n=n, rows=1)
    assert lookup_torch((n,), dim=-1) == lookup_torch((n,), dim=0)


def test_torch_verdicts_construct():
    for shape, dim in [((32, 300), 1), ((300, 32), 0), ((1000, 64), 1), ((64, 1000), 0)]:
        verdict = lookup_torch(shape, dim=dim)
        assert verdict.supported
        _construct(verdict)


@pytest.mark.parametrize("shape,dim", [((2, 3, 4), 1), ((2, 2, 2, 2), 0)])
def test_torch_rejects_higher_rank(shape, dim):
    with pytest.raises(ValueError, match="dimensions"):
        lookup_torch(shape, dim=dim)


@pytest.mark.parametrize("dim", [2, -3, 99])
def test_torch_rejects_out_of_range_dim(dim):
    with pytest.raises(ValueError, match="out of range"):
        lookup_torch((32, 300), dim=dim)


def test_torch_dim_choice_actually_changes_the_answer():
    """The shape is required, not just dim's size: the same tensor routes to
    different apps depending on which axis is reduced."""
    shape = (300, 16)
    assert lookup_torch(shape, dim=1).app_name == "softmax_rows_partial"
    assert lookup_torch(shape, dim=0).app_name == "softmax_columns_packed"


def test_torch_agrees_with_real_pytorch_semantics():
    """Pin lookup_torch's dim convention to actual torch, not just to our own
    reading of it: for each shape/dim, the axis torch reduces (the one whose
    output sums to 1) must be the axis the verdict routed to.

    torch is not a project dependency (the apps' own tests reference numpy), so
    this is skipped when it isn't installed rather than adding a build dep.
    """
    torch = pytest.importorskip("torch")

    for shape, dim in [((32, 300), 1), ((300, 32), 0), ((8, 128), -1),
                       ((300, 16), 0), ((64, 1000), -2), ((129, 128), 1)]:
        x = torch.randn(*shape)
        y = torch.softmax(x, dim=dim)
        # The reduced axis is the one that sums to 1.
        assert torch.allclose(y.sum(dim=dim), torch.ones_like(y.sum(dim=dim)), atol=1e-5)

        verdict = lookup_torch(shape, dim=dim)
        assert verdict.supported
        reduced_is_last = (dim % 2) == 1
        rows_apps = {"softmax_rows", "softmax_rows_partial", "softmax_rows_long"}
        assert (verdict.app_name in rows_apps) == reduced_is_last, (
            f"{shape} dim={dim}: torch reduces "
            f"{'along rows' if reduced_is_last else 'down columns'} but the "
            f"router chose {verdict.app_name}"
        )


def test_catalog_names_every_app():
    text = catalog()
    for app in ("softmax_rows", "softmax_rows_partial", "softmax_rows_long",
                "softmax_columns", "softmax_columns_packed"):
        assert app in text
    assert "none" in text  # no open gaps
