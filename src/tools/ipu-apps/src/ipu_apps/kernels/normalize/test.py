"""LayerNorm family tests, beyond each kernel's own cases and test.py.

* **Documented routing** -- every row of the ``layernorm`` table in
  ``docs/content/kernels/normalization-and-shaping.md`` resolves to the kernel
  it names. The table is checked-in text, so without this it is an unverified
  second copy of the routing rules.
* **Completeness** -- every declared layernorm kernel appears in that table.
* **Axis order** -- the query is ``shape=(channels, tokens)``. The
  ``residual_add`` table on the same page lists its columns the other way
  round (tokens first), so a swap here is an easy mistake that routing alone
  would not always catch; :func:`test_the_axis_order_is_not_reversible` pins
  it.
"""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry import kernels, resolve
from ipu_apps.kernel_registry.testing import doc_page, doc_table
from ipu_apps.kernels.normalize.app import OP

KERNELS = sorted(spec.name for spec in kernels(OP))

_DOC = doc_page("normalization-and-shaping.md")
_ROWS = doc_table(_DOC, f"`{OP}`") if _DOC else []
_SKIP = pytest.mark.skipif(
    _DOC is None,
    reason="docs/content/kernels/normalization-and-shaping.md is not in the runfiles")


def _shape(row):
    """The registry query a documented row describes: ``(channels, tokens)``."""
    return (int(row["Channels"]), int(row["Tokens"]))


@_SKIP
def test_the_doc_actually_contains_a_layernorm_table():
    """Guards the parser: a doc rewrite that renames the heading or reshapes
    the table must not turn this file into a test that silently checks
    nothing."""
    assert len(_ROWS) >= 3, f"parsed only {len(_ROWS)} rows from {_DOC}"
    assert set(_ROWS[0]) == {"Kernel", "Channels", "Tokens"}, set(_ROWS[0])


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[r["Kernel"] for r in _ROWS] or None)
def test_documented_shape_routes_to_documented_kernel(row):
    shape = _shape(row)
    verdict = resolve(OP, shape=shape)
    assert verdict.supported, f"{row['Kernel']}: {verdict.reason}"
    assert verdict.app_name == row["Kernel"], (
        f"docs say (channels={shape[0]}, tokens={shape[1]}) is {row['Kernel']}, "
        f"registry answers {verdict.app_name}. Re-probe the table in {_DOC.name}."
    )
    assert verdict.alternatives == (), verdict.alternatives


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[r["Kernel"] for r in _ROWS] or None)
def test_the_axis_order_is_not_reversible(row):
    """Reading the table's columns the wrong way round must not still resolve
    to the same kernel -- otherwise the documented axis order is unverifiable."""
    channels, tokens = _shape(row)
    if channels == tokens:
        pytest.skip("square shape: the two orders are indistinguishable")
    verdict = resolve(OP, shape=(tokens, channels))
    assert verdict.app_name != row["Kernel"], (
        f"{row['Kernel']} claims both (channels={channels}, tokens={tokens}) and "
        f"its reverse, so the documented axis order proves nothing"
    )


@_SKIP
def test_every_kernel_is_documented():
    assert sorted({row["Kernel"] for row in _ROWS}) == KERNELS
