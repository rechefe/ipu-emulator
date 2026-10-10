"""Residual-add family tests, beyond each kernel's own cases and test.py.

* **Documented routing** -- every row of the ``residual_add`` table in
  ``docs/content/kernels/normalization-and-shaping.md`` resolves to the kernel
  it names. The table is checked-in text, so without this it is an unverified
  second copy of the routing rules.
* **Completeness** -- every token x channel kernel appears in that table. The
  CHW ``residual_add`` is deliberately excluded: it answers the same operation
  from a different layout and is documented on the convolutions page instead,
  so :data:`CHW_KERNEL` is named here rather than the table being trusted to
  be short by one.
* **Axis order** -- this table lists ``Tokens`` before ``Channels``, the
  opposite of the ``layernorm`` table directly above it on the same page,
  while both queries are ``shape=(tokens, channels)`` and
  ``shape=(channels, tokens)`` respectively. That is the trap this file pins.
"""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry import kernels, resolve
from ipu_apps.kernel_registry.testing import doc_page, doc_table
from ipu_apps.kernels.elementwise.app import RESIDUAL_ADD_OP as OP

# Same operation, different memory layout (channels x height x width), and
# documented with the convolutions it is used alongside.
CHW_KERNEL = "residual_add"

KERNELS = sorted(spec.name for spec in kernels(OP) if spec.name != CHW_KERNEL)

_DOC = doc_page("normalization-and-shaping.md")
_ROWS = doc_table(_DOC, f"`{OP}`") if _DOC else []
_SKIP = pytest.mark.skipif(
    _DOC is None,
    reason="docs/content/kernels/normalization-and-shaping.md is not in the runfiles")


def _shape(row):
    """The registry query a documented row describes: ``(tokens, channels)``."""
    return (int(row["Tokens"]), int(row["Channels"]))


@_SKIP
def test_the_doc_actually_contains_a_residual_add_table():
    """Guards the parser: a doc rewrite that renames the heading or reshapes
    the table must not turn this file into a test that silently checks
    nothing."""
    assert len(_ROWS) >= 3, f"parsed only {len(_ROWS)} rows from {_DOC}"
    assert set(_ROWS[0]) == {"Kernel", "Tokens", "Channels"}, set(_ROWS[0])


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[r["Kernel"] for r in _ROWS] or None)
def test_documented_shape_routes_to_documented_kernel(row):
    shape = _shape(row)
    verdict = resolve(OP, shape=shape)
    assert verdict.supported, f"{row['Kernel']}: {verdict.reason}"
    assert verdict.app_name == row["Kernel"], (
        f"docs say (tokens={shape[0]}, channels={shape[1]}) is {row['Kernel']}, "
        f"registry answers {verdict.app_name}. Re-probe the table in {_DOC.name}."
    )


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[r["Kernel"] for r in _ROWS] or None)
def test_the_axis_order_is_not_reversible(row):
    """This table's columns are tokens-first while the layernorm table above
    it is channels-first, so reading either the wrong way round must fail."""
    tokens, channels = _shape(row)
    if tokens == channels:
        pytest.skip("square shape: the two orders are indistinguishable")
    verdict = resolve(OP, shape=(channels, tokens))
    assert verdict.app_name != row["Kernel"], (
        f"{row['Kernel']} claims both (tokens={tokens}, channels={channels}) and "
        f"its reverse, so the documented axis order proves nothing"
    )


@_SKIP
def test_every_token_by_channel_kernel_is_documented():
    assert sorted({row["Kernel"] for row in _ROWS}) == KERNELS


def test_the_chw_kernel_is_a_separate_layout_not_a_missing_row():
    """If the CHW kernel ever starts answering token x channel queries, the
    exclusion above is hiding a real gap in the table."""
    assert CHW_KERNEL in {spec.name for spec in kernels(OP)}
    assert CHW_KERNEL not in {row["Kernel"] for row in _ROWS}
