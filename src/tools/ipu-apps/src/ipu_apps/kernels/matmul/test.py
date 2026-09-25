"""Matmul family tests, beyond each kernel's own cases and test.py.

* **Documented routing** -- every row of the ``matmul`` table in
  ``docs/content/kernels/linear-layers.md`` resolves to the kernel it names.
  The table is checked-in text, so without this it is an unverified second
  copy of the routing rules.
* **Completeness** -- every declared matmul kernel appears in that table, so
  adding a kernel without documenting it fails here rather than leaving the
  page quietly short.
* **Activation contract** -- a kernel that fuses ``silu`` into its store must
  refuse the same shape asked for with ``activation="none"``, and vice versa.
  A plain query that silently returned ``silu(C)`` was a real routing bug.
"""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry import kernels, resolve
from ipu_apps.kernel_registry.testing import doc_page, doc_table
from ipu_apps.kernels.matmul.app import ACTIVATIONS, OP

KERNELS = sorted(spec.name for spec in kernels(OP))

_DOC = doc_page("linear-layers.md")
_ROWS = doc_table(_DOC, f"`{OP}`") if _DOC else []
_SKIP = pytest.mark.skipif(_DOC is None,
                           reason="docs/content/kernels/linear-layers.md is not in the runfiles")


def _query(row):
    """The registry query a documented row describes.

    ``shape_b_t`` is the *transposed* weight shape ``(N, K)``, not ``(K, N)``.
    """
    m, k, n = int(row["M"]), int(row["K"]), int(row["N"])
    return dict(shape_a=(m, k), shape_b_t=(n, k), activation=row["activation"])


def _id(row):
    return f"{row['Kernel']}-{row['activation']}"


@_SKIP
def test_the_doc_actually_contains_a_matmul_table():
    """Guards the parser: a doc rewrite that renames the heading or reshapes
    the table must not turn this file into a test that silently checks
    nothing."""
    assert len(_ROWS) >= 10, f"parsed only {len(_ROWS)} rows from {_DOC}"
    assert set(_ROWS[0]) == {"Kernel", "M", "K", "N", "activation"}, set(_ROWS[0])


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[_id(r) for r in _ROWS] or None)
def test_documented_shape_routes_to_documented_kernel(row):
    verdict = resolve(OP, **_query(row))
    assert verdict.supported, f"{_id(row)}: {verdict.reason}"
    assert verdict.app_name == row["Kernel"], (
        f"docs say (M, K, N) == ({row['M']}, {row['K']}, {row['N']}) with "
        f"activation={row['activation']!r} is {row['Kernel']}, registry answers "
        f"{verdict.app_name}. Re-probe the table in {_DOC.name}."
    )
    assert verdict.alternatives == (), verdict.alternatives


@_SKIP
@pytest.mark.parametrize("row", _ROWS, ids=[_id(r) for r in _ROWS] or None)
def test_the_wrong_activation_is_refused(row):
    """Each kernel fuses one activation into its store, so the same shape
    asked for with any other activation must be refused rather than silently
    computing something else."""
    for other in ACTIVATIONS:
        if other == row["activation"]:
            continue
        verdict = resolve(OP, **{**_query(row), "activation": other})
        assert verdict.app_name != row["Kernel"], (
            f"{row['Kernel']} fuses activation={row['activation']!r} but also "
            f"claims an activation={other!r} query"
        )


@_SKIP
def test_every_kernel_is_documented():
    assert sorted({row["Kernel"] for row in _ROWS}) == KERNELS


@_SKIP
def test_documented_activations_are_real_activations():
    assert {row["activation"] for row in _ROWS} <= set(ACTIVATIONS)
