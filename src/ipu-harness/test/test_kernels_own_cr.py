"""Kernels that address each table through the table's own CR.

A table is reached through the CR that holds its base, and the LR holds only the
offset within it. These kernels used to read or store through the hardwired CR0
with an absolute row in an LR; the tables checker found those accesses in the
table whose rows held the row, and the manifests declared ``also_via = ["CR0"]``
for it. Now no access goes through CR0 and no table declares a second CR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.runner import RunOptions, run_kernel
from ipu_harness.variants import variant

KERNELS = Path(__file__).resolve().parents[3] / "kernels"

OWN_CR = [
    "attention/attn_v_bcast_36",
    "normalize/packed_layernorm",
    "projections/packed_output_linear",
    "projections/packed_output_linear_p4",
    "projections/proj_p4",
    "projections/proj_p4_2tg",
    "projections/proj_p4_2tg_silu",
    "reshape/packed_pack",
    "reshape/packed_unpack",
    "softmax/softmax_columns_packed",
    "softmax/softmax_rows_partial",
]


@pytest.mark.parametrize("kernel", OWN_CR)
def test_the_default_case_reaches_no_table_through_cr0(kernel):
    manifest = load_manifest(KERNELS / kernel)
    case = next(c for c in expand(manifest) if c.name == "default")
    result = run_kernel(manifest, case, variant("fp32"), RunOptions(tables=True))

    assert result.passed, result.describe()
    report = result.tables
    assert report is not None
    assert [t.label for t in report.tables if t.cr == 0 or t.views] == []
    assert [f.describe() for f in report.findings if f.severity == "error"] == []
    assert [f.describe() for f in report.findings if "CR0" in f.message] == []
