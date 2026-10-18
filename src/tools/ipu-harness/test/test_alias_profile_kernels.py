"""Profiling keeps every kernel case's results and accounting (moved from
ipu-apps, E07-25).

Every case of every kernel of the tree runs twice in its first format, plain
and with an alias profile (``RunOptions(alias_profile=True)``): the two runs
retire the same words, write the same output bytes and count the same
statistics, and the profile reports every alias detector. Heavy tier
(nightly): it repeats the work of every kernel's own tests.
"""

from __future__ import annotations

import pytest

from ipu_emu.alias_detectors import default_registry as alias_registry
from ipu_harness.cases import Case, at_values, expand
from ipu_harness.manifest import KernelManifest
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import RunOptions, run_kernel

_REGISTRY = Registry.load(kernel_tree())

CASES = [
    pytest.param(manifest, case, id=f"{name}-{case.name}")
    for name, manifest in sorted(_REGISTRY.manifests.items())
    for case in expand(manifest)
]


def _profiled_equals_plain(manifest: KernelManifest, case: Case) -> dict:
    fmt = case.formats[0]
    plain = run_kernel(manifest, case, fmt)
    profiled = run_kernel(manifest, case, fmt, RunOptions(alias_profile=True))
    assert plain.passed, plain.describe()
    assert profiled.passed, profiled.describe()
    profile = profiled.alias_profile
    assert profile is not None
    assert profiled.cycles == plain.cycles == profile.cycles
    assert profiled.raw_outputs == plain.raw_outputs
    assert profiled.stats == plain.stats
    report = profile.to_dict()
    assert set(report["aliases"]) == set(alias_registry().entries)
    assert report["totals"]["unique_read_bytes"] <= report["totals"]["read_bytes"]
    return report


def test_the_tree_has_cases():
    assert not _REGISTRY.invalid, _REGISTRY.invalid
    assert len(CASES) > len(_REGISTRY.manifests)


@pytest.mark.parametrize(("manifest", "case"), CASES)
def test_a_profiled_case_equals_a_plain_one(manifest, case):
    report = _profiled_equals_plain(manifest, case)
    if manifest.name.startswith("softmax"):
        # The kernels write their log2(e) row once, as an input; the detector
        # verifies the exponential without a declaration.
        exponential = report["aliases"]["A9_EXP"]["measurements"]
        assert exponential and all(row["status"] == "verified" for row in exponential)


@pytest.mark.parametrize("width", [17, 50])
def test_the_packed_softmax_tail_mask_profiles_the_same(width):
    manifest = _REGISTRY.kernel("softmax_columns_packed")
    default = next(c for c in expand(manifest) if c.name == "default")
    _profiled_equals_plain(
        manifest, at_values(default, f"tail_width_{width}", {"rows": 17, "width": width})
    )
