"""fully_connected beyond its golden: its multiplier and alias accounting, and
that a missing data file fails the case (E07-21).

Each format runs once, profiled (E06-12): the profiled run is held to the
golden too, so the counts are those of a correct run.
"""

from __future__ import annotations

import shutil

import pytest

from ipu_harness.cases import expand
from ipu_harness.family import load_kernel
from ipu_harness.params import PlanError
from ipu_harness.runner import RunOptions, run_kernel

#: What the profile sees in every format, as (verified, candidate) counts.
#: A16: each of the 10 samples is stored with the emulator-only ST.ACC (the
#: one ISA alias the ipu_apps test counted, ``alias_hits``). The profile's
#: sequence detectors also report post-increment addressing (A12), a narrow
#: MULT (A13: 64 outputs on 128 lanes) and multiple accumulators (A14); see
#: docs/content/isa-alias-catalogue.md.
ALIASES = {
    "A16_STR_ACC_REG": (10, 0),
    "A12_ADDRESSING": (3869, 0),
    "A13_NARROW_MULT": (1280, 0),
    "A14_MULTIPLE_ACC": (1161, 0),
}


@pytest.fixture(scope="module")
def default(cases):
    (case,) = [c for c in cases if c.name == "default"]
    return case


@pytest.mark.parametrize("fmt", ["int8", "int32", "e4", "e5"])
def test_mac_accounting_and_alias_hits(kernel, default, run_case, fmt):
    (format_,) = [f for f in default.formats if f.value == fmt]
    result = run_case(default, format_, RunOptions(alias_profile=True))
    assert result.passed, result.describe()
    stats = result.stats
    assert stats is not None and stats.mult_active_cycles > 0
    # The kernel pads 64 outputs to 128 lanes and declares no narrower width,
    # so every multiply counts all 128 lanes.
    assert stats.mult_lane_ops == stats.mult_active_cycles * 128
    assert result.alias_profile is not None
    aliases = result.alias_profile.to_dict()["aliases"]
    assert observed(aliases) == ALIASES


def observed(aliases):
    """Each alias the profile saw, with its (verified, candidate) counts."""
    counts = {}
    for alias, entry in aliases.items():
        seen = {"verified": 0, "candidate": 0}
        for row in entry["measurements"]:
            seen[row["status"]] += row["count"]
        if any(seen.values()):
            counts[alias] = (seen["verified"], seen["candidate"])
    return counts


@pytest.mark.parametrize("missing", ["e4/x.bin", "int8/w.bin", "e5/y.bin"])
def test_a_missing_data_file_fails_the_case(kernel, tmp_path, missing):
    # A copy of the family folder (the kernel and the family's pack.py) with
    # one data file removed: the case fails with an error naming it, before
    # or after the run, and is never skipped.
    family = tmp_path / kernel.directory.parent.name
    shutil.copytree(kernel.directory.parent, family, ignore=shutil.ignore_patterns("__pycache__"))
    copy = family / kernel.directory.name
    (copy / "data" / missing).unlink()
    manifest = load_kernel(str(copy))
    (case,) = [c for c in expand(manifest) if c.name == "default"]
    fmt = {"int8": "int8", "e4": "e4", "e5": "e5"}[missing.split("/")[0]]
    (format_,) = [f for f in case.formats if f.value == fmt]
    with pytest.raises(PlanError, match=missing.split("/")[1]):
        run_kernel(manifest, case, format_)
