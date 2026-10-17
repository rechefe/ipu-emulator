"""fully_connected beyond its golden: its multiplier and alias accounting (E07-21).

Each format runs once, profiled (E06-12): the profiled run is held to the
golden too, so the counts are those of a correct run.
"""

from __future__ import annotations

import pytest

from ipu_harness.runner import RunOptions

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
