"""The recorder on real kernels.

Three kernels (the brief's Stage 1 list): ``residual_add``, a pointwise 1x1
convolution and a small transformer matmul.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ipu_memsim.models import Ideal
from ipu_memsim.record import record
from ipu_memsim.replay import replay
from ipu_memsim.trace import OTHER, READ, WRITE, load_trace, save_trace
from ipu_memsim.workset import working_set

KERNELS = ["residual_add", "pointwise_conv_unified", "matmul_small"]


def run(kernel, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ST.ACC is an emulator-only instruction
        return record(kernel, **kwargs)


@pytest.fixture(scope="module", params=KERNELS)
def trace(request):
    return run(request.param)


def test_entries_match_the_emulators_own_statistics(trace):
    stats = trace.header["stats"]
    assert int((trace.op == READ).sum()) == stats["xmem_reads"]
    assert int((trace.op == WRITE).sum()) == stats["xmem_writes"]
    assert trace.header["stats_match"]
    assert len(trace) > 0


def test_cycles_are_in_order_and_inside_the_run(trace):
    assert (np.diff(trace.cycle) >= 0).all()
    assert trace.cycle.min() >= 0 and trace.cycle.max() < trace.ideal_cycles


def test_every_write_lands_in_an_output_or_scratch_region(trace):
    roles = [r["role"] for r in trace.header["regions"]]
    for region in trace.region[trace.op == WRITE].tolist():
        assert region != OTHER and roles[region] in ("output", "scratch")


def test_replaying_with_ideal_gives_the_emulator_cycle_count(trace):
    assert replay(trace, Ideal()).total_cycles == trace.ideal_cycles


def test_recording_is_deterministic(trace, tmp_path):
    again = run(trace.header["kernel"])
    save_trace(trace, tmp_path / "a.csv")
    save_trace(again, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert load_trace(tmp_path / "a.csv").header == trace.header


def test_residual_add_by_hand():
    """y = a + b over 64 rows: each row of a and b is read once, each row of y
    written once, front to back."""
    trace = run("residual_add")
    assert trace.header["params"]["num_channels"] == 64
    by_name = {w.name: w for w in working_set(trace)}
    for name in ("a", "b"):
        assert (by_name[name].reads, by_name[name].footprint, by_name[name].look_back) == (
            64,
            64,
            0,
        )
    assert (by_name["y"].writes, by_name["y"].footprint) == (64, 64)
    # The regions are where kernel.toml puts them: rows 0, 1024 and 2048.
    assert [r["base_row"] for r in trace.header["regions"]] == [0, 1024, 2048]


def test_other_parameter_values_and_cases():
    bigger = run("residual_add", values={"num_channels": 96})
    assert bigger.header["params"]["num_channels"] == 96
    assert int((bigger.op == WRITE).sum()) == 96
    with pytest.raises(KeyError, match="no case"):
        run("residual_add", case="nope")
