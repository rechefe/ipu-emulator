"""The trace file: what is saved is what is loaded, and saving is deterministic."""

from __future__ import annotations

import numpy as np
import pytest
from support import make_trace

from ipu_memsim.trace import OTHER, READ, WRITE, load_trace, save_trace

REGIONS = [("a", "input", 0, 4), ("y", "output", 100, 4)]
ACCESSES = [(0, READ, 0), (0, WRITE, 100), (1, READ, 1), (2, READ, 999), (3, WRITE, 101)]


@pytest.mark.parametrize("name", ["t.csv", "t.csv.gz"])
def test_round_trip(tmp_path, name):
    trace = make_trace(ACCESSES, REGIONS, ideal_cycles=7)
    save_trace(trace, tmp_path / name)
    loaded = load_trace(tmp_path / name)
    assert loaded.header == trace.header
    for column in ("cycle", "op", "region", "row", "nbytes"):
        assert np.array_equal(getattr(loaded, column), getattr(trace, column)), column
    assert loaded.ideal_cycles == 7


def test_rows_outside_every_region_are_other(tmp_path):
    trace = make_trace(ACCESSES, REGIONS)
    assert trace.region.tolist() == [0, 1, 0, OTHER, 1]
    save_trace(trace, tmp_path / "t.csv")
    assert "2,R,other,999,512" in (tmp_path / "t.csv").read_text().splitlines()


@pytest.mark.parametrize("name", ["t.csv", "t.csv.gz"])
def test_saving_twice_gives_the_same_bytes(tmp_path, name):
    trace = make_trace(ACCESSES, REGIONS)
    save_trace(trace, tmp_path / "one" / name)
    save_trace(trace, tmp_path / "two" / name)
    assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_a_file_that_is_no_trace_is_refused(tmp_path):
    (tmp_path / "x.csv").write_text("cycle,op\n1,R\n")
    with pytest.raises(ValueError, match="not a memsim trace"):
        load_trace(tmp_path / "x.csv")
