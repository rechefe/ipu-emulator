"""The replay engine and the toy models, on traces small enough to check by hand."""

from __future__ import annotations

import pytest
from support import make_trace

from ipu_memsim.config import make_model, rows_for_bytes
from ipu_memsim.models import AllMiss, Banked, Cache, Ideal, PageFirstTouch
from ipu_memsim.replay import replay
from ipu_memsim.trace import READ, WRITE
from ipu_memsim.workset import working_set

REGIONS = [("a", "input", 0, 4096), ("y", "output", 8192, 4096)]


def stream(n=3000):
    """Read row i and write output row i, one pair per cycle; 10 idle cycles at the end."""
    accesses = []
    for i in range(n):
        accesses += [(i, READ, i), (i, WRITE, 8192 + i)]
    return make_trace(accesses, REGIONS, ideal_cycles=n + 10)


def reuse():
    """Reads of rows 0 1 2 3 0 4 0, one per cycle."""
    return make_trace([(i, READ, r) for i, r in enumerate([0, 1, 2, 3, 0, 4, 0])], REGIONS)


# -- the mandatory sanity tests ----------------------------------------------


def test_ideal_reproduces_the_emulator_cycle_count():
    result = replay(stream(), Ideal())
    assert (result.total_cycles, result.stall_cycles) == (3010, 0)


@pytest.mark.parametrize("trace", [stream(), reuse()])
def test_an_infinite_preloaded_cache_reproduces_the_ideal_count(trace):
    result = replay(trace, Cache(size_rows=None, miss_penalty=50, preloaded=True))
    assert result.total_cycles == trace.ideal_cycles
    assert result.counters["misses"] == 0


def test_an_infinite_cold_cache_misses_once_per_line():
    result = replay(reuse(), Cache(size_rows=None, miss_penalty=50))
    assert result.counters == {"hits": 2, "misses": 5, "evictions": 0}
    assert result.stall_cycles == 5 * 50


@pytest.mark.parametrize("trace", [stream(), reuse()])
def test_a_cache_that_never_hits_stalls_accesses_times_penalty(trace):
    result = replay(trace, Cache(size_rows=0, miss_penalty=7))
    assert result.counters["hits"] == 0
    assert result.stall_cycles == len(trace) * 7
    assert result.total_cycles == trace.ideal_cycles + len(trace) * 7
    assert replay(trace, AllMiss(7)).stall_cycles == len(trace) * 7


@pytest.mark.parametrize(
    "config",
    [
        {"model": {"kind": "ideal"}},
        {"model": {"kind": "all_miss", "penalty": 3}},
        {"model": {"kind": "page_first_touch", "page_rows": 1024, "penalty": 100}},
        {"model": {"kind": "banked", "num_banks": 4, "busy_cycles": 3}},
        {"model": {"kind": "cache", "size_rows": 64, "line_rows": 4, "assoc": 2}},
        {
            "model": {"kind": "cache_unit", "num_banks": 4},
            "dram": {"kind": "fixed", "rows_per_cycle": 0.5},
        },
    ],
)
def test_the_same_trace_and_config_give_the_same_result(config):
    first = replay(stream(), make_model(config))
    second = replay(stream(), make_model(config))
    assert first == second
    # A model object can be reused: start() forgets the previous replay.
    model = make_model(config)
    assert replay(stream(), model) == replay(stream(), model) == first


# -- each toy model, by hand ---------------------------------------------------


def test_the_cache_docstring_example():
    # size 4, fully associative: 0 1 2 3 miss, 0 hits, 4 evicts row 1, 0 hits.
    result = replay(reuse(), Cache(size_rows=4, miss_penalty=10))
    assert result.counters == {"hits": 2, "misses": 5, "evictions": 1}


def test_a_preloaded_cache_pays_only_for_lines_that_come_back():
    # Capacity 2, rows 0 1 2 0: the first touches are free; row 0 was evicted
    # by row 2 and its return is the one miss.
    trace = make_trace([(i, READ, r) for i, r in enumerate([0, 1, 2, 0])], REGIONS)
    result = replay(trace, Cache(size_rows=2, miss_penalty=10, preloaded=True))
    assert result.counters == {"hits": 3, "misses": 1, "evictions": 2}


def test_cache_lines_and_free_write_allocation():
    # Lines of 4 rows: the 3000 input rows are 750 lines; writes are free.
    result = replay(
        stream(), Cache(size_rows=None, line_rows=4, miss_penalty=10, write_allocate_free=True)
    )
    assert result.stall_cycles == 750 * 10
    assert result.regions["a"].stalled == 750
    assert result.regions["y"].stall_cycles == 0


def test_direct_mapped_cache_conflicts():
    # 4 lines, 1 way: rows 0 and 4 share a set and evict each other.
    trace = make_trace([(i, READ, r) for i, r in enumerate([0, 4, 0, 4])], REGIONS)
    assert replay(trace, Cache(size_rows=4, assoc=1)).counters["misses"] == 4
    assert replay(trace, Cache(size_rows=4, assoc=2)).counters["misses"] == 2


def test_page_first_touch():
    # 3000 input rows and 3000 output rows: 3 pages each.
    result = replay(stream(), PageFirstTouch(page_rows=1024, penalty=100))
    assert result.counters == {"pages_touched": 6}
    assert result.stall_cycles == 600


def test_the_banked_docstring_example():
    trace = make_trace([(10, READ, 0), (11, READ, 2), (12, READ, 1)], REGIONS)
    result = replay(trace, Banked(num_banks=2, busy_cycles=3, mapping="interleave"))
    assert result.stall_cycles == 2  # only row 2 waits
    assert result.counters == {"bank_waits": 1}


def test_banked_mappings_differ():
    # Rows 0 and 1 in the same cycle: different banks interleaved, the same bank blocked.
    trace = make_trace([(0, READ, 0), (0, READ, 1)], REGIONS)
    assert replay(trace, Banked(4, 1, "interleave")).stall_cycles == 0
    assert replay(trace, Banked(4, 1, "blocked", bank_rows=1024)).stall_cycles == 1
    with pytest.raises(ValueError, match="mapping"):
        Banked(4, 1, "random")


# -- the replay's accounting ---------------------------------------------------


def test_stalls_serialize_and_later_accesses_see_the_shifted_time():
    """Three accesses, 50 cycles of stall each, in 1000 cycles: 1150 cycles,
    and the third (recorded at cycle 700) reaches the model at time 800."""
    seen = []

    class Spy(AllMiss):
        def access(self, t, op, row, region):
            seen.append(t)
            return 50

    trace = make_trace([(100, READ, 0), (400, READ, 1), (700, READ, 2)], REGIONS, 1000)
    assert replay(trace, Spy(50)).total_cycles == 1150
    assert seen == [100, 450, 800]


def test_per_region_statistics_add_up():
    result = replay(stream(), AllMiss(2))
    assert result.regions["a"].reads == 3000 and result.regions["a"].writes == 0
    assert result.regions["y"].writes == 3000
    assert sum(r.stall_cycles for r in result.regions.values()) == result.stall_cycles
    assert result.slowdown == pytest.approx((3010 + 12000) / 3010)


def test_config_errors_name_the_problem():
    with pytest.raises(ValueError, match="unknown model kind"):
        make_model({"model": {"kind": "magic"}})
    with pytest.raises(ValueError, match="no \\[model\\]"):
        make_model({})
    with pytest.raises(ValueError, match="no key 'bogus'"):
        make_model({"model": {"kind": "all_miss", "penalty": 1, "bogus": 2}})
    assert make_model({"model": {"kind": "cache", "size_rows": "inf"}}).size_rows is None
    assert rows_for_bytes(2 * 2**20) == 16384


# -- working set ---------------------------------------------------------------


def test_working_set_by_hand():
    # Rows 0 1 2 1 2 3 2 3 4 (a sliding 3-row window, like a 3x3 convolution).
    rows = [0, 1, 2, 1, 2, 3, 2, 3, 4]
    trace = make_trace([(i, READ, r) for i, r in enumerate(rows)], REGIONS)
    (a,) = working_set(trace)
    # Row 1 is live over accesses 1..3, row 2 over 2..6, row 3 over 5..7.
    # Accesses 2 and 3 have rows 1 and 2 live; 5 and 6 have rows 2 and 3 live:
    # never more than two (row 0 ends at access 0, row 4 starts at access 8).
    assert (a.name, a.footprint, a.peak_live, a.look_back) == ("a", 5, 2, 1)
    assert (a.reads, a.writes) == (9, 0)
