"""The cache_unit model against schedules worked out by hand.

The DRAM in every test needs 76 + rows cycles per transfer, so a full
1024-row page takes 1100 cycles.
"""

from __future__ import annotations

import pytest
from support import make_trace

from ipu_memsim.cache_unit import CacheUnit
from ipu_memsim.dram import FixedDram, TableDram, make_dram
from ipu_memsim.replay import replay
from ipu_memsim.trace import READ, WRITE

DRAM = FixedDram(latency_cycles=76, rows_per_cycle=1.0)


def unit(**arguments) -> CacheUnit:
    return CacheUnit(dram=DRAM, **{"num_banks": 2, **arguments})


def read_stream(rows=2048, passes=1):
    """Read an input table front to back, one row per cycle, ``passes`` times."""
    accesses = [(p * rows + i, READ, i) for p in range(passes) for i in range(rows)]
    return make_trace(accesses, [("x", "input", 0, rows)])


def write_stream(rows=2048):
    return make_trace([(i, WRITE, 5000 + i) for i in range(rows)], [("y", "output", 5000, rows)])


def test_the_docstring_example():
    # Page 0: demand fetch, 1100 cycles. Page 1 is prefetched from 1100 to
    # 2200; the IPU arrives at 1100 + 1024 = 2124 and waits 76.
    result = replay(read_stream(), unit())
    assert result.stall_cycles == 1100 + 76
    table = result.counters["tables"]["x"]
    assert (table["banks"], table["jump_back"]) == (2, 0)
    assert (table["fetches"], table["prefetches"], table["waits"]) == (1, 1, 1)


def test_prefetch_before_start_removes_the_cold_start():
    result = replay(read_stream(), unit(prefetch_before_start=True))
    assert result.stall_cycles == 0
    assert result.counters["dma_transfers"] == 0


def test_no_prefetch_means_one_demand_fetch_per_page():
    result = replay(read_stream(), unit(prefetch=False))
    assert result.stall_cycles == 2 * 1100
    assert result.counters["tables"]["x"]["fetches"] == 2


def test_one_bank_per_table_cannot_prefetch():
    # Page 1 needs the bank page 0 is in: a conflict, then a demand fetch.
    result = replay(read_stream(), unit(num_banks=1))
    assert result.stall_cycles == 2 * 1100
    assert result.counters["tables"]["x"]["conflicts"] == 1


def test_a_long_stream_keeps_up_when_the_dma_is_faster_than_the_ipu():
    # 8 pages through 2 banks. A page takes the DMA 1100 cycles and the IPU
    # 1024, so the IPU falls 76 cycles further behind on every page.
    result = replay(read_stream(8 * 1024), unit())
    assert result.stall_cycles == 1100 + 7 * 76
    assert result.counters["tables"]["x"]["prefetches"] == 7


def test_writes_claim_banks_for_free_and_drain_in_the_background():
    result = replay(write_stream(), unit())
    assert result.stall_cycles == 0
    table = result.counters["tables"]["y"]
    assert (table["claims"], table["fetches"], table["drains"]) == (2, 0, 2)
    # Page 0 drains from 1024 to 2124; page 1 after the kernel (2048), from
    # 2124 to 3224: the tail is 3224 - 2048.
    assert result.counters["tail_drain_cycles"] == 3224 - 2048


def test_a_written_page_must_drain_before_its_bank_is_reused():
    # One bank: page 1 has to wait for page 0's drain (1100 cycles).
    result = replay(write_stream(), unit(num_banks=1))
    assert result.stall_cycles == 1100
    assert result.counters["tables"]["y"]["conflicts"] == 1


def test_a_table_read_twice_stays_resident_when_it_has_the_banks():
    # jump_back "auto" sees the kernel go back 2047 rows: no bank is released.
    result = replay(read_stream(passes=2), unit())
    table = result.counters["tables"]["x"]
    assert table["jump_back"] == 2047
    assert table["refetches"] == 0
    assert result.stall_cycles == 1100 + 76  # the first pass only


def test_a_table_read_twice_through_one_bank_is_fetched_again():
    result = replay(read_stream(passes=2), unit(num_banks=1))
    table = result.counters["tables"]["x"]
    assert (table["fetches"], table["refetches"]) == (2, 2)
    assert result.stall_cycles == 4 * 1100


def test_a_fixed_jump_back_delays_the_release():
    # jump_back = 500: page 0's bank is released at page-1 row 500, so the
    # prefetch of page 2 starts 500 cycles later than with jump_back = 0.
    eager = replay(read_stream(3 * 1024), unit())
    late = replay(read_stream(3 * 1024), unit(jump_back=500))
    assert late.counters["tables"]["x"]["jump_back"] == 500
    assert late.stall_cycles == eager.stall_cycles + 500
    never = replay(read_stream(3 * 1024), unit(jump_back="never"))
    assert never.counters["tables"]["x"]["conflicts"] == 1


def test_scratch_tables_never_go_to_dram():
    accesses = [(i, WRITE, i) for i in range(1500)] + [(1500 + i, READ, i) for i in range(1500)]
    trace = make_trace(accesses, [("s", "scratch", 0, 1500)])
    result = replay(trace, unit())
    assert result.stall_cycles == 0
    assert result.counters["dma_transfers"] == 0


def test_banks_are_shared_in_proportion_to_table_size():
    accesses = [(0, READ, 0), (1, READ, 10000), (2, WRITE, 20000)]
    regions = [
        ("big", "input", 0, 6 * 1024),
        ("small", "input", 10000, 10),
        ("y", "output", 20000, 2048),
    ]
    trace = make_trace(accesses, regions)
    tables = replay(trace, unit(num_banks=6)).counters["tables"]
    assert {n: t["banks"] for n, t in tables.items()} == {"big": 4, "small": 1, "y": 1}
    # More banks than pages: nobody gets more than it can use.
    result = replay(trace, unit(num_banks=16))
    assert {n: t["banks"] for n, t in result.counters["tables"].items()} == {
        "big": 6, "small": 1, "y": 2,
    }  # fmt: skip
    assert result.counters["banks_used"] == 9
    explicit = replay(trace, unit(num_banks=6, bank_allocation={"big": 1, "small": 1, "y": 2}))
    assert explicit.counters["tables"]["y"]["banks"] == 2
    with pytest.raises(ValueError, match="3 tables need at least 3 banks"):
        replay(trace, unit(num_banks=2))
    with pytest.raises(ValueError, match="does not fit"):
        replay(trace, unit(num_banks=2, bank_allocation={"big": 2, "small": 1, "y": 1}))


def test_only_the_rows_a_table_has_are_moved():
    trace = make_trace([(0, READ, 0)], [("x", "input", 0, 64)])
    assert replay(trace, unit()).stall_cycles == 76 + 64
    assert replay(trace, unit(fill_whole_bank=True)).stall_cycles == 76 + 1024


def test_two_dma_channels_fetch_two_tables_at_once():
    regions = [("a", "input", 0, 1024), ("b", "input", 2000, 1024)]
    trace = make_trace([(0, READ, 0), (0, READ, 2000)], regions)
    assert replay(trace, unit()).stall_cycles == 2200
    assert (
        replay(trace, unit(dma_channels=2)).stall_cycles == 2200
    )  # demand fetches follow each other
    # ... but a prefetch no longer delays another table's demand fetch.
    regions = [("a", "input", 0, 2048), ("b", "input", 3000, 1024)]
    trace = make_trace([(0, READ, 0), (1, READ, 3000)], regions)
    assert replay(trace, unit(num_banks=3)).stall_cycles == 1100 + 2200 - 1  # behind a's prefetch
    assert replay(trace, unit(num_banks=3, dma_channels=2)).stall_cycles == 1100 + 1100


def test_rows_outside_every_table_never_stall():
    trace = make_trace([(0, READ, 777)], [("x", "input", 0, 64)])
    result = replay(trace, unit())
    assert result.stall_cycles == 0 and result.counters["untracked_accesses"] == 1


def test_dram_backends():
    assert FixedDram(100, 0.5).transfer_cycles(1024, write=False) == 2148  # the docstring example
    table = TableDram({64: 200, 1024: 1500})
    assert table.transfer_cycles(64, False) == 200
    assert table.transfer_cycles(544, False) == 850  # halfway between
    assert table.transfer_cycles(2048, True) == 3000  # beyond the table: by its last rate
    assert isinstance(make_dram(None), FixedDram)
    assert make_dram({"kind": "table", "fill": {"1024": 1100}}).transfer_cycles(1024, False) == 1100
    with pytest.raises(ValueError, match="unknown \\[dram\\] kind"):
        make_dram({"kind": "magic"})
