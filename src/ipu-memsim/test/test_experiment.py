"""The parameter registry, the fill-time model and the experiment API."""

from __future__ import annotations

import csv
import inspect
import sys
import xml.etree.ElementTree as ET

import pytest
from support import make_trace

from ipu_memsim import experiment
from ipu_memsim.cache_unit import CacheUnit
from ipu_memsim.config import KINDS, make_model
from ipu_memsim.dram import BankFillDram, FixedDram, TableDram, make_dram
from ipu_memsim.experiment import WorkItem, break_even, run_sweep, sweep_values, write_results
from ipu_memsim.params import (
    ASSUMED,
    CONFIRMED,
    MODEL_KINDS,
    REGISTRY,
    UNKNOWN,
    build_config,
    coerce,
    param,
    params_for,
    placeholders_in_use,
    validate_config,
)  # fmt: skip
from ipu_memsim.replay import replay
from ipu_memsim.toml_out import to_toml
from ipu_memsim.trace import READ, WRITE

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def stream(rows=4096, period=4):
    """Read row i and write output row i every ``period`` cycles."""
    accesses = []
    for i in range(rows):
        accesses += [(period * i, READ, i), (period * i, WRITE, 10000 + i)]
    return make_trace(accesses, [("x", "input", 0, rows), ("y", "output", 10000, rows)])


def reread(rows=2048, passes=3):
    """Read a weight table front to back several times."""
    accesses = [(p * rows + i, READ, i) for p in range(passes) for i in range(rows)]
    return make_trace(accesses, [("w", "input", 0, rows)])


WORKLOAD = [WorkItem("stream", "stream", 2, stream()), WorkItem("reread", "reread", 1, reread())]


def spec(**changes):
    base = {
        "name": "t",
        "workload": {"kind": "trace", "trace": "unused"},
        "model": {"kind": "cache_unit"},
        "fixed": {"xmem_mb": 2},
        "sweep": [{"param": "fill_cycles_per_bank", "values": [0, 500, 1000, 2111, 4000]}],
    }
    return {**base, **changes}


# -- the registry ----------------------------------------------------------------


def test_the_registry_covers_every_key_a_config_accepts():
    """Every constructor argument of every model, and every [dram] key, is a
    registry entry, and the registry lists nothing a model does not take."""
    for kind, model_class in KINDS.items():
        arguments = set(inspect.signature(model_class.__init__).parameters) - {
            "self",
            "dram",
            "args",
            "kwargs",
        }
        registered = {p.config_key for p in params_for(kind) if p.section == "model"}
        assert registered == arguments, kind
    backend_arguments = set()
    for backend in (BankFillDram, FixedDram, TableDram):
        backend_arguments |= set(inspect.signature(backend.__init__).parameters) - {"self"}
    registered = {p.config_key for p in REGISTRY if p.section == "dram"} - {"kind"}
    assert registered == backend_arguments
    assert set(KINDS) == set(MODEL_KINDS)


def test_every_entry_is_complete():
    for p in REGISTRY:
        assert p.status in (CONFIRMED, ASSUMED, UNKNOWN), p.name
        assert p.description.endswith(".") and p.source, p.name
        assert p.section in ("model", "dram", "derived"), p.name
        if p.default is not None:
            coerce(p, p.default)  # the default is a legal value
    names = [(kind, p.name) for kind in MODEL_KINDS for p in params_for(kind)]
    assert len(names) == len(set(names))  # unique per model


def test_the_defaults_of_every_model_build_and_run():
    for kind in MODEL_KINDS:
        config = build_config(kind)
        validate_config(config)
        assert replay(stream(64), make_model(config)).ideal_cycles == 253


def test_coerce_checks_type_range_and_choices():
    assert coerce(param("cache_unit", "num_banks"), "16") == 16
    assert coerce(param("cache_unit", "prefetch"), "false") is False
    assert coerce(param("cache_unit", "jump_back"), "auto") == "auto"
    assert coerce(param("cache", "size_rows"), "inf") == "inf"
    for name, bad, message in [
        ("num_banks", 0, "outside"),
        ("num_banks", 2.5, "whole number"),
        ("num_banks", "many", "must be a number"),
        ("residency", "sometimes", "must be one of"),
        ("prefetch", "maybe", "true or false"),
    ]:
        with pytest.raises(ValueError, match=message):
            coerce(param("cache_unit", name), bad)
    with pytest.raises(ValueError, match="has no parameter 'penalty'"):
        param("cache_unit", "penalty")


def test_build_config_keeps_only_the_chosen_dram_kind_and_derives_the_size():
    config = build_config("cache_unit", {"xmem_mb": 0.5, "fill_cycles_per_bank": 2111})
    assert config["model"]["num_banks"] == 4  # 0.5 MB of 1024-row banks
    assert config["dram"] == {"kind": "bank_fill", "fill_cycles_per_bank": 2111.0, "clock_ns": 1.0}
    fixed = build_config("cache_unit", {"dram_kind": "fixed"})
    assert fixed["dram"] == {"kind": "fixed", "latency_cycles": 100, "rows_per_cycle": 1.0}
    assert build_config("cache", {"xmem_mb": 2})["model"]["size_rows"] == 16384
    with pytest.raises(ValueError, match="no parameter"):
        build_config("ideal", {"num_banks": 4})


def test_validate_config_names_the_bad_key():
    with pytest.raises(ValueError, match="no key 'fill_time'"):
        validate_config({"model": {"kind": "cache_unit"}, "dram": {"fill_time": 3}})
    with pytest.raises(ValueError, match="unknown config table"):
        validate_config({"model": {"kind": "ideal"}, "extra": {}})
    with pytest.raises(ValueError, match="outside"):
        make_model({"model": {"kind": "cache_unit", "num_banks": 0}})


def test_placeholders_name_what_the_result_rests_on():
    names = {p.name for p in placeholders_in_use(build_config("cache_unit"))}
    assert {"fill_cycles_per_bank", "bank_allocation", "jump_back", "residency"} <= names
    assert "num_banks" not in names  # CONFIRMED
    assert "rows_per_cycle" not in names  # another dram kind's
    assert "drain_cycles_per_bank" not in names  # optional and not set
    assert placeholders_in_use(build_config("ideal")) == []


# -- the fill time ---------------------------------------------------------------


def test_bank_fill_dram():
    dram = BankFillDram(fill_cycles_per_bank=2111)
    assert dram.transfer_cycles(1024, write=False) == 2111
    assert dram.transfer_cycles(64, write=False) == 132  # the docstring example
    assert dram.transfer_cycles(1024, write=True) == 2111  # a drain takes as long, unless given
    assert BankFillDram(1000, drain_cycles_per_bank=10).transfer_cycles(1024, write=True) == 10
    # 2111 ns on a 0.5 ns clock is 4222 cycles.
    assert BankFillDram(fill_ns_per_bank=2111, clock_ns=0.5).transfer_cycles(1024, False) == 4222
    assert (
        make_dram({"kind": "bank_fill", "fill_ns_per_bank": 500}).transfer_cycles(1024, False)
        == 500
    )
    with pytest.raises(ValueError, match="not both"):
        BankFillDram(fill_cycles_per_bank=1, fill_ns_per_bank=1)
    with pytest.raises(ValueError, match="clock_ns"):
        BankFillDram(1, clock_ns=0)


@pytest.mark.parametrize("trace", [stream(), reread(), stream(8192, period=1)])
def test_a_fill_time_of_zero_reproduces_the_ideal_cycle_count(trace):
    for banks in (2, 16):
        result = replay(trace, CacheUnit(banks, dram=BankFillDram(0)))
        assert result.total_cycles == trace.ideal_cycles


@pytest.mark.parametrize("trace", [stream(), reread(), stream(8192, period=1)])
@pytest.mark.parametrize("banks", [2, 4, 16])
def test_stall_never_falls_when_the_fill_time_grows(trace, banks):
    stalls = [
        replay(trace, CacheUnit(banks, dram=BankFillDram(fill))).stall_cycles
        for fill in (0, 250, 500, 1000, 2111, 4000, 8000)
    ]
    assert stalls == sorted(stalls)


def test_prefetch_distance_lets_the_dma_run_further_ahead():
    """A fast reader (one row per cycle) and a DMA that needs 1100 cycles per
    page. With distance 1 the DMA starts page p+1 only when the IPU reaches
    page p; with distance 3 (4 banks) it works without a pause."""
    trace = make_trace([(i, READ, i) for i in range(8 * 1024)], [("x", "input", 0, 8 * 1024)])
    dram = BankFillDram(1100)
    near = replay(trace, CacheUnit(4, dram=dram, prefetch_distance=1))
    far = replay(trace, CacheUnit(4, dram=dram, prefetch_distance=3))
    none = replay(trace, CacheUnit(4, dram=dram, prefetch_distance=0))
    assert far.stall_cycles <= near.stall_cycles < none.stall_cycles
    assert none.stall_cycles == 8 * 1100  # every page a demand fetch
    # The DMA is the bottleneck either way (1100 > 1024 cycles per page): the
    # IPU ends when the DMA has delivered the last page, 8 * 1100 after the
    # start, having read 1024 rows after that... so it cannot beat this bound.
    assert far.total_cycles >= 8 * 1100
    # A slow reader (4 cycles per row): the DMA keeps ahead with distance 1 already.
    slow = make_trace([(4 * i, READ, i) for i in range(4096)], [("x", "input", 0, 4096)])
    assert (
        replay(slow, CacheUnit(4, dram=dram, prefetch_distance=1)).stall_cycles
        == replay(slow, CacheUnit(4, dram=dram, prefetch_distance=3)).stall_cycles
        == 1100
    )


# -- the residency switch ----------------------------------------------------------


def test_stays_in_xmem_removes_the_activation_round_trip():
    regions = [("x", "input", 0, 1024), ("w", "input", 5000, 1024), ("y", "output", 9000, 1024)]
    accesses = []
    for i in range(1024):
        accesses += [(i, READ, i), (i, READ, 5000 + i), (i, WRITE, 9000 + i)]
    trace = make_trace(accesses, regions)
    dram = BankFillDram(1000)
    round_trip = replay(trace, CacheUnit(dram=dram))
    stays = replay(trace, CacheUnit(dram=dram, residency="stays_in_xmem"))
    # Round trip: x and w are fetched (2 x 1000 cycles of stall), y is drained after the end.
    assert round_trip.stall_cycles == 2000 and round_trip.counters["tail_drain_cycles"] > 0
    # Stays: x is there already and y is left in XMEM; only the weights are fetched.
    assert stays.stall_cycles == 1000
    assert stays.counters["tables"]["x"]["fetches"] == 0
    assert stays.counters["tables"]["y"]["drains"] == 0 and stays.counters["tail_drain_cycles"] == 0
    # Which inputs are weights is a parameter: here both are, so nothing changes for inputs.
    both = replay(trace, CacheUnit(dram=dram, residency="stays_in_xmem", weight_tables="x,w"))
    assert both.stall_cycles == 2000
    with pytest.raises(ValueError, match="residency"):
        CacheUnit(residency="maybe")


# -- experiments -------------------------------------------------------------------


def test_run_sweep_gives_one_row_per_group_and_value():
    table = run_sweep(spec(), workload=WORKLOAD)
    assert [p.name for p in table.params] == ["fill_cycles_per_bank"]
    assert table.groups == ["stream", "reread"]
    assert len(table.rows) == 2 * 5
    assert set(table.rows[0]) == {
        "group", "fill_cycles_per_bank", "ideal_cycles", "stall_cycles", "total_cycles",
        "stall_percent", "note",
    }  # fmt: skip
    first = table.rows[0]
    # Weighted by the item's count (2), and a fill time of 0 is the ideal run.
    assert first["ideal_cycles"] == 2 * stream().ideal_cycles
    assert (first["fill_cycles_per_bank"], first["stall_cycles"], first["stall_percent"]) == (
        0,
        0,
        0.0,
    )
    for row in table.rows:
        assert row["total_cycles"] == row["ideal_cycles"] + row["stall_cycles"]
        assert row["stall_percent"] == pytest.approx(
            100 * row["stall_cycles"] / row["ideal_cycles"]
        )
    assert table.replays == 10
    # The swept parameter is the subject, not a placeholder; the others are named.
    names = {p.name for p in table.placeholders}
    assert "fill_cycles_per_bank" not in names and "bank_allocation" in names
    assert run_sweep(spec(), workload=WORKLOAD).rows == table.rows  # deterministic


def test_a_two_parameter_sweep_is_a_grid_and_reports_what_cannot_run():
    grid = spec(fixed={}, sweep=[
        {"param": "fill_cycles_per_bank", "values": [500, 4000]},
        {"param": "num_banks", "values": [1, 2, 16]},
    ])  # fmt: skip
    seen = []
    table = run_sweep(
        grid, workload=WORKLOAD, progress=lambda done, total: seen.append((done, total))
    )
    assert len(table.rows) == 2 * 2 * 3
    assert seen[-1] == (12, 12)
    one_bank = [row for row in table.rows if row["num_banks"] == 1 and row["group"] == "stream"]
    # One bank cannot hold the stream's two tables: the row says so.
    assert all(row["stall_cycles"] is None and "2 tables need" in row["note"] for row in one_bank)
    assert all(row["stall_cycles"] is not None for row in table.rows if row["group"] == "reread")
    assert experiment.grid_size(grid) == 6


def test_sweep_values_from_a_range():
    p, values = sweep_values("cache_unit", {"param": "fill_cycles_per_bank", "min": 250, "max": 8000,
                                            "steps": 6, "scale": "log"})  # fmt: skip
    assert values == [250, 500, 1000, 2000, 4000, 8000]
    assert sweep_values("cache_unit", {"param": "num_banks", "min": 4, "max": 16, "steps": 4})[
        1
    ] == [4, 8, 12, 16]
    assert sweep_values("cache_unit", {"param": "prefetch", "values": [True, False]})[1] == [
        True,
        False,
    ]
    for bad, message in [
        ({"param": "weight_tables", "values": ["w"]}, "cannot be swept"),
        ({"param": "num_banks", "min": 4, "max": 2}, "max > min"),
        ({"param": "num_banks", "min": 0, "max": 2, "scale": "log"}, "min > 0"),
        ({"param": "num_banks", "values": [0]}, "outside"),
    ]:
        with pytest.raises(ValueError, match=message):
            sweep_values("cache_unit", bad)
    with pytest.raises(ValueError, match="one or two"):
        run_sweep(spec(sweep=[]), workload=WORKLOAD)
    with pytest.raises(ValueError, match="both swept and fixed"):
        run_sweep(spec(fixed={"fill_cycles_per_bank": 1}), workload=WORKLOAD)


def test_break_even_by_hand():
    table = run_sweep(spec(), workload=WORKLOAD)
    for row in table.rows:  # put known numbers in: 0, 2, 4, 12, 30 percent
        if row["group"] == "stream":
            row["stall_percent"] = {0: 0.0, 500: 2.0, 1000: 4.0, 2111: 12.0, 4000: 30.0}[
                row["fill_cycles_per_bank"]
            ]
        else:
            row["stall_percent"] = 1.0
    stream_edge, reread_edge = break_even(table, 10.0)
    assert stream_edge["first_above"] == 2111
    # Between 1000 (4%) and 2111 (12%): 1000 + (10 - 4) / (12 - 4) * 1111.
    assert stream_edge["crossing"] == pytest.approx(1000 + 0.75 * 1111)
    assert (reread_edge["first_above"], reread_edge["crossing"]) == (None, None)
    assert break_even(table, 1.5)[0]["first_above"] == 500


def test_results_are_saved_with_the_spec_that_made_them(tmp_path):
    table = run_sweep(spec(report={"threshold_percent": 5}), workload=WORKLOAD)
    written = write_results(table, tmp_path)
    assert {p.name for p in written} == {"results.csv", "spec.toml", "break_even.csv", "t.svg"}
    with open(tmp_path / "results.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10 and rows[0]["group"] == "stream"
    # The saved spec is a complete experiment: running it again gives the same table.
    saved = tomllib.loads((tmp_path / "spec.toml").read_text())
    assert saved["report"]["threshold_percent"] == 5
    assert saved["sweep"] == [
        {"param": "fill_cycles_per_bank", "values": [0, 500, 1000, 2111, 4000]}
    ]
    assert run_sweep(saved, workload=WORKLOAD).rows == table.rows
    svg = (tmp_path / "t.svg").read_text()
    ET.fromstring(svg)
    assert "TOY MODEL" in svg and "DRAMsim3 HBM2 anchor" in svg
    # Two parameters: one heatmap per group.
    grid = spec(fixed={}, sweep=[{"param": "fill_cycles_per_bank", "values": [500, 4000]},
                                 {"param": "xmem_mb", "values": [0.125, 2]}])  # fmt: skip
    names = {p.name for p in write_results(run_sweep(grid, workload=WORKLOAD), tmp_path / "grid")}
    assert {"t_stream.svg", "t_reread.svg"} <= names
    assert (
        "n/a" in (tmp_path / "grid" / "t_stream.svg").read_text()
    )  # 0.125 MB = 1 bank: cannot run


def test_toml_round_trip():
    data = {"name": "x y", "n": 3, "f": 0.5, "flag": True, "list": [1, 2.5, "a"],
            "table": {"k": "v", "inner": {"a": 1}}, "sweep": [{"param": "p", "values": [1, 2]}]}  # fmt: skip
    assert tomllib.loads(to_toml(data)) == data


# -- sensitivity -------------------------------------------------------------------


def sensitivity_spec():
    return {
        "name": "s",
        "workload": {"kind": "trace", "trace": "unused"},
        "model": {"kind": "cache_unit"},
        "fixed": {"xmem_mb": 2, "fill_ns_per_bank": 2111},
        "vary": [
            {"param": "fill_ns_per_bank", "low": 250, "high": 8000, "reason": "range"},
            {"param": "clock_ns", "low": 0.5, "high": 2.0},
            {"param": "dma_channels", "low": 1, "high": 4},
            {"param": "residency", "low": "stays_in_xmem", "high": "dram_roundtrip"},
        ],
    }


def test_sensitivity_varies_one_parameter_at_a_time():
    from ipu_memsim import sensitivity

    rows = sensitivity.run_sensitivity(sensitivity_spec(), workload=WORKLOAD)
    assert len(rows) == 4 * 2  # parameters x groups
    by = {(row.param, row.group): row for row in rows}
    # The baseline column is the plain run at the fixed values, for every parameter.
    baseline = run_sweep(
        spec(fixed={"xmem_mb": 2}, sweep=[{"param": "fill_ns_per_bank", "values": [2111]}]),
        workload=WORKLOAD,
    )
    for row in rows:
        expected = next(r["stall_percent"] for r in baseline.rows if r["group"] == row.group)
        assert row.stall_percent[1] == pytest.approx(expected), row.param
        known = [s for s in row.stall_percent if s is not None]
        assert row.spread == pytest.approx(max(known) - min(known))
    fill = by[("fill_ns_per_bank", "stream")]
    assert fill.settings == (250, 2111, 8000) and fill.reason == "range"
    assert fill.stall_percent[0] < fill.stall_percent[1] < fill.stall_percent[2]
    # Halving the clock period doubles the fill time in cycles: the same as 4222 ns at 1 ns.
    clock = by[("clock_ns", "stream")]
    doubled = run_sweep(
        spec(fixed={"xmem_mb": 2}, sweep=[{"param": "fill_ns_per_bank", "values": [4222]}]),
        workload=WORKLOAD,
    )
    assert clock.stall_percent[0] == pytest.approx(doubled.rows[0]["stall_percent"])
    # The ranking is by the largest spread, and deterministic.
    ranking = sensitivity.rank(rows)
    spreads = [entry["max_spread"] for entry in ranking]
    assert spreads == sorted(spreads, reverse=True) and ranking[0]["param"] == "fill_ns_per_bank"
    assert (
        sensitivity.rank(sensitivity.run_sensitivity(sensitivity_spec(), workload=WORKLOAD))
        == ranking
    )


def test_sensitivity_results_are_saved_with_their_spec_and_a_pair_heatmap(tmp_path):
    from ipu_memsim import sensitivity

    one = sensitivity_spec()
    rows = sensitivity.run_sensitivity(one, workload=WORKLOAD)
    pair = sensitivity.run_pair(one, "fill_ns_per_bank", "residency", workload=WORKLOAD)
    assert len(pair.rows) == 2 * 3 * 2  # groups x fill settings x residency settings
    written = sensitivity.write_sensitivity(one, rows, tmp_path, pair)
    names = {p.name for p in written}
    assert {
        "sensitivity.csv",
        "ranking.csv",
        "ranking.md",
        "spec.toml",
        "s_pair_stream.svg",
    } <= names
    saved = tomllib.loads((tmp_path / "spec.toml").read_text())
    assert (
        sensitivity.run_sensitivity(saved, workload=WORKLOAD) == rows
    )  # reproducible from the saved spec
    assert "| 1 | `fill_ns_per_bank` |" in (tmp_path / "ranking.md").read_text()


def test_bank_allocation_rules():
    regions = [
        ("big", "input", 0, 6 * 1024),
        ("small", "input", 10000, 10),
        ("y", "output", 20000, 4096),
    ]
    trace = make_trace([(0, READ, 0), (1, READ, 10000), (2, WRITE, 20000)], regions)

    def banks(rule, total=9):
        tables = replay(trace, CacheUnit(total, bank_allocation=rule)).counters["tables"]
        return [t["banks"] for t in tables.values()]

    assert banks("proportional") == [5, 1, 3]  # by size: 6 and 4 pages share the 8 spare banks
    assert banks("equal") == [4, 1, 4]  # evenly, but nobody gets more than it has pages
    assert banks("one_each") == [1, 1, 1]
    with pytest.raises(ValueError, match="unknown bank_allocation"):
        replay(trace, CacheUnit(9, bank_allocation="random"))
