"""The working-set formulas and the knee finder."""

from __future__ import annotations

import pytest
from support import make_trace

from ipu_memsim import formulas
from ipu_memsim.experiment import WorkItem, run_sweep
from ipu_memsim.formulas import TableNeed, banks_needed, kernel_banks, measured_banks, tables_for
from ipu_memsim.knee import knees
from ipu_memsim.trace import READ, WRITE


def test_banks_needed_by_hand():
    assert banks_needed(8192, 0) == 1  # streamed front to back: one bank, however long
    assert (
        banks_needed(1440, 1439) == 2
    )  # re-read from the start: all of it (the docstring example)
    assert banks_needed(8192, 63) == 2  # a 64-row window can straddle two banks
    assert banks_needed(8448, 512, bank_rows=256) == 3  # a 513-row window: at most 3 banks of 256
    assert banks_needed(32, 31) == 1  # never more banks than the table has pages
    assert banks_needed(1, 0) == 1


def test_formulas_at_the_mobilevit_shapes():
    """Rows and look-back as the recorded traces have them (STATUS.md lists all)."""
    by = {t.name: t for t in tables_for("matmul_432x144_x128", {"m": 256, "k": 144, "n": 432})}
    assert (by["b"].rows, by["a"].rows, by["y"].rows) == (288, 864, 864)
    assert (by["b"].look_back, by["a"].look_back, by["y"].look_back) == (287, 863, 431)
    conv = {
        t.name: t
        for t in tables_for(
            "conv_universal", {"height": 64, "width": 64, "in_channels": 32, "out_channels": 32}
        )
    }
    # The project book's INT8 result, on the FP32 kernel: a look-back of 3 * C_in - 1.
    assert (conv["x"].rows, conv["x"].look_back, conv["filters"].rows, conv["y"].rows) == (
        1056,
        95,
        128,
        1024,
    )
    point = {
        t.name: t
        for t in tables_for(
            "pointwise_conv_unified",
            {"height": 128, "width": 128, "in_channels": 64, "out_channels": 32},
        )
    }
    assert (point["x"].rows, point["x"].look_back, point["w"].rows, point["y"].rows) == (
        8192,
        63,
        32,
        4096,
    )
    depth = {
        t.name: t
        for t in tables_for(
            "depthwise_conv_universal", {"height": 64, "width": 64, "in_channels": 256}
        )
    }
    assert (depth["x"].rows, depth["x"].look_back, depth["filters"].rows) == (8448, 512, 20)
    norm = tables_for("layernorm_256x144", {"channels": 144, "tokens": 256})
    assert len(norm) == 11 and kernel_banks(norm) == 11  # eleven tables, one bank each
    assert tables_for("unfold_32x32x144", {"c": 144}) is None  # not covered
    assert tables_for("attn_v_bcast_36", {}) is None


def test_measured_banks_on_hand_made_patterns():
    # A table streamed with a 100-row look-back, in banks of 64 rows: the
    # 101-row window straddles at most ceil(100 / 64) + 1 = 3 banks.
    accesses = []
    for i in range(100, 1000):
        accesses += [(2 * i, READ, i), (2 * i + 1, READ, i - 100)]
    trace = make_trace(accesses, [("x", "input", 0, 1000)])
    assert measured_banks(trace, bank_rows=64) == {"x": 3} and banks_needed(1000, 100, 64) == 3
    # Read twice, front to back: everything must stay (16 pages of 64).
    again = make_trace([(i, READ, i % 1000) for i in range(2000)], [("w", "input", 0, 1000)])
    assert measured_banks(again, bank_rows=64) == {"w": 16} and banks_needed(1000, 999, 64) == 16
    # An output written in order: one bank is enough (it drains behind the writer).
    out = make_trace([(i, WRITE, i) for i in range(1000)], [("y", "output", 0, 1000)])
    assert measured_banks(out, bank_rows=64) == {"y": 1}


def test_validate_reports_match_safe_and_under(monkeypatch):
    accesses = []
    for i in range(100, 1000):
        accesses += [(2 * i, READ, i), (2 * i + 1, READ, i - 100)]
    trace = make_trace(accesses, [("x", "input", 0, 1000)])
    trace.header["kernel"], trace.header["params"] = "fake", {}

    def check(look_back):
        monkeypatch.setitem(
            formulas.FORMULAS, "fake", lambda p: [TableNeed("x", "input", 1000, look_back)]
        )
        (row,) = formulas.validate(trace, (64,))
        return row["verdict"], row["look_back_trace"], row["banks_formula"], row["banks_replay"]

    assert check(100) == ("match", 100, 3, 3)
    assert check(400) == ("safe", 100, 8, 3)  # asks for more than needed
    assert check(0) == ("UNDER", 100, 1, 3)  # asks for too few: a wrong formula is caught


def test_knees_by_hand():
    # The module's example: 900%, 40%, 4.3%, 4.3%, 4.3% at 3, 4, 5, 8, 16 banks.
    stream = make_trace([(4 * i, READ, i) for i in range(64)], [("x", "input", 0, 64)])
    spec = {"workload": {"kind": "trace", "trace": "x"}, "model": {"kind": "cache_unit"},
            "sweep": [{"param": "num_banks", "values": [3, 4, 5, 8, 16]}]}  # fmt: skip
    table = run_sweep(spec, workload=[WorkItem("g", "g", 1, stream)])
    for row, stall in zip(table.rows, [900.0, 40.0, 4.3, 4.3, 4.3], strict=True):
        row["stall_percent"] = stall
    (knee,) = knees(table, bank_rows=1024, threshold_percent=10.0)
    assert (knee["threshold_knee_banks"], knee["plateau_knee_banks"]) == (5, 5)
    assert (knee["plateau_knee_rows"], knee["plateau_knee_mb"]) == (5120, 0.625)
    # A looser threshold moves only the threshold knee.
    assert knees(table, threshold_percent=50.0)[0]["threshold_knee_banks"] == 4
    # Stall above the threshold even with the most banks: no threshold knee, still a plateau.
    for row in table.rows:
        row["stall_percent"] = 102.0
    (flat,) = knees(table)
    assert (flat["threshold_knee_banks"], flat["plateau_knee_banks"]) == (None, 3)
    with pytest.raises(ValueError, match="num_banks alone"):
        table.params.clear()
        knees(table)
