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


# -- network-only tables ---------------------------------------------------------------


def test_layer_stats_by_hand():
    from ipu_memsim.netstats import LayerStats, layer_stats

    # 3x3, 16 -> 32 on 8x8 with a batch norm: 9*16*32 weights + 2*32; one multiply per weight and pixel.
    conv = {
        "op": "conv",
        "kernel_size": 3,
        "in_channels": 16,
        "out_channels": 32,
        "height": 8,
        "width": 8,
        "affine": 2,
    }
    assert layer_stats(conv) == LayerStats(4608 + 64, 16 * 64, 32 * 64, 4608 * 64)
    # Stride 2 quarters the output map; depthwise has one input channel per filter.
    depthwise = {**conv, "out_channels": 16, "groups": 16, "stride": 2, "affine": 0}
    assert layer_stats(depthwise) == LayerStats(144, 16 * 64, 16 * 16, 144 * 16)
    linear = {"op": "linear", "in_features": 192, "out_features": 768, "tokens": 197}
    assert layer_stats(linear) == LayerStats(192 * 768 + 768, 192 * 197, 768 * 197, 192 * 768 * 197)
    assert layer_stats({**linear, "bias": False}).params == 192 * 768
    attention = {"op": "attention", "tokens": 197, "heads": 3, "head_dim": 64}
    assert layer_stats(attention) == LayerStats(0, 3 * 197 * 192, 197 * 192, 2 * 3 * 197 * 197 * 64)
    assert layer_stats({"op": "layernorm", "channels": 192, "tokens": 197}).params == 384
    with pytest.raises(ValueError, match="unknown op"):
        layer_stats({"op": "magic"})


def test_the_network_files_reproduce_the_published_parameter_counts():
    """A hand-written layer list that misses or doubles a layer would not add
    up to the model's published total. All three do, exactly."""
    from pathlib import Path

    from ipu_memsim.netstats import network_table

    folder = Path(__file__).resolve().parents[1] / "networks"
    for name, published in [("deit_tiny", 5717416), ("swin_t", 28288354), ("yolov8n", 3157200)]:
        table = network_table(folder / f"{name}.toml")
        assert table["published_params"] == published
        assert table["total_params"] == published, name
        assert all(row["operations_per_byte"] >= 0 for row in table["rows"])
    # What no formula covers is listed, never dropped.
    yolo = network_table(folder / "yolov8n.toml")
    reasons = {item["reason"] for item in yolo["not_covered"]}
    assert "no formula for a stride-2 convolution" in reasons
    assert any("over the convolution kernels' limit" in reason for reason in reasons)
    # The templates hold no layers: nothing is invented for them.
    for template in ("hybrid_cnn_vit_TEMPLATE", "physical_ai_TEMPLATE"):
        assert network_table(folder / f"{template}.toml")["rows"] == []


def test_layer_knee_uses_the_validated_formulas_and_says_when_it_cannot():
    from ipu_memsim.netstats import layer_knee

    # A linear layer is the matmul kernel: 197 tokens (2 groups), 192 -> 768.
    banks, note = layer_knee(
        {"op": "linear", "in_features": 192, "out_features": 768, "tokens": 197}
    )
    assert (
        banks == kernel_banks(formulas.matmul_x128({"m": 197, "k": 192, "n": 768}))
        and "matmul" in note
    )
    assert layer_knee({"op": "conv", "kernel_size": 3, "stride": 2, "in_channels": 16, "out_channels": 32,
                       "height": 64, "width": 64})[0] is None  # fmt: skip
    assert layer_knee({"op": "pool"}) == (None, "no formula for op 'pool'")
