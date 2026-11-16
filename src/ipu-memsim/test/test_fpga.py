"""The FPGA comparison harness, on SYNTHETIC measurements made from the model."""

from __future__ import annotations

import pytest
from support import make_trace

from ipu_memsim import fpga
from ipu_memsim.trace import READ, WRITE


def stream(rows, period):
    accesses = []
    for i in range(rows):
        accesses += [(period * i, READ, i), (period * i, WRITE, 20000 + i)]
    return make_trace(accesses, [("x", "input", 0, rows), ("y", "output", 20000, rows)])


TRACES = {
    "small": stream(64, 5),
    "medium": stream(2048, 3),
    "large": stream(6000, 2),
    "slow": stream(3000, 9),
}
CONFIGS = {
    "banks16": {"model": {"kind": "cache_unit", "num_banks": 16}},
    "banks4": {"model": {"kind": "cache_unit", "num_banks": 4}},
}
RUNS = [(kernel, config) for kernel in TRACES for config in CONFIGS]


def test_spearman_by_hand():
    assert fpga.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert fpga.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Ranks 1 2 3 4 against 1 3 2 4: 1 - 6 * (0 + 1 + 1 + 0) / (4 * 15) = 0.8.
    assert fpga.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert fpga.spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)  # ties share a rank
    assert fpga.spearman([7], [3]) == 1.0


def test_a_model_compared_with_its_own_output_has_no_error():
    measured = fpga.synthetic_measurements(RUNS, TRACES, CONFIGS, fill_cycles=3000, noise=0.0)
    assert all("SYNTHETIC" in m.note for m in measured)
    at_truth = {name: fpga._with_fill(config, 3000) for name, config in CONFIGS.items()}
    same = fpga.compare(measured, TRACES, at_truth)
    assert (
        same["runs"] == 8
        and same["max_abs_rel_error"] == 0.0
        and same["rank_correlation"] == pytest.approx(1.0)
    )
    assert all(row["fetch_count_matches"] for row in same["rows"])
    # With the wrong fill time the model is off, in a known direction: too optimistic.
    wrong = fpga.compare(
        measured, TRACES, {name: fpga._with_fill(c, 500) for name, c in CONFIGS.items()}
    )
    assert wrong["bias"] < 0 and wrong["max_abs_rel_error"] > 0.05
    assert wrong["rank_correlation"] > 0.8  # ...but it still orders most runs correctly


def test_calibration_recovers_the_fill_time_and_holds_on_unseen_runs():
    measured = fpga.synthetic_measurements(
        RUNS, TRACES, CONFIGS, fill_cycles=3000, noise=0.02, seed=1
    )
    calibration, held_out = measured[0::2], measured[1::2]
    fitted, _ = fpga.fit_fill_time(calibration, TRACES, CONFIGS)
    assert fitted == pytest.approx(3000, rel=0.1)  # 2% noise on the cycles, not on the fill time
    check = fpga.compare(
        held_out, TRACES, {name: fpga._with_fill(c, fitted) for name, c in CONFIGS.items()}
    )
    assert check["mean_abs_rel_error"] < 0.03 and check["rank_correlation"] > 0.9
    text = fpga.report(check, "held out", fitted)
    assert "Rank correlation" in text and "Relative error = (model - measured) / measured" in text
    assert text.count("\n| ") == 1 + len(held_out)  # the header and one line per run


def test_the_import_file_round_trips_and_bad_files_are_refused(tmp_path):
    measured = fpga.synthetic_measurements(RUNS[:3], TRACES, CONFIGS, fill_cycles=1000)
    measured[0].stall_cycles = measured[0].fetches = measured[0].bank_fill_cycles = (
        None  # optional columns
    )
    fpga.save_measurements(measured, tmp_path / "m.csv", "SYNTHETIC")
    assert (tmp_path / "m.csv").read_text().startswith("# SYNTHETIC\nkernel,config,measured_cycles")
    assert fpga.load_measurements(tmp_path / "m.csv") == measured
    (tmp_path / "bad.csv").write_text("kernel,measured_cycles\nx,5\n")
    with pytest.raises(ValueError, match="missing column"):
        fpga.load_measurements(tmp_path / "bad.csv")
    (tmp_path / "empty.csv").write_text("kernel,config,measured_cycles\nx,c,\n")
    with pytest.raises(ValueError, match="line 2: no measured_cycles"):
        fpga.load_measurements(tmp_path / "empty.csv")
    with pytest.raises(KeyError):
        fpga.compare([fpga.Measurement("unknown", "banks16", 10)], TRACES, CONFIGS)
