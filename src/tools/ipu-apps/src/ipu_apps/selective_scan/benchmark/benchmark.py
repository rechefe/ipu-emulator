"""Benchmark: selective_scan on the MambaVision shapes.

Usage::

    PYTHONPATH=... python -m ipu_apps.selective_scan.benchmark.benchmark

WHERE THE SHAPES COME FROM (mamba_vision.py, checked against the repo)
---------------------------------------------------------------------
Only stages 3 and 4 run a scan: ``MambaVisionLayer`` builds ConvBlocks for
``i in (0, 1)`` and ``Block``s for ``i in (2, 3)``, and each ``Block`` that is
not in ``transformer_blocks`` holds a ``MambaVisionMixer``.

* ``d_state=8`` and ``expand=1`` are hard-coded at the ``MambaVisionMixer(...)``
  call site, so ``d_inner = dim`` and the scan runs on ``d_inner//2 = dim/2``
  channels -- HALF the stage width, because ``in_proj`` splits into ``x`` and
  ``z`` and only ``x`` is scanned.
* stage ``i`` has width ``dim * 2**i``.
* At 224x224 the feature maps are 14x14 (stage 3) and 7x7 (stage 4), and
  ``window_size=[8, 8, 14, 7]`` makes each stage exactly one window, so the
  scan sequence length is 196 and 49 with no extra window batching.

    variant  dim   stage-3 (L=196)   stage-4 (L=49)
    T / T2    80   D = 160           D = 320
    S         96   D = 192           D = 384
    B        128   D = 256           D = 512

MambaVision-T runs 4 mixer blocks in stage 3 (depth 8, transformer_blocks =
range(4, 8)) and 2 in stage 4 (depth 4, range(2, 4)), so one 224x224 image is
4 x (D=160, L=196) + 2 x (D=320, L=49) scans.

``BenchRow.cyc_per_row`` carries CYCLES PER TIMESTEP per 128-channel tile --
the natural unit for a scan, since the recurrence is serial in t and every
timestep does the same work. The table header still says "cyc/row" because the
helper is shared with the softmax family; the config labels say ``t=`` to make
the unit unambiguous.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.selective_scan import SelectiveScanApp
from ipu_apps.softmax.benchmarking import BenchRow, print_and_write_table

ASM_PATH = Path(__file__).resolve().parent.parent / "selective_scan.asm"

# (batch, d_model, d_state, seq_len). d_model here is the SCAN channel count
# (dim/2), not the stage width. Batch is images -- at 224x224 stages 3 and 4 are
# one window each, so window partitioning does not multiply it.
CONFIGS = [
    (1, 160, 8, 196),    # MambaVision-T stage 3          (2 tiles: 128 + 32)
    (1, 320, 8, 49),     # MambaVision-T stage 4          (3 tiles: 128+128+64)
    (4, 160, 8, 196),    # MambaVision-T stage 3, batch 4
    (4, 320, 8, 49),     # MambaVision-T stage 4, batch 4
    (1, 192, 8, 196),    # MambaVision-S stage 3          (2 tiles: 128 + 64)
    (1, 256, 8, 196),    # MambaVision-B stage 3          (2 full tiles)
    (1, 128, 16, 196),   # NOT MambaVision: vanilla Mamba d_state=16, for the
                         # 4N term -- one exact tile keeps it comparable
]


def run_config(batch: int, d_model: int, d_state: int, seq_len: int,
               asm_text: str) -> BenchRow:
    rng = np.random.RandomState(0)
    kw = {
        "u": (rng.randn(batch, d_model, seq_len)).astype(np.float32),
        "dt": (rng.randn(batch, d_model, seq_len) * 0.5 - 1.0).astype(np.float32),
        "A": (-np.abs(rng.randn(d_model, d_state))).astype(np.float32),
        "B": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "C": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "D_skip": (rng.randn(d_model) * 0.5).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "selective_scan.bin"
        assemble_to_bin_file(asm_text, str(inst_file))
        app = SelectiveScanApp(inst_path=inst_file, **kw)
        state, cycles = app.run(max_cycles=40_000_000)

    out = app.read_output(state)
    max_err = float(np.abs(out - app.reference()).max())
    timesteps = app.instances * app.seq_len

    return BenchRow(
        label=f"b={batch} D={d_model} N={d_state} t={seq_len}",
        cycles=cycles,
        cyc_per_row=cycles / timesteps,
        mult_utilization=state.stats.mult_utilization,
        acc_utilization=state.stats.acc_utilization,
        max_err=max_err,
        correct=max_err < 1e-4,
    )


def main() -> None:
    asm_text = ASM_PATH.read_text()
    rows = [run_config(*cfg, asm_text) for cfg in CONFIGS]
    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("selective_scan", rows, out_path)


if __name__ == "__main__":
    main()
