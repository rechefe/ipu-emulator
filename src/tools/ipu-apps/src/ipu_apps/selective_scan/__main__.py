"""Debug runner for the selective-scan app.

Usage::

    bazel run //src/tools/ipu-apps:selective_scan
    bazel run //src/tools/ipu-apps:selective_scan -- --stage 4 --batch 4
    PYTHONPATH=... python -m ipu_apps.selective_scan            # same thing

Assembles the kernel, runs one random config, cross-checks against the numpy
recurrence and prints cycles per timestep.

The ``--stage`` presets are the real MambaVision shapes (see
benchmark/benchmark.py for where they come from): stages 3 and 4 are the only
ones that run a scan, ``d_state=8``, and the scan width is ``dim/2`` because
``in_proj`` splits into x and z and only x is scanned.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.selective_scan import SelectiveScanApp

ASM_PATH = Path(__file__).resolve().parent / "selective_scan.asm"

# variant -> (stage-3 scan channels, stage-4 scan channels); dim * 2**i / 2.
VARIANTS = {"T": (160, 320), "S": (192, 384), "B": (256, 512)}
# 224x224: stage 3 is one 14x14 window, stage 4 one 7x7 window.
STAGE_TOKENS = {3: 196, 4: 49}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the selective-scan kernel")
    ap.add_argument("--variant", default="T", choices=sorted(VARIANTS),
                    help="MambaVision size (default: T)")
    ap.add_argument("--stage", type=int, default=3, choices=(3, 4),
                    help="MambaVision stage; only 3 and 4 run a scan (default: 3)")
    ap.add_argument("--batch", type=int, default=1, help="images (default: 1)")
    ap.add_argument("--d-state", type=int, default=8,
                    help="N; MambaVision hard-codes 8 (default: 8)")
    ap.add_argument("--d-model", type=int, default=None,
                    help="override the scan channel count from --variant/--stage")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override the token count from --stage")
    ap.add_argument("--max-cycles", type=int, default=40_000_000)
    args = ap.parse_args()

    d_model = args.d_model or VARIANTS[args.variant][args.stage - 3]
    seq_len = args.seq_len or STAGE_TOKENS[args.stage]
    batch, d_state = args.batch, args.d_state

    rng = np.random.RandomState(0)
    # A_log is the checkpoint parameter; the harness folds it to cA itself.
    kw = {
        "u": (rng.randn(batch, d_model, seq_len)).astype(np.float32),
        "dt": (rng.randn(batch, d_model, seq_len) * 0.5 - 1.0).astype(np.float32),
        "A_log": np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float32), (d_model, 1))),
        "B": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "C": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "D_skip": np.ones(d_model, dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "selective_scan.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        app = SelectiveScanApp(inst_path=inst_file, **kw)
        state, cycles = app.run(max_cycles=args.max_cycles)

    out = app.read_output(state)
    max_err = float(np.abs(out - app.reference()).max())
    timesteps = app.instances * app.seq_len

    print(f"config          MambaVision-{args.variant} stage {args.stage}: "
          f"batch={batch} D={d_model} N={d_state} L={seq_len}")
    print(f"instances       {app.instances} (batch x {app.tiles} channel tiles)")
    print(f"cycles          {cycles}")
    print(f"cyc/timestep    {cycles / timesteps:.2f}   (per 128-channel tile)")
    print(f"mult util       {state.stats.mult_utilization * 100:.1f}%")
    print(f"acc util        {state.stats.acc_utilization * 100:.1f}%")
    print(f"max abs err     {max_err:.3e}   {'PASS' if max_err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
