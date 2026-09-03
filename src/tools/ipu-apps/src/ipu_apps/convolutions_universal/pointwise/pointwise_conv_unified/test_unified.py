"""Standalone correctness test for pointwise_conv_unified (FP32 wide-vector).

Assembles the asm fresh each run (no persisted binary) and compares output
against a real ``torch.nn.functional.conv2d`` reference (tolerance-based,
since IPU FP32 accumulation order differs from PyTorch's). Designed to be
runnable directly:

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.test_unified
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
)


ASM_PATH = Path(__file__).resolve().parent / "pointwise_conv_unified.asm"

# (height, width, in_channels, out_channels)
# One representative per distinct code path, not an exhaustive parameter
# sweep -- keep this list small so the suite stays fast.
TEST_CONFIGS = [
    (16, 16,  16,  8),  # single-pass (in_ch <= 128)
    (16, 16, 128,  8),  # single-pass, in_ch at the 128 boundary
    (16, 16, 144,  8),  # multi-pass: 1 full + tail 16
    (16, 16, 256,  8),  # multi-pass: exact multiple, no tail
    (32, 32,  96, 32),  # larger spatial / out_ch variety
    (3, 5, 144, 8),     # non-power-of-2 spatial shape (internal padding)
    (1, 1, 8, 4),       # degenerate 1x1 spatial shape
]


def reference_pointwise(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference. Returns (out_ch, height, width) float32."""
    import torch
    import torch.nn.functional as F

    out_ch, in_ch = weights.shape
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).reshape(out_ch, in_ch, 1, 1)
    return F.conv2d(x, w).squeeze(0).numpy()


def run_one(inst_file: Path, height: int, width: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + height + width)
    weights = (rng.randn(out_ch, in_ch) * 0.3).astype(np.float32)
    input_chw = (rng.randn(in_ch, height, width) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        output_file = tmp / "output.bin"
        input_file.write_bytes(input_chw.tobytes())
        kernel_file.write_bytes(weights.tobytes())

        app = PointwiseConvUnifiedApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=output_file,
            height=height, width=width, in_channels=in_ch, out_channels=out_ch,
        )
        max_cyc = 50 * in_ch * out_ch * app.row_groups + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(
            out_ch, height, width
        )
        expected = reference_pointwise(weights, input_chw)

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, actual, expected


def main() -> None:
    template_src = ASM_PATH.read_text()

    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "unified.bin"
        print("Assembling pointwise_conv_unified.asm ...", flush=True)
        assemble_to_bin_file(template_src, str(inst_file))

        print(f"\n{'config':>26} {'cycles':>10} {'max_diff':>12} {'status':>8}")
        print("-" * 62)

        all_ok = True
        tol = 1e-3
        for height, width, in_ch, out_ch in TEST_CONFIGS:
            label = f"{height}x{width} ic={in_ch} oc={out_ch}"
            try:
                cycles, max_diff, actual, expected = run_one(
                    inst_file, height, width, in_ch, out_ch
                )
                ok = max_diff < tol
                status = "PASS" if ok else "FAIL"
                print(f"{label:>26} {cycles:>10} {max_diff:>12.3e} {status:>8}")
                if not ok:
                    all_ok = False
                    print(f"  first OC actual: {actual[0, 0, :8]}")
                    print(f"  first OC expect: {expected[0, 0, :8]}")
            except Exception as e:
                all_ok = False
                print(f"{label:>26} {'ERROR':>10} {'-':>12} {'FAIL':>8}")
                print(f"  {type(e).__name__}: {e}")

        print("-" * 62)
        print(f"Overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
