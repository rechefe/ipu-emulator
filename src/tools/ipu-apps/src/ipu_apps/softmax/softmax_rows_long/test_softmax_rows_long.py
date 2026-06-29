"""Standalone runner + config list for softmax_rows_long.

Shared by the pytest wrapper in ipu-apps/test/. Computes a numpy softmax
reference over rows x N FP32 logits (N>128, N%128 != 0) and compares.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp

ASM_PATH = (
    Path(__file__).resolve().parent / "softmax_rows_long.asm"
)

# (rows, n, scale, seed)
TEST_CONFIGS = [
    (1, 129, 3.0, 0),     # smallest long row: 1 full chunk + 1 tail
    (4, 200, 4.0, 1),     # 1 full + 72 tail
    (8, 256 + 1, 3.0, 2), # 2 full + 1 tail
    (8, 300, 5.0, 3),     # 2 full + 44 tail
    (16, 130, 50.0, 4),   # numerical stability (large |x|)
    (8, 300, 0.01, 5),    # near-uniform
    (32, 384 + 17, 3.0, 6),  # 3 full + 17 tail
    (128, 129, 4.0, 7),   # max rows in one group
]


def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def run_one(inst_file: Path, rows: int, n: int, scale: float, seed: int):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, n) * scale).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "in.bin"
        output_file = Path(tmp) / "out.bin"
        input_file.write_bytes(x.tobytes())
        app = SoftmaxRowsLongApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=output_file,
            rows=rows,
            n=n,
        )
        _, cycles = app.run(max_cycles=8_000_000)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(rows, n)
    ref = _reference(x)
    max_abs = float(np.abs(out - ref).max())
    sums = out.sum(axis=1)
    return cycles, max_abs, sums, out, ref


if __name__ == "__main__":
    from ipu_as.lark_tree import assemble_to_bin_file

    with tempfile.TemporaryDirectory() as tmp:
        inst = Path(tmp) / "softmax_rows_long.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
        for rows, n, scale, seed in TEST_CONFIGS:
            cycles, max_abs, sums, _, _ = run_one(inst, rows, n, scale, seed)
            ok = max_abs < 1e-4 and np.allclose(sums, 1.0, atol=1e-5)
            print(f"rows={rows:3d} n={n:3d} scale={scale:<5} "
                  f"max_abs={max_abs:.2e} sum~1={np.allclose(sums,1.0,atol=1e-5)} "
                  f"cyc={cycles} {'OK' if ok else 'FAIL'}")
