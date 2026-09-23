"""Shared concat cases: random FP32 rows checked bit-exact against ``np.concatenate``.

Each input channel is one opaque row -- no spatial packing is encoded inside
a row, since the kernel itself never interprets one. The copy passes through
``MULT x1.0`` and ``ACC.ADD.FIRST``, which are exact, so the output must equal
``np.concatenate((A, B))`` byte for byte.
"""
import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, check_output_bytes


def concat_cases(app, *, seed=0xC047):
    """``CASES`` for the concat kernel whose harness module is ``app``."""

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        a = rng.uniform(-1.0, 1.0, size=(app.C_A, LANES)).astype(np.float32)
        b = rng.uniform(-1.0, 1.0, size=(app.C_B, LANES)).astype(np.float32)
        a_path, b_path = workspace / "a.bin", workspace / "b.bin"
        out, expected = workspace / "output.bin", workspace / "expected.bin"
        a_path.write_bytes(a.tobytes())
        b_path.write_bytes(b.tobytes())
        expected.write_bytes(np.concatenate((a, b)).tobytes())
        return PreparedCase(
            {"shape": (app.H, app.W, app.C_A, app.C_B)},
            {"input_a_path": a_path, "input_b_path": b_path, "output_path": out},
            lambda: check_output_bytes(out, expected),
        )

    return {"default": KernelCase(prepare, {"seed": seed}, 5_000_000)}
