"""Shared elementwise cases: random FP32 residual-add operands checked against A + B.

A residual-add kernel's ``cases.py`` is one call, :func:`residual_add_cases`,
passing its app module and its input recipe. The module's
``N_ROWS``, ``N_TOK`` and ``N_CH`` constants give the geometry.

File layout (what every residual-add harness reads): A and B are ``N_ROWS``
FP32 rows of ``LANES`` lanes each, the first ``valid_lanes`` of every row
holding data and the rest zero padding (one channel per row, except for
``residual_add_256x144``, whose rows are all ``LANES`` lanes of data). The
output is the same rows, cropped to ``valid_lanes`` by the harness when
``cropped_output`` is set.
"""
import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase


def residual_add_cases(app, *, seed, valid_lanes, rtol, atol, cropped_output=False,
                       max_cycles=5_000_000):
    """``CASES`` for the residual-add kernel whose harness module is ``app``.

    Args:
        seed:           Default ``RandomState`` seed.
        valid_lanes:    Data lanes per row; the rest is zero padding.
        rtol, atol:     Tolerance the kernel was validated at.
        cropped_output: The harness crops each output row to ``valid_lanes``.
    """
    rows = app.N_ROWS

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        a = np.zeros((rows, LANES), dtype=np.float32)
        b = np.zeros((rows, LANES), dtype=np.float32)
        a[:, :valid_lanes] = rng.uniform(-1.0, 1.0, size=(rows, valid_lanes))
        b[:, :valid_lanes] = rng.uniform(-1.0, 1.0, size=(rows, valid_lanes))
        expected = (a + b)[:, :valid_lanes]

        a_path, b_path, out = (workspace / name for name in
                               ("a_fp32.bin", "b_fp32.bin", "output.bin"))
        a_path.write_bytes(a.tobytes())
        b_path.write_bytes(b.tobytes())

        def check():
            raw = np.fromfile(out, dtype="<f4")
            lanes = valid_lanes if cropped_output else LANES
            if raw.size != rows * lanes:
                raise ValueError(f"output has {raw.size} FP32 values, expected {rows * lanes}")
            got = raw.reshape(rows, lanes)[:, :valid_lanes]
            np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol,
                                       err_msg="residual add output does not match A + B")

        params = {"shape": (app.N_TOK, app.N_CH)}
        return PreparedCase(params, {"input_a_path": a_path, "input_b_path": b_path,
                                     "output_path": out}, check)

    return {"default": KernelCase(prepare, {"seed": seed}, max_cycles)}
