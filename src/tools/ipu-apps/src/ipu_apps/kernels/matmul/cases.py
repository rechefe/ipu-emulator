"""Shared matmul cases: random FP32 operands checked against a NumPy reference.

A kernel's ``cases.py`` is one call, :func:`matmul_cases`, passing its app
module; the module's constants say which file layout it uses (see the family
``app.py``). Inputs are uniform in ``[-1, 1)``; the tolerance is the one the
kernels were validated at (IPU FP32 accumulation order differs from NumPy's).
"""
import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase

RTOL, ATOL = 1e-4, 1e-3


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def matmul_cases(app, *, activation="none", max_cycles=5_000_000):
    """``CASES`` for the matmul kernel whose harness module is ``app``.

    ``app`` defines ``K`` and either ``M`` and ``N`` (row-major layout) or
    ``N_OUT``, ``N_TOK`` and optionally ``N_TG`` (channel-major layout).
    """
    channel_major = hasattr(app, "N_TOK")
    k = app.K
    n = app.N_OUT if channel_major else app.N
    groups = getattr(app, "N_TG", 1)
    tokens = app.N_TOK * groups if channel_major else app.M

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        a = rng.uniform(-1.0, 1.0, size=(tokens, k)).astype(np.float32)
        w = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
        expected = a.astype(np.float64) @ w.T.astype(np.float64)
        if activation == "silu":
            expected = silu(expected)

        inp, weights, out = (workspace / name for name in ("input.bin", "weights.bin", "output.bin"))
        # Channel-major kernels read D = A^T, as the transformer blocks hold it.
        inp.write_bytes(np.ascontiguousarray(a.T if channel_major else a).tobytes())
        weights.write_bytes(w.tobytes())

        def check():
            raw = np.fromfile(out, dtype="<f4")
            if channel_major:
                want = groups * n * LANES
                if raw.size != want:
                    raise ValueError(f"output has {raw.size} FP32 values, expected {want}")
                # (N_TG, N, 128) rows; lanes past N_TOK are padding.
                got = raw.reshape(groups, n, LANES)[..., :app.N_TOK]
                got = got.transpose(0, 2, 1).reshape(tokens, n)
            else:
                if raw.size != tokens * n:
                    raise ValueError(f"output has {raw.size} FP32 values, expected {tokens * n}")
                got = raw.reshape(tokens, n)
            np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)

        params = {"shape_a": (tokens, k), "shape_b_t": (n, k), "activation": activation}
        return PreparedCase(params, {"input_path": inp, "weights_path": weights,
                                     "output_path": out}, check)

    return {
        "default": KernelCase(prepare, {"seed": 0}, max_cycles),
    }
