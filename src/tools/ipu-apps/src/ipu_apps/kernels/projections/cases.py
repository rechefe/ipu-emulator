"""Shared projection cases: random FP32 streams checked against a NumPy reference.

A kernel's ``cases.py`` is one call, :func:`projection_cases`, passing its app
module (which defines ``ACTIVATION`` and ``SPEC``). Inputs are uniform in
``[-1, 1)``, generated per stream then W from a fixed seed; the reference is
computed per (stream, token group) in FP32, ``C[p][tg] = act(W @ D[p][tg])``,
and compared at a tolerance that allows for IPU FP32 accumulation order
differing from NumPy's.
"""
from __future__ import annotations

import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.projections.app import N_STREAM, ProjectionLayout

RTOL, ATOL = 1e-4, 1e-3
SEED = 0xC0FFEE


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))       # silu = x * sigmoid(x)


def reference(d, w, activation):
    """``act(W @ D[p][tg])`` for every stream and token group, in FP32.

    ``d`` is ``(N_STREAM, N_TG, K, N_TOK)``; the result ``(N_STREAM, N_TG, N_OUT, N_TOK)``.
    """
    pre_act = np.matmul(w, d)                   # C[p][tg][j, t] = sum_k W[j,k] * D[p][tg][k,t]
    return silu(pre_act) if activation == "silu" else pre_act


def read_output(path, layout: ProjectionLayout):
    """The output file's valid lanes, ``(N_STREAM, N_TG, N_OUT, N_TOK)``; raise on a size mismatch."""
    raw = np.fromfile(path, dtype="<f4")
    want = int(np.prod(layout.output_shape))
    if raw.size != want:
        raise ValueError(f"output has {raw.size} FP32 values, expected {want}")
    return raw.reshape(layout.output_shape)[..., :layout.n_tok]


def projection_case(workspace, layout: ProjectionLayout, d, w, activation, check=None):
    """Write ``d`` (``(N_STREAM, N_TG, K, N_TOK)``) and ``w`` (``(N_OUT, K)``).

    The returned case's check compares the output with :func:`reference`, or,
    if given, calls ``check(got)`` with the output's valid lanes instead.
    """
    d = np.ascontiguousarray(d, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    if d.shape != layout.input_shape:
        raise ValueError(f"D must be {layout.input_shape}; got {d.shape}")
    if w.shape != (layout.n_out, layout.k):
        raise ValueError(f"W must be {(layout.n_out, layout.k)}; got {w.shape}")
    inp, weights, out = (workspace / name for name in ("input.bin", "weights.bin", "output.bin"))
    inp.write_bytes(d.tobytes())
    weights.write_bytes(w.tobytes())

    def compare(got):
        expected = reference(d, w, activation)
        for p in range(N_STREAM):
            for tg in range(layout.n_tg):
                np.testing.assert_allclose(
                    got[p, tg], expected[p, tg], rtol=RTOL, atol=ATOL,
                    err_msg=f"stream {p} tg {tg} output mismatch",
                )

    params = {"k": layout.k, "n_out": layout.n_out, "n_streams": N_STREAM,
              "activation": activation}
    return PreparedCase(
        params,
        {"input_path": inp, "weights_path": weights, "output_path": out},
        lambda: (check or compare)(read_output(out, layout)),
    )


def projection_cases(app, *, max_cycles):
    """``CASES`` for the projection kernel whose harness module is ``app``."""
    layout = app.SPEC.app_class.layout
    activation = app.ACTIVATION

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        d = np.stack([
            rng.uniform(-1.0, 1.0, size=(layout.n_tg, layout.k, layout.n_tok)).astype(np.float32)
            for _ in range(N_STREAM)
        ])
        w = rng.uniform(-1.0, 1.0, size=(layout.n_out, layout.k)).astype(np.float32)
        return projection_case(workspace, layout, d, w, activation)

    return {
        "default": KernelCase(prepare, {"seed": SEED}, max_cycles),
    }
