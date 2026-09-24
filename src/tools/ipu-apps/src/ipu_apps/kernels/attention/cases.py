"""Shared attention cases: FP32 staging, datapath-mirroring references, checks.

No checked-in goldens: every case generates FP32 inputs from a fixed seed and
computes the expected result directly. Two references mirror
the emulator's datapath exactly, because the two chains round differently and
must never share a golden:

* :func:`agg_fold` -- ``MULT.RC.VV`` + ``AGG.SUM[.FIRST]`` (query-major
  ``attn_v``): float32 lane products, a float64 left-fold per 128-lane key
  chunk rounded once to float32 on the R_ACC write, and each later chunk added
  in float32 to the already-rounded partial.
* :func:`acc_fold` -- ``MULT.RC.VE`` + ``ACC.ADD[.FIRST]`` (key-major
  ``attn_v_bcast``, and every scores kernel's contraction over channels): a
  per-lane float32 running sum rounded on every step.

Expected values are computed inside each case's ``check``, never in
``prepare``: routing-only callers (the registry tests) prepare every case.
"""
from __future__ import annotations

import numpy as np

from ipu_emu.emulator import run_test
from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry.cases import (
    KernelCase, PreparedCase, assemble_kernel, load_cases, run_case,
)
from ipu_apps.kernel_registry.registry import create_harness

MAX_CYCLES = 20_000_000


def uniform(rng: np.random.RandomState, shape) -> np.ndarray:
    """``rng.uniform(-1, 1, shape)`` as FP32 -- every attention test's inputs."""
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def pad_rows(x: np.ndarray, *, width: int = LANES, fill: float = 0.0) -> np.ndarray:
    """Place each row of 2-D ``x`` in the leading lanes of a ``width``-lane row.

    One channel (or query, or key) per whole row: rows are never shared, and
    the trailing ``width - x.shape[1]`` lanes hold ``fill``.
    """
    rows, n = x.shape
    out = np.full((rows, width), fill, dtype=np.float32)
    out[:, :n] = x
    return out


def agg_fold(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]`` through the AGG datapath.

    ``AGG.SUM.FIRST`` (ipu.py ``_agg_sum_lanes``) left-folds the active
    MULT_RES lanes starting from a Python float (float64) and rounds the total
    ONCE to float32 on the R_ACC write; the per-lane products themselves are
    float32 (MULT writes ``<f`` lanes). A second 128-key chunk (N_TOK=256) is a
    second float64 left-fold, added by ``AGG.SUM`` -- in float32 -- to the
    already-rounded first-chunk partial, then rounded once more. A plain
    ``np.einsum`` accumulates in one pass with different rounding and cannot
    discriminate a wrong cross-chunk carry from a right one.

    ``np.cumsum`` over float64 is a strict sequential left-fold, so this is the
    same computation as the scalar Python loop, vectorised.
    """
    prod = (P[:, None, :, :].astype(np.float32)
            * V[:, :, None, :].astype(np.float32)).astype(np.float32)  # [b, t, i, s]
    out = None
    for lo in range(0, prod.shape[-1], LANES):
        chunk = np.cumsum(prod[..., lo:lo + LANES].astype(np.float64), axis=-1)[..., -1]
        chunk = chunk.astype(np.float32)
        out = chunk if out is None else (
            chunk.astype(np.float64) + out.astype(np.float64)).astype(np.float32)
    return out


def acc_fold(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]`` through the ACC datapath.

    Each step forms float32 lane products (lane = i) and ``ACC.ADD`` writes the
    running sum back as float32 (``execute_acc_add`` packs ``<f`` each cycle);
    ``ACC.ADD.FIRST`` seeds it at ``s = 0``. One continuous float32 left-fold
    over ``s``, rounded at every step -- not a float64 accumulation and not
    AGG's single-rounding fold.
    """
    P = P.astype(np.float32)
    V = V.astype(np.float32)
    acc = None
    for s in range(P.shape[-1]):
        prod = (P[:, None, :, s] * V[:, :, None, s]).astype(np.float32)   # [b, t, i]
        acc = prod if acc is None else (acc + prod).astype(np.float32)
    return acc


def assert_close(got, expected, *, rtol, atol, what):
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, err_msg=f"{what} mismatch")


def read_fp32(path, count: int) -> np.ndarray:
    """An output file as FP32, refusing a size other than ``count`` elements."""
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != count:
        raise ValueError(f"output has {raw.size} FP32 values, expected {count}")
    return raw


def run_poked(kernel: str, workspace, poke, **options):
    """Run ``kernel``'s default case with ``poke(state)`` between setup and execution.

    Stages everything through the kernel's own harness, then lets a test
    overwrite XMEM (e.g. fill padding lanes with garbage a chained producer
    would leave there) before the program runs. Returns the halted state; the
    case's own check is not applied.
    """
    case = load_cases(kernel)["default"]
    prepared = case.prepare(workspace, **(dict(case.defaults) | options))
    inst = assemble_kernel(kernel, workspace)
    app = create_harness(kernel, params=prepared.params,
                         bindings={**prepared.bindings, "inst_path": inst})
    state = app.make_state()
    app.setup(state)
    poke(state)
    state, cycles = run_test(inst_path=inst, setup=lambda s: None, teardown=app.teardown,
                             max_cycles=case.max_cycles, state=state)
    if not state.is_halted or cycles <= 0:
        raise RuntimeError(f"{kernel} did not complete within {case.max_cycles} cycles")
    return state


# -- per-head layouts (L3 256x36 and L5 16x60) --------------------------------
#
# The L3 and L5 kernels share one file layout per op -- they differ only in the
# module constants the helpers below read -- so their cases are built here. The
# L4 (64x48) kernels batch 16 (stream, head) blocks and write their own cases.


def _paths(workspace, *names):
    return [workspace / f"{name}.bin" for name in names]


def qk_scores_cases(app, *, seed):
    """``CASES`` for a single-head query-major scores kernel (``app`` module).

    ``S[i, s] = sum_c Q[i, c] * K[s, c]`` against a float32 matmul; the output
    is query-major, row ``(i, g)`` holding keys ``[g*N_TPG, (g+1)*N_TPG)``.
    """
    N, D, N_TG, N_TPG = app.N, app.D, app.N_TG, app.N_TPG

    def prepare(workspace, *, seed):
        rng = np.random.RandomState(seed)
        # Inputs are channel-major: element [token t, channel c] at (c*N + t).
        Q = uniform(rng, (D, N))
        K = uniform(rng, (D, N))
        q_path, k_path, out = _paths(workspace, "q_fp32", "k_fp32", "output")
        q_path.write_bytes(Q.tobytes())
        k_path.write_bytes(K.tobytes())

        def check():
            expected = Q.T @ K                      # S[i, s] = sum_c Q[c, i] * K[c, s]
            # Query-major: one WHOLE row per (query, key group); the first
            # N_TPG lanes are live.
            got = read_fp32(out, N * N_TG * LANES).reshape(N, N_TG, LANES)
            for g in range(N_TG):
                lo = g * N_TPG
                assert_close(got[:, g, :N_TPG], expected[:, lo:lo + N_TPG],
                             rtol=1e-4, atol=1e-3, what=f"QK^T scores for key group {g}")

        return PreparedCase({"n_tok": N, "d": D},
                            {"query_path": q_path, "key_path": k_path, "output_path": out}, check)

    return {"default": KernelCase(prepare, {"seed": seed}, MAX_CYCLES)}


def attn_v_cases(app, *, seed, rtol, atol):
    """``CASES`` for a query-major P + AGG attn@V kernel (``app`` module).

    ``O[h, i, t] = sum_s P[h, i, s] * V[h, t, s]`` through :func:`agg_fold`.
    ``pad`` fills the unused lanes of every P and V row: AGG reduces across
    lanes, so the result must not depend on it (``valid_elements`` excludes
    padding structurally, not the harness's zero-fill).
    """
    N_TOK, D, N_HEAD, N_CHAN = app.N_TOK, app.D, app.N_HEAD, app.N_CHAN
    width = app.PV_STRIDE_ROWS * LANES
    out_width = app.O_CHAN_ROWS * LANES

    def prepare(workspace, *, seed, pad):
        rng = np.random.RandomState(seed)
        # P[h, i, s] -- attention probabilities; V[h, t, s] -- values, channel-major.
        P = uniform(rng, (N_HEAD, N_TOK, N_TOK))
        V = uniform(rng, (N_HEAD, D, N_TOK))
        p_path, v_path, out = _paths(workspace, "p_fp32", "v_fp32", "output")
        # P is staged QUERY-major (row i = all keys for query i), V channel-major.
        p_path.write_bytes(pad_rows(P.reshape(-1, N_TOK), width=width, fill=pad).tobytes())
        v_path.write_bytes(pad_rows(V.reshape(-1, N_TOK), width=width, fill=pad).tobytes())

        def check():
            # Channel (h*D + t) occupies O_CHAN_ROWS whole rows; query i is lane i.
            got = read_fp32(out, N_CHAN * out_width).reshape(N_CHAN, out_width)[:, :N_TOK]
            assert_close(got.reshape(N_HEAD, D, N_TOK), agg_fold(P, V),
                         rtol=rtol, atol=atol, what="attn@V (query-major + AGG)")

        return PreparedCase({"n_tok": N_TOK, "d": D},
                            {"p_path": p_path, "v_path": v_path, "output_path": out}, check)

    return {"default": KernelCase(prepare, {"seed": seed, "pad": 0.0}, MAX_CYCLES)}


def scores_km_cases(app, *, seed, head, reference, rtol, atol):
    """``CASES`` for a single-head key-major scores kernel (``app`` module).

    The Q/K files hold all ``N_HEADS`` heads channel-major; ``head`` selects
    one. The output is key-major, row ``(s, g)`` holding queries
    ``[g*N_TPG, (g+1)*N_TPG)``. ``reference`` is ``"acc"`` (the per-channel
    ACC.ADD float32 fold) or ``"matmul"`` (a float32 matmul).
    """
    N_TOK, D, N_TG, N_TPG, N_HEADS = app.N_TOK, app.D, app.N_TG, app.N_TPG, app.N_HEADS

    def prepare(workspace, *, seed, head):
        rng = np.random.RandomState(seed)
        # Canonical channel-major files: element [token t, channel h*D+c] at
        # (h*D + c)*N_TOK + t, for all N_HEADS heads.
        Q = uniform(rng, (N_HEADS * D, N_TOK))
        K = uniform(rng, (N_HEADS * D, N_TOK))
        q_path, k_path, out = _paths(workspace, "q_fp32", "k_fp32", "output")
        q_path.write_bytes(Q.tobytes())
        k_path.write_bytes(K.tobytes())

        def check():
            lo = head * D
            q_head, k_head = Q[lo:lo + D], K[lo:lo + D]        # [D, N_TOK]
            if reference == "acc":
                # S[s, i]: lane = query i, key s, fold over channels c.
                expected = acc_fold(q_head.T[None], k_head.T[None])[0]
            else:
                expected = (q_head.T @ k_head).T                # [key, query]
            got = read_fp32(out, N_TOK * N_TG * LANES).reshape(N_TOK, N_TG, LANES)
            for g in range(N_TG):
                lo_q = g * N_TPG
                assert_close(got[:, g, :N_TPG], expected[:, lo_q:lo_q + N_TPG],
                             rtol=rtol, atol=atol, what=f"key-major scores for query group {g}")

        return PreparedCase({"n_tok": N_TOK, "d": D, "head": head},
                            {"input_path": q_path, "weights_path": k_path, "output_path": out},
                            check)

    return {"default": KernelCase(prepare, {"seed": seed, "head": head}, MAX_CYCLES)}


def attn_v_bcast_cases(app, *, seed, rtol, atol):
    """``CASES`` for a key-major P + broadcast attn@V kernel (``app`` module).

    ``O[h, i, t] = sum_s P[h, i, s] * V[h, t, s]`` through :func:`acc_fold`.
    ``pad`` fills the unused lanes of every P and V row: lanes accumulate
    independently (no AGG), so the result must not depend on it.
    """
    N_TOK, D, N_HEAD, N_CHAN = app.N_TOK, app.D, app.N_HEAD, app.N_CHAN
    width = app.PV_STRIDE_ROWS * LANES
    out_width = app.O_CHAN_ROWS * LANES

    def prepare(workspace, *, seed, pad):
        rng = np.random.RandomState(seed)
        # P[h, i, s] -- attention probabilities; V[h, t, s] -- values, channel-major.
        P = uniform(rng, (N_HEAD, N_TOK, N_TOK))
        V = uniform(rng, (N_HEAD, D, N_TOK))
        p_path, v_path, out = _paths(workspace, "p_fp32", "v_fp32", "output")
        # P is staged KEY-major (row s = all queries for key s), V channel-major.
        p_path.write_bytes(pad_rows(P.transpose(0, 2, 1).reshape(-1, N_TOK),
                                    width=width, fill=pad).tobytes())
        v_path.write_bytes(pad_rows(V.reshape(-1, N_TOK), width=width, fill=pad).tobytes())

        def check():
            # Channel (h*D + t) occupies O_CHAN_ROWS whole rows; query i is lane i.
            got = read_fp32(out, N_CHAN * out_width).reshape(N_CHAN, out_width)[:, :N_TOK]
            assert_close(got.reshape(N_HEAD, D, N_TOK), acc_fold(P, V),
                         rtol=rtol, atol=atol, what="attn@V (key-major + broadcast)")

        return PreparedCase({"n_tok": N_TOK, "d": D},
                            {"p_path": p_path, "v_path": v_path, "output_path": out}, check)

    return {"default": KernelCase(prepare, {"seed": seed, "pad": 0.0}, MAX_CYCLES)}


def padded_runs(kernel: str, workspace, pads) -> list[np.ndarray]:
    """Run ``kernel``'s default case once per ``pad`` value; return each output.

    Each run is also checked against the case's reference, so garbage padding
    must leave the result correct, not merely unchanged.
    """
    case = load_cases(kernel)["default"]
    outputs = []
    for i, pad in enumerate(pads):
        out = workspace / f"output_{i}.bin"
        run_case(kernel, case, options={"pad": pad}, output_path=out, workspace=workspace / f"run_{i}")
        outputs.append(np.fromfile(out, dtype="<f4"))
    return outputs
