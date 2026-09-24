"""Runtime cases for attn_v_64x48 (Layer 4): O[b] = P[b] V[b] through AGG.

This is the QUERY-major P + AGG variant of attn@V; it pairs with
``qk_scores_64x48``. ``attn_v_bcast_48`` is the key-major broadcast kernel and
shares V's and O's layouts, but produces bit-different results by design and
carries its own golden -- the two must never share one.

The golden mirrors AGG's datapath rather than calling einsum (``agg_fold``):
the emulator accumulates the MULT_RES lanes as a float64 left-fold and rounds
the sum to float32 exactly once, on the R_ACC write. ``pad`` fills the unused
64 lanes of every P and V row (see ``test.py``).
"""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.attention.attn_v_64x48.app import D, N_BLOCK, N_CHAN, N_TOK
from ipu_apps.kernels.attention.cases import (
    MAX_CYCLES, agg_fold, assert_close, pad_rows, read_fp32, uniform,
)


def prepare(workspace, *, seed, pad):
    rng = np.random.RandomState(seed)
    # P[b, i, s] -- attention probabilities (query-major); V[b, t, s] -- values.
    P = uniform(rng, (N_BLOCK, N_TOK, N_TOK))
    V = uniform(rng, (N_BLOCK, D, N_TOK))
    p_path, v_path, out = (workspace / f"{n}.bin" for n in ("p_fp32", "v_fp32", "output"))
    # P is staged QUERY-major: one whole row per query, keys in leading lanes.
    p_path.write_bytes(pad_rows(P.reshape(-1, N_TOK), fill=pad).tobytes())
    # V is channel-major: one whole row per value channel, keys leading.
    v_path.write_bytes(pad_rows(V.reshape(-1, N_TOK), fill=pad).tobytes())

    def check():
        # Output: channel (b*D + t) is one cropped row of N_TOK FP32 queries.
        got = read_fp32(out, N_CHAN * N_TOK).reshape(N_BLOCK, D, N_TOK)
        assert_close(got, agg_fold(P, V), rtol=1e-4, atol=1e-3,
                     what="attn@V (query-major + AGG)")

    return PreparedCase({"n_tok": N_TOK, "d": D},
                        {"p_path": p_path, "v_path": v_path, "output_path": out}, check)


CASES = {"default": KernelCase(prepare, {"seed": 0xA48, "pad": 0.0}, MAX_CYCLES)}
