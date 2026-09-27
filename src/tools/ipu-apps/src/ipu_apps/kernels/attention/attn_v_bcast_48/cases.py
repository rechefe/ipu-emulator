"""Runtime cases for attn_v_bcast_48 (Layer 4): O[b] = P[b] V[b], broadcast ACC.

This is the KEY-major P + broadcast variant of attn@V; it pairs with
``attn_scores_km_64x48``. ``attn_v_64x48`` is the query-major + AGG kernel and
shares V's and O's layouts, but produces bit-different results by design and
carries its own golden -- the two must never share one.

This kernel uses **no AGG**: ``MULT.RC.VE`` broadcasts the scalar V[s,t] over
the key-major score row and ``ACC.ADD[.FIRST]`` runs a per-lane float32
accumulation over the 64 keys, so the golden is a plain float32 running sum
(``acc_fold``) -- NOT the float64 left-fold that the AGG sibling's golden needs.
``pad`` fills the unused 64 lanes of every P and V row (see ``test.py``).
"""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.attention.attn_v_bcast_48.app import D, N_BLOCK, N_CHAN, N_TOK
from ipu_apps.kernels.attention.cases import (
    MAX_CYCLES, acc_fold, assert_close, pad_rows, read_fp32, uniform,
)


def prepare(workspace, *, seed, pad):
    rng = np.random.RandomState(seed)
    # P[b, i, s] -- attention probabilities; V[b, t, s] -- values (channel-major).
    P = uniform(rng, (N_BLOCK, N_TOK, N_TOK))
    V = uniform(rng, (N_BLOCK, D, N_TOK))
    p_path, v_path, out = (workspace / f"{n}.bin" for n in ("p_fp32", "v_fp32", "output"))
    # P is staged KEY-major: one whole row per key, holding that key's 64 query
    # scores in the leading lanes -- exactly attn_scores_km_64x48's output.
    p_path.write_bytes(pad_rows(P.transpose(0, 2, 1).reshape(-1, N_TOK), fill=pad).tobytes())
    # V is channel-major: one whole row per value channel, keys leading.
    # Identical to attn_v_64x48's V layout.
    v_path.write_bytes(pad_rows(V.reshape(-1, N_TOK), fill=pad).tobytes())

    def check():
        # Output: channel (b*D + t) is one cropped row of N_TOK FP32 queries --
        # the same shape attn_v_64x48 emits.
        got = read_fp32(out, N_CHAN * N_TOK).reshape(N_BLOCK, D, N_TOK)
        assert_close(got, acc_fold(P, V), rtol=1e-4, atol=1e-3,
                     what="attn@V (key-major + broadcast)")

    return PreparedCase({"n_tok": N_TOK, "d": D},
                        {"p_path": p_path, "v_path": v_path, "output_path": out}, check)


CASES = {"default": KernelCase(prepare, {"seed": 0xB48, "pad": 0.0}, MAX_CYCLES)}
