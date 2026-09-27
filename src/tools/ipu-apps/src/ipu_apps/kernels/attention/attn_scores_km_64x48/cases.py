"""Runtime cases for attn_scores_km_64x48 (Layer 4): key-major scores, one head.

S[p, i, s] = sum_c Q[p, i, c] * K[p, s, c] for the selected head across all
P = 4 streams. Scores are stored KEY-major (each key owns a row holding its 64
query scores), which is the distinguishing property of this kernel versus the
query-major ``qk_scores_64x48``. This kernel is the head of the KEY-MAJOR chain
and pairs only with ``attn_v_bcast_48``; the two chains produce bit-different
results by design and never share a golden.

The producer emits full 512 B rows (producers write full rows, the final consumer crops) -- attn_v_bcast_48 stages this output verbatim as whole
rows, so the check crops to the valid N lanes itself. ``head`` selects which
head is scored (the default exercises a non-zero one).
"""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.attention.attn_scores_km_64x48.app import D, LANES, N, N_HEAD, P
from ipu_apps.kernels.attention.cases import MAX_CYCLES, assert_close, read_fp32, uniform


def prepare(workspace, *, seed, head):
    rng = np.random.RandomState(seed)
    # Canonical channel-major files over all (stream, head) blocks: element
    # [stream p, head h, token t, channel c] at ((p*N_HEAD + h)*D + c)*N + t.
    Q = uniform(rng, (P, N_HEAD, D, N))
    K = uniform(rng, (P, N_HEAD, D, N))
    q_path, k_path, out = (workspace / f"{n}.bin" for n in ("q_fp32", "k_fp32", "output"))
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    def check():
        # The selected head, per stream: S[p] = Q[p].T @ K[p] with the channel
        # axis contracted -> [queries, keys], then transposed to key-major.
        expected = np.einsum("pci,pcs->psi", Q[:, head], K[:, head])   # [P, key, query]
        got = read_fp32(out, P * N * LANES).reshape(P, N, LANES)[:, :, :N]
        assert_close(got, expected, rtol=1e-4, atol=1e-3,
                     what=f"key-major score (head {head})")

    return PreparedCase({"n_tok": N, "d": D, "head": head},
                        {"input_path": q_path, "weights_path": k_path, "output_path": out},
                        check)


CASES = {"default": KernelCase(prepare, {"seed": 0xD48, "head": 2}, MAX_CYCLES)}
