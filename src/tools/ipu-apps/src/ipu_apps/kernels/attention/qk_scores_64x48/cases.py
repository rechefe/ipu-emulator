"""Runtime cases for qk_scores_64x48 (Layer 4): S[b] = Q[b]^T K[b] per block.

S[b] = Q[b].T @ K[b] is computed directly for each of the 16 (stream, head)
blocks of Layer 4. This is the QUERY-major score kernel; it pairs with
``attn_v_64x48``. The key-major variant ``attn_scores_km_64x48`` produces
bit-different results by design and has its own golden -- the two must never
share one.
"""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernels.attention.cases import MAX_CYCLES, assert_close, read_fp32, uniform
from ipu_apps.kernels.attention.qk_scores_64x48.app import D, N, N_BLOCK, N_TG, N_TPG


def prepare(workspace, *, seed):
    rng = np.random.RandomState(seed)
    # Inputs are channel-major per block: [block b, token t, channel c]
    # at element (b*D + c)*N + t.
    Q = uniform(rng, (N_BLOCK, D, N))
    K = uniform(rng, (N_BLOCK, D, N))
    q_path, k_path, out = (workspace / f"{n}.bin" for n in ("q_fp32", "k_fp32", "output"))
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    def check():
        # S[b, i, s] = sum_c Q[b, c, i] * K[b, c, s]
        expected = np.einsum("bci,bcs->bis", Q, K)              # [N_BLOCK, N, N]
        # The teardown crops each whole stored row to its N valid scores.
        got = read_fp32(out, N_BLOCK * N * N_TG * N_TPG).reshape(N_BLOCK, N, N_TPG)
        assert_close(got, expected, rtol=1e-4, atol=1e-3, what="QK^T query-major score")

    return PreparedCase({"n_tok": N, "d": D},
                        {"query_path": q_path, "key_path": k_path, "output_path": out}, check)


CASES = {"default": KernelCase(prepare, {"seed": 0x4C0}, MAX_CYCLES)}
