"""Runtime cases for attn_scores_km_256x36: key-major scores for one head.

Scores are stored KEY-major (each key column contiguous), which is the
distinguishing property of this kernel vs the query-major ``qk_scores_256x36``.

L3 is the only shape with TWO query groups (N_TOK=256 spans two 128-lane
groups g=0,1), so it is the only place this kernel's cross-group addressing is
exercised. The reference mirrors the emulator's datapath rather than a plain
matmul: each channel step is ``MULT.RC.VE`` (lane = query) followed by
``ACC.ADD[.FIRST]``, a per-lane float32 running sum rounded on every one of the
D channel steps; each query group starts its own fresh R_ACC. Head 1 exercises
the head slicing.
"""
from ipu_apps.kernels.attention.attn_scores_km_256x36 import app
from ipu_apps.kernels.attention.cases import scores_km_cases

CASES = scores_km_cases(app, seed=0xD00, head=1, reference="acc", rtol=1e-6, atol=1e-6)
