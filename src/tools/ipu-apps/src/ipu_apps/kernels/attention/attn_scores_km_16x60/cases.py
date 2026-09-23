"""Runtime cases for attn_scores_km_16x60 (Layer 5): key-major scores, one head.

Scores are stored KEY-major (key s's score column occupies one WHOLE row, one
query per lane), which is the distinguishing property of this kernel vs the
query-major ``qk_scores_16x60``. This is the first half of the KEY-MAJOR chain
(``attn_scores_km_16x60`` -> ``attn_v_bcast_60``); the query-major chain is
bit-different BY DESIGN and the two must never share expectations.

Contraction is ACC.ADD over 60 channels (no AGG), which rounds to float32 at
every step, so a float32 matmul reference is compared at rtol=1e-4/atol=1e-3.
Head 1 exercises the head slicing.
"""
from ipu_apps.kernels.attention.attn_scores_km_16x60 import app
from ipu_apps.kernels.attention.cases import scores_km_cases

CASES = scores_km_cases(app, seed=0xD60, head=1, reference="matmul", rtol=1e-4, atol=1e-3)
