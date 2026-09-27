"""Runtime cases for attn_v_bcast_60 (Layer 5): O = P V, broadcast ACC datapath.

This is the key-major P + BROADCAST variant of attn@V, the second half of the
KEY-MAJOR chain (``attn_scores_km_16x60`` -> ``attn_v_bcast_60``). The
query-major sibling ``attn_v_16x60`` computes the same mathematical product
with a DIFFERENT datapath (MULT.RC.VV + AGG.SUM); the two chains are
bit-different BY DESIGN and must never share expectations, so the reference is
the ACC.ADD float32 fold (``acc_fold``), not the AGG float64 fold.
"""
from ipu_apps.kernels.attention.attn_v_bcast_60 import app
from ipu_apps.kernels.attention.cases import attn_v_bcast_cases

CASES = attn_v_bcast_cases(app, seed=0xB60, rtol=1e-4, atol=1e-3)
