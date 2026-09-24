"""Runtime cases for qk_scores_16x60 (Layer 5): S = Q^T K against NumPy.

This is the QUERY-MAJOR score kernel; it pairs with ``attn_v_16x60``. The
key-major chain (``attn_scores_km_16x60`` + ``attn_v_bcast_60``) computes the
same mathematical scores through a different mapping and has its OWN golden --
the two are bit-different by design and must never share expectations.
"""
from ipu_apps.kernels.attention.cases import qk_scores_cases
from ipu_apps.kernels.attention.qk_scores_16x60 import app

CASES = qk_scores_cases(app, seed=0x5C0)
