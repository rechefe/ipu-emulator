"""Runtime cases for attn_v_16x60 (Layer 5): O = P V through the AGG datapath.

This is the query-major P + AGG variant of attn@V, the second half of the
QUERY-MAJOR chain (``qk_scores_16x60`` -> ``attn_v_16x60``). ``attn_v_bcast_60``
is the key-major broadcast kernel; it shares V's and O's layouts but is a
DIFFERENT mapping with its own golden -- the two chains are bit-different by
design and must never share expectations.
"""
from ipu_apps.kernels.attention.attn_v_16x60 import app
from ipu_apps.kernels.attention.cases import attn_v_cases

CASES = attn_v_cases(app, seed=0xA60, rtol=1e-4, atol=1e-3)
