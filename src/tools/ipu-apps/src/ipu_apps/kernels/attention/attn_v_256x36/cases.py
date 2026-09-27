"""Runtime cases for attn_v_256x36: O = P V through the two-chunk AGG datapath.

This is the query-major P + AGG variant of attn@V; ``attn_v_bcast_36`` is the
key-major broadcast kernel and shares V's and O's layouts.

L3 is the only shape with TWO key chunks (N_TOK=256 spans two 128-lane
groups), so it is the only place cross-chunk AGG accumulation is exercised.
The reference (``agg_fold``) therefore mirrors AGG's actual datapath rather
than calling ``np.einsum``, and is compared at rtol = atol = 1e-6.
"""
from ipu_apps.kernels.attention.attn_v_256x36 import app
from ipu_apps.kernels.attention.cases import attn_v_cases

CASES = attn_v_cases(app, seed=0xA18, rtol=1e-6, atol=1e-6)
