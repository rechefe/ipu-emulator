"""Runtime cases for attn_v_bcast_36: O = P V through the broadcast ACC datapath.

This is the key-major P variant of attn@V; ``attn_v_256x36`` is the
query-major + AGG kernel and shares V's and O's layouts.

There is no AGG in this kernel: the contraction over all 256 keys is a single
continuous ACC.ADD (ACC.ADD.FIRST at s=0) per-lane float32 running sum,
rounded on every step, with no group split or reset (the 128/128 R0/R1 split
is only how V's scalar source is staged). The reference (``acc_fold``) mirrors
that per-step fold exactly, so it agrees with the kernel's output exactly
(rtol = atol = 0), not merely within a loose tolerance.
"""
from ipu_apps.kernels.attention.attn_v_bcast_36 import app
from ipu_apps.kernels.attention.cases import attn_v_bcast_cases

CASES = attn_v_bcast_cases(app, seed=0xA17, rtol=0, atol=0)
