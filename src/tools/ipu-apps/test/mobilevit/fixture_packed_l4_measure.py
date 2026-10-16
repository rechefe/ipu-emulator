"""Measurement helpers for the L4 packed-layout kernel family.

Standalone test support (not a shipped ipu_apps kernel module) -- uses the
direct assemble_to_bin_file + IpuState pattern. Builds only the packed
kernels; never imports or runs layernorm_64x192, qk_scores_64x48,
attn_v_64x48, attn_scores_km_64x48, attn_v_bcast_48, residual_add_64x192,
proj_*_p4, or any softmax kernel.

Its instruction counting is the L5 fixture's (fixture_packed_l5_measure.py):
this part of the harness has zero L4-specific content.
"""

from fixture_packed_l5_measure import InstrCount, count_instructions

__all__ = ["InstrCount", "count_instructions"]
