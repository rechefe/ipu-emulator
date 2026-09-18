"""Attention applications.

Twelve kernels implementing two independent QKT/attn@V chains, ported across
all three MobileViT layers (L3/L4/L5):

- **query-major:** ``qk_scores_*`` -> ``attn_v_*`` (attn@V via AGG)
- **key-major:**   ``attn_scores_km_*`` -> ``attn_v_bcast_*`` (attn@V via
  broadcast ACC.ADD, no AGG)

The two chains produce bit-different results by design and must never be
mixed -- see ``kernel_docs/kernel_layer_map.md``'s "Two attention mappings"
section.
"""
