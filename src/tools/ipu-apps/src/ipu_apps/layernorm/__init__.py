"""LayerNorm applications.

Four kernels: one per MobileViT layer (L3/L4/L5), plus ``layernorm_128x16``,
a 16-channel ancestor kernel not tied to any of the three layer shapes.
"""
