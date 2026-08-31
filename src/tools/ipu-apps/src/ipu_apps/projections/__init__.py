"""Multi-stream projection applications.

Twelve kernels: the all-stream (P=4) counterparts of the single-stream
``matmul_*_x128`` projection matmuls in :mod:`ipu_apps.matmuls`, one set per
MobileViT layer (L3/L4/L5) and role (``qkv``, ``outproj``, ``ffn1``,
``ffn2``). Each loops all 4 pixel-streams internally, sharing one weight
matrix across streams, instead of one host round-trip per stream.
"""
