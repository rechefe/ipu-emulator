"""Matmul applications.

Sixteen kernels: 4 generic, layer-independent harnesses (``matmul_128x128``,
``matmul_128x64x128``, ``matmul_128x64x64``, ``matmul_64x64x64``) plus the 12
MobileViT L3/L4/L5 single-stream projection matmuls (OutProj/FFN1/FFN2/QKV,
suffixed ``_x128``). All compute ``C = A @ W^T``; the ``proj_*_p4`` family in
:mod:`ipu_apps.projections` wraps the identical arithmetic to loop all 4
pixel-streams in one invocation.
"""
