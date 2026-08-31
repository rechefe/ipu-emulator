"""Unfold (space-to-depth) applications.

Three kernels, one per MobileViT layer (L3/L4/L5), each splitting a spatial
feature map into 4 stride-2 phase streams ahead of the transformer stack --
see :mod:`ipu_apps.unfold.unfold_32x32x144` for the derivation shared by all
three.
"""
