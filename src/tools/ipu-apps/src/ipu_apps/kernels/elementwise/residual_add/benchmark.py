"""Benchmark configs for residual_add (``bazel run :benchmark_residual_add``):
the MobileViT-S residual channel counts."""

CONFIGS = [dict(num_channels=n) for n in (64, 96, 128, 160)]
PER = "num_channels"
