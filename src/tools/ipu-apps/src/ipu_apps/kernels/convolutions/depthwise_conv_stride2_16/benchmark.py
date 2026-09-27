"""Benchmark configs for depthwise_conv_stride2_16
(``bazel run :benchmark_depthwise_conv_stride2_16``): channel-count sweep."""

CONFIGS = [
    dict(channels=2),     # minimal case
    dict(channels=16),    # small channel count
    dict(channels=320),   # MobileViT-S's actual stage-5 shape
]
PER = "channels"
MAX_CYCLES = 50_000_000
