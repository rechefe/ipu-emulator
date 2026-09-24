"""Benchmark configs for conv_universal_wide384
(``bazel run :benchmark_conv_universal_wide384``)."""

CONFIGS = [
    dict(width=384, height=8, in_channels=1, out_channels=2),    # minimal cpr=3 case
    dict(width=384, height=16, in_channels=3, out_channels=4),   # moderate spatial + channels
    dict(width=512, height=8, in_channels=2, out_channels=4),    # cpr=4
    dict(width=384, height=8, in_channels=16, out_channels=4),   # in_channels > FPB=14: reload path
    dict(width=640, height=4, in_channels=1, out_channels=2),    # cpr=5
]
MAX_CYCLES = 50_000_000
