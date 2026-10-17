# Convolutions

## Routing

For `has_bias=False, apply_relu=False`. With `has_bias=True, apply_relu=True`
the same ranges route to the `_bn_activation` kernels (`conv_universal`,
`depthwise_conv_universal` and `pointwise_conv_unified` only).

```bash
bazel run //src/tools/ipu-apps:query -- conv2d in_channels=8 out_channels=8 \
    kernel_size=3 stride=1 padding=1 dilation=1 groups=1 \
    has_bias=False apply_relu=False height=64 width=w --sweep w=1..400
```

**Standard conv** (`groups=1, kernel_size=3, stride=1`), by width, height=64:

```
w 1..128                     conv_universal
w 129..383                   (not covered)
w = 384                      conv_universal_wide384
w 385..400                   (not covered)
```

`conv_universal` and `conv_universal_bn_activation` run as
`//kernels/convolutions:conv_universal` and
`//kernels/convolutions:conv_universal_bn_activation`, two entries of
`kernels/convolutions/registry.toml` over one program,
`kernels/convolutions/conv_universal_tpl`, and answer `bazel run //kernels:query`
with the same eleven parameters, every one required. The `_bn_activation`
entry takes `apply_relu=True` with or without a bias (a zero one);
`has_bias=True, apply_relu=False` has no kernel.

The pointwise and depthwise twins (`pointwise_conv_unified`,
`pointwise_conv_unified_bn_activation`, `depthwise_conv_universal`,
`depthwise_conv_universal_bn_activation`) are entries of the same
`registry.toml` over two programs, `kernels/convolutions/pointwise_conv_tpl` and
`kernels/convolutions/depthwise_conv_tpl`, and run as
`//kernels/convolutions:<name>`. `pointwise_conv_unified_bn_activation` takes
`apply_relu=True` with or without a bias (a zero one);
`depthwise_conv_universal_bn_activation` requires both the bias and ReLU.

`conv_first_layer` covers only `3x256x256 -> 16x128x128`, stride 2, bias+ReLU.
It runs as `//kernels/convolutions/conv_first_layer` and answers
`bazel run //kernels:query -- conv2d ...` with the same eleven parameters.

**Pointwise** (`groups=1, kernel_size=1`), by width, height=64:

```
w 1..128                     pointwise_conv_unified
w 129..140                   (not covered)
```

`in_channels` must be a multiple of 8, `out_channels` a multiple of 4.

**Depthwise, stride=1** (`groups=in_channels, kernel_size=3`), by width, height=64:

```
w 1..128                     depthwise_conv_universal
w 129..140                   (not covered)
```

**Depthwise, stride=2** (`groups=in_channels, kernel_size=3`), by width, height=128:

```
w 1..15                      (not covered)
w = 16                       depthwise_conv_stride2_narrow
w 17..31                     (not covered)
w = 32                       depthwise_conv_stride2_narrow
w 33..63                     (not covered)
w = 64                       depthwise_conv_stride2_narrow
w 65..127                    (not covered)
w = 128                      depthwise_conv_stride2_128
w 129..140                   (not covered)
```

**Depthwise, stride=2, width=16**, by height:

```
h 1..3                       (not covered)
h = 4                        depthwise_conv_stride2_narrow
h 5..15                      (not covered)
h = 16                       depthwise_conv_stride2_16
h 17..31                     (not covered)
h = 32                       depthwise_conv_stride2_narrow
h 33..35                     (not covered)
h = 36                       depthwise_conv_stride2_narrow
h 37..63                     (not covered)
h = 64                       depthwise_conv_stride2_narrow
```

`kernel_size` is 1 or 3, `groups` is 1 or `in_channels`, `dilation` is 1.

## Tiled kernels

`conv1x1`, `conv3x3_relu` and `conv3x3_relu_cin1` (`//kernels/convolutions/<name>`)
take their own query vocabulary: `in_channels`, `height`, `width`,
`out_channels`, `kernel_size`, `stride`, `padding` and `activation`, every one
required. A universal query (it names `dilation`, `groups`, `has_bias` and
`apply_relu`) never reaches them, and theirs never reaches a universal kernel.

```bash
bazel run //kernels:query -- conv2d in_channels=1 height=64 width=64 \
    out_channels=64 kernel_size=3 stride=1 padding=1 activation=relu
```

| Kernel | `kernel_size` | `padding` | `activation` | Note |
|---|---|---|---|---|
| `conv1x1` | 1 | 0 | `none` | bias added, no ReLU |
| `conv3x3_relu` | 3 | 1 | `relu` | |
| `conv3x3_relu_cin1` | 3 | 1 | `relu` | `in_channels = 1` only; preferred there (cost 0 against 1) |

Each takes `x` (`[in_channels, height, width]`), `w` (`[out_channels,
in_channels, k, k]`) and `b` (`[out_channels]`), and writes `y`
(`[out_channels, height, width]`), all FP32. The program's tiled layout (one
row per channel, padded row and column tile; the 3x3 kernels' tiles carry a
one-column halo) is the family's pack transforms (`kernels/convolutions/pack.py`).

## Stride-2 depthwise

Stage 1 runs `depthwise_conv_universal` at full resolution; stage 2 decimates
with `ACC.STRIDE`. `depthwise_conv_stride2_16` packs two channels per output
chunk.

## `residual_add`

Elementwise `y = a + b` over `[channels, height, width]` FP32, one channel per
128-element row: `//kernels/elementwise/residual_add` (E07-06). Queried by
`num_channels` (at most 1024):

```bash
bazel run //kernels:query -- residual_add num_channels=64
bazel run //kernels/elementwise/residual_add -- --benchmark
```

## Data layout

`input_path` / `output_path`: raw `[channels, height, width]` float32.
`kernel_path`: `[out, in / groups, k, k]` float32. `bias_path` (bias kernels):
`[out_channels]` float32.
