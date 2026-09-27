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

`conv_first_layer` covers only `3x256x256 -> 16x128x128`, stride 2, bias+ReLU.

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

## Stride-2 depthwise

Stage 1 runs `depthwise_conv_universal` at full resolution; stage 2 decimates
with `ACC.STRIDE`. `depthwise_conv_stride2_16` packs two channels per output
chunk.

## `residual_add`

Elementwise `C = A + B` over `[channels, height, width]` FP32, one channel per
128-element chunk. Queried by `num_channels` (at most 1024):

```bash
bazel run //src/tools/ipu-apps:query -- residual_add num_channels=64
```

## Data layout

`input_path` / `output_path`: raw `[channels, height, width]` float32.
`kernel_path`: `[out, in / groups, k, k]` float32. `bias_path` (bias kernels):
`[out_channels]` float32.
