# Kernels

A **kernel** is one assembly program plus the Python harness that feeds it: an
`.asm` file, an `IpuApp` subclass that lays out XMEM and reads results back, and
a declaration saying which computations it handles.

This section covers what exists, how the emulator knows what exists, and how to
add more.

## The pages

| Page | What it answers |
|---|---|
| [Softmax](softmax.md) | Which softmax kernels exist, which shape each handles, and what they cost |
| [Convolutions](convolutions.md) | Which conv2d / CHW residual_add kernels exist, which shape each handles, and what they cost |
| [Linear layers](linear-layers.md) | Matmul and multi-stream projection kernels for MobileViT-S's transformer blocks |
| [Normalization and shaping](normalization-and-shaping.md) | LayerNorm, residual add, and spatial unfold / fold / concat kernels |
| [Attention](attention.md) | The two non-interchangeable QKᵀ / attn·V mappings and their twelve kernels |
| [MobileViT-S](mobilevit.md) | Kernels per MobileViT-S transformer layer, and chaining rules |
| [Application coverage](../app-coverage.md) | How the emulator knows which kernel implements a computation |
| [Adding applications](../adding-applications.md) | How to contribute a kernel |

## Finding a kernel for a computation

Rather than reading each kernel's docstring, ask the registry. It answers from
the kernels themselves, so it cannot drift out of date:

```bash
bazel run //src/tools/ipu-apps:query -- softmax shape=32,300 dim=1
bazel run //src/tools/ipu-apps:query                                 # every op and kernel
```

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

# PyTorch-native: the layer plus the shape it will receive
lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))

# framework-free
resolve("softmax", shape=(32, 300), dim=1)
```

Both return a verdict carrying the app class, its constructor arguments, why
that kernel was chosen, and any caveats that still apply — or, when nothing
covers the shape, what each candidate objected to.

## Currently registered

89 kernels across 21 operations; `bazel run //src/tools/ipu-apps:query` prints
the current list.

| Operation | Kernels |
|---|---|
| `softmax` | 5 — see [Softmax](softmax.md) |
| `conv2d` | 14 — see [Convolutions](convolutions.md) |
| `matmul` | 16 — see [Linear layers](linear-layers.md) |
| `projection` | 12 — see [Linear layers](linear-layers.md) |
| `layernorm` | 4 — see [Normalization and shaping](normalization-and-shaping.md) |
| `residual_add` | 4 — three MobileViT shapes ([Normalization and shaping](normalization-and-shaping.md)) plus CHW `residual_add` ([Convolutions](convolutions.md)) |
| `unfold`, `fold`, `concat` | 3 each — see [Normalization and shaping](normalization-and-shaping.md) |
| `qk_scores`, `attn_scores_km`, `attn_v`, `attn_v_bcast` | 3 each — see [Attention](attention.md) |
| `maxpool2d` | 5 (`maxpool2d_window`, `_nms7`, `_nms9`, `_stride2`, `_stride2_tail`) |
| `sample_descriptors` | 2 (`sample_descriptors`, `sample_descriptors_separable`) |
| `channel_peak`, `score_threshold`, `l2_normalize`, `depth_to_space`, `identity`, `fully_connected` | 1 each |

See [Building applications](../building-applications.md) for the structure they
share.
