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
| [Linear layers](linear-layers.md) | Matmul and multi-stream projection kernels for MobileViT-S's transformer blocks |
| [Normalization and shaping](normalization-and-shaping.md) | LayerNorm, residual add, and spatial unfold kernels |
| [Attention](attention.md) | The two non-interchangeable QKᵀ / attn·V mappings and their twelve kernels |
| [Application coverage](../app-coverage.md) | How the emulator knows which kernel implements a computation |
| [Adding applications](../adding-applications.md) | How to contribute a kernel |

## Finding a kernel for a computation

Rather than reading each kernel's docstring, ask the registry. It answers from
the kernels themselves, so it cannot drift out of date:

```bash
python -m ipu_apps.softmax --shape 32,300 --dim 1
python -m ipu_apps.softmax --catalog
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

| Operation | Kernels |
|---|---|
| `softmax` | 5 — see [Softmax](softmax.md) |
| `matmul` | 16 — see [Linear layers](linear-layers.md) |
| `projection` | 12 — see [Linear layers](linear-layers.md) |
| `layernorm` | 4 — see [Normalization and shaping](normalization-and-shaping.md) |
| `residual_add` | 3 — see [Normalization and shaping](normalization-and-shaping.md) |
| `unfold` | 3 — see [Normalization and shaping](normalization-and-shaping.md) |
| `qk_scores` | 3 — see [Attention](attention.md) |
| `attn_scores_km` | 3 — see [Attention](attention.md) |
| `attn_v` | 3 — see [Attention](attention.md) |
| `attn_v_bcast` | 3 — see [Attention](attention.md) |

Convolution, pointwise, depthwise and fully-connected applications exist in the
tree but are not yet registered; they are used directly rather than through the
registry. See [Building applications](../building-applications.md) for the
general application structure they follow.
