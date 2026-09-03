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
| [Convolutions](convolutions.md) | Which conv2d/residual_add kernels exist, which shape each handles, and what they cost |
| [Application coverage](../app-coverage.md) | How the emulator knows which kernel implements a computation |
| [Adding applications](../adding-applications.md) | How to contribute a kernel |

## Finding a kernel for a computation

Rather than reading each kernel's docstring, ask the registry. It answers from
the kernels themselves, so it cannot drift out of date:

```bash
python -m ipu_apps.softmax --shape 32,300 --dim 1
python -m ipu_apps.softmax --catalog

python -m ipu_apps.convolutions_universal --in-channels 32 --out-channels 32 \
    --kernel-size 3 --stride 1 --padding 1 --groups 32 --height 64 --width 64
python -m ipu_apps.convolutions_universal --catalog
```

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

# PyTorch-native: the layer plus the shape it will receive
lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))
lookup_layer(nn.Conv2d(32, 32, 3, groups=32, padding=1), input_shape=(32, 64, 64))

# framework-free
resolve("softmax", shape=(32, 300), dim=1)
resolve("conv2d", in_channels=32, out_channels=32, kernel_size=3, stride=1,
        padding=1, dilation=1, groups=32, has_bias=False, apply_relu=False,
        height=64, width=64)
```

Both return a verdict carrying the app class, its constructor arguments, why
that kernel was chosen, and any caveats that still apply — or, when nothing
covers the shape, what each candidate objected to.

## Currently registered

| Operation | Kernels |
|---|---|
| `softmax` | 5 — see [Softmax](softmax.md) |
| `conv2d` | 11 — see [Convolutions](convolutions.md) |
| `residual_add` | 1 — see [Convolutions](convolutions.md) |

Fully-connected applications exist in the tree but are not yet registered;
they are used directly rather than through the registry. See
[Building applications](../building-applications.md) for the general
application structure they follow.
