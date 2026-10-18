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
| [Kernel coverage](../explanation/kernel-coverage.md) | How a computation resolves to a kernel, and every registered kernel (generated) |
| [Write a kernel](../how-to/write-a-kernel.md) | How to contribute a kernel |

## Finding a kernel for a computation

Rather than reading each kernel's docstring, ask the registry. It answers from
the kernels themselves, so it cannot drift out of date:

```bash
bazel run //kernels:query -- softmax rows=32 n=300 dim=1
bazel run //kernels:query                                            # the coverage report
```

```python
from ipu_harness.registry import default_registry

registry = default_registry()

# PyTorch-native: the layer plus the shape it will receive
registry.from_layer(nn.Softmax(dim=1), (32, 300))

# framework-free
registry.resolve("softmax", rows=32, n=300, dim=1)
```

Both return a verdict carrying the kernel, its parameters and why that kernel
was chosen — or, when nothing covers the query, what each candidate objected
to. Every family is a folder of `kernels/`; the
[coverage page](../explanation/kernel-coverage.md) lists them.

## Currently registered

[Kernel coverage](../explanation/kernel-coverage.md#current-coverage) lists
every operation and its kernels, generated from the registry when the docs
are built. `bazel run //kernels:query` prints the same list.

[Write a kernel](../how-to/write-a-kernel.md) shows the structure of a kernel
under `kernels/`.
