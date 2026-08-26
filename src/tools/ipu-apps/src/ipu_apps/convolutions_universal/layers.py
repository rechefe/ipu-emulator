"""Standalone PyTorch ``nn.Conv2d`` adapter for the convolutions_universal apps.

Runs a real ``torch.nn.Conv2d`` module (or a framework-free equivalent
description) against the IPU emulator by resolving through
``ipu_apps.kernel_registry`` -- every app in this package declares its own
``SPEC`` (see each app's ``__init__.py``), so this module is now a thin
wrapper: build a query, resolve it, construct the winning app, run, read
back the raw FP32 output. All shape/layout/packing plumbing lives inside the
apps themselves (see ``pointwise_conv_unified``'s class docstring for the
file-layout contract every app follows: ``input_path``/``output_path`` hold
the caller's raw, unpadded tensor).

This module also registers a ``register_layer("Conv2d")`` adapter (see
``ipu_apps.kernel_registry.layers``), so ``from_layer(nn.Conv2d(...),
input_shape)`` works the same way softmax's adapters do.

All apps in this package are FP32-only (wide-vector debug mode) -- there is
no INT8/quantized path anymore (see git history / conv-int8-snapshot branch
for the prior INT8 implementation).
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_apps.kernel_registry import registry
from ipu_apps.kernel_registry.layers import UnsupportedLayer, register_layer

if TYPE_CHECKING:
    import torch


# ============================================================================
# Layer description (framework-free)
# ============================================================================

@dataclass(frozen=True)
class Conv2dDescription:
    """Framework-free description of what :func:`resolve`/:func:`run_layer` need.

    Mirrors the fields of ``torch.nn.Conv2d`` that matter for dispatch; build
    one directly (no torch needed) or via :func:`from_torch_conv2d`.

    ``apply_relu`` has no ``torch.nn.Conv2d`` equivalent -- a plain Conv2d
    layer never implies an activation, so this must be requested explicitly
    (see :func:`run_layer`'s ``apply_relu`` kwarg) rather than inferred from
    ``has_bias``. The two are independent: every bias-capable kernel in this
    package unconditionally applies ReLU, so ``has_bias=True,
    apply_relu=False`` has no matching kernel and is refused.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    has_bias: bool
    apply_relu: bool = False
    height: int = 0
    width: int = 0

    def params(self) -> dict:
        return {
            "in_channels": self.in_channels, "out_channels": self.out_channels,
            "kernel_size": self.kernel_size, "stride": self.stride,
            "padding": self.padding, "dilation": self.dilation,
            "groups": self.groups, "has_bias": self.has_bias,
            "apply_relu": self.apply_relu, "height": self.height,
            "width": self.width,
        }


def from_torch_conv2d(
    layer: "torch.nn.Conv2d",
    input_shape: tuple[int, int, int] | None = None,
    *,
    apply_relu: bool = False,
) -> Conv2dDescription:
    """Build a :class:`Conv2dDescription` from a real ``torch.nn.Conv2d``.

    ``apply_relu`` must be passed explicitly by the caller -- see the
    ``apply_relu`` field's docstring for why it can't be inferred from the
    layer itself (a plain ``Conv2d`` has no activation concept).

    ``input_shape``, if given, is ``[in_channels, height, width]`` -- height
    and width are part of every kernel's query (region layout and
    column-alignment constraints are shape-dependent), so they default to 0
    (deliberately unresolvable) when omitted.
    """
    def _one(v) -> int:
        if isinstance(v, tuple):
            if v[0] != v[1]:
                raise UnsupportedLayer(
                    f"non-square parameter {v} is not supported (kernel_size, "
                    "stride, padding, and dilation must be scalars or square "
                    "tuples)"
                )
            return v[0]
        return int(v)

    height, width = (0, 0)
    if input_shape is not None:
        _, height, width = input_shape

    return Conv2dDescription(
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=_one(layer.kernel_size),
        stride=_one(layer.stride),
        padding=_one(layer.padding),
        dilation=_one(layer.dilation),
        groups=layer.groups,
        has_bias=layer.bias is not None,
        apply_relu=apply_relu,
        height=height,
        width=width,
    )


# ============================================================================
# Dispatch + run (via kernel_registry)
# ============================================================================

def resolve(desc: Conv2dDescription):
    """Resolve ``desc`` to a :class:`~ipu_apps.kernel_registry.spec.Verdict`.

    Does not run anything -- inspect ``verdict.kernel.name`` for the chosen
    app, or ``verdict.reason``/``bool(verdict)`` for why it was refused.
    """
    return registry.resolve("conv2d", **desc.params())


def run_layer(
    layer: "torch.nn.Conv2d",
    input_tensor: "torch.Tensor",
    *,
    apply_relu: bool = False,
    max_cycles: int = 50_000_000,
) -> "torch.Tensor":
    """Run ``layer`` on ``input_tensor`` through the matching IPU app.

    ``input_tensor`` must be ``[in_channels, height, width]`` (no batch
    dimension -- these apps process one image at a time), any real-valued
    tensor (cast to float32). Returns an ``[out_channels, out_height,
    out_width]`` float32 tensor.

    ``apply_relu`` must be passed explicitly -- a plain ``torch.nn.Conv2d``
    has no activation concept, so it can never be inferred from ``layer``
    (see :class:`Conv2dDescription`'s docstring). If ``layer.bias`` is set,
    ``apply_relu=True`` is required (every bias-capable kernel unconditionally
    applies ReLU).
    """
    import torch

    in_ch, height, width = input_tensor.shape
    desc = from_torch_conv2d(
        layer, (in_ch, height, width), apply_relu=apply_relu,
    )
    verdict = resolve(desc)
    if not verdict:
        raise UnsupportedLayer(verdict.reason)

    input_np = input_tensor.detach().cpu().numpy().astype(np.float32)
    weight_np = layer.weight.detach().cpu().numpy().astype(np.float32)
    bias_np = (
        layer.bias.detach().cpu().numpy().astype(np.float32)
        if desc.has_bias else None
    )

    kernel = verdict.kernel
    app_kwargs = dict(verdict.kwargs)

    with tempfile.TemporaryDirectory(prefix="run_layer_") as tmp_s:
        tmp = Path(tmp_s)
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        input_file.write_bytes(input_np.tobytes())

        ctor_params = _ctor_param_names(kernel.app_class)
        ctor_kwargs = dict(
            input_path=input_file, output_path=output_file, **app_kwargs,
        )
        if "kernel" in ctor_params:
            ctor_kwargs["kernel"] = weight_np
        else:
            # This app only accepts a kernel_path=, not a raw array (e.g.
            # pointwise_conv_unified) -- write the weights out ourselves.
            kernel_file = tmp / "kernel.bin"
            kernel_file.write_bytes(weight_np.tobytes())
            ctor_kwargs["kernel_path"] = kernel_file
        if desc.has_bias and "bias" in ctor_params:
            ctor_kwargs["bias"] = bias_np

        self_assembles = getattr(kernel.app_class, "SELF_ASSEMBLES", False)
        if kernel.asm is not None and not self_assembles:
            from ipu_as.lark_tree import assemble_to_bin_file

            app_module = importlib.import_module(kernel.app_class.__module__)
            app_dir = Path(app_module.__file__).resolve().parent
            asm_path = app_dir / kernel.asm
            bin_path = tmp / "assembled.bin"
            assemble_to_bin_file(asm_path.read_text(), str(bin_path))
            ctor_kwargs["inst_path"] = bin_path

        app = kernel.app_class(**ctor_kwargs)
        app.run(max_cycles=max_cycles)

        out_raw = np.frombuffer(output_file.read_bytes(), dtype=np.float32)
        out_shape = verdict.shapes.get("output") if verdict.shapes else None
        if out_shape is not None:
            out_raw = out_raw.reshape(out_shape)
        return torch.from_numpy(out_raw.copy())


def _ctor_param_names(app_class: type) -> set[str]:
    try:
        sig = inspect.signature(app_class.__init__)
    except (TypeError, ValueError):
        return set()
    return set(sig.parameters)


# ============================================================================
# Framework-layer adapter
# ============================================================================

@register_layer("Conv2d")
def _conv2d_layer(layer, input_shape):
    """``nn.Conv2d`` -> the ``conv2d`` operation.

    ``apply_relu`` has no ``torch.nn.Conv2d`` equivalent (see
    :class:`Conv2dDescription`'s docstring) -- routed here as ``False``, the
    only value derivable from the layer alone. Callers that want the ReLU
    twin should build a :class:`Conv2dDescription` directly (via
    :func:`from_torch_conv2d(layer, input_shape, apply_relu=True)`) and call
    :func:`resolve`/:func:`run_layer`, rather than going through
    ``from_layer``.
    """
    desc = from_torch_conv2d(layer, input_shape, apply_relu=False)
    return "conv2d", desc.params()
