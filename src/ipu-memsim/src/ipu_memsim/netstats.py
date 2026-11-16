"""Network-only tables: what a network asks of memory, from its layer shapes
alone. No kernel, no emulator and no hardware model is involved.

A network file lists layers; a layer that has an ``op`` and a shape can be
counted here even when no kernel computes it::

    [[layer]]
    type = "mlp"                 # the row of the table it adds to
    name = "block fc1 192->768"
    op = "linear"                # conv | linear | attention | layernorm | add | embedding | pool
    in_features = 192
    out_features = 768
    tokens = 197
    count = 12

Per layer, and summed per layer type (each layer times its ``count``):

- **weights**: parameters, in bytes at one byte per element (ASSUMED: the
  hardware's 8-bit element formats);
- **input / output activations**: elements read and written, in bytes;
- **operations**: multiply-accumulates (for ``layernorm`` and ``add``, which
  have none, one operation per element);
- **operations per byte**: operations over (weights + input + output) bytes.
  A low value means the layer mostly moves data, which is where memory
  stalls hurt (compare elementwise against the matmuls in STATUS.md).

:func:`layer_knee` applies the working-set formulas of
``ipu_memsim.formulas`` to the shape, to predict the banks the layer needs.
Those formulas were validated on MobileViT-S shapes only, so every number
from it is **EXTRAPOLATED, not replayed**. A layer no formula covers gets
``None`` and a reason.

How each op is counted (``H_out x W_out`` is the output map; ``affine`` is
the extra parameters per output channel: 1 for a bias, 2 for a folded batch
norm's scale and shift):

==========  ===================================  ==================================
op          parameters                           operations
==========  ===================================  ==================================
conv        k*k*(C_in/groups)*C_out + affine*C_out  k*k*(C_in/groups)*C_out*H_out*W_out
linear      in*out + (out if bias)               in*out*tokens
attention   relative-position table, if any      2 * heads * tokens^2 * head_dim
                                                 (scores, then attention x V)
layernorm   2*channels                           channels*tokens
add         0                                    elements
embedding   elements                             0
pool        0                                    0 (not counted)
==========  ===================================  ==================================
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipu_memsim import formulas

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["LayerStats", "layer_knee", "layer_stats", "network_table"]

MAX_CONV_WIDTH = 128  # the convolution kernels' limit (kernel.toml: width max = 128)


@dataclass(frozen=True)
class LayerStats:
    """One layer's counts, for a single occurrence (see the module)."""

    params: int
    input_elements: int
    output_elements: int
    operations: int


def layer_stats(layer: Mapping[str, Any]) -> LayerStats:
    """Count one layer (see the module docstring's table).

    Raises:
        ValueError: An unknown ``op``.
    """
    op = layer["op"]
    if op == "conv":
        k, cin, cout = layer["kernel_size"], layer["in_channels"], layer["out_channels"]
        groups, stride = layer.get("groups", 1), layer.get("stride", 1)
        h_out, w_out = layer["height"] // stride, layer["width"] // stride
        weights = k * k * (cin // groups) * cout
        return LayerStats(
            weights + layer.get("affine", 0) * cout,
            cin * layer["height"] * layer["width"],
            cout * h_out * w_out,
            weights * h_out * w_out,
        )
    if op == "linear":
        fin, fout, tokens = layer["in_features"], layer["out_features"], layer.get("tokens", 1)
        bias = fout if layer.get("bias", True) else 0
        return LayerStats(fin * fout + bias, fin * tokens, fout * tokens, fin * fout * tokens)
    if op == "attention":
        n, heads, d = layer["tokens"], layer["heads"], layer["head_dim"]
        return LayerStats(
            layer.get("table_params", 0), 3 * n * heads * d, n * heads * d, 2 * heads * n * n * d
        )
    if op == "layernorm":
        elements = layer["channels"] * layer.get("tokens", 1)
        return LayerStats(2 * layer["channels"], elements, elements, elements)
    if op == "add":
        return LayerStats(0, 2 * layer["elements"], layer["elements"], layer["elements"])
    if op == "embedding":
        return LayerStats(layer["elements"], 0, 0, 0)
    if op == "pool":
        return LayerStats(0, layer.get("in_elements", 0), layer.get("out_elements", 0), 0)
    raise ValueError(f"unknown op {op!r} in layer {layer.get('name')!r}")


def layer_knee(layer: Mapping[str, Any], bank_rows: int = 1024) -> tuple[int | None, str]:  # noqa: C901 (one branch per op)
    """The banks the working-set formulas predict for a layer, and a note.

    EXTRAPOLATED: the formulas describe this repository's kernels and were
    validated on MobileViT-S shapes. ``(None, reason)`` when no formula
    applies.
    """
    op = layer["op"]
    if op == "conv":
        k, stride, groups = layer["kernel_size"], layer.get("stride", 1), layer.get("groups", 1)
        shape = {key: layer[key] for key in ("height", "width", "in_channels", "out_channels")}
        if layer["width"] > MAX_CONV_WIDTH:
            return (
                None,
                f"width {layer['width']} is over the convolution kernels' limit of {MAX_CONV_WIDTH}",
            )
        if stride != 1:
            return None, "no formula for a stride-2 convolution"
        if k == 1 and groups == 1:
            return formulas.kernel_banks(
                formulas.pointwise_conv(shape), bank_rows
            ), "pointwise formula"
        if k == 3 and groups == 1:
            return formulas.kernel_banks(formulas.conv3x3(shape), bank_rows), "3x3 formula"
        if k == 3 and groups == layer["in_channels"]:
            return formulas.kernel_banks(
                formulas.depthwise_conv3x3(shape), bank_rows
            ), "depthwise formula"
        return None, f"no formula for a {k}x{k} convolution with groups = {groups}"
    if op == "linear":
        shape = {"m": layer.get("tokens", 1), "k": layer["in_features"], "n": layer["out_features"]}
        return formulas.kernel_banks(formulas.matmul_x128(shape), bank_rows), "matmul formula"
    if op == "attention":
        scores = {"n_tok": layer["tokens"], "d": layer["head_dim"], "n_block": layer["heads"]}
        values = {"n_tok": layer["tokens"], "d": layer["head_dim"], "n_head": layer["heads"]}
        # Two kernels run one after the other: the layer needs the larger of the two.
        banks = max(formulas.kernel_banks(formulas.qk_scores(scores), bank_rows),
                    formulas.kernel_banks(formulas.attn_v(values), bank_rows))  # fmt: skip
        return banks, "scores and attention-times-V formulas (softmax not included)"
    if op == "layernorm":
        shape = {"channels": layer["channels"], "tokens": layer.get("tokens", 1)}
        return formulas.kernel_banks(formulas.layernorm(shape), bank_rows), "LayerNorm formula"
    if op == "add" and "channels" in layer:
        shape = {"channels": layer["channels"], "tokens": layer.get("tokens", 1)}
        return formulas.kernel_banks(
            formulas.residual_add(shape), bank_rows
        ), "residual-add formula"
    return None, f"no formula for op {op!r}"


def network_table(path: Path, bank_rows: int = 1024) -> dict[str, Any]:
    """The network-only table of one network file.

    Returns:
        ``{"name", "published_params", "total_params", "rows", "not_covered"}``.
        ``rows`` has one dict per layer type: ``layers`` (occurrences),
        ``weight_bytes``, ``input_bytes``, ``output_bytes``, ``operations``,
        ``operations_per_byte``, ``knee_banks`` (the largest predicted knee
        among the type's covered layers, EXTRAPOLATED) and ``covered`` (how
        many of its layers a formula covers). ``not_covered`` lists the
        layers without a formula, with the reason.
    """
    with open(path, "rb") as handle:
        description = tomllib.load(handle)
    layers = [
        layer
        for layer in description.get("layer", [])
        if "op" in layer and int(layer.get("count", 1))
    ]
    rows, not_covered = [], []
    total_params = 0
    for layer_type in dict.fromkeys(layer["type"] for layer in layers):
        group = [layer for layer in layers if layer["type"] == layer_type]
        weights = inputs = outputs = operations = occurrences = covered = 0
        knee: int | None = None
        for layer in group:
            stats, count = layer_stats(layer), int(layer.get("count", 1))
            weights += stats.params * count
            inputs += stats.input_elements * count
            outputs += stats.output_elements * count
            operations += stats.operations * count
            occurrences += count
            banks, note = layer_knee(layer, bank_rows)
            if banks is None:
                not_covered.append(
                    {"type": layer_type, "name": layer.get("name", ""), "reason": note}
                )
            else:
                covered += 1
                knee = banks if knee is None else max(knee, banks)
        total_params += weights
        size = weights + inputs + outputs
        rows.append(
            {
                "type": layer_type,
                "layers": occurrences,
                "weight_bytes": weights,
                "input_bytes": inputs,
                "output_bytes": outputs,
                "operations": operations,
                "operations_per_byte": operations / size if size else 0.0,
                "knee_banks": knee,
                "covered": f"{covered}/{len(group)}",
            }
        )
    return {
        "name": description.get("name", path.stem),
        "published_params": description.get("published_params"),
        "total_params": total_params,
        "rows": rows,
        "not_covered": not_covered,
    }
