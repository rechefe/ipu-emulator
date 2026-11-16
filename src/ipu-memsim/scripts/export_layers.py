"""Export a PyTorch model's layer shapes as an ``ipu-memsim`` network file.

Run this on a machine where torch installs (it is **not** used by the tests
and was **not run on the machine this tool was written on**, where torch
cannot be installed; treat its first output with care and compare the
parameter total it prints against the model's own).

    python export_layers.py --timm deit_tiny_patch16_224 --input 1 3 224 224 --out deit_tiny.toml
    python export_layers.py --torchvision swin_t --input 1 3 224 224 --out swin_t.toml

It runs one forward pass with a hook on every convolution, linear layer and
LayerNorm, records the shape each one saw, merges identical shapes into one
``[[layer]]`` with a ``count``, and writes the TOML file. What it cannot see
(attention as a plain matrix product inside a block, residual additions,
embeddings stored as bare parameters) is listed at the end of the file as a
comment, with the parameters it did not account for, so the gap is visible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def layer_key(module, inputs, output):
    """One layer's shape as a sorted tuple of (field, value), or None to skip it."""
    import torch.nn as nn

    x = inputs[0]
    if isinstance(module, nn.Conv2d):
        k, groups = module.kernel_size[0], module.groups
        kind = (
            "pointwise_conv"
            if k == 1
            else (
                "depthwise_conv" if groups == module.in_channels and groups > 1 else "standard_conv"
            )
        )
        return (("affine", 1 if module.bias is not None else 0), ("groups", groups), ("height", x.shape[2]),
                ("in_channels", module.in_channels), ("kernel_size", k), ("op", "conv"),
                ("out_channels", module.out_channels), ("stride", module.stride[0]), ("type", kind), ("width", x.shape[3]))  # fmt: skip
    if isinstance(module, nn.Linear):
        tokens = int(
            x.numel() // (x.shape[0] * module.in_features)
        )  # elements per sample over the feature size
        return (("bias", module.bias is not None), ("in_features", module.in_features), ("op", "linear"),
                ("out_features", module.out_features), ("tokens", tokens), ("type", "mlp"))  # fmt: skip
    if isinstance(module, nn.LayerNorm):
        channels = module.normalized_shape[-1]
        return (
            ("channels", channels),
            ("op", "layernorm"),
            ("tokens", int(x.numel() // (x.shape[0] * channels))),
            ("type", "normalization"),
        )
    return None


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timm", help="a timm model name")
    parser.add_argument("--torchvision", help="a torchvision.models constructor name")
    parser.add_argument(
        "--input", type=int, nargs=4, default=[1, 3, 224, 224], help="input shape N C H W"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.timm:
        import timm

        model, name = timm.create_model(args.timm, pretrained=False), args.timm
    elif args.torchvision:
        import torchvision

        model, name = getattr(torchvision.models, args.torchvision)(weights=None), args.torchvision
    else:
        parser.error("give --timm or --torchvision")
    model.eval()

    seen: Counter = Counter()
    counted_params = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal counted_params
        key = layer_key(module, inputs, output)
        if key is not None:
            seen[key] += 1
            counted_params += sum(p.numel() for p in module.parameters(recurse=False))

    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(*args.input))
    for handle in hooks:
        handle.remove()

    total = sum(p.numel() for p in model.parameters())
    lines = [
        f"# Exported by scripts/export_layers.py from {name}, input {args.input}.",
        "# Shapes are what the model's own layers saw in one forward pass.",
        f'name = "{name} (exported)"',
        f"published_params = {total}",
        "",
    ]
    for key, count in seen.items():
        fields = dict(key)
        lines.append("[[layer]]")
        lines.append(f'type = "{fields.pop("type")}"')
        lines.append(
            f'name = "{fields["op"]} '
            + " ".join(f"{k}={v}" for k, v in fields.items() if k != "op")
            + '"'
        )
        for field, value in fields.items():
            text = (
                f'"{value}"'
                if isinstance(value, str)
                else str(value).lower()
                if isinstance(value, bool)
                else str(value)
            )
            lines.append(f"{field} = {text}")
        lines.append(f"count = {count}")
        lines.append("")
    lines.append(f"# Parameters in exported layers: {counted_params} of {total}.")
    lines.append(
        "# Not exported (add by hand): attention products, residual additions, batch norms,"
    )
    lines.append(
        "# and parameters held outside conv / linear / LayerNorm modules (tokens, position tables)."
    )
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.out}: {len(seen)} distinct layers, {counted_params} of {total} parameters accounted for"
    )


if __name__ == "__main__":
    main()
