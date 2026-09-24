"""Shared convolution cases: seeded FP32 data checked against independent NumPy convolutions.

The references are NumPy convolutions (no torch dependency).

**Memory-layout kernels** (:func:`make_cases`): ``conv1x1`` and the
``conv3x3_relu*`` kernels read the tiled fixture layout of
:mod:`~ipu_apps.kernel_registry.case_support`.

**Universal kernels** (:func:`conv2d_case`): a kernel's ``cases.py`` names the
query it answers (kernel size, stride, depthwise, bias+ReLU) and the default
options; the data generator, file bindings and check are shared. IPU FP32
accumulation order differs from the reference's, so the comparison is
``max |actual - expected| < tol`` rather than bit-exact.
"""
from __future__ import annotations

import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernel_registry.case_support import random_values, tile_input, tiled_case

def make_cases(app):
    def prepare(workspace, *, channels, height, width, out_channels):
        k = app.kernel_size
        params = dict(shape=(channels, height, width), out_channels=out_channels,
                      kernel_size=k, stride=1, padding=k//2,
                      activation="none" if k == 1 else "relu")
        layout = app.memory_layout(**params)
        x = random_values(params['shape'])
        weights = random_values((out_channels, channels, k, k)) / 4
        bias = random_values((out_channels,))
        group_size = 128 if k == 1 else 14
        groups = (channels + group_size - 1) // group_size

        def weights_and_bias(image):
            for o in range(out_channels):
                for g in range(groups):
                    chunk = weights[o, g*group_size:(g+1)*group_size].ravel()
                    image[layout.crs[4] + o*groups + g, :len(chunk)] = chunk
                image[layout.crs[5] + o, 0] = bias[o]

        padded = np.pad(x, ((0, 0), (k//2,)*2, (k//2,)*2))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(1, 2))
        expected = np.einsum('cyxij,ocij->oyx', windows, weights) + bias[:, None, None]
        if k == 3:
            expected = np.maximum(expected, 0)
        return tiled_case(workspace, params, layout,
                          tile_input(x, halo=k//2, guard=not app.single_channel), expected,
                          columns=128 if k == 1 else 126, constants=weights_and_bias)

    def case(c, h, w, o):
        return KernelCase(prepare, dict(channels=c, height=h, width=w, out_channels=o))

    cases = {'default': case(1 if app.single_channel else 3, 3, 8, 2),
             'tile_boundary': case(1 if app.single_channel else 2, 2, 133, 2),
             'single_pixel': case(1, 1, 1, 1)}
    if not app.single_channel:
        cap = 128 if app.kernel_size == 1 else 14
        cases['full_group'] = case(cap, 1, 2, 1)
        cases['partial_group'] = case(cap + 1, 1, 2, 2)
    return cases


# Tolerances: 3x3 kernels 1e-2, pointwise 1e-3 (max abs difference).
TOL_3X3 = 1e-2
TOL_1X1 = 1e-3


def conv2d_reference(x, weight, bias=None, *, stride=1, padding=0, groups=1,
                     relu=False):
    """NumPy ``conv2d`` of one ``[C, H, W]`` image, ``weight`` ``[O, C/groups, k, k]``."""
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    out_ch, group_in, k, _ = weight.shape
    group_out = out_ch // groups
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(1, 2))
    windows = windows[:, ::stride, ::stride]
    out = np.concatenate([
        np.einsum("cyxij,ocij->oyx",
                  windows[g * group_in:(g + 1) * group_in],
                  weight[g * group_out:(g + 1) * group_out])
        for g in range(groups)
    ])
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float64)[:, None, None]
    if relu:
        out = np.maximum(out, 0.0)
    return out


def check_close(actual, expected, tol):
    """Raise unless ``max |actual - expected| < tol``."""
    if actual.shape != expected.shape:
        raise ValueError(f"output shape {actual.shape}, expected {expected.shape}")
    diff = float(np.abs(actual - expected).max()) if actual.size else 0.0
    if not diff < tol:
        raise AssertionError(
            f"max diff {diff:.3e} >= {tol:g}\n"
            f"  actual[0,0,:8]:   {actual.reshape(actual.shape[0], -1)[0, :8]}\n"
            f"  expected[0,0,:8]: {expected.reshape(expected.shape[0], -1)[0, :8]}"
        )


def _data(seed, in_ch, out_ch, group_in, k, height, width, bias):
    rng = np.random.RandomState(seed)
    weights = (rng.randn(out_ch, group_in, k, k) * (0.2 if k == 3 else 0.3)).astype(np.float32)
    x = (rng.randn(in_ch, height, width) * 0.5).astype(np.float32)
    b = (rng.randn(out_ch) * 0.3).astype(np.float32) if bias else None
    return x, weights, b


def conv2d_case(*, kernel_size=3, stride=1, depthwise=False, bias=False,
                defaults, data="random", max_cycles=20_000_000):
    """A :class:`KernelCase` for one universal conv2d configuration family.

    ``defaults`` are the case options: ``height``, ``width``, ``seed`` and
    either ``channels`` (depthwise) or ``in_channels`` + ``out_channels``.
    ``bias=True`` selects a ``_bn_activation``-style query (bias + ReLU).

    ``data`` picks the generator:

    * ``"random"``        -- random weights/input (and bias), checked against
      :func:`conv2d_reference` within the family tolerance;
    * ``"negative"``      -- all weights -1 on a constant positive input, so
      every output is negative: a kernel without ReLU must keep the sign;
    * ``"negative_bias"`` -- zero weights and a negative bias, so a ReLU
      kernel must write exact zeros everywhere.
    """
    padding = kernel_size // 2
    tol = TOL_3X3 if kernel_size == 3 else TOL_1X1

    def prepare(workspace, **options):
        height, width, seed = options["height"], options["width"], options["seed"]
        if depthwise:
            in_ch = out_ch = options["channels"]
        else:
            in_ch, out_ch = options["in_channels"], options["out_channels"]
        groups = in_ch if depthwise else 1
        group_in = in_ch // groups
        x, weights, b = _data(seed, in_ch, out_ch, group_in, kernel_size, height, width, bias)
        if data == "negative":
            weights = -np.ones_like(weights)
            x = np.full_like(x, 5.0)
        elif data == "negative_bias":
            weights = np.zeros_like(weights)
            x = np.ones_like(x)
            b = -np.abs(b) - 0.1
        elif data != "random":
            raise ValueError(f"unknown data generator {data!r}")

        params = dict(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=1, groups=groups,
            has_bias=bias, apply_relu=bias, height=height, width=width,
        )
        paths = {name: workspace / f"{name}.bin" for name in ("input", "kernel", "bias", "output")}
        paths["input"].write_bytes(x.tobytes())
        paths["kernel"].write_bytes(weights.tobytes())
        bindings = {"input_path": paths["input"], "kernel_path": paths["kernel"],
                    "output_path": paths["output"]}
        if bias:
            paths["bias"].write_bytes(b.tobytes())
            bindings["bias_path"] = paths["bias"]
        expected = conv2d_reference(x, weights, b, stride=stride, padding=padding,
                                    groups=groups, relu=bias)

        def check():
            raw = np.fromfile(paths["output"], dtype="<f4")
            if raw.size != expected.size:
                raise ValueError(f"output has {raw.size} FP32 values, expected {expected.size}")
            actual = raw.reshape(expected.shape)
            if data == "negative" and not np.all(actual < 0):
                raise AssertionError("expected all-negative outputs (no ReLU applied)")
            if data == "negative_bias" and not np.all(actual == 0.0):
                raise AssertionError("ReLU should zero all negative-bias outputs")
            check_close(actual, expected, tol)

        return PreparedCase(params, bindings, check)

    return KernelCase(prepare, defaults, max_cycles)
