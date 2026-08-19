"""Tests for the standalone PyTorch nn.Conv2d adapter (convolutions_universal.layers).

Covers: the width pad/pack/unpack/truncate roundtrip (framework-free), then
run_layer end-to-end against F.conv2d/reference math for each dispatch path
(plain conv, depthwise, depthwise stride=2, +bias/ReLU), the small-image
sweep (width 1-63 padded to the next power of 2, PLUS row padding for the
num_chunks>=2 / rows>=4 floors small images can miss even after column
padding), and the specific refusal cases layers.py documents (non-square
params, unsupported groups, stride=2 for plain conv, bias without
apply_relu, apply_relu with stride=2, width > 128).

CI status: like test_conv_universal_bn_activation_pytorch.py, this is a
**local-only cross-check** -- importorskip'd, not in the Bazel BUILD targets
(no conv app has a Bazel target yet; see BUILD.bazel's TODO-shaped gap).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from ipu_apps.convolutions_universal.layers import (  # noqa: E402
    Conv2dDescription,
    UnsupportedLayer,
    pack_input_chunked,
    pad_width_to_cols,
    quantize_int8,
    resolve,
    run_layer,
    truncate_width,
    unpack_output_chunked,
)
from ipu_apps.convolutions_universal import (  # noqa: E402
    CHUNK_BYTES,
    XmemOverflow,
    allocate_regions,
)


class TestAllocateRegions:
    """allocate_regions -- the dynamic XMEM layout that replaced pointwise_conv_unified's
    fixed *_BASE_ADDR gaps (see module docstring in convolutions_universal/__init__.py
    for why: a fixed 64 KiB kernel gap silently overflowed into the mask region for
    in_channels=512, out_channels=160, corrupting output with no error)."""

    def test_bases_are_sequential_and_chunk_aligned(self) -> None:
        bases = allocate_regions([("a", 100), ("b", 200), ("c", 128)])
        assert bases["a"] == 0
        # "a" is 100 B -> next base rounds up to the 128 B chunk boundary
        assert bases["b"] == CHUNK_BYTES
        # "b" is 200 B starting at 128 -> ends at 328 -> rounds up to 384
        assert bases["c"] == 3 * CHUNK_BYTES
        for base in bases.values():
            assert base % CHUNK_BYTES == 0

    def test_zero_size_region_takes_no_space(self) -> None:
        bases = allocate_regions([("a", 0), ("b", 128)])
        assert bases["a"] == 0
        assert bases["b"] == 0

    def test_overflow_raises_with_breakdown(self) -> None:
        with pytest.raises(XmemOverflow, match="a=.*b="):
            allocate_regions([("a", 3 * 1024 * 1024), ("b", 128)])

    def test_exact_fit_does_not_raise(self) -> None:
        from ipu_apps.convolutions_universal import XMEM_BYTES
        bases = allocate_regions([("only", XMEM_BYTES)])
        assert bases["only"] == 0


class TestWidthPaddingRoundtrip:
    """Framework-free: pad -> pack -> unpack -> truncate must be identity."""

    @pytest.mark.parametrize("width", [65, 99, 100, 101, 125, 126, 127, 128])
    def test_roundtrip(self, width: int) -> None:
        rng = np.random.RandomState(width)
        x = rng.randint(-8, 9, size=(3, 5, width)).astype(np.int8)
        cols = 128 if width > 64 else 64
        padded = pad_width_to_cols(x, cols)
        assert padded.shape == (3, 5, cols)
        packed = pack_input_chunked(padded)
        unpacked = unpack_output_chunked(packed, 3, 5, cols)
        assert np.array_equal(unpacked, padded)
        truncated = truncate_width(unpacked, width)
        assert np.array_equal(truncated, x)


class TestQuantizeInt8:
    def test_integer_valued_tensor_ok(self) -> None:
        t = torch.tensor([-4.0, 0.0, 3.0, 127.0, -128.0])
        arr = quantize_int8(t)
        assert arr.dtype == np.int8
        assert list(arr) == [-4, 0, 3, 127, -128]

    def test_fractional_tensor_refused(self) -> None:
        t = torch.tensor([1.5, 2.0])
        with pytest.raises(UnsupportedLayer, match="non-integer"):
            quantize_int8(t)

    def test_out_of_range_refused(self) -> None:
        t = torch.tensor([200.0])
        with pytest.raises(UnsupportedLayer, match="INT8 range"):
            quantize_int8(t)


def _reference_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    *, stride: int, groups: int, relu: bool, padding: int = 1,
) -> torch.Tensor:
    """IPU-math-equivalent reference: exact-int F.conv2d, +bias, ReLU, clamp.

    ``padding`` defaults to 1 ("same" for the k=3 apps every other caller in
    this file exercises); 1x1 (pointwise) callers pass padding=0 explicitly,
    since a 1x1 kernel has no neighbourhood to pad for.
    """
    out = F.conv2d(
        x.unsqueeze(0).to(torch.int64), weight.to(torch.int64),
        bias=None, padding=padding, stride=stride, groups=groups,
    )[0]
    if bias is not None:
        out = out + bias.to(torch.int64).view(-1, 1, 1)
    if relu:
        out = out.clamp(min=0)
    return out.clamp(-128, 127).to(torch.int8)


class TestRunLayerStride1:

    def test_plain_conv_no_bias_non_power_of_2_width(self) -> None:
        torch.manual_seed(0)
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        layer.weight.data = torch.randint(-8, 9, (4, 3, 3, 3)).to(torch.float32)
        x = torch.randint(-8, 9, (3, 8, 100)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False)
        assert out.shape == expected.shape == (4, 8, 100)
        assert torch.equal(out, expected)

    def test_plain_conv_bias_relu_boundary_width_127(self) -> None:
        torch.manual_seed(3)
        layer = torch.nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
        layer.weight.data = torch.randint(-8, 9, (5, 3, 3, 3)).to(torch.float32)
        layer.bias.data = torch.randint(-40, 41, (5,)).to(torch.float32)
        x = torch.randint(-8, 9, (3, 4, 127)).to(torch.float32)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=1, relu=True,
        )
        assert torch.equal(out, expected)

    def test_depthwise_bias_relu_width_65(self) -> None:
        torch.manual_seed(1)
        c = 4
        layer = torch.nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        layer.weight.data = torch.randint(-4, 5, (c, 1, 3, 3)).to(torch.float32)
        layer.bias.data = torch.randint(-20, 21, (c,)).to(torch.float32)
        x = torch.randint(-4, 5, (c, 6, 65)).to(torch.float32)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=c, relu=True,
        )
        assert torch.equal(out, expected)

    def test_depthwise_no_bias_width_128(self) -> None:
        torch.manual_seed(2)
        c = 3
        layer = torch.nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        layer.weight.data = torch.randint(-4, 5, (c, 1, 3, 3)).to(torch.float32)
        x = torch.randint(-4, 5, (c, 5, 128)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=c, relu=False)
        assert torch.equal(out, expected)


class TestRunLayerStride2:

    def test_depthwise_stride2_non_power_of_2_width(self) -> None:
        torch.manual_seed(1)
        c = 4
        layer = torch.nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c, bias=False)
        layer.weight.data = torch.randint(-4, 5, (c, 1, 3, 3)).to(torch.float32)
        x = torch.randint(-4, 5, (c, 8, 100)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=2, groups=c, relu=False)
        assert out.shape == expected.shape
        assert torch.equal(out, expected)


class TestSmallImages:
    """Widths 1-63 (padded to the next power of 2 in {16,32,64}) and small
    row counts that miss the underlying apps' own size floors even after
    column padding -- num_chunks>=2 for stride 1, rows>=4 (mult. of 4) for
    stride 2 -- and therefore need row padding too. Both floors are handled
    internally by run_layer; verified here against direct app calls that
    previously raised ValueError for these exact shapes."""

    @pytest.mark.parametrize("width", [1, 2, 5, 8, 15, 16, 17, 31, 63])
    @pytest.mark.parametrize("rows", [1, 2, 3, 4])
    def test_stride1_small_width_small_rows(self, width: int, rows: int) -> None:
        torch.manual_seed(9)
        layer = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
        layer.weight.data = torch.randint(-4, 5, (3, 2, 3, 3)).to(torch.float32)
        x = torch.randint(-4, 5, (2, rows, width)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False)
        assert out.shape == expected.shape == (3, rows, width)
        assert torch.equal(out, expected)

    @pytest.mark.parametrize("width", [1, 2, 5, 8, 15, 16, 31, 63])
    @pytest.mark.parametrize("rows", [1, 2, 3, 5, 7])
    def test_stride2_small_width_small_rows(self, width: int, rows: int) -> None:
        torch.manual_seed(9)
        c = 3
        layer = torch.nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c, bias=False)
        layer.weight.data = torch.randint(-4, 5, (c, 1, 3, 3)).to(torch.float32)
        x = torch.randint(-4, 5, (c, rows, width)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=2, groups=c, relu=False)
        assert out.shape == expected.shape
        assert torch.equal(out, expected)


class TestRunLayerPointwise:
    """1x1 (pointwise) dispatch: no spatial neighbourhood, so no mask/edge
    concerns -- see layers.py's _pointwise_pad_shape docstring. Covers the
    plain single-pass case, the multi-pass kernel-region-overflow shape that
    was found broken in pointwise_conv_unified itself (in_channels=512,
    out_channels=160 needs 81,920 B against the app's old fixed 64 KiB kernel
    gap -- fixed by dynamic region allocation, see allocate_regions in
    convolutions_universal/__init__.py), and the 8x8 spatial floor a plain
    1x1 conv has no other reason to reject."""

    def test_plain_1x1_no_bias(self) -> None:
        torch.manual_seed(4)
        layer = torch.nn.Conv2d(16, 8, kernel_size=1, bias=False)
        layer.weight.data = torch.randint(-4, 5, (8, 16, 1, 1)).to(torch.float32)
        x = torch.randint(-4, 5, (16, 16, 16)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False, padding=0)
        assert out.shape == expected.shape == (8, 16, 16)
        assert torch.equal(out, expected)

    def test_1x1_bias_relu(self) -> None:
        torch.manual_seed(5)
        layer = torch.nn.Conv2d(16, 8, kernel_size=1, bias=True)
        layer.weight.data = torch.randint(-4, 5, (8, 16, 1, 1)).to(torch.float32)
        layer.bias.data = torch.randint(-40, 41, (8,)).to(torch.float32)
        x = torch.randint(-4, 5, (16, 16, 16)).to(torch.float32)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=1, relu=True, padding=0,
        )
        assert torch.equal(out, expected)

    def test_1x1_8x8_spatial_floor(self) -> None:
        """8x8 has no reason to be rejected for a 1x1 conv (no 3x3
        neighbourhood, no {16,32,64,128} floor) -- verifies width pads only
        to 16 (not to 64+), leaving row padding untouched."""
        torch.manual_seed(6)
        layer = torch.nn.Conv2d(160, 160, kernel_size=1, bias=False)
        layer.weight.data = torch.randint(-3, 4, (160, 160, 1, 1)).to(torch.float32)
        x = torch.randint(-3, 4, (160, 8, 8)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False, padding=0)
        assert out.shape == expected.shape == (160, 8, 8)
        assert torch.equal(out, expected)

    def test_1x1_multipass_kernel_region_overflow_shape(self) -> None:
        """in_channels=512 (4 passes) x out_channels=160 needs a kernel
        region of 81,920 B -- over pointwise_conv_unified's old fixed 64 KiB
        gap, which silently corrupted output channel 128 onward with no
        error before the dynamic-region-allocation fix. Regression guard for
        that specific shape, at the 8x8 floor used by MobileViT's
        stages.4.0.conv3_1x1."""
        torch.manual_seed(7)
        layer = torch.nn.Conv2d(512, 160, kernel_size=1, bias=False)
        layer.weight.data = torch.randint(-3, 4, (160, 512, 1, 1)).to(torch.float32)
        x = torch.randint(-3, 4, (512, 8, 8)).to(torch.float32)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False, padding=0)
        assert out.shape == expected.shape == (160, 8, 8)
        assert torch.equal(out, expected)


class TestRefusals:

    def _depthwise_layer(self, **kw) -> torch.nn.Conv2d:
        c = kw.pop("channels", 4)
        return torch.nn.Conv2d(c, c, kernel_size=3, groups=c, **kw)

    def test_wrong_kernel_size(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=5, padding=2)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="kernel_size"):
            run_layer(layer, x)

    def test_dilation(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=2, dilation=2)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="dilation"):
            run_layer(layer, x)

    def test_non_same_padding(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=0)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="padding"):
            run_layer(layer, x)

    def test_unsupported_stride(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=3)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="stride"):
            run_layer(layer, x)

    def test_partial_groups(self) -> None:
        layer = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
        x = torch.zeros(4, 8, 8)
        with pytest.raises(UnsupportedLayer, match="groups"):
            run_layer(layer, x)

    def test_depthwise_multiplier(self) -> None:
        layer = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=4)
        x = torch.zeros(4, 8, 8)
        with pytest.raises(UnsupportedLayer, match="depthwise multiplier"):
            run_layer(layer, x)

    def test_stride2_plain_conv_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=2)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="stride=2 for a plain"):
            run_layer(layer, x)

    def test_bias_without_apply_relu_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="apply_relu=False"):
            run_layer(layer, x, apply_relu=False)

    def test_apply_relu_with_stride2_refused(self) -> None:
        c = 4
        layer = self._depthwise_layer(channels=c, padding=1, stride=2, bias=False)
        layer.weight.data = torch.randint(-4, 5, (c, 1, 3, 3)).to(torch.float32)
        x = torch.zeros(c, 8, 8)
        with pytest.raises(UnsupportedLayer, match="apply_relu=True is not supported for stride=2"):
            run_layer(layer, x, apply_relu=True)

    def test_width_over_128_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        x = torch.zeros(3, 8, 200)
        with pytest.raises(UnsupportedLayer, match="only spatial dims up to 128"):
            run_layer(layer, x)

    def test_non_square_stride_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=(1, 2), bias=False)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="non-square"):
            run_layer(layer, x)

    def test_1x1_stride2_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, stride=2, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer, match="stride"):
            run_layer(layer, x)

    def test_1x1_groups_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, groups=8, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer, match="groups"):
            run_layer(layer, x)

    def test_1x1_in_channels_not_multiple_of_8_refused(self) -> None:
        layer = torch.nn.Conv2d(6, 8, kernel_size=1, bias=False)
        x = torch.zeros(6, 8, 8)
        with pytest.raises(UnsupportedLayer, match="in_channels"):
            run_layer(layer, x)

    def test_1x1_out_channels_not_multiple_of_4_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 6, kernel_size=1, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer, match="out_channels"):
            run_layer(layer, x)

    def test_1x1_bias_without_apply_relu_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, bias=True)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer, match="apply_relu=False"):
            run_layer(layer, x, apply_relu=False)


class TestResolve:
    """Framework-free dispatch check (no torch layer needed, just the dataclass)."""

    def test_plain_conv_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=3, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, has_bias=False,
        )
        assert resolve(desc) == "conv_universal"

    def test_plain_conv_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=3, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, has_bias=True, apply_relu=True,
        )
        assert resolve(desc) == "conv_universal_bn_activation"

    def test_depthwise_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=4, has_bias=False,
        )
        assert resolve(desc) == "depthwise_conv_universal"

    def test_depthwise_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=4, has_bias=True, apply_relu=True,
        )
        assert resolve(desc) == "depthwise_conv_universal_bn_activation"

    def test_depthwise_stride2(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=2,
            padding=1, dilation=1, groups=4, has_bias=False,
        )
        assert resolve(desc) == "depthwise_conv_stride2_128"

    def test_pointwise_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=8, out_channels=8, kernel_size=1, stride=1,
            padding=0, dilation=1, groups=1, has_bias=False,
        )
        assert resolve(desc) == "pointwise_conv_unified"

    def test_pointwise_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=8, out_channels=8, kernel_size=1, stride=1,
            padding=0, dilation=1, groups=1, has_bias=True, apply_relu=True,
        )
        assert resolve(desc) == "pointwise_conv_unified_bn_activation"
