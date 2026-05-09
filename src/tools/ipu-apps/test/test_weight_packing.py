"""Unit tests for the dense / legacy weight packers in
``ipu_apps.convolutions_universal.weights``.

The legacy pointwise packers are copied verbatim into this file (rather than
imported) so that future refactors of the app source that route the app
harnesses through the new packer cannot silently make these tests tautological.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipu_emu.ipu_math import DType
from ipu_apps.convolutions_universal.weights import (
    pack_conv_weights_dense,
    pack_depthwise_weights_dense,
    pack_pointwise_weights_universal,
    pack_pointwise_weights_8x8,
)


FPB_3 = 14  # 128 // 9

_RNG = np.random.default_rng(0x1C00F)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _ref_dense_conv(weights_int8: np.ndarray, kernel_size: int) -> bytes:
    """Direct reference for the dense conv layout — triple-loop spec."""
    out_ch, in_ch, kh, kw = weights_int8.shape
    assert kh == kw == kernel_size
    k2 = kernel_size * kernel_size
    fpb = 128 // k2
    blocks_per_filter = math.ceil(in_ch / fpb)
    total = out_ch * blocks_per_filter * 128
    buf = bytearray(total)
    raw = weights_int8.astype(np.int8).view(np.uint8).ravel()
    for f in range(out_ch):
        for b in range(blocks_per_filter):
            block_base = (f * blocks_per_filter + b) * 128
            for s in range(fpb):
                ic = b * fpb + s
                if ic >= in_ch:
                    continue
                src = (f * in_ch + ic) * k2
                dst = block_base + s * k2
                buf[dst:dst + k2] = bytes(raw[src:src + k2])
    return bytes(buf)


def _ref_dense_depthwise(weights_int8: np.ndarray, kernel_size: int) -> bytes:
    channels, mid, kh, kw = weights_int8.shape
    assert mid == 1 and kh == kw == kernel_size
    k2 = kernel_size * kernel_size
    fpb = 128 // k2
    num_blocks = math.ceil(channels / fpb)
    buf = bytearray(num_blocks * 128)
    raw = weights_int8.astype(np.int8).view(np.uint8).ravel()
    for b in range(num_blocks):
        block_base = b * 128
        for s in range(fpb):
            c = b * fpb + s
            if c >= channels:
                continue
            src = c * k2
            dst = block_base + s * k2
            buf[dst:dst + k2] = bytes(raw[src:src + k2])
    return bytes(buf)


def _legacy_compute_G(in_channels: int) -> int:
    g = in_channels & (-in_channels)
    return min(g, 128)


def _legacy_pointwise_universal_pack(
    raw_kernel: bytes, in_channels: int, out_channels: int,
) -> bytes:
    """Copy of ``PointwiseConvUniversalApp._pack_kernel`` (minus self.)."""
    G = _legacy_compute_G(in_channels)
    num_groups = in_channels // G
    oc_per_reg = 128 // G
    num_batches = out_channels // (2 * oc_per_reg)

    total_size = num_batches * 2 * num_groups * 128
    packed = bytearray(total_size)

    for batch in range(num_batches):
        for half in range(2):
            for group in range(num_groups):
                dest_base = ((batch * 2 + half) * num_groups + group) * 128
                for j in range(oc_per_reg):
                    oc = batch * 2 * oc_per_reg + half * oc_per_reg + j
                    for i in range(G):
                        ic = group * G + i
                        packed[dest_base + j * G + i] = raw_kernel[
                            oc * in_channels + ic
                        ]
    return bytes(packed)


def _legacy_pointwise_8x8_pack(
    raw_kernel: bytes, in_channels: int, out_channels: int,
) -> bytes:
    """Copy of ``pointwise_8x8._build_kernel_data``."""
    ic_pairs = in_channels // 2
    oc_pairs = out_channels // 2
    blocks_per_pair = math.ceil(ic_pairs / 32)
    kernel_bytes_per_pair = blocks_per_pair * 128
    packed = bytearray(oc_pairs * kernel_bytes_per_pair)
    for p in range(oc_pairs):
        f0 = 2 * p
        f1 = 2 * p + 1
        for j in range(ic_pairs):
            block = j // 32
            pos_in_block = j % 32
            dst = p * kernel_bytes_per_pair + block * 128 + pos_in_block * 4
            ic_even = 2 * j
            ic_odd = 2 * j + 1
            packed[dst] = raw_kernel[f0 * in_channels + ic_even]
            packed[dst + 1] = raw_kernel[f0 * in_channels + ic_odd]
            packed[dst + 2] = raw_kernel[f1 * in_channels + ic_even]
            packed[dst + 3] = raw_kernel[f1 * in_channels + ic_odd]
    return bytes(packed)


# ---------------------------------------------------------------------------
# Dense conv — int8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_ch,in_ch", [(4, 16), (4, 14), (4, 28), (4, 10)])
def test_pack_conv_weights_dense_int8(out_ch: int, in_ch: int) -> None:
    w = _RNG.integers(-128, 128, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    got = pack_conv_weights_dense(w, DType.INT8)

    blocks_per_filter = math.ceil(in_ch / FPB_3)
    assert len(got) == out_ch * blocks_per_filter * 128

    assert got == _ref_dense_conv(w, 3)

    # Spot-check a few explicit bytes: pick (f=1, b=0, s=2, dr=1, dc=2).
    f, b, s, dr, dc = 1, 0, 2, 1, 2
    ic = b * FPB_3 + s
    if ic < in_ch:
        expected = int(w[f, ic, dr, dc]) & 0xFF
        offset = (f * blocks_per_filter + b) * 128 + s * 9 + (dr * 3 + dc)
        assert got[offset] == expected

    # Padding tail of every block must be zero (bytes [126, 128)).
    for f in range(out_ch):
        for b in range(blocks_per_filter):
            tail = (f * blocks_per_filter + b) * 128
            assert got[tail + FPB_3 * 9:tail + 128] == b"\x00\x00"

    # Partial last block: any unused slots must be zero.
    partial = in_ch % FPB_3
    if partial != 0:
        last_b = blocks_per_filter - 1
        for f in range(out_ch):
            base = (f * blocks_per_filter + last_b) * 128
            assert got[base + partial * 9:base + FPB_3 * 9] == bytes(
                (FPB_3 - partial) * 9
            )


# ---------------------------------------------------------------------------
# Dense depthwise — int8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [16, 10, 14])
def test_pack_depthwise_weights_dense_int8(channels: int) -> None:
    w = _RNG.integers(-128, 128, size=(channels, 1, 3, 3), dtype=np.int8)
    got = pack_depthwise_weights_dense(w, DType.INT8)

    num_blocks = math.ceil(channels / FPB_3)
    assert len(got) == num_blocks * 128
    assert got == _ref_dense_depthwise(w, 3)

    # Tail padding bytes [126, 128) are zero in every block.
    for b in range(num_blocks):
        base = b * 128
        assert got[base + FPB_3 * 9:base + 128] == b"\x00\x00"

    # Partial last block check.
    partial = channels % FPB_3
    if partial != 0:
        last_b = num_blocks - 1
        base = last_b * 128
        assert got[base + partial * 9:base + FPB_3 * 9] == bytes(
            (FPB_3 - partial) * 9
        )


# ---------------------------------------------------------------------------
# Pointwise universal — byte-identical to legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_ch,in_ch", [(8, 32), (16, 16)])
def test_pack_pointwise_weights_universal_matches_legacy(
    out_ch: int, in_ch: int,
) -> None:
    w = _RNG.integers(-128, 128, size=(out_ch, in_ch), dtype=np.int8)
    got = pack_pointwise_weights_universal(w, in_ch, out_ch, DType.INT8)
    expected = _legacy_pointwise_universal_pack(
        w.view(np.uint8).tobytes(), in_ch, out_ch,
    )
    assert got == expected


def test_pack_pointwise_weights_universal_accepts_4d_singleton() -> None:
    out_ch, in_ch = 8, 32
    w2d = _RNG.integers(-128, 128, size=(out_ch, in_ch), dtype=np.int8)
    w4d = w2d.reshape(out_ch, in_ch, 1, 1)
    got_2d = pack_pointwise_weights_universal(w2d, in_ch, out_ch, DType.INT8)
    got_4d = pack_pointwise_weights_universal(w4d, in_ch, out_ch, DType.INT8)
    assert got_2d == got_4d


# ---------------------------------------------------------------------------
# Pointwise 8x8 — byte-identical to legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_ch,in_ch", [(4, 16), (6, 66)])
def test_pack_pointwise_weights_8x8_matches_legacy(
    out_ch: int, in_ch: int,
) -> None:
    w = _RNG.integers(-128, 128, size=(out_ch, in_ch), dtype=np.int8)
    got = pack_pointwise_weights_8x8(w, DType.INT8)
    expected = _legacy_pointwise_8x8_pack(
        w.view(np.uint8).tobytes(), in_ch, out_ch,
    )
    assert got == expected


def test_pack_pointwise_weights_8x8_accepts_4d_singleton() -> None:
    out_ch, in_ch = 4, 16
    w2d = _RNG.integers(-128, 128, size=(out_ch, in_ch), dtype=np.int8)
    w4d = w2d.reshape(out_ch, in_ch, 1, 1)
    assert (
        pack_pointwise_weights_8x8(w2d, DType.INT8)
        == pack_pointwise_weights_8x8(w4d, DType.INT8)
    )


# ---------------------------------------------------------------------------
# FP8 smoke tests (DType.E4 == FP8_E4M3)
# ---------------------------------------------------------------------------

def _nonzero_fp32(shape: tuple[int, ...]) -> np.ndarray:
    # Uniform in [-1, 1] with a tiny offset to avoid zero bins — FP8_E4M3
    # encodes these as nonzero bytes.
    return (_RNG.uniform(-1.0, 1.0, size=shape).astype(np.float32) + 0.125)


def test_fp8_e4m3_smoke_pack_conv_dense() -> None:
    out_ch, in_ch = 4, 16
    w = _nonzero_fp32((out_ch, in_ch, 3, 3))
    packed = pack_conv_weights_dense(w, DType.E4)
    assert len(packed) == out_ch * math.ceil(in_ch / FPB_3) * 128
    # At least one data byte per filter is nonzero.
    blocks_per_filter = math.ceil(in_ch / FPB_3)
    for f in range(out_ch):
        base = f * blocks_per_filter * 128
        # first block's data region:
        assert any(b != 0 for b in packed[base:base + FPB_3 * 9])


def test_fp8_e4m3_smoke_pack_depthwise_dense() -> None:
    channels = 14
    w = _nonzero_fp32((channels, 1, 3, 3))
    packed = pack_depthwise_weights_dense(w, DType.E4)
    num_blocks = math.ceil(channels / FPB_3)
    assert len(packed) == num_blocks * 128
    assert any(b != 0 for b in packed[:FPB_3 * 9])


def test_fp8_e4m3_smoke_pack_pointwise_universal() -> None:
    out_ch, in_ch = 8, 32
    w = _nonzero_fp32((out_ch, in_ch))
    packed = pack_pointwise_weights_universal(w, in_ch, out_ch, DType.E4)
    # Legacy formula for output length:
    G = _legacy_compute_G(in_ch)
    num_groups = in_ch // G
    oc_per_reg = 128 // G
    num_batches = out_ch // (2 * oc_per_reg)
    assert len(packed) == num_batches * 2 * num_groups * 128
    assert any(b != 0 for b in packed)


def test_fp8_e4m3_smoke_pack_pointwise_8x8() -> None:
    out_ch, in_ch = 4, 16
    w = _nonzero_fp32((out_ch, in_ch))
    packed = pack_pointwise_weights_8x8(w, DType.E4)
    ic_pairs = in_ch // 2
    oc_pairs = out_ch // 2
    blocks_per_pair = math.ceil(ic_pairs / 32)
    assert len(packed) == oc_pairs * blocks_per_pair * 128
    assert any(b != 0 for b in packed)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

def test_conv_dense_wrong_ndim() -> None:
    w = np.zeros((4, 16), dtype=np.int8)
    with pytest.raises(ValueError, match="ndim"):
        pack_conv_weights_dense(w, DType.INT8)


def test_conv_dense_wrong_kernel_size() -> None:
    w = np.zeros((4, 16, 5, 5), dtype=np.int8)
    with pytest.raises(ValueError, match="trailing dims"):
        pack_conv_weights_dense(w, DType.INT8)


def test_conv_dense_dtype_mismatch_int8_gets_float32() -> None:
    w = np.zeros((4, 16, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="int8"):
        pack_conv_weights_dense(w, DType.INT8)


def test_conv_dense_dtype_mismatch_fp8_gets_int8() -> None:
    w = np.zeros((4, 16, 3, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="float32"):
        pack_conv_weights_dense(w, DType.E4)


def test_depthwise_dense_wrong_mid_dim() -> None:
    w = np.zeros((4, 2, 3, 3), dtype=np.int8)
    with pytest.raises(ValueError, match=r"\[channels, 1, K, K\]"):
        pack_depthwise_weights_dense(w, DType.INT8)


def test_pointwise_universal_in_channels_too_small() -> None:
    w = np.zeros((4, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="divisible by at least 4"):
        pack_pointwise_weights_universal(w, 3, 4, DType.INT8)


def test_pointwise_universal_out_channels_indivisible() -> None:
    # in_channels=8 -> G=8, oc_per_reg=16, out_channels must be multiple of 32
    w = np.zeros((16, 8), dtype=np.int8)
    with pytest.raises(ValueError, match="out_channels"):
        pack_pointwise_weights_universal(w, 8, 16, DType.INT8)


def test_pointwise_universal_shape_mismatch() -> None:
    # Shape check fires before any G/divisibility check.
    w = np.zeros((8, 16), dtype=np.int8)
    with pytest.raises(ValueError, match="shape"):
        pack_pointwise_weights_universal(w, 32, 8, DType.INT8)


def test_pointwise_8x8_odd_in_channels() -> None:
    w = np.zeros((4, 15), dtype=np.int8)
    with pytest.raises(ValueError, match="in_channels"):
        pack_pointwise_weights_8x8(w, DType.INT8)


def test_pointwise_8x8_odd_out_channels() -> None:
    w = np.zeros((3, 16), dtype=np.int8)
    with pytest.raises(ValueError, match="out_channels"):
        pack_pointwise_weights_8x8(w, DType.INT8)
