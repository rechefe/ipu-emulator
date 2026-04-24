"""Shared utilities for convolutions_universal benchmark scripts.

Provides:
  - Data generation (INT8 random tensors, multi-channel memory packing)
  - PyTorch INT8 reference computation for each conv type
  - Output reading from emulator state (unpacked INT32 -> float32 for comparison)
  - Table printer (cycles, MACs/cycle, correctness)

Speed-of-light: 128 MAC/cycle (one mult.ve fires 128 multiplies in 1 cycle).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap (allows running scripts directly without installing packages)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[7]  # repo root
for _pkg in ("ipu-emu-py/src", "ipu-common/src", "ipu-apps/src", "ipu-as-py/src"):
    _p = str(_ROOT / "src" / "tools" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 128  # MAC/cycle

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def gen_int8(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Random INT8 array in [-3, 3] (small values avoid overflow in reference)."""
    rng = np.random.RandomState(seed)
    return rng.randint(-3, 4, size=shape, dtype=np.int8)


def pack_input_multichannel(
    input_chw: np.ndarray,
    rows: int,
    cols: int,
    channels: int,
) -> bytes:
    """Pack CHW input into the multi-channel memory layout.

    Layout: row_groups × channels × 128 bytes.
    rows_per_chunk = 128 // cols spatial rows per 128-byte chunk.
    input_chw shape: (channels, rows, cols).
    """
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    packed = bytearray(row_groups * channels * 128)
    for rg in range(row_groups):
        for ch in range(channels):
            dst = (rg * channels + ch) * 128
            for r in range(rows_per_chunk):
                spatial_row = rg * rows_per_chunk + r
                src_row = input_chw[ch, spatial_row, :]
                packed[dst + r * cols : dst + r * cols + cols] = (
                    src_row.view(np.uint8).tobytes()
                )
    return bytes(packed)


# ---------------------------------------------------------------------------
# PyTorch INT8 reference (float32 arithmetic on int8-range values)
# ---------------------------------------------------------------------------

def _to_torch_int8(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def ref_pointwise(
    input_chw: np.ndarray,
    kernel_oihw: np.ndarray,
) -> torch.Tensor:
    """Reference 1x1 conv: input (C,H,W), kernel (OC,IC,1,1) -> (OC,H,W) float32."""
    x = _to_torch_int8(input_chw).unsqueeze(0)          # (1,IC,H,W)
    w = _to_torch_int8(kernel_oihw)                      # (OC,IC,1,1)
    return F.conv2d(x, w, padding=0).squeeze(0)          # (OC,H,W)


def ref_depthwise(
    input_chw: np.ndarray,
    kernel_c33: np.ndarray,
    stride: int = 1,
) -> torch.Tensor:
    """Reference 3x3 depthwise conv: input (C,H,W), kernel (C,3,3) -> (C,H',W') float32."""
    c = input_chw.shape[0]
    x = _to_torch_int8(input_chw).unsqueeze(0)           # (1,C,H,W)
    w = _to_torch_int8(kernel_c33).unsqueeze(1)          # (C,1,3,3)
    return F.conv2d(x, w, padding=1, stride=stride, groups=c).squeeze(0)  # (C,H',W')


def ref_conv3x3(
    input_chw: np.ndarray,
    kernel_oihw: np.ndarray,
    stride: int = 1,
) -> torch.Tensor:
    """Reference 3x3 standard conv: input (IC,H,W), kernel (OC,IC,3,3) -> (OC,H',W')."""
    x = _to_torch_int8(input_chw).unsqueeze(0)
    w = _to_torch_int8(kernel_oihw)
    return F.conv2d(x, w, padding=1, stride=stride).squeeze(0)


def ref_depthwise_stride2_quantized(
    input_chw: np.ndarray,
    kernel_c33: np.ndarray,
) -> np.ndarray:
    """Reference stride-2 depthwise 3x3 conv with INT8 output quantization.

    Matches the IPU: conv with stride=2 using only even-indexed output columns
    (horizontal stride decimation), then clamp to [-128, 127].

    The IPU computes full-row convolution then decimates (picks every 2nd element),
    so the reference uses stride=2 directly which gives the same result.

    Returns INT8 numpy array (channels, out_rows, out_cols).
    """
    result_f = ref_depthwise(input_chw, kernel_c33, stride=2)  # (C, H/2, W/2)
    clamped = result_f.numpy().clip(-128, 127).round().astype(np.int8)
    return clamped


# ---------------------------------------------------------------------------
# Output reading from emulator state
# ---------------------------------------------------------------------------

def read_acc_chunks(state: Any, base_addr: int, num_chunks: int) -> np.ndarray:
    """Read num_chunks × 512-byte ACC blocks; return (num_chunks*128,) float32 array.

    Each 512-byte block: 128 lanes × 4 bytes INT32 (little-endian).
    """
    total = num_chunks * 512
    raw = state.xmem.read_address(base_addr, total)
    return np.frombuffer(raw, dtype="<i4").astype(np.float32)


def read_acc_paired_8x8(
    state: Any,
    base_addr: int,
    out_channels: int,
    spatial: int,
) -> np.ndarray:
    """Read paired-output ACC layout from 8x8 apps (OC pairs, 128 lanes).

    Paired layout: oc_pairs × 512 bytes.
      Lanes 0..spatial-1:       even OC spatial values
      Lanes spatial..2*spatial-1: odd OC spatial values

    Returns (out_channels, spatial) float32 array.
    """
    oc_pairs = out_channels // 2
    raw = state.xmem.read_address(base_addr, oc_pairs * 512)
    vals = np.frombuffer(raw, dtype="<i4").astype(np.float32)
    # vals shape: (oc_pairs, 128)
    vals = vals.reshape(oc_pairs, 128)
    result = np.empty((out_channels, spatial), dtype=np.float32)
    for p in range(oc_pairs):
        result[2 * p]     = vals[p, :spatial]
        result[2 * p + 1] = vals[p, spatial:2 * spatial]
    return result


def read_int8_multichannel_output(
    state: Any,
    base_addr: int,
    num_chunks: int,
    channels: int,
    rows_per_chunk: int,
    cols: int,
) -> np.ndarray:
    """Read INT8 multi-channel output (chunk-interleaved, 128 bytes/chunk).

    Layout: num_chunks × channels × 128 bytes.
    Returns (channels, rows_per_chunk*num_chunks, cols) int8 array.
    """
    raw = state.xmem.read_address(base_addr, num_chunks * channels * 128)
    vals = np.frombuffer(raw, dtype=np.uint8).reshape(num_chunks, channels, 128)
    out_rows = num_chunks * rows_per_chunk
    result = np.empty((channels, out_rows, cols), dtype=np.int8)
    for rg in range(num_chunks):
        for ch in range(channels):
            block = vals[rg, ch, : rows_per_chunk * cols].view(np.int8)
            result[ch, rg * rows_per_chunk : (rg + 1) * rows_per_chunk, :] = (
                block.reshape(rows_per_chunk, cols)
            )
    return result


def read_universal_output(
    state: Any,
    base_addr: int,
    num_chunks: int,
    channels: int,
    rows_per_chunk: int,
    cols: int,
) -> np.ndarray:
    """Read universal-harness output (chunk-interleaved layout).

    Layout: num_chunks × channels × 512 bytes (128 × INT32 each).
    Each 512-byte block holds rows_per_chunk rows × cols elements for one channel.

    Returns (channels, rows_per_chunk*num_chunks, cols) float32.
    """
    chunk_bytes = 512
    raw = state.xmem.read_address(base_addr, num_chunks * channels * chunk_bytes)
    vals = np.frombuffer(raw, dtype="<i4").astype(np.float32)
    vals = vals.reshape(num_chunks, channels, 128)
    out_rows = num_chunks * rows_per_chunk
    result = np.empty((channels, out_rows, cols), dtype=np.float32)
    for rg in range(num_chunks):
        for ch in range(channels):
            block = vals[rg, ch, : rows_per_chunk * cols]
            result[ch, rg * rows_per_chunk : (rg + 1) * rows_per_chunk, :] = (
                block.reshape(rows_per_chunk, cols)
            )
    return result


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correct(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    atol: float = 0.5,
) -> bool:
    """Compare actual vs expected arrays; return True if all within atol."""
    if actual.shape != expected.shape:
        print(f"  [SHAPE MISMATCH] {label}: actual {actual.shape} vs expected {expected.shape}")
        return False
    a = actual.astype(np.float32)
    e = expected.astype(np.float32)
    if not np.allclose(a, e, atol=atol):
        diff = np.abs(a - e)
        bad = int(np.sum(diff > atol))
        print(f"  [MISMATCH] {label}: {bad}/{actual.size} elements differ (max={diff.max():.1f})")
        return False
    return True


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table_header() -> None:
    print()
    print(f"{'Configuration':<40} {'cycles':>10}  {'MAC/cycle':>9}  {'% SOL':>7}  {'correct':>8}")
    print("-" * 82)


def print_table_row(
    name: str,
    cycles: int,
    macs: int,
    correct: bool,
) -> None:
    mac_per_cycle = macs / cycles if cycles else 0.0
    pct_sol = 100.0 * mac_per_cycle / SPEED_OF_LIGHT
    ok = "OK" if correct else "FAIL"
    print(
        f"{name:<40} {cycles:>10,}  {mac_per_cycle:>9.2f}  {pct_sol:>6.1f}%  {ok:>8}"
    )


def print_table_footer(title: str) -> None:
    print(f"\n  Speed-of-light: {SPEED_OF_LIGHT} MAC/cycle  |  {title}")
