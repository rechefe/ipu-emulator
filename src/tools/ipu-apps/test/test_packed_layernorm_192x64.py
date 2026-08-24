"""Packed LayerNorm, L4 shape: 192 channels x 64 tokens, PACKED 2
channels/row (96 rows). Measures correctness against numpy float64,
executed instruction count, cycles, and XMEM activation bytes, packed vs
the existing unpacked layernorm_64x192 kernel.

L4 port of test_packed_layernorm_240x16.py: partition_size(64)=64 ->
PACK=2 (see docs/isa_friction_log.md), vs L5's PACK=8.

Standalone/throwaway: no BUILD.bazel target. layernorm_64x192 is used
READ-ONLY (as the unpacked baseline for comparison) -- not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jinja2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import encode_dstructure

from fixture_packed_l4_measure import count_instructions

from ipu_apps.layernorm_64x192 import LayerNorm64x192App

_PACKED_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_layernorm_192x64.asm"
_UNPACKED_ASM_SRC = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/layernorm_64x192/layernorm_64x192.asm"
)

N_CH = 192
N_TOK = 64
LANES = 128
ROW_BYTES = 512
PACK = 2
N_PACKED_ROWS = N_CH // PACK
assert N_CH % PACK == 0

DATA_BASE_ROW = 0
NEG_MEAN_TILE_ROW = DATA_BASE_ROW + N_PACKED_ROWS
CENTERED_BASE_ROW = NEG_MEAN_TILE_ROW + 1
INVSTD_TILE_ROW = CENTERED_BASE_ROW + N_PACKED_ROWS
GAMMA_TILE_BASE_ROW = INVSTD_TILE_ROW + 1
BETA_TILE_BASE_ROW = GAMMA_TILE_BASE_ROW + N_PACKED_ROWS
OUTPUT_BASE_ROW = BETA_TILE_BASE_ROW + N_PACKED_ROWS
MASK_ROW = OUTPUT_BASE_ROW + N_PACKED_ROWS
ALLONES_MASK_ROW = MASK_ROW + 1
SCRATCH64_ROW = ALLONES_MASK_ROW + 1


def _pack(x: np.ndarray) -> np.ndarray:
    """x: [N_CH, N_TOK] -> [N_PACKED_ROWS, LANES], PACK channels/row."""
    assert x.shape == (N_CH, N_TOK)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = x[ch]
    return rows


def _unpack(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((N_CH, N_TOK), dtype=np.float64)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            out[ch] = rows[r, p * N_TOK:(p + 1) * N_TOK]
    return out


def _replicate_per_channel(vals: np.ndarray) -> np.ndarray:
    """vals: [N_CH] -> [N_PACKED_ROWS, LANES] tile, each channel's scalar
    repeated across its own 64-lane window (per-channel affine params, NOT
    per-token broadcast)."""
    assert vals.shape == (N_CH,)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = vals[ch]
    return rows


def _numpy_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """x: [N_CH, N_TOK] float64. Reduce over channel axis (axis=0) per token."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    invstd = 1.0 / np.sqrt(var)
    normalized = (x - mean) * invstd
    return normalized * gamma[:, None] + beta[:, None]


def test_packed_layernorm_correctness_and_cost(tmp_path: Path) -> None:
    rng = np.random.RandomState(11)
    X = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=(N_CH,)).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=(N_CH,)).astype(np.float32)

    expected = _numpy_layernorm(X.astype(np.float64), gamma.astype(np.float64), beta.astype(np.float64))

    # ---- Packed run ----
    reset_labels()
    packed_bin = tmp_path / "packed_layernorm.bin"
    template_text = _PACKED_ASM_SRC.read_text()
    rendered = jinja2.Template(template_text).render()
    assemble_to_bin_file(rendered, str(packed_bin))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, packed_bin)

    state.xmem.write_address(DATA_BASE_ROW * ROW_BYTES, bytearray(_pack(X).tobytes()))

    neg_inv_n_row = np.full((LANES,), -1.0 / N_CH, dtype=np.float32)
    inv_n_row = np.full((LANES,), 1.0 / N_CH, dtype=np.float32)
    state.xmem.write_address(NEG_MEAN_TILE_ROW * ROW_BYTES, bytearray(neg_inv_n_row.tobytes()))
    state.xmem.write_address(INVSTD_TILE_ROW * ROW_BYTES, bytearray(inv_n_row.tobytes()))

    gamma_tile = _replicate_per_channel(gamma)
    beta_tile = _replicate_per_channel(beta)
    state.xmem.write_address(GAMMA_TILE_BASE_ROW * ROW_BYTES, bytearray(gamma_tile.tobytes()))
    state.xmem.write_address(BETA_TILE_BASE_ROW * ROW_BYTES, bytearray(beta_tile.tobytes()))

    mrow = bytearray(128)
    for p_out in range(PACK):
        bits = 0
        for b in range(N_TOK * p_out, N_TOK * p_out + N_TOK):
            bits |= (1 << b)
        mrow[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    state.xmem.write_address(MASK_ROW * ROW_BYTES, bytes(mrow))
    state.xmem.write_address(ALLONES_MASK_ROW * ROW_BYTES, bytes([0xFF] * 128))

    # CR assignments -- must match asm_packed_layernorm_192x64.asm's register-name block
    state.regfile.set_cr(2, DATA_BASE_ROW)
    state.regfile.set_cr(3, N_PACKED_ROWS)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(5, SCRATCH64_ROW)
    state.regfile.set_cr(6, NEG_MEAN_TILE_ROW)
    state.regfile.set_cr(7, CENTERED_BASE_ROW)
    state.regfile.set_cr(8, INVSTD_TILE_ROW)
    state.regfile.set_cr(9, GAMMA_TILE_BASE_ROW)
    state.regfile.set_cr(10, BETA_TILE_BASE_ROW)
    state.regfile.set_cr(11, OUTPUT_BASE_ROW)
    state.regfile.set_cr(12, MASK_ROW)
    state.regfile.set_cr(13, encode_dstructure(valid_elements=N_TOK))
    state.regfile.set_cr(14, encode_dstructure(valid_elements=128))
    state.regfile.set_cr(15, ALLONES_MASK_ROW)

    with count_instructions() as packed_counts:
        packed_cycles = run_until_complete(state, max_cycles=200_000)

    packed_out_raw = state.xmem.read_address(OUTPUT_BASE_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    packed_out_rows = np.frombuffer(bytes(packed_out_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES)
    packed_out = _unpack(packed_out_rows)
    packed_err = float(np.max(np.abs(packed_out - expected)))

    packed_xmem_activation_bytes = (
        (N_PACKED_ROWS * 5 + 5) * ROW_BYTES
    )

    # ---- Unpacked run (existing layernorm_64x192, read-only baseline) ----
    reset_labels()
    unpacked_bin = tmp_path / "unpacked_layernorm.bin"
    assemble_to_bin_file(_UNPACKED_ASM_SRC.read_text(), str(unpacked_bin))

    x_path = tmp_path / "x.bin"
    gamma_path = tmp_path / "gamma.bin"
    beta_path = tmp_path / "beta.bin"
    out_path = tmp_path / "out.bin"

    x_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    x_rows[:, :N_TOK] = X
    x_path.write_bytes(x_rows.tobytes())

    gamma_path.write_bytes(gamma.tobytes())
    beta_path.write_bytes(beta.tobytes())

    unpacked_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    app = LayerNorm64x192App(
        inst_path=unpacked_bin, input_path=x_path, gamma_path=gamma_path, beta_path=beta_path,
        output_path=out_path,
    )
    with count_instructions() as unpacked_counts:
        _, unpacked_cycles = app.run(max_cycles=200_000, state=unpacked_state)

    unpacked_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(N_CH, LANES)[:, :N_TOK].astype(np.float64)
    unpacked_err = float(np.max(np.abs(unpacked_out - expected)))

    unpacked_xmem_activation_bytes = (N_CH * 3 + 4) * ROW_BYTES

    print("=== Packed vs unpacked LayerNorm, 192ch x 64tok (L4) ===")
    print(f"packed:   cycles={packed_cycles:6d}  instrs={packed_counts.total:6d}  "
          f"xmem_activation_bytes={packed_xmem_activation_bytes:8d}  max_abs_err={packed_err:.6e}")
    print(f"unpacked: cycles={unpacked_cycles:6d}  instrs={unpacked_counts.total:6d}  "
          f"xmem_activation_bytes={unpacked_xmem_activation_bytes:8d}  max_abs_err={unpacked_err:.6e}")
    print(f"packed/unpacked cycles ratio: {packed_cycles / unpacked_cycles:.4f}")
    print(f"packed/unpacked instr ratio:  {packed_counts.total / unpacked_counts.total:.4f}")
    print(f"packed/unpacked xmem ratio:   {packed_xmem_activation_bytes / unpacked_xmem_activation_bytes:.4f}")
    print(f"LAYERNORM_L4_PACKED_CYCLES={packed_cycles}")
    print(f"LAYERNORM_L4_PACKED_INSTRUCTIONS={packed_counts.total}")
    print(f"LAYERNORM_L4_PACKED_XMEM_BYTES={packed_xmem_activation_bytes}")
    print(f"LAYERNORM_L4_UNPACKED_CYCLES={unpacked_cycles}")
    print(f"LAYERNORM_L4_UNPACKED_INSTRUCTIONS={unpacked_counts.total}")
    print(f"LAYERNORM_L4_UNPACKED_XMEM_BYTES={unpacked_xmem_activation_bytes}")

    assert packed_err < 1e-3, f"packed layernorm wrong: max abs error {packed_err:.6e}"
    assert unpacked_err < 1e-3, f"unpacked layernorm wrong: max abs error {unpacked_err:.6e}"
