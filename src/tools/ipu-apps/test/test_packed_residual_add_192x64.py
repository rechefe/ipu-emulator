"""Packed elementwise residual add, L4 shape: 192 channels x 64 tokens,
PACKED 2 channels/row -- the "easy case" (no cross-partition combine).
Measures correctness, executed instruction count, cycles, and XMEM
activation bytes, packed vs the existing unpacked residual_add_64x192
kernel.

L4 port of test_packed_residual_add_240x16.py: partition_size(64)=64 ->
PACK=2 (see docs/isa_friction_log.md), vs L5's PACK=8. No width-dependent
constant changes beyond the packed row count -- elementwise add is
partition-local regardless of partition width.

Standalone/throwaway: no BUILD.bazel target for the new packed kernel,
follows the direct assemble_to_bin_file + IpuState pattern used by
test_decisive_l5_uncropped.py. residual_add_64x192 itself is used
READ-ONLY (as the unpacked baseline for comparison) -- not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import encode_dstructure

from fixture_packed_l4_measure import count_instructions

from ipu_apps.residual_add.residual_add_64x192 import ResidualAdd64x192App

_PACKED_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_residual_add_192x64.asm"
_UNPACKED_ASM_SRC = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/residual_add/residual_add_64x192/residual_add_64x192.asm"
)

N_CH = 192
N_TOK = 64
LANES = 128
ROW_BYTES = 512
PACK = 2                    # channels per packed row
N_PACKED_ROWS = N_CH // PACK
assert N_CH % PACK == 0

A_BASE_ROW = 0
B_BASE_ROW = A_BASE_ROW + N_PACKED_ROWS
OUT_BASE_ROW = B_BASE_ROW + N_PACKED_ROWS


def _pack(x: np.ndarray) -> bytes:
    """x: [N_CH, N_TOK] -> packed bytes, PACK channels/row, 64 lanes/channel,
    zero-padded to 128 lanes/row."""
    assert x.shape == (N_CH, N_TOK)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = x[ch]
    return rows.tobytes()


def _unpack(raw: bytes) -> np.ndarray:
    rows = np.frombuffer(raw, dtype=np.float32).reshape(N_PACKED_ROWS, LANES)
    out = np.zeros((N_CH, N_TOK), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            out[ch] = rows[r, p * N_TOK:(p + 1) * N_TOK]
    return out


def test_packed_residual_add_correctness_and_cost(tmp_path: Path) -> None:
    rng = np.random.RandomState(7)
    A = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)
    B = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)
    expected = (A.astype(np.float64) + B.astype(np.float64))

    # ---- Packed run ----
    reset_labels()
    packed_bin = tmp_path / "packed.bin"
    assemble_to_bin_file(_PACKED_ASM_SRC.read_text(encoding="utf-8"), str(packed_bin))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, packed_bin)
    state.xmem.write_address(A_BASE_ROW * ROW_BYTES, bytearray(_pack(A)))
    state.xmem.write_address(B_BASE_ROW * ROW_BYTES, bytearray(_pack(B)))
    state.regfile.set_cr(9, B_BASE_ROW)
    state.regfile.set_cr(3, OUT_BASE_ROW)
    state.regfile.set_cr(4, 0)
    state.regfile.set_cr(5, -1)
    state.regfile.set_cr(6, N_PACKED_ROWS)
    state.regfile.set_cr(7, 1)
    state.regfile.set_cr(8, 1)
    state.regfile.set_cr(10, 1)
    state.regfile.set_cr(15, encode_dstructure(valid_elements=128))

    with count_instructions() as packed_counts:
        packed_cycles = run_until_complete(state, max_cycles=100_000)

    packed_raw = bytes(state.xmem.read_address(OUT_BASE_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES))
    packed_out = _unpack(packed_raw)
    packed_err = float(np.max(np.abs(packed_out.astype(np.float64) - expected)))

    packed_xmem_activation_bytes = 3 * N_PACKED_ROWS * ROW_BYTES  # A + B + C resident

    # ---- Unpacked run (existing residual_add_64x192, read-only baseline) ----
    reset_labels()
    unpacked_bin = tmp_path / "unpacked.bin"
    assemble_to_bin_file(_UNPACKED_ASM_SRC.read_text(encoding="utf-8"), str(unpacked_bin))

    a_path = tmp_path / "a.bin"
    b_path = tmp_path / "b.bin"
    # residual_add_64x192 expects one full 512-byte row per channel (64
    # tokens live, 64 lanes padding) -- ONE CHANNEL PER ROW, per its own
    # header comment.
    a_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    b_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    a_rows[:, :N_TOK] = A
    b_rows[:, :N_TOK] = B
    a_path.write_bytes(a_rows.tobytes())
    b_path.write_bytes(b_rows.tobytes())
    out_path = tmp_path / "out.bin"

    unpacked_state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    app = ResidualAdd64x192App(
        inst_path=unpacked_bin, input_a_path=a_path, input_b_path=b_path, output_path=out_path,
    )
    with count_instructions() as unpacked_counts:
        _, unpacked_cycles = app.run(max_cycles=100_000, state=unpacked_state)

    unpacked_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(N_CH, LANES)[:, :N_TOK]
    unpacked_err = float(np.max(np.abs(unpacked_out.astype(np.float64) - expected)))

    unpacked_xmem_activation_bytes = 3 * N_CH * ROW_BYTES  # A + B + C resident, one channel/row

    print("=== packed vs unpacked residual add, 192ch x 64tok (L4) ===")
    print(f"packed:   cycles={packed_cycles:6d}  instrs={packed_counts.total:6d}  "
          f"xmem_activation_bytes={packed_xmem_activation_bytes:8d}  max_abs_err={packed_err:.6e}")
    print(f"unpacked: cycles={unpacked_cycles:6d}  instrs={unpacked_counts.total:6d}  "
          f"xmem_activation_bytes={unpacked_xmem_activation_bytes:8d}  max_abs_err={unpacked_err:.6e}")
    print(f"packed/unpacked cycles ratio: {packed_cycles / unpacked_cycles:.4f}")
    print(f"packed/unpacked instr ratio:  {packed_counts.total / unpacked_counts.total:.4f}")
    print(f"packed/unpacked xmem ratio:   {packed_xmem_activation_bytes / unpacked_xmem_activation_bytes:.4f}")
    print(f"RESIDUAL_L4_PACKED_CYCLES={packed_cycles}")
    print(f"RESIDUAL_L4_PACKED_INSTRUCTIONS={packed_counts.total}")
    print(f"RESIDUAL_L4_PACKED_XMEM_BYTES={packed_xmem_activation_bytes}")
    print(f"RESIDUAL_L4_UNPACKED_CYCLES={unpacked_cycles}")
    print(f"RESIDUAL_L4_UNPACKED_INSTRUCTIONS={unpacked_counts.total}")
    print(f"RESIDUAL_L4_UNPACKED_XMEM_BYTES={unpacked_xmem_activation_bytes}")

    assert packed_err < 1e-4, f"packed residual add wrong: max abs error {packed_err:.6e}"
    assert unpacked_err < 1e-4, f"unpacked residual add wrong: max abs error {unpacked_err:.6e}"
