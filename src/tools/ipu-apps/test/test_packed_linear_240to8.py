"""Kernel C (L5 packed-viability task, hard case): packed linear layer,
240 input channels -> 8 output channels, 16 tokens. Two packed paths,
measured against the unpacked baseline (test_linear_240to8_unpacked_baseline.py):

  path (a) -- pre-replicated weights (compute-optimal): weight scalar
    replicated 16x per partition, one MULT.RC.VV per chunk (30 chunks/output
    instead of 240 MULT.RC.VE calls), one primitive-A combine per output.
    16x weight memory.

  path (b) -- masked passes (memory-optimal): 8 masked MULT.RC.VE calls per
    chunk (one scalar each, R_MASK-gated to one partition), no weight
    replication, but back to 240 MULT ops per output (same as unpacked) --
    activation-memory win only, no compute win.

Standalone/throwaway: no BUILD.bazel target. Does not modify or import
residual_add_16x240, qk_scores_16x60, attn_v_16x60, or any softmax kernel.
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

from fixture_packed_l5_measure import count_instructions
from test_linear_240to8_unpacked_baseline import (
    run_unpacked_linear, K, N_OUT, N_TOK, LANES, ROW_BYTES,
)

_ASM_A_SRC = Path(__file__).resolve().parent / "asm_packed_linear_240to8_replicated.asm"
_ASM_B_SRC = Path(__file__).resolve().parent / "asm_packed_linear_240to8_masked.asm"

PACK = 8
N_CHUNKS = K // PACK  # 30
assert K % PACK == 0

W_CHUNKS = 2
TAIL_WIDTH = K - LANES * (W_CHUNKS - 1)  # 112

DATA_BASE_ROW_B = 0
WEIGHTS_BASE_ROW_B = DATA_BASE_ROW_B + N_CHUNKS
OUTPUT_BASE_ROW_B = WEIGHTS_BASE_ROW_B + N_OUT * W_CHUNKS

DATA_BASE_ROW_A = 0
WEIGHTS_BASE_ROW_A = DATA_BASE_ROW_A + N_CHUNKS
SCRATCH_BASE_ROW_A = WEIGHTS_BASE_ROW_A + N_OUT * N_CHUNKS
OUTPUT_BASE_ROW_A = SCRATCH_BASE_ROW_A + 1


def _pack_x(X: np.ndarray) -> np.ndarray:
    """X: [K, N_TOK] -> packed [N_CHUNKS, LANES], PACK channels/row."""
    rows = np.zeros((N_CHUNKS, LANES), dtype=np.float32)
    for c in range(N_CHUNKS):
        for p in range(PACK):
            k = c * PACK + p
            rows[c, p * N_TOK:(p + 1) * N_TOK] = X[k]
    return rows


def test_packed_linear_path_a_replicated_weights(tmp_path: Path) -> None:
    reset_labels()
    bin_path = tmp_path / "packed_a.bin"
    assemble_to_bin_file(_ASM_A_SRC.read_text(), str(bin_path))

    rng = np.random.RandomState(11)
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)

    packed_x = _pack_x(X)
    state.xmem.write_address(DATA_BASE_ROW_A * ROW_BYTES, bytearray(packed_x.tobytes()))

    # Weight rows: one per (o, chunk), W[o, 8c+p] replicated across
    # partition p's 16 lanes.
    w_rows = np.zeros((N_OUT * N_CHUNKS, LANES), dtype=np.float32)
    for o in range(N_OUT):
        for c in range(N_CHUNKS):
            row = np.zeros(LANES, dtype=np.float32)
            for p in range(PACK):
                k = c * PACK + p
                row[p * 16:(p + 1) * 16] = W[o, k]
            w_rows[o * N_CHUNKS + c] = row
    state.xmem.write_address(WEIGHTS_BASE_ROW_A * ROW_BYTES, bytearray(w_rows.tobytes()))

    state.regfile.set_cr(2, DATA_BASE_ROW_A - 1)  # pre-increment bias: co-issued LDR+ADD reads ptr+1
    state.regfile.set_cr(3, WEIGHTS_BASE_ROW_A)
    state.regfile.set_cr(4, OUTPUT_BASE_ROW_A)
    state.regfile.set_cr(5, N_CHUNKS)
    state.regfile.set_cr(6, SCRATCH_BASE_ROW_A)
    state.regfile.set_cr(7, 16)
    state.regfile.set_cr(8, 32)
    state.regfile.set_cr(9, 48)
    state.regfile.set_cr(10, 64)
    state.regfile.set_cr(11, 80)
    state.regfile.set_cr(12, 96)
    state.regfile.set_cr(13, N_OUT * N_CHUNKS)
    state.regfile.set_cr(14, encode_dstructure(valid_elements=128))
    state.regfile.set_cr(15, encode_dstructure(valid_elements=16))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = bytes(state.xmem.read_address(OUTPUT_BASE_ROW_A * ROW_BYTES, N_OUT * ROW_BYTES))
    out_rows = np.frombuffer(out_raw, dtype=np.float32).reshape(N_OUT, LANES)
    OUT = out_rows[:, :N_TOK]

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))

    # XMEM activation bytes: packed X (30 rows) + packed OUT (8 rows,
    # unavoidably 1 channel/row since only 8 output channels exist -- no
    # packing possible/needed below 8 rows). Weight bytes tracked
    # separately since weights are parameters, not activations, but
    # reported for the 16x-memory claim.
    packed_activation_bytes = (N_CHUNKS + N_OUT) * ROW_BYTES
    packed_weight_bytes = N_OUT * N_CHUNKS * ROW_BYTES
    unpacked_weight_bytes = N_OUT * K * 4  # 8*240 FP32 scalars, no padding needed at scalar granularity

    print("=== Kernel C path (a): packed linear 240->8, pre-replicated weights ===")
    print(f"cycles={cycles} instrs={counts.total} max_abs_err={err:.6e}")
    print(f"  by slot: {counts.by_slot}")
    print(f"  activation bytes: {packed_activation_bytes}")
    print(f"  weight bytes: {packed_weight_bytes} (unpacked equivalent: {unpacked_weight_bytes}, "
          f"ratio {packed_weight_bytes / unpacked_weight_bytes:.2f}x)")
    print(f"KERNEL_C_PATH_A_CYCLES={cycles}")
    print(f"KERNEL_C_PATH_A_INSTRUCTIONS={counts.total}")
    print(f"KERNEL_C_PATH_A_ACTIVATION_BYTES={packed_activation_bytes}")
    print(f"KERNEL_C_PATH_A_WEIGHT_BYTES={packed_weight_bytes}")

    assert err < 1e-3, f"path (a) packed linear wrong: max abs error {err:.6e}"


def test_packed_linear_path_b_masked_passes(tmp_path: Path) -> None:
    reset_labels()
    bin_path = tmp_path / "packed_b.bin"
    assemble_to_bin_file(_ASM_B_SRC.read_text(), str(bin_path))

    rng = np.random.RandomState(11)
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)

    packed_x = _pack_x(X)
    state.xmem.write_address(DATA_BASE_ROW_B * ROW_BYTES, bytearray(packed_x.tobytes()))

    # Weight rows: unpacked layout, identical to the unpacked baseline --
    # W_CHUNKS rows/output, up to 128 raw (unreplicated) scalars/row.
    w_rows = np.zeros((N_OUT * W_CHUNKS, LANES), dtype=np.float32)
    for o in range(N_OUT):
        w_rows[o * W_CHUNKS + 0, :LANES] = W[o, 0:LANES]
        w_rows[o * W_CHUNKS + 1, :TAIL_WIDTH] = W[o, LANES:K]
    state.xmem.write_address(WEIGHTS_BASE_ROW_B * ROW_BYTES, bytearray(w_rows.tobytes()))

    state.regfile.set_cr(2, DATA_BASE_ROW_B - 1)  # pre-increment bias
    state.regfile.set_cr(3, WEIGHTS_BASE_ROW_B)
    state.regfile.set_cr(4, OUTPUT_BASE_ROW_B)
    state.regfile.set_cr(8, W_CHUNKS)
    state.regfile.set_cr(9, N_OUT)
    state.regfile.set_cr(10, 16)
    state.regfile.set_cr(11, -16)  # NEG_SIXTEEN: rc_idx_reg pre-increment seed bias
    state.regfile.set_cr(12, -1)   # NEG_ONE: k_idx pre-increment seed bias
    state.regfile.set_cr(15, encode_dstructure(valid_elements=N_TOK))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = bytes(state.xmem.read_address(OUTPUT_BASE_ROW_B * ROW_BYTES, N_OUT * ROW_BYTES))
    out_rows = np.frombuffer(out_raw, dtype=np.float32).reshape(N_OUT, LANES)
    OUT = out_rows[:, :N_TOK]

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))

    packed_activation_bytes = (N_CHUNKS + N_OUT) * ROW_BYTES
    unpacked_weight_bytes = N_OUT * K * 4
    weight_load_instrs = N_OUT * W_CHUNKS  # 16 total, same as unpacked -- see docstring

    print("=== Kernel C path (b): packed linear 240->8, masked passes (unpacked weights) ===")
    print(f"cycles={cycles} instrs={counts.total} max_abs_err={err:.6e}")
    print(f"  by slot: {counts.by_slot}")
    print(f"  activation bytes: {packed_activation_bytes}")
    print(f"  weight bytes: {unpacked_weight_bytes} (same as unpacked, no replication)")
    print(f"  weight-load instructions: {weight_load_instrs} (2/output, same as unpacked)")
    print(f"KERNEL_C_PATH_B_CYCLES={cycles}")
    print(f"KERNEL_C_PATH_B_INSTRUCTIONS={counts.total}")
    print(f"KERNEL_C_PATH_B_ACTIVATION_BYTES={packed_activation_bytes}")
    print(f"KERNEL_C_PATH_B_WEIGHT_BYTES={unpacked_weight_bytes}")

    assert err < 1e-3, f"path (b) packed linear wrong: max abs error {err:.6e}"
