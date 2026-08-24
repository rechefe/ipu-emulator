"""Validates the user's proposed "packed-output linear kernel" construction
that would refute last session's item-3 conclusion ("path (b) never
produces packed output, therefore packing can't chain").

Construction: rc_idx = 16*(p_in - p_out) mod 512 lands input partition
p_in's 16 lanes at mult_res lanes 16*p_out..16*p_out+15 for ANY output
partition p_out (not just p_in==p_out as the original path (b) always
used), masked to that 16-lane window via mask_offset=p_out. ACC.ADD
(never reset between output channels) accumulates disjoint lane-ranges of
ONE shared r_acc, so all 8 output channels land packed in one row after
one store.

TINY validation shape: K=8 (exactly one packed chunk), N_OUT=8, N_TOK=16 --
the smallest shape that exercises every (p_in, p_out) pair. Standalone,
throwaway: no BUILD.bazel target, does not import or modify
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
from ipu_emu.ipu_config import encode_dstructure, Partition

from fixture_packed_l5_measure import count_instructions

_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_output_linear_tiny.asm"

LANES = 128
ROW_BYTES = 512
K = 8
N_OUT = 8
N_TOK = 16

DATA_BASE_ROW = 0
WEIGHTS_BASE_ROW = 1
OUTPUT_BASE_ROW = 9
MASK_ROW = 10


def _build_mask_row() -> bytes:
    row = bytearray(128)
    for p_out in range(8):
        bits = 0
        for b in range(16 * p_out, 16 * p_out + 16):
            bits |= (1 << b)
        row[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    return bytes(row)


def test_packed_output_linear_tiny_correctness(tmp_path: Path) -> None:
    reset_labels()
    bin_path = tmp_path / "packed_output_tiny.bin"
    assemble_to_bin_file(_ASM_SRC.read_text(), str(bin_path))

    rng = np.random.RandomState(7)
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)

    packed_x = np.zeros(LANES, dtype=np.float32)
    for p_in in range(8):
        packed_x[p_in * 16:(p_in + 1) * 16] = X[p_in]
    state.xmem.write_address(DATA_BASE_ROW * ROW_BYTES, bytearray(packed_x.tobytes()))

    w_rows = np.zeros((N_OUT, LANES), dtype=np.float32)
    for p_out in range(N_OUT):
        w_rows[p_out, :K] = W[p_out]
    state.xmem.write_address(WEIGHTS_BASE_ROW * ROW_BYTES, bytearray(w_rows.tobytes()))

    state.xmem.write_address(MASK_ROW * ROW_BYTES, _build_mask_row())

    state.regfile.set_cr(2, DATA_BASE_ROW)
    state.regfile.set_cr(3, WEIGHTS_BASE_ROW)
    state.regfile.set_cr(4, OUTPUT_BASE_ROW)
    state.regfile.set_cr(5, 16)
    state.regfile.set_cr(6, -16)
    state.regfile.set_cr(7, -1)
    state.regfile.set_cr(8, MASK_ROW)
    state.regfile.set_cr(9, 128)
    state.regfile.set_cr(10, 256)
    state.regfile.set_cr(11, 384)
    state.regfile.set_cr(14, encode_dstructure(valid_elements=128, partition=Partition.P8))
    state.regfile.set_cr(15, encode_dstructure(valid_elements=128))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = bytes(state.xmem.read_address(OUTPUT_BASE_ROW * ROW_BYTES, ROW_BYTES))
    out_packed = np.frombuffer(out_raw, dtype=np.float32).reshape(N_OUT, 16)

    err = float(np.max(np.abs(out_packed.astype(np.float64) - expected)))

    print(f"packed-output linear tiny (K=8,N_OUT=8): cycles={cycles} instrs={counts.total} "
          f"max_abs_err={err:.6e}")
    print(f"  by slot: {counts.by_slot}")
    print(f"  store count: {counts.by_slot.get('store', 0) + counts.by_slot.get('aaq', 0)}")
    print(f"  output activation bytes: {ROW_BYTES} (1 packed row vs {N_OUT}*{ROW_BYTES}={N_OUT*ROW_BYTES} unpacked)")

    assert err < 1e-3, f"packed-output linear wrong: max abs error {err:.6e}"
