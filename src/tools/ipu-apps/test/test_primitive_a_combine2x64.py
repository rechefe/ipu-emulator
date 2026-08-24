"""Primitive A -- cross-partition combine, L4 shape, in isolation.

Given r_acc holding 2 partitions of 64 lanes, produces 64 results =
elementwise sum across the 2 partitions. Measures instruction count and
cycles for this primitive ALONE, isolated from any surrounding kernel.

L4 port of test_primitive_a_combine8x16.py: partition_size(64)=64 ->
parts_per_chunk=128/64=2 (see docs/isa_friction_log.md), so this primitive
combines 2 terms instead of L5's 8.

Standalone/throwaway: no BUILD.bazel app target, follows the direct
assemble_to_bin_file + IpuState pattern used by test_decisive_l5_uncropped.py.
Does not modify or import residual_add_64x192, qk_scores_64x48,
attn_v_64x48, attn_scores_km_64x48, attn_v_bcast_48, or any softmax kernel.
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

_ASM_SRC = Path(__file__).resolve().parent / "asm_primitive_a_combine2x64.asm"

ROW_BYTES = 512
SCRATCH_BASE_ROW = 0
OUT_BASE_ROW = 1

# CR indices, must match asm_primitive_a_combine2x64.asm's register-name block.
CR_OFF64 = 3
CR_DTYPE_ONE = 10
CR_DSTRUCT_WIDE = 14
CR_DSTRUCT_NARR = 15


def test_primitive_a_combine2x64(tmp_path: Path) -> None:
    reset_labels()
    bin_path = tmp_path / "a.bin"
    assemble_to_bin_file(_ASM_SRC.read_text(), str(bin_path))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)

    # 2 partitions x 64 lanes of known FP32 values.
    rng = np.random.RandomState(42)
    partitions = rng.uniform(-5.0, 5.0, size=(2, 64)).astype(np.float32)
    r_acc_bytes = partitions.reshape(-1).tobytes()
    assert len(r_acc_bytes) == 512
    state.regfile.set_r_acc_bytes(bytearray(r_acc_bytes))

    state.regfile.set_cr(0, SCRATCH_BASE_ROW)
    state.regfile.set_cr(2, OUT_BASE_ROW)
    state.regfile.set_cr(CR_OFF64, 64)
    state.regfile.set_cr(CR_DTYPE_ONE, 1)
    state.regfile.set_cr(CR_DSTRUCT_WIDE, encode_dstructure(valid_elements=128))
    state.regfile.set_cr(CR_DSTRUCT_NARR, encode_dstructure(valid_elements=64))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=1000)

    print(f"primitive A (2x64): {cycles} cycles, {counts.total} dynamic instructions")
    print(f"  by slot: {counts.by_slot}")

    out_bytes = state.xmem.read_address(OUT_BASE_ROW * ROW_BYTES, ROW_BYTES)
    got = np.frombuffer(bytes(out_bytes), dtype=np.float32)[:64]
    expected = partitions.sum(axis=0)

    err = float(np.max(np.abs(got - expected)))
    print(f"  max abs error vs numpy: {err:.6e}")

    assert err < 1e-4, f"combine result wrong: max abs error {err:.6e}"

    print(f"PRIMITIVE_A_L4_CYCLES={cycles}")
    print(f"PRIMITIVE_A_L4_INSTRUCTIONS={counts.total}")
