"""On-chip pack/unpack kernels for the attention seam: packed (2 channels/
row, 96 rows) <-> unpacked (one channel per row, 192 rows), 192 channels x
64 tokens (L4 shape). Measures correctness and cost for both directions.

L4 port of test_packed_pack_unpack_240x16.py: partition_size(64)=64 ->
PACK=128/64=2 (see docs/isa_friction_log.md), vs L5's PACK=8.

Standalone/throwaway: no BUILD.bazel target. Does not modify or import
qk_scores_64x48, attn_v_64x48, attn_scores_km_64x48, attn_v_bcast_48,
layernorm_64x192, or any softmax kernel.
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

_UNPACK_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_unpack_192x64.asm"
_PACK_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_pack_192x64.asm"

N_CH = 192
N_TOK = 64
LANES = 128
ROW_BYTES = 512
PACK = 2
N_PACKED_ROWS = N_CH // PACK
assert N_CH % PACK == 0


def _pack(x: np.ndarray) -> np.ndarray:
    assert x.shape == (N_CH, N_TOK)
    rows = np.zeros((N_PACKED_ROWS, LANES), dtype=np.float32)
    for r in range(N_PACKED_ROWS):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * N_TOK:(p + 1) * N_TOK] = x[ch]
    return rows


def _mask_row() -> bytes:
    mrow = bytearray(128)
    for p_out in range(PACK):
        bits = 0
        for b in range(N_TOK * p_out, N_TOK * p_out + N_TOK):
            bits |= (1 << b)
        mrow[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    return bytes(mrow)


def test_unpack_correctness_and_cost(tmp_path: Path) -> None:
    rng = np.random.RandomState(21)
    X = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)

    PACKED_BASE_ROW = 0
    UNPACKED_BASE_ROW = PACKED_BASE_ROW + N_PACKED_ROWS
    MASK_ROW = UNPACKED_BASE_ROW + N_CH

    reset_labels()
    rendered = jinja2.Template(_UNPACK_ASM_SRC.read_text()).render()
    binpath = tmp_path / "unpack.bin"
    assemble_to_bin_file(rendered, str(binpath))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, binpath)
    state.xmem.write_address(PACKED_BASE_ROW * ROW_BYTES, bytearray(_pack(X).tobytes()))
    state.xmem.write_address(MASK_ROW * ROW_BYTES, _mask_row())

    state.regfile.set_cr(2, PACKED_BASE_ROW)
    state.regfile.set_cr(3, N_PACKED_ROWS)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(5, UNPACKED_BASE_ROW)
    state.regfile.set_cr(6, MASK_ROW)
    state.regfile.set_cr(7, encode_dstructure(valid_elements=N_TOK))
    state.regfile.set_cr(8, encode_dstructure(valid_elements=128))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = state.xmem.read_address(UNPACKED_BASE_ROW * ROW_BYTES, N_CH * ROW_BYTES)
    out_rows = np.frombuffer(bytes(out_raw), dtype=np.float32).reshape(N_CH, LANES)
    out = out_rows[:, :N_TOK].astype(np.float64)
    err = float(np.max(np.abs(out - X.astype(np.float64))))

    activation_bytes = (N_PACKED_ROWS + N_CH + 1) * ROW_BYTES

    print(f"UNPACK: cycles={cycles} instrs={counts.total} err={err:.6e} "
          f"activation_bytes={activation_bytes}")
    print(f"UNPACK_L4_CYCLES={cycles}")
    print(f"UNPACK_L4_INSTRUCTIONS={counts.total}")
    print(f"UNPACK_L4_ACTIVATION_BYTES={activation_bytes}")

    assert err < 1e-5, f"unpack wrong: max abs error {err:.6e}"


def test_pack_correctness_and_cost(tmp_path: Path) -> None:
    rng = np.random.RandomState(22)
    X = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)

    UNPACKED_BASE_ROW = 0
    PACKED_BASE_ROW = UNPACKED_BASE_ROW + N_CH
    MASK_ROW = PACKED_BASE_ROW + N_PACKED_ROWS

    reset_labels()
    rendered = jinja2.Template(_PACK_ASM_SRC.read_text()).render()
    binpath = tmp_path / "pack.bin"
    assemble_to_bin_file(rendered, str(binpath))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, binpath)

    x_rows = np.zeros((N_CH, LANES), dtype=np.float32)
    x_rows[:, :N_TOK] = X
    state.xmem.write_address(UNPACKED_BASE_ROW * ROW_BYTES, bytearray(x_rows.tobytes()))
    state.xmem.write_address(MASK_ROW * ROW_BYTES, _mask_row())

    state.regfile.set_cr(2, UNPACKED_BASE_ROW)
    state.regfile.set_cr(3, N_PACKED_ROWS)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(5, PACKED_BASE_ROW)
    state.regfile.set_cr(6, MASK_ROW)
    state.regfile.set_cr(7, encode_dstructure(valid_elements=128))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = state.xmem.read_address(PACKED_BASE_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    out_rows = np.frombuffer(bytes(out_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES)
    expected_packed = _pack(X)
    err = float(np.max(np.abs(out_rows.astype(np.float64) - expected_packed.astype(np.float64))))

    activation_bytes = (N_CH + N_PACKED_ROWS + 1) * ROW_BYTES

    print(f"PACK: cycles={cycles} instrs={counts.total} err={err:.6e} "
          f"activation_bytes={activation_bytes}")
    print(f"PACK_L4_CYCLES={cycles}")
    print(f"PACK_L4_INSTRUCTIONS={counts.total}")
    print(f"PACK_L4_ACTIVATION_BYTES={activation_bytes}")

    assert err < 1e-5, f"pack wrong: max abs error {err:.6e}"


def test_pack_unpack_roundtrip(tmp_path: Path) -> None:
    """pack(unpack(X)) == X, chained on-chip with no host-side conversion
    between the two kernel runs (only XMEM reads/writes by the harness,
    which is legal load-time staging, not a conversion between two
    kernels' data already resident in XMEM)."""
    rng = np.random.RandomState(23)
    X = rng.uniform(-3.0, 3.0, size=(N_CH, N_TOK)).astype(np.float32)

    PACKED_BASE_ROW = 0
    UNPACKED_BASE_ROW = PACKED_BASE_ROW + N_PACKED_ROWS
    MASK_ROW = UNPACKED_BASE_ROW + N_CH

    reset_labels()
    unpack_rendered = jinja2.Template(_UNPACK_ASM_SRC.read_text()).render()
    unpack_bin = tmp_path / "unpack_rt.bin"
    assemble_to_bin_file(unpack_rendered, str(unpack_bin))

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, unpack_bin)
    state.xmem.write_address(PACKED_BASE_ROW * ROW_BYTES, bytearray(_pack(X).tobytes()))
    state.xmem.write_address(MASK_ROW * ROW_BYTES, _mask_row())
    state.regfile.set_cr(2, PACKED_BASE_ROW)
    state.regfile.set_cr(3, N_PACKED_ROWS)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(5, UNPACKED_BASE_ROW)
    state.regfile.set_cr(6, MASK_ROW)
    state.regfile.set_cr(7, encode_dstructure(valid_elements=N_TOK))
    state.regfile.set_cr(8, encode_dstructure(valid_elements=128))
    run_until_complete(state, max_cycles=100_000)

    # ---- Re-pack the SAME state's XMEM (no host touch) using the pack kernel ----
    PACKED2_BASE_ROW = MASK_ROW + 1
    MASK2_ROW = PACKED2_BASE_ROW + N_PACKED_ROWS

    reset_labels()
    pack_rendered = jinja2.Template(_PACK_ASM_SRC.read_text()).render()
    pack_bin = tmp_path / "pack_rt.bin"
    assemble_to_bin_file(pack_rendered, str(pack_bin))
    load_program_from_binary(state, pack_bin)
    state.program_counter = 0
    state.xmem.write_address(MASK2_ROW * ROW_BYTES, _mask_row())
    state.regfile.set_cr(2, UNPACKED_BASE_ROW)
    state.regfile.set_cr(3, N_PACKED_ROWS)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(5, PACKED2_BASE_ROW)
    state.regfile.set_cr(6, MASK2_ROW)
    state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
    run_until_complete(state, max_cycles=100_000)

    roundtrip_raw = state.xmem.read_address(PACKED2_BASE_ROW * ROW_BYTES, N_PACKED_ROWS * ROW_BYTES)
    roundtrip = np.frombuffer(bytes(roundtrip_raw), dtype=np.float32).reshape(N_PACKED_ROWS, LANES)
    expected = _pack(X)
    err = float(np.max(np.abs(roundtrip.astype(np.float64) - expected.astype(np.float64))))
    print(f"ROUNDTRIP max abs err: {err:.6e}")
    assert err < 1e-5, f"pack(unpack(X)) roundtrip wrong: max abs error {err:.6e}"
