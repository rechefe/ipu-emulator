"""Packed-OUTPUT linear layer, L4 shape (K variable, N_OUT=2/call). Measures
correctness against numpy float64, and cost, at L4's real K shapes
(QKV/OutProj K=192, FFN2 K=384).

L4 port of test_packed_output_linear_generic.py: partition_size(64)=64 ->
parts_per_chunk=128/64=2 (see docs/isa_friction_log.md), so this kernel's
rc_idx formula is 64*(p_in-p_out) mod 512, N_OUT is fixed at 2/call (not
8), and -- the key re-derived finding -- replication needs 2 R_CYCLIC
slots (0 and 3), not L5's validated 1-slot optimization (that finding was
specific to L5's rc_idx range and does not carry over; see the .asm
header's enumeration).

Standalone/throwaway: no BUILD.bazel target.
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
from ipu_emu.ipu_config import encode_dstructure, Partition

from fixture_packed_l4_measure import count_instructions

_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_output_linear_generic_p4.asm"

LANES = 128
ROW_BYTES = 512
PACK = 2


def _chunk_widths(k: int) -> list[int]:
    widths = []
    remaining = k
    while remaining > 0:
        w = min(LANES, remaining)
        assert w % PACK == 0
        widths.append(w)
        remaining -= w
    return widths


def _mask_row_2() -> bytes:
    mrow = bytearray(128)
    for p_out in range(PACK):
        bits = 0
        for b in range(64 * p_out, 64 * p_out + 64):
            bits |= (1 << b)
        mrow[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    return bytes(mrow)


def run_packed_output_linear_l4(state: IpuState, *, asm_src: Path, data_base_row: int, k: int,
                                 n_out: int, weight_slices: list, output_base_row: int,
                                 scratch_base_row: int, tmp_path: Path,
                                 max_cycles: int = 4_000_000):
    """weight_slices: list of n_out//2 arrays, each [2, k]. Writes n_out//2
    packed output rows starting at output_base_row."""
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    mask_row = scratch_base_row
    weights_base_row = mask_row + 1

    rendered = jinja2.Template(asm_src.read_text()).render(chunk_widths=widths)
    state.xmem.write_address(mask_row * ROW_BYTES, _mask_row_2())

    total_cycles = 0
    total_counts: dict[str, int] = {}

    for group in range(n_out // PACK):
        W2 = weight_slices[group]
        this_weights_base = weights_base_row + group * PACK * w_chunks
        this_output_row = output_base_row + group

        reset_labels()
        bin_path = tmp_path / f"lin_{group}.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)
        state.program_counter = 0

        w_rows = np.zeros((PACK * w_chunks, LANES), dtype=np.float32)
        for c, width in enumerate(widths):
            off = sum(widths[:c])
            for p_out in range(PACK):
                w_rows[c * PACK + p_out, :width] = W2[p_out, off:off + width]
        state.xmem.write_address(this_weights_base * ROW_BYTES, bytearray(w_rows.tobytes()))

        state.regfile.set_cr(2, data_base_row)
        state.regfile.set_cr(3, this_weights_base)
        state.regfile.set_cr(4, this_output_row)
        state.regfile.set_cr(5, mask_row)
        state.regfile.set_cr(6, encode_dstructure(valid_elements=128, partition=Partition.P2))
        state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
        for p_out in range(PACK):
            seed = (512 - 64 * p_out - 64) % 512
            state.regfile.set_cr(8 + p_out, seed)

        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)
        total_cycles += cycles
        for slot, n in counts.by_slot.items():
            total_counts[slot] = total_counts.get(slot, 0) + n

    return total_cycles, total_counts


def _pack_output(x_packed_rows: np.ndarray, n_out: int, n_tok: int) -> np.ndarray:
    out = np.zeros((n_out, n_tok))
    for r in range(n_out // PACK):
        for p in range(PACK):
            out[r * PACK + p] = x_packed_rows[r, p * n_tok:(p + 1) * n_tok]
    return out


def _pack_input(x: np.ndarray, k: int, n_tok: int) -> np.ndarray:
    n_packed_rows = k // PACK
    rows = np.zeros((n_packed_rows, LANES), dtype=np.float32)
    for r in range(n_packed_rows):
        for p in range(PACK):
            ch = r * PACK + p
            rows[r, p * n_tok:(p + 1) * n_tok] = x[ch]
    return rows


def test_packed_output_linear_k192_n8(tmp_path: Path) -> None:
    """K=192 (QKV/OutProj's real K), small N_OUT=8 for a fast correctness
    check with a runtime multi-weight-chunk loop exercised (chunk_widths =
    [128, 64])."""
    K = 192
    N_OUT = 8
    N_TOK = 64
    rng = np.random.RandomState(101)
    X = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.1, 0.1, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    DATA_BASE_ROW = 0
    SCRATCH_BASE_ROW = DATA_BASE_ROW + K // PACK
    OUTPUT_BASE_ROW = SCRATCH_BASE_ROW + 1 + PACK * len(_chunk_widths(K)) * (N_OUT // PACK)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(DATA_BASE_ROW * ROW_BYTES, bytearray(_pack_input(X, K, N_TOK).tobytes()))

    weight_slices = [W[g * PACK:(g + 1) * PACK] for g in range(N_OUT // PACK)]
    cycles, counts = run_packed_output_linear_l4(
        state, asm_src=_ASM_SRC, data_base_row=DATA_BASE_ROW, k=K, n_out=N_OUT,
        weight_slices=weight_slices, output_base_row=OUTPUT_BASE_ROW,
        scratch_base_row=SCRATCH_BASE_ROW, tmp_path=tmp_path,
    )

    out_raw = state.xmem.read_address(OUTPUT_BASE_ROW * ROW_BYTES, (N_OUT // PACK) * ROW_BYTES)
    out_rows = np.frombuffer(bytes(out_raw), dtype=np.float32).reshape(N_OUT // PACK, LANES)
    out = _pack_output(out_rows, N_OUT, N_TOK)
    err = float(np.max(np.abs(out - expected)))

    print(f"PACKED_OUTPUT_LINEAR_L4_K192N8_CYCLES={cycles}")
    print(f"PACKED_OUTPUT_LINEAR_L4_K192N8_INSTRUCTIONS={sum(counts.values())}")
    print(f"PACKED_OUTPUT_LINEAR_L4_K192N8_ERROR={err:.6e}")
    assert err < 1e-3, f"packed output linear (K=192,N_OUT=8) wrong: {err:.6e}"


def test_packed_output_linear_k384_silu(tmp_path: Path) -> None:
    """K=384 (FFN2's real K, 3 weight-chunks of [128,128,128]) with the
    silu variant (FFN1's own activation, exercised here on the K=384 shape
    purely to also exercise the 3-weight-chunk runtime loop path with a
    non-identity activation)."""
    asm_src = Path(__file__).resolve().parent / "asm_packed_output_linear_silu_p4.asm"
    K = 384
    N_OUT = 4
    N_TOK = 64
    rng = np.random.RandomState(102)
    X = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.05, 0.05, size=(N_OUT, K)).astype(np.float32)

    def silu_np(x):
        return x / (1.0 + np.exp(-x))

    expected = silu_np(W.astype(np.float64) @ X.astype(np.float64))

    DATA_BASE_ROW = 0
    SCRATCH_BASE_ROW = DATA_BASE_ROW + K // PACK
    OUTPUT_BASE_ROW = SCRATCH_BASE_ROW + 1 + PACK * len(_chunk_widths(K)) * (N_OUT // PACK)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    state.xmem.write_address(DATA_BASE_ROW * ROW_BYTES, bytearray(_pack_input(X, K, N_TOK).tobytes()))

    weight_slices = [W[g * PACK:(g + 1) * PACK] for g in range(N_OUT // PACK)]
    cycles, counts = run_packed_output_linear_l4(
        state, asm_src=asm_src, data_base_row=DATA_BASE_ROW, k=K, n_out=N_OUT,
        weight_slices=weight_slices, output_base_row=OUTPUT_BASE_ROW,
        scratch_base_row=SCRATCH_BASE_ROW, tmp_path=tmp_path,
    )

    out_raw = state.xmem.read_address(OUTPUT_BASE_ROW * ROW_BYTES, (N_OUT // PACK) * ROW_BYTES)
    out_rows = np.frombuffer(bytes(out_raw), dtype=np.float32).reshape(N_OUT // PACK, LANES)
    out = _pack_output(out_rows, N_OUT, N_TOK)
    err = float(np.max(np.abs(out - expected)))

    print(f"PACKED_OUTPUT_LINEAR_L4_K384SILU_CYCLES={cycles}")
    print(f"PACKED_OUTPUT_LINEAR_L4_K384SILU_INSTRUCTIONS={sum(counts.values())}")
    print(f"PACKED_OUTPUT_LINEAR_L4_K384SILU_ERROR={err:.6e}")
    assert err < 1e-3, f"packed output linear (K=384,silu) wrong: {err:.6e}"


def test_replication_slot_enumeration() -> None:
    """Direct enumeration (not assumption) of which R_CYCLIC slots the
    rc_idx=64*(p_in-p_out) mod 512 formula actually touches, for all 4
    (p_in, p_out) pairs in L4's 2-partition range. This is the evidence
    behind the "2 slots (0 and 3), not 1, not 4" finding stated in the
    .asm header and docs/isa_friction_log.md -- a direct CONTRADICTION of
    the L5 packed-output-linear kernel's validated "1 slot suffices"
    result (that finding was proven specific to L5's rc_idx range, not a
    general ISA property)."""
    ps = 64
    touched_slots = set()
    pairs = []
    for p_in in range(2):
        for p_out in range(2):
            rc_idx = (ps * (p_in - p_out)) % 512
            max_read = rc_idx + ps - 1
            slot_start = rc_idx // 128
            slot_end = max_read // 128
            pairs.append((p_in, p_out, rc_idx, max_read, slot_start, slot_end))
            touched_slots.add(slot_start)
            touched_slots.add(slot_end)

    for p_in, p_out, rc_idx, max_read, slot_start, slot_end in pairs:
        print(f"p_in={p_in} p_out={p_out} rc_idx={rc_idx} max_read={max_read} "
              f"slots=[{slot_start},{slot_end}]")

    print(f"REPLICATION_SLOTS_TOUCHED={sorted(touched_slots)}")
    assert touched_slots == {0, 3}, (
        f"expected L4's rc_idx formula to touch exactly slots {{0,3}}, got {touched_slots}"
    )
