"""Task item 4: replication-count optimization for asm_packed_output_linear_
generic.asm's replicate_chunk() macro.

The task brief asked whether 2-slot replication suffices (down from the
original 4 slots) for rc_idx = 16*((p_in - p_out) mod 8). Direct
enumeration (see asm_packed_output_linear_1slot.asm's header) shows the
16-lane read window [rc_idx, rc_idx+15] never exceeds R_CYCLIC element 127
for ANY (p_in, p_out) pair -- the construction never leaves slot 0 at all,
so ONE slot suffices, not two. This is verified here against numpy float64
at K=240 (real L5 out-proj/QKV/FFN1 shape) and the K in {240,480,720}
boundary set, and measured against the original 4-slot kernel for cycle/
instruction savings.

Standalone/throwaway: no BUILD.bazel target. Does not modify
asm_packed_output_linear_generic.asm (kept as the read-only baseline for
comparison) or any production kernel.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jinja2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import encode_dstructure, Partition

from fixture_packed_l5_measure import count_instructions

_1SLOT_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_output_linear_1slot.asm"
_BASELINE_ASM_SRC = Path(__file__).resolve().parent / "asm_packed_output_linear_generic.asm"

LANES = 128
ROW_BYTES = 512
PACK = 8
N_TOK = 16


def _pack_x(X: np.ndarray, k: int) -> np.ndarray:
    n_chunks = k // PACK
    rows = np.zeros((n_chunks, LANES), dtype=np.float32)
    for c in range(n_chunks):
        for p in range(PACK):
            kk = c * PACK + p
            rows[c, p * N_TOK:(p + 1) * N_TOK] = X[kk]
    return rows


def _chunk_widths(k: int) -> list[int]:
    widths = []
    remaining = k
    while remaining > 0:
        w = min(LANES, remaining)
        assert w % 8 == 0
        widths.append(w)
        remaining -= w
    return widths


def _run(asm_path: Path, state: IpuState, X: np.ndarray, W: np.ndarray, *,
         k: int, n_out: int, max_cycles: int = 4_000_000):
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    n_chunks = k // PACK

    rendered = jinja2.Template(asm_path.read_text()).render(chunk_widths=widths)

    data_base_row = 0
    weights_base_row = data_base_row + n_chunks
    mask_row = weights_base_row + 8 * w_chunks * (n_out // 8)
    output_base_row = mask_row + 1

    state.xmem.write_address(data_base_row * ROW_BYTES, bytearray(_pack_x(X, k).tobytes()))

    mrow = bytearray(128)
    for p_out in range(8):
        bits = 0
        for b in range(16 * p_out, 16 * p_out + 16):
            bits |= (1 << b)
        mrow[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    state.xmem.write_address(mask_row * ROW_BYTES, bytes(mrow))

    total_cycles = 0
    total_counts: dict[str, int] = {}
    OUT = np.zeros((n_out, N_TOK), dtype=np.float32)

    for group in range(n_out // 8):
        W8 = W[group * 8:(group + 1) * 8]
        this_weights_base = weights_base_row + group * 8 * w_chunks
        this_output_row = output_base_row + group

        reset_labels()
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "k.bin"
            assemble_to_bin_file(rendered, str(bin_path))
            load_program_from_binary(state, bin_path)
            state.program_counter = 0

            w_rows = np.zeros((8 * w_chunks, LANES), dtype=np.float32)
            for c, width in enumerate(widths):
                off = sum(widths[:c])
                for p_out in range(8):
                    w_rows[c * 8 + p_out, :width] = W8[p_out, off:off + width]
            state.xmem.write_address(this_weights_base * ROW_BYTES, bytearray(w_rows.tobytes()))

            state.regfile.set_cr(2, data_base_row)
            state.regfile.set_cr(3, this_weights_base)
            state.regfile.set_cr(4, this_output_row)
            state.regfile.set_cr(5, mask_row)
            state.regfile.set_cr(6, encode_dstructure(valid_elements=128, partition=Partition.P8))
            state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
            for p_out in range(8):
                seed = (512 - 16 * p_out - 16) % 512
                state.regfile.set_cr(8 + p_out, seed)

            with count_instructions() as counts:
                cycles = run_until_complete(state, max_cycles=max_cycles)

            total_cycles += cycles
            for slot, n in counts.by_slot.items():
                total_counts[slot] = total_counts.get(slot, 0) + n

            out_raw = state.xmem.read_address(this_output_row * ROW_BYTES, ROW_BYTES)
            OUT[group * 8:(group + 1) * 8] = np.frombuffer(out_raw, dtype=np.float32).reshape(8, 16)

    return total_cycles, total_counts, OUT


def test_1slot_matches_baseline_correctness_and_saves_cycles() -> None:
    rng = np.random.RandomState(101)
    K, N_OUT = 240, 8
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state_1slot = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    cycles_1, counts_1, out_1 = _run(_1SLOT_ASM_SRC, state_1slot, X, W, k=K, n_out=N_OUT)
    err_1 = float(np.max(np.abs(out_1.astype(np.float64) - expected)))

    state_base = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    cycles_base, counts_base, out_base = _run(_BASELINE_ASM_SRC, state_base, X, W, k=K, n_out=N_OUT)
    err_base = float(np.max(np.abs(out_base.astype(np.float64) - expected)))

    print(f"1-slot:   cycles={cycles_1} instrs={sum(counts_1.values())} load_instrs={counts_1['load']} err={err_1:.6e}")
    print(f"4-slot:   cycles={cycles_base} instrs={sum(counts_base.values())} load_instrs={counts_base['load']} err={err_base:.6e}")
    print(f"cycles ratio (1slot/4slot): {cycles_1 / cycles_base:.4f}")
    print(f"load-instr ratio (1slot/4slot): {counts_1['load'] / counts_base['load']:.4f}")
    print(f"REPLICATION_1SLOT_CYCLES={cycles_1}")
    print(f"REPLICATION_4SLOT_CYCLES={cycles_base}")
    print(f"REPLICATION_1SLOT_LOAD_INSTRS={counts_1['load']}")
    print(f"REPLICATION_4SLOT_LOAD_INSTRS={counts_base['load']}")

    assert err_1 < 1e-3, f"1-slot variant wrong: {err_1:.6e}"
    assert err_base < 1e-3, f"4-slot baseline wrong: {err_base:.6e}"
    assert cycles_1 < cycles_base, "1-slot variant should be faster, not slower"


def test_1slot_boundary_shapes() -> None:
    rng = np.random.RandomState(202)
    N_OUT = 8
    for K in (240, 480, 720):
        X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
        W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
        expected = W.astype(np.float64) @ X.astype(np.float64)

        state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        cycles, counts, out = _run(_1SLOT_ASM_SRC, state, X, W, k=K, n_out=N_OUT)
        err = float(np.max(np.abs(out.astype(np.float64) - expected)))
        print(f"K={K}: cycles={cycles} instrs={sum(counts.values())} err={err:.6e}")
        assert err < 1e-3, f"K={K}: 1-slot variant wrong: {err:.6e}"
