"""Packed-OUTPUT linear layer at REAL L5 sizes: QKV (240->720), out-proj
(240->240), FFN1 (240->480), FFN2 (480->240).

Each real shape needs N_OUT/8 separate KERNEL RUNS (asm_packed_output_linear_
generic.asm always produces exactly one packed row of 8 output channels per
run -- the R_MASK 8-slot / 8-partition alignment this construction depends
on only holds at exactly 8 outputs per packed row). This is still zero
host-side data conversion between runs: each run reads the SAME packed X
(already resident in XMEM, never touched by numpy) and a different 8-channel
weight slice (chosen by which CRs the harness loads before each run -- a
compile/load-time choice, explicitly legal per the brief), and writes its
own independent packed output row. No run's output feeds another run's
input; nothing is unpacked and repacked between calls.

Standalone/throwaway: no BUILD.bazel target. Does not modify or import
residual_add_16x240, qk_scores_16x60, attn_v_16x60, or any softmax kernel.
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

from fixture_packed_l5_measure import count_instructions

_ASM_TEMPLATE_SRC = Path(__file__).resolve().parent / "asm_packed_output_linear_generic.asm"

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


def run_packed_output_linear_full(state: IpuState, X: np.ndarray, W: np.ndarray, *,
                                   k: int, n_out: int, max_cycles: int = 4_000_000):
    """X: [k, 16], W: [n_out, k]. Runs the kernel n_out/8 times (one packed
    row of 8 outputs per run). Returns (total_cycles, total_instr_counts_by_slot,
    OUT [n_out, 16], per_run_cycles list).
    """
    assert n_out % 8 == 0
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    n_chunks = k // PACK

    template_text = _ASM_TEMPLATE_SRC.read_text()
    rendered = jinja2.Template(template_text).render(chunk_widths=widths)

    data_base_row = 0
    weights_base_row = data_base_row + n_chunks
    mask_row = weights_base_row + 8 * w_chunks * (n_out // 8)
    output_base_row = mask_row + 1

    packed_x = _pack_x(X, k)
    state.xmem.write_address(data_base_row * ROW_BYTES, bytearray(packed_x.tobytes()))

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
    per_run_cycles = []

    for group in range(n_out // 8):
        W8 = W[group * 8:(group + 1) * 8]
        this_weights_base = weights_base_row + group * 8 * w_chunks
        this_output_row = output_base_row + group

        reset_labels()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "packed_output.bin"
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
            per_run_cycles.append(cycles)
            for slot, n in counts.by_slot.items():
                total_counts[slot] = total_counts.get(slot, 0) + n

            out_raw = bytes(state.xmem.read_address(this_output_row * ROW_BYTES, ROW_BYTES))
            out_packed = np.frombuffer(out_raw, dtype=np.float32).reshape(8, 16)
            OUT[group * 8:(group + 1) * 8] = out_packed

    return total_cycles, total_counts, OUT, per_run_cycles


SHAPES = {
    "QKV":      dict(k=240, n_out=720),
    "outproj":  dict(k=240, n_out=240),
    "FFN1":     dict(k=240, n_out=480),
    "FFN2":     dict(k=480, n_out=240),
}


def _run_one(name: str, k: int, n_out: int, seed: int):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.0, 2.0, size=(k, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(n_out, k)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    cycles, counts, OUT, per_run = run_packed_output_linear_full(state, X, W, k=k, n_out=n_out)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    n_chunks = k // PACK
    output_activation_bytes = (n_out // 8) * ROW_BYTES
    input_activation_bytes = n_chunks * ROW_BYTES
    total_instrs = sum(counts.values())

    print(f"{name}: PACKED-OUTPUT k={k} n_out={n_out} cycles={cycles} instrs={total_instrs} "
          f"err={err:.6e} input_act_bytes={input_activation_bytes} "
          f"output_act_bytes={output_activation_bytes} runs={n_out // 8} "
          f"cycles_per_run_range=[{min(per_run)},{max(per_run)}]")
    print(f"  by slot totals: {counts}")

    assert err < 1e-3, f"{name}: packed-output wrong: max abs error {err:.6e}"
    return dict(cycles=cycles, instrs=total_instrs, err=err,
                input_act_bytes=input_activation_bytes,
                output_act_bytes=output_activation_bytes)


def test_packed_output_qkv():
    _run_one("QKV", **SHAPES["QKV"], seed=101)


def test_packed_output_outproj():
    _run_one("outproj", **SHAPES["outproj"], seed=102)


def test_packed_output_ffn1():
    _run_one("FFN1", **SHAPES["FFN1"], seed=103)


def test_packed_output_ffn2():
    _run_one("FFN2", **SHAPES["FFN2"], seed=104)
