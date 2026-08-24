"""Kernel C unpacked baseline (L5 packed-viability task): 240 input
channels -> 8 output channels, 16 tokens, ONE CHANNEL PER ROW (the standing
L5 convention). Standalone smoke test to confirm the baseline is correct
before comparing packed variants against it.

run_unpacked_linear_generic also backs the real-size extrapolation
(QKV/outproj/FFN1/FFN2) in test_l5_real_size_extrapolation.py -- it is
parameterized over (K, N_OUT), not hardcoded to 240->8.
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

_ASM_SRC = Path(__file__).resolve().parent / "asm_unpacked_linear_240to8.asm"

K = 240
N_OUT = 8
N_TOK = 16
LANES = 128
ROW_BYTES = 512
W_CHUNKS = 2
TAIL_WIDTH = K - LANES * (W_CHUNKS - 1)  # 112

DATA_BASE_ROW = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + K
OUTPUT_BASE_ROW = WEIGHTS_BASE_ROW + N_OUT * W_CHUNKS


def run_unpacked_linear(state: IpuState, X: np.ndarray, W: np.ndarray):
    """X: [K, N_TOK], W: [N_OUT, K]. Loads data, runs, returns (cycles, counts, OUT).

    Fixed at the module-level K=240/N_OUT=8/W_CHUNKS=2 shape.
    """
    x_rows = np.zeros((K, LANES), dtype=np.float32)
    x_rows[:, :N_TOK] = X
    state.xmem.write_address(DATA_BASE_ROW * ROW_BYTES, bytearray(x_rows.tobytes()))

    w_rows = np.zeros((N_OUT * W_CHUNKS, LANES), dtype=np.float32)
    for o in range(N_OUT):
        w_rows[o * W_CHUNKS + 0, :LANES] = W[o, 0:LANES]
        w_rows[o * W_CHUNKS + 1, :TAIL_WIDTH] = W[o, LANES:K]
    state.xmem.write_address(WEIGHTS_BASE_ROW * ROW_BYTES, bytearray(w_rows.tobytes()))

    state.regfile.set_cr(2, DATA_BASE_ROW)
    state.regfile.set_cr(3, WEIGHTS_BASE_ROW)
    state.regfile.set_cr(4, OUTPUT_BASE_ROW)
    state.regfile.set_cr(5, -1)
    state.regfile.set_cr(6, LANES - 2)
    state.regfile.set_cr(7, TAIL_WIDTH - 2)
    state.regfile.set_cr(8, W_CHUNKS)
    state.regfile.set_cr(9, N_OUT)
    state.regfile.set_cr(10, W_CHUNKS - 1)
    state.regfile.set_cr(15, encode_dstructure(valid_elements=N_TOK))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=100_000)

    out_raw = bytes(state.xmem.read_address(OUTPUT_BASE_ROW * ROW_BYTES, N_OUT * ROW_BYTES))
    out_rows = np.frombuffer(out_raw, dtype=np.float32).reshape(N_OUT, LANES)
    OUT = out_rows[:, :N_TOK]
    return cycles, counts, OUT


def run_unpacked_linear_generic(state: IpuState, X: np.ndarray, W: np.ndarray, *,
                                 k: int, n_out: int, n_tok: int = 16, max_cycles: int = 2_000_000):
    """General (K, N_OUT) version of run_unpacked_linear, for real L5 shapes.

    X: [k, n_tok], W: [n_out, k]. Returns (cycles, counts, OUT).
    """
    w_chunks = -(-k // LANES)
    tail_width = k - LANES * (w_chunks - 1)
    data_base_row = 0
    weights_base_row = data_base_row + k
    output_base_row = weights_base_row + n_out * w_chunks

    x_rows = np.zeros((k, LANES), dtype=np.float32)
    x_rows[:, :n_tok] = X
    state.xmem.write_address(data_base_row * ROW_BYTES, bytearray(x_rows.tobytes()))

    w_rows = np.zeros((n_out * w_chunks, LANES), dtype=np.float32)
    for o in range(n_out):
        for c in range(w_chunks):
            lo, hi = c * LANES, min(c * LANES + LANES, k)
            w_rows[o * w_chunks + c, :hi - lo] = W[o, lo:hi]
    state.xmem.write_address(weights_base_row * ROW_BYTES, bytearray(w_rows.tobytes()))

    state.regfile.set_cr(2, data_base_row)
    state.regfile.set_cr(3, weights_base_row)
    state.regfile.set_cr(4, output_base_row)
    state.regfile.set_cr(5, -1)
    state.regfile.set_cr(6, LANES - 2)
    state.regfile.set_cr(7, tail_width - 2)
    state.regfile.set_cr(8, w_chunks)
    state.regfile.set_cr(9, n_out)
    state.regfile.set_cr(10, w_chunks - 1)
    state.regfile.set_cr(15, encode_dstructure(valid_elements=n_tok))

    with count_instructions() as counts:
        cycles = run_until_complete(state, max_cycles=max_cycles)

    out_raw = bytes(state.xmem.read_address(output_base_row * ROW_BYTES, n_out * ROW_BYTES))
    out_rows = np.frombuffer(out_raw, dtype=np.float32).reshape(n_out, LANES)
    OUT = out_rows[:, :n_tok]
    return cycles, counts, OUT


def test_unpacked_linear_240to8_correctness(tmp_path: Path) -> None:
    reset_labels()
    bin_path = tmp_path / "unpacked_linear.bin"
    assemble_to_bin_file(_ASM_SRC.read_text(), str(bin_path))

    rng = np.random.RandomState(11)
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(N_OUT, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)
    cycles, counts, OUT = run_unpacked_linear(state, X, W)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    print(f"unpacked linear 240->8: cycles={cycles} instrs={counts.total} max_abs_err={err:.6e}")
    print(f"  by slot: {counts.by_slot}")
    assert err < 1e-3, f"unpacked linear wrong: max abs error {err:.6e}"


def test_unpacked_linear_multichunk_correctness(tmp_path: Path) -> None:
    """K=480 (W_CHUNKS=4) regression test for the missing bound_sel check
    (fixed in asm_unpacked_linear_240to8.asm's chunk_loop): with only
    TAIL_BOUND ever selected, middle chunks under-ran their k-loop and
    produced silently wrong sums. Invisible at K<=256 (W_CHUNKS<=2), where
    chunk_loop runs at most once and that iteration IS the last chunk.
    """
    reset_labels()
    bin_path = tmp_path / "unpacked_linear_mc.bin"
    assemble_to_bin_file(_ASM_SRC.read_text(), str(bin_path))

    k, n_out = 480, 2
    rng = np.random.RandomState(23)
    X = rng.uniform(-2.0, 2.0, size=(k, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(n_out, k)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    load_program_from_binary(state, bin_path)
    cycles, counts, OUT = run_unpacked_linear_generic(state, X, W, k=k, n_out=n_out)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    print(f"unpacked linear {k}->{n_out} (W_CHUNKS=4): cycles={cycles} max_abs_err={err:.6e}")
    assert err < 1e-3, f"unpacked linear (multi-chunk) wrong: max abs error {err:.6e}"
