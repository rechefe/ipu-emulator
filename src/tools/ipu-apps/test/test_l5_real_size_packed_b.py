"""L5 packed-viability task, item 1 & 2 follow-up: path (b) (masked passes,
memory-optimal, no weight replication) measured at the four REAL L5 linear-
layer shapes, after the IPC bundling fix validated at 240->8 in
asm_packed_linear_240to8_masked.asm.

Uses a GENERIC Jinja template (asm_packed_linear_masked_generic.asm),
parameterized over K/N_OUT via a pre-rendered `chunk_widths` list (weight-
chunk widths, each a multiple of 8, summing to K) -- computed in Python and
rendered through jinja2.Template(...).render(...) BEFORE handing the text to
assemble_to_bin_file (which only does a no-context self-render, so the loop
bounds must already be baked into literal text by the time it sees the
source). This keeps the kernel fully unrolled/static per shape -- no runtime
branching -- matching the same structure validated at 240->8.

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
from ipu_emu.ipu_config import encode_dstructure

from fixture_packed_l5_measure import count_instructions
from test_linear_240to8_unpacked_baseline import run_unpacked_linear_generic

_ASM_TEMPLATE_SRC = Path(__file__).resolve().parent / "asm_packed_linear_masked_generic.asm"

N_TOK = 16
LANES = 128
ROW_BYTES = 512
PACK = 8


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
        assert w % 8 == 0, f"weight-chunk width {w} not a multiple of 8 (k={k})"
        widths.append(w)
        remaining -= w
    return widths


def run_packed_linear_b_generic(state: IpuState, X: np.ndarray, W: np.ndarray, *,
                                 k: int, n_out: int, n_tok: int = 16, max_cycles: int = 4_000_000):
    """X: [k, n_tok], W: [n_out, k]. Returns (cycles, counts, OUT)."""
    assert k % PACK == 0
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    n_chunks = k // PACK

    template_text = _ASM_TEMPLATE_SRC.read_text()
    rendered = jinja2.Template(template_text).render(chunk_widths=widths)

    reset_labels()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "packed_b_generic.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)

        data_base_row = 0
        weights_base_row = data_base_row + n_chunks
        output_base_row = weights_base_row + n_out * w_chunks

        packed_x = _pack_x(X, k)
        state.xmem.write_address(data_base_row * ROW_BYTES, bytearray(packed_x.tobytes()))

        w_rows = np.zeros((n_out * w_chunks, LANES), dtype=np.float32)
        for o in range(n_out):
            off = 0
            for c, width in enumerate(widths):
                w_rows[o * w_chunks + c, :width] = W[o, off:off + width]
                off += width
        state.xmem.write_address(weights_base_row * ROW_BYTES, bytearray(w_rows.tobytes()))

        state.regfile.set_cr(2, data_base_row - 1)  # pre-increment bias
        state.regfile.set_cr(3, weights_base_row)
        state.regfile.set_cr(4, output_base_row)
        state.regfile.set_cr(8, w_chunks)
        state.regfile.set_cr(9, n_out)
        state.regfile.set_cr(10, 16)
        state.regfile.set_cr(11, -16)
        state.regfile.set_cr(12, -1)
        state.regfile.set_cr(15, encode_dstructure(valid_elements=n_tok))

        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)

        out_raw = bytes(state.xmem.read_address(output_base_row * ROW_BYTES, n_out * ROW_BYTES))
        out_rows = np.frombuffer(out_raw, dtype=np.float32).reshape(n_out, LANES)
        OUT = out_rows[:, :n_tok]
        weight_load_instrs = n_out * w_chunks
        return cycles, counts, OUT, weight_load_instrs


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
    cycles, counts, OUT, weight_load_instrs = run_packed_linear_b_generic(
        state, X, W, k=k, n_out=n_out)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    n_chunks = k // PACK
    packed_activation_bytes = (n_chunks + n_out) * ROW_BYTES
    unpacked_weight_bytes = n_out * k * 4
    ipc = counts.total / cycles if cycles else 0.0

    print(f"{name}: PACKED(b) k={k} n_out={n_out} cycles={cycles} instrs={counts.total} "
          f"ipc={ipc:.3f} err={err:.6e} act_bytes={packed_activation_bytes} "
          f"weight_bytes={unpacked_weight_bytes} weight_load_instrs={weight_load_instrs}")
    print(f"  by slot: {counts.by_slot}")

    assert err < 1e-3, f"{name}: packed(b) wrong: max abs error {err:.6e}"
    return dict(cycles=cycles, instrs=counts.total, err=err,
                act_bytes=packed_activation_bytes, weight_bytes=unpacked_weight_bytes,
                weight_load_instrs=weight_load_instrs)


def test_packed_b_qkv():
    _run_one("QKV", **SHAPES["QKV"], seed=101)


def test_packed_b_outproj():
    _run_one("outproj", **SHAPES["outproj"], seed=102)


def test_packed_b_ffn1():
    _run_one("FFN1", **SHAPES["FFN1"], seed=103)


def test_packed_b_ffn2():
    _run_one("FFN2", **SHAPES["FFN2"], seed=104)
