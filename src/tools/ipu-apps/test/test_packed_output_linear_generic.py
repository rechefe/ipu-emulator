"""Generic (K, N_OUT=8-per-call) packed-OUTPUT linear kernel, extending the
validated construction in asm_packed_output_linear_tiny.asm to real K sizes.

Validates: rc_idx = 16*(p_in - p_out) mod 512 with R_CYCLIC replicated into
all 4 slots per packed chunk and R_MASK pre-built per p_out, producing a
packed row of 8 output channels from packed input, no host-side conversion.
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
        assert w % 8 == 0, f"weight-chunk width {w} not a multiple of 8 (k={k})"
        widths.append(w)
        remaining -= w
    return widths


def _build_mask_row() -> bytes:
    row = bytearray(128)
    for p_out in range(8):
        bits = 0
        for b in range(16 * p_out, 16 * p_out + 16):
            bits |= (1 << b)
        row[p_out * 16:(p_out + 1) * 16] = bits.to_bytes(16, "little")
    return bytes(row)


def run_packed_output_linear_8ch(state: IpuState, X: np.ndarray, W8: np.ndarray, *,
                                  k: int, max_cycles: int = 4_000_000):
    """X: [k, 16], W8: [8, k] (exactly 8 output channels, one packed-output row).

    Returns (cycles, counts, OUT_packed_row [8,16]).
    """
    widths = _chunk_widths(k)
    w_chunks = len(widths)
    n_chunks = k // PACK

    template_text = _ASM_TEMPLATE_SRC.read_text()
    rendered = jinja2.Template(template_text).render(chunk_widths=widths)

    reset_labels()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "packed_output_generic.bin"
        assemble_to_bin_file(rendered, str(bin_path))
        load_program_from_binary(state, bin_path)

        data_base_row = 0
        weights_base_row = data_base_row + n_chunks
        mask_row = weights_base_row + 8 * w_chunks
        output_row = mask_row + 1

        packed_x = _pack_x(X, k)
        state.xmem.write_address(data_base_row * ROW_BYTES, bytearray(packed_x.tobytes()))

        # Weight-chunk-major, p_out-minor: row = c*8 + p_out. The kernel
        # reloads R0 once per (packed chunk, p_out) via w_row_ptr = w_ptr +
        # p_out, where w_ptr already points at the current weight-chunk's
        # row 0 (p_out=0); advancing w_ptr by 8 between weight-chunks.
        w_rows = np.zeros((8 * w_chunks, LANES), dtype=np.float32)
        for c, width in enumerate(widths):
            off = sum(widths[:c])
            for p_out in range(8):
                w_rows[c * 8 + p_out, :width] = W8[p_out, off:off + width]
        state.xmem.write_address(weights_base_row * ROW_BYTES, bytearray(w_rows.tobytes()))

        state.xmem.write_address(mask_row * ROW_BYTES, _build_mask_row())

        # CR0/CR1 are hardwired read-only (0 and 1); every real value lives
        # in cr2-cr15, matching asm_packed_output_linear_generic.asm's CR map.
        state.regfile.set_cr(2, data_base_row)
        state.regfile.set_cr(3, weights_base_row)
        state.regfile.set_cr(4, output_row)
        state.regfile.set_cr(5, mask_row)
        state.regfile.set_cr(6, encode_dstructure(valid_elements=128, partition=Partition.P8))
        state.regfile.set_cr(7, encode_dstructure(valid_elements=128))
        for p_out in range(8):
            seed = (512 - 16 * p_out - 16) % 512
            state.regfile.set_cr(8 + p_out, seed)

        with count_instructions() as counts:
            cycles = run_until_complete(state, max_cycles=max_cycles)

        out_raw = bytes(state.xmem.read_address(output_row * ROW_BYTES, ROW_BYTES))
        out_packed = np.frombuffer(out_raw, dtype=np.float32).reshape(8, 16)
        return cycles, counts, out_packed


def test_packed_output_linear_240to8():
    K = 240
    rng = np.random.RandomState(11)
    X = rng.uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-0.5, 0.5, size=(8, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    cycles, counts, OUT = run_packed_output_linear_8ch(state, X, W, k=K)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    replication_loads = counts.by_slot.get("load", 0)
    print(f"packed-OUTPUT linear 240->8: cycles={cycles} instrs={counts.total} "
          f"max_abs_err={err:.6e}")
    print(f"  by slot: {counts.by_slot}")
    print(f"  store+aaq count: {counts.by_slot.get('store', 0) + counts.by_slot.get('aaq', 0)}")
    print(f"  output activation bytes: {ROW_BYTES} (1 packed row) vs unpacked 8*{ROW_BYTES}={8*ROW_BYTES}")

    assert err < 1e-3, f"packed-output linear 240->8 wrong: max abs error {err:.6e}"


def test_packed_output_linear_multichunk_regression():
    """K=480 (4 weight-chunks) regression test: caught the wc_bound
    off-by-one that silently dropped the LAST weight-chunk's 12 packed
    chunks (chunk_ctr topped out at 48/59), and the w_ptr stale +1 advance
    (should be +8, one row per p_out per weight-chunk) that only manifests
    once there's more than one post-peeled weight-chunk transition.
    """
    K = 480
    rng = np.random.RandomState(23)
    X = np.random.RandomState(23).uniform(-2.0, 2.0, size=(K, N_TOK)).astype(np.float32)
    W = np.random.RandomState(23).uniform(-0.5, 0.5, size=(8, K)).astype(np.float32)
    expected = W.astype(np.float64) @ X.astype(np.float64)

    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    cycles, counts, OUT = run_packed_output_linear_8ch(state, X, W, k=K, max_cycles=4_000_000)

    err = float(np.max(np.abs(OUT.astype(np.float64) - expected)))
    print(f"packed-output linear 480->8 (4 weight-chunks): cycles={cycles} "
          f"instrs={counts.total} max_abs_err={err:.6e}")
    assert err < 1e-3, f"packed-output linear 480->8 wrong: max abs error {err:.6e}"
