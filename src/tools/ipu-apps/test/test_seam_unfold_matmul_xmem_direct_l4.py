"""Seam investigation: does unfold_16x16x192's raw per-stream XMEM output
already satisfy matmul_576x192_x128's raw XMEM DATA-region input contract,
with NO file round-trip and NO ``_load_data()`` re-packing step at all?
(L4 QKV; unfold's C=192 output channels match matmul_576x192_x128's K=192.)

Structural question under test: unfold_16x16x192 emits, per stream, N_OUT=192
channel-major rows of 512 B, each row holding only N_TOK=64 valid FP32 tokens
(256 B) followed by 256 B of STALE r_acc garbage (STR_ACC_REG always writes
the full register -- see the module docstring). This is structurally
DIFFERENT from LayerNorm's zero-padded rows. matmul_576x192_x128's own
_load_data pads each channel's N_TOK=64 elements out to a whole row with
ZEROS, and DATA_STRIDE_ROWS=1 (one row per input channel) -- i.e. exactly the
row layout unfold already produces, garbage tail and all.

Whether that garbage tail is inert depends on whether the matmul's datapath
ever reads past lane 64 in a way that reaches the valid output. Reading
matmul_576x192_x128.asm: the per-k contraction step is
    MULT.RC.VE r0[fixed_idx] x r_cyclic[:] ; ACC.ADD[.FIRST]
This is the ACC.ADD family (per kernel_docs/kernel_layer_map.md's
"cross-lane reduction vs per-lane independence" section): MULT.RC.VE/ACC.ADD
accumulate each of the 128 SIMD LANES independently -- lane i's accumulator
never depends on lane j. The K-dimension chunk loop (lr6=126, lr11=62) walks
CHANNELS/ROWS, not lanes within a row, so it does not touch lanes 64..127
either. Since D's SIMD lane axis IS the token axis (t in C[j,t]) and the
kernel is per-lane-independent, garbage in lanes 64..127 of every input row
can only ever contaminate OUTPUT lanes 64..127 -- which this test crops away
(MM_N_TOK=64) -- never output lanes 0..63. This test confirms that
empirically: run unfold for real, capture its raw per-stream output bytes
directly via state.xmem.read_address, feed one stream's raw bytes verbatim
into the matmul's DATA region (bypassing _load_data), and compare against an
independent reference computed straight from the unfold definition (stride-2
decimation) and the matmul definition (C = W @ D) -- never from either
kernel's own golden/internals.

Mutation-first: the control test corrupts one channel's row in the captured
unfold stream before the direct handoff and confirms the result diverges
from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_16x16x192 import (
    Unfold16x16x192App, H as UF_H, W as UF_W, C as UF_C, N_STRIPES as UF_N_STRIPES,
    N_STREAMS as UF_N_STREAMS, N_OUT as UF_N_OUT, N_TOK as UF_N_TOK, LANES as UF_LANES,
    DST_BASE as UF_DST_BASE, DST_BASE_ROW as UF_DST_BASE_ROW,
)
from ipu_apps.matmuls.matmul_576x192_x128 import (
    MatMul576x192x128App, K as MM_K, N_OUT as MM_N_OUT, N_TOK as MM_N_TOK,
    LANES as MM_LANES, ROW_BYTES as MM_ROW_BYTES,
    DATA_BASE as MM_DATA_BASE, OUTPUT_BASE as MM_OUTPUT_BASE,
    OUTPUT_ROW_BYTES as MM_OUTPUT_ROW_BYTES, DATA_STRIDE_ROWS as MM_DATA_STRIDE_ROWS,
    OUTPUT_STRIDE_ROWS as MM_OUTPUT_STRIDE_ROWS, W_STRIDE_ROWS as MM_W_STRIDE_ROWS,
    WEIGHTS_BASE_ROW as MM_WEIGHTS_BASE_ROW, OUTPUT_BASE_ROW as MM_OUTPUT_BASE_ROW,
)

_UF_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/unfold_16x16x192/unfold_16x16x192.asm"
)
_uf_tmpdir = tempfile.TemporaryDirectory()
_UF_INST_BIN = Path(_uf_tmpdir.name) / "unfold_16x16x192.bin"
assemble_to_bin_file(_UF_ASM_PATH.read_text(encoding="utf-8"), str(_UF_INST_BIN))

_MM_INST_BIN = Path(os.environ["MATMUL_576X192_X128_INST_BIN"])

_POISON = 1e3

# Structural precondition: unfold's per-stream channel count must equal the
# matmul's K (unfold's output feeds the matmul's contraction dimension), and
# both must agree on the row/lane geometry before any byte handoff.
assert UF_C == UF_N_OUT == MM_K == 192
assert UF_LANES == MM_LANES == 128
assert UF_N_TOK == 64
assert MM_N_TOK == 64
_STRIPE_H = UF_H // UF_N_STRIPES     # 8 spatial rows per stripe

_STREAM = 0   # TL: phase (0, 0) -- arbitrary but fixed single-stream choice


def _run_unfold_capture_stream_xmem(
    x: np.ndarray, tmp_path: Path, tag: str,
) -> bytes:
    """Run the real unfold_16x16x192 kernel; return stream _STREAM's raw XMEM
    output bytes (N_OUT * ROW_BYTES = 192 * 512 B), read directly via
    state.xmem -- no file round-trip. x is [C, H, W].
    """
    src = np.zeros((UF_N_STRIPES * UF_C, UF_LANES), dtype=np.float32)
    for stripe in range(UF_N_STRIPES):
        r0 = stripe * _STRIPE_H
        for ch in range(UF_C):
            block = x[ch, r0 : r0 + _STRIPE_H, :]
            src[stripe * UF_C + ch, : _STRIPE_H * UF_W] = block.reshape(-1)
    input_path = tmp_path / f"uf_x_{tag}.bin"
    input_path.write_bytes(src.tobytes())

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    row_bytes = UF_LANES * 4
    state.xmem.write_address(
        UF_DST_BASE,
        bytearray(np.full(UF_N_STREAMS * UF_N_OUT * UF_LANES, _POISON, dtype=np.float32).tobytes()),
    )
    app = Unfold16x16x192App(inst_path=_UF_INST_BIN, input_path=input_path)
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    stream_base = UF_DST_BASE + _STREAM * UF_N_OUT * row_bytes
    raw = bytes(state.xmem.read_address(stream_base, UF_N_OUT * row_bytes))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(UF_N_OUT, UF_LANES)
    assert not np.all(rows == _POISON, axis=1).any(), "unfold left poisoned rows untouched"
    # NOTE: the module docstring's "256 B of stale r_acc garbage" is stale
    # content relative to whatever ACC.STRIDE's r_acc buffer held BEFORE this
    # store (see ipu.py execute_acc_stride: it writes only its computed
    # out_indices slots, leaving the rest of r_acc whatever it was). Since
    # r_acc starts life zeroed and nothing else writes its upper half before
    # this kernel's first store, that stale content reads back as 0.0 here --
    # NOT proof the tail is zero-padded by design (it structurally is NOT: no
    # code path clears it, it just happens to start at zero). Do not assert a
    # specific tail value; the seam question is whether whatever IS there
    # (zero in this run, potentially non-zero after a real prior kernel wrote
    # r_acc) can reach the matmul's valid output -- answered structurally
    # below via the per-lane-independence argument, and empirically by the
    # clean-path test not needing the tail to be any particular value.
    return raw


def _run_matmul_576x192_direct_xmem_handoff(
    uf_raw: bytes, W: np.ndarray, tmp_path: Path, tag: str,
) -> np.ndarray:
    """Feed uf_raw straight into matmul_576x192_x128's DATA region via
    state.xmem.write_address -- _load_data() is NEVER called. Only weights go
    through the normal file-staged _load_weights path. Returns the cropped
    [N_OUT, N_TOK] result.
    """
    assert len(uf_raw) == UF_N_OUT * UF_LANES * 4 == MM_K * MM_ROW_BYTES, (
        "byte-length mismatch between unfold's raw per-stream output region "
        "and matmul's raw DATA region -- direct handoff is not even "
        "byte-shape compatible"
    )

    weights_path = tmp_path / f"mm_w_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_path = tmp_path / f"mm_out_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    state.xmem.write_address(
        MM_DATA_BASE,
        bytearray(np.full(MM_K * MM_LANES, _POISON, dtype=np.float32).tobytes()),
    )
    state.xmem.write_address(
        MM_OUTPUT_BASE,
        bytearray(np.full(MM_N_OUT * MM_LANES, _POISON, dtype=np.float32).tobytes()),
    )

    # THE ACTUAL SEAM UNDER TEST: raw bytes, verbatim, no repacking -- the
    # 256 B garbage tail of every row rides along untouched.
    state.xmem.write_address(MM_DATA_BASE, bytearray(uf_raw))

    app = MatMul576x192x128App(
        inst_path=_MM_INST_BIN,
        input_path="/dev/null",   # placeholder; _load_data is bypassed below
        weights_path=weights_path,
        output_path=output_path,
    )
    from ipu_apps.matmuls.matmul_576x192_x128 import _load_weights as mm_load_weights

    def setup_no_load_data(state: "IpuState") -> None:
        mm_load_weights(state, app.weights_path)
        state.regfile.set_cr(0, 0)
        state.regfile.set_cr(9, MM_WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, MM_WEIGHTS_BASE_ROW + 1)
        state.regfile.set_cr(3, MM_OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, -MM_DATA_STRIDE_ROWS)
        state.regfile.set_cr(8, -1)
        state.regfile.set_lr(0, 0)
        state.regfile.set_lr(2, MM_DATA_STRIDE_ROWS)
        state.regfile.set_lr(3, MM_OUTPUT_STRIDE_ROWS)
        state.regfile.set_lr(6, 126)
        state.regfile.set_lr(7, 0)
        state.regfile.set_lr(8, 0)
        state.regfile.set_lr(9, 0)
        state.regfile.set_lr(10, MM_N_OUT)
        state.regfile.set_lr(11, (MM_K - MM_LANES) - 2)
        state.regfile.set_lr(12, MM_W_STRIDE_ROWS)

    app.setup = setup_no_load_data
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(MM_OUTPUT_BASE, MM_N_OUT * MM_LANES * 4))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(MM_N_OUT, MM_LANES)
    assert not np.all(rows == _POISON, axis=1).any(), "matmul left poisoned output rows untouched"
    return rows[:, :MM_N_TOK]


def _unfold_stream_reference(x: np.ndarray) -> np.ndarray:
    """Independent reference for stream _STREAM (TL, phase (0,0)): the
    standard stride-2 space-to-depth decimation, matching
    unfold_16x16x192's module docstring exactly (stream s takes every other
    spatial row/col at phase (s//2, s%2)) -- not the kernel's own
    implementation.
    """
    r_ph, c_ph = _STREAM // 2, _STREAM % 2
    return x[:, r_ph::2, c_ph::2].reshape(UF_C, UF_N_TOK).astype(np.float64)  # [C, N_TOK]


def test_unfold_16x16x192_output_is_not_byte_compatible_with_matmul_576x192_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    channel's VALID lanes (0:N_TOK) in the captured unfold stream before
    handing it to the matmul and confirm the result diverges from the
    reference. (Corrupting the stale tail deliberately would prove nothing,
    since the whole point under test is that the tail is inert.)
    """
    rng = np.random.RandomState(0x4A11)
    x = rng.uniform(-1.0, 1.0, size=(UF_C, UF_H, UF_W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    uf_raw = bytearray(_run_unfold_capture_stream_xmem(x, tmp_path, tag="mut"))
    row_bytes = UF_LANES * 4
    ch = 9
    corrupt_valid = np.full(UF_N_TOK, 999.0, dtype=np.float32).tobytes()
    off = ch * row_bytes
    uf_raw[off : off + UF_N_TOK * 4] = corrupt_valid

    got = _run_matmul_576x192_direct_xmem_handoff(bytes(uf_raw), W, tmp_path, tag="mut")

    d_expected = _unfold_stream_reference(x)          # [K, N_TOK]
    expected = (W.astype(np.float64) @ d_expected)     # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_unfold_tail_lanes_are_structurally_inert_when_poisoned(tmp_path: Path) -> None:
    """Direct test of the inertness claim itself: rather than relying on
    whatever the emulator happens to leave in the tail lanes (empirically
    0.0 in this run -- see the note in _run_unfold_capture_stream_xmem), fill
    lanes [N_TOK:LANES] of EVERY row with a large, distinctive, non-zero
    marker (999.0) before the handoff -- the worst case the docstring's
    "stale garbage" language warns about -- and confirm the matmul result is
    still bit-for-bit determined only by lanes [0:N_TOK], matching the
    reference. This proves inertness structurally rather than by lucky
    accident of what the emulator's reset state happened to be.
    """
    rng = np.random.RandomState(0x4A13)
    x = rng.uniform(-1.0, 1.0, size=(UF_C, UF_H, UF_W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    uf_raw = bytearray(_run_unfold_capture_stream_xmem(x, tmp_path, tag="poison_tail"))
    rows = np.frombuffer(bytes(uf_raw), dtype=np.float32).reshape(UF_N_OUT, UF_LANES).copy()
    rows[:, UF_N_TOK:] = 999.0
    uf_raw = bytearray(rows.tobytes())

    got = _run_matmul_576x192_direct_xmem_handoff(bytes(uf_raw), W, tmp_path, tag="poison_tail")

    d_expected = _unfold_stream_reference(x)
    expected = (W.astype(np.float64) @ d_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"unfold tail-lane poison probe max abs error (valid lanes only) = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "matmul_576x192_x128's valid-lane output changed when unfold's "
            "tail lanes were poisoned -- the tail is NOT structurally inert, "
            "contradicting the per-lane-independence argument"
        ),
    )


def test_unfold_16x16x192_feeds_matmul_576x192_via_direct_xmem_no_file_staging_l4(
    tmp_path: Path,
) -> None:
    """The real question: with the corruption removed, does the verbatim
    byte handoff (no _load_data, no file round-trip, stale r_acc tail lanes
    included as-is) produce the correct matmul result end to end -- proving
    the garbage tail lanes are structurally inert under matmul_576x192_x128's
    per-lane-independent ACC.ADD datapath?
    """
    rng = np.random.RandomState(0x4A12)
    x = rng.uniform(-1.0, 1.0, size=(UF_C, UF_H, UF_W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    uf_raw = _run_unfold_capture_stream_xmem(x, tmp_path, tag="clean")
    got = _run_matmul_576x192_direct_xmem_handoff(uf_raw, W, tmp_path, tag="clean")

    d_expected = _unfold_stream_reference(x)
    expected = (W.astype(np.float64) @ d_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam unfold_16x16x192(direct XMEM, stream {_STREAM})->matmul_576x192_x128 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "unfold_16x16x192's raw per-stream XMEM output (with its stale "
            "r_acc garbage tail), handed to matmul_576x192_x128 verbatim "
            "with NO file staging and NO _load_data repack, does not match "
            "an independent reference -- either the garbage tail is not "
            "inert, or some other seam defect exists"
        ),
    )
