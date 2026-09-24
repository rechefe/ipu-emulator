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
This is the ACC.ADD family: MULT.RC.VE/ACC.ADD
accumulate each of the 128 SIMD LANES independently -- lane i's accumulator
never depends on lane j. The K-dimension chunk loop (lr6=126, lr11=62) walks
CHANNELS/ROWS, not lanes within a row, so it does not touch lanes 64..127
either. Since D's SIMD lane axis IS the token axis (t in C[j,t]) and the
kernel is per-lane-independent, garbage in lanes 64..127 of every input row
can only ever contaminate OUTPUT lanes 64..127 -- which this test crops away
(MM_N_TOK=64) -- never output lanes 0..63. This test confirms that
empirically: run unfold for real, capture its raw per-stream output bytes
directly via state.xmem.read_address, feed one stream's raw bytes verbatim
into the matmul's DATA region (the matmul harness runs its own setup() with
_load_data disabled), and compare against an independent reference
computed straight from the unfold definition (stride-2
decimation) and the matmul definition (C = W @ D) -- never from either
kernel's own golden/internals.

Mutation-first: the control test corrupts one channel's row in the captured
unfold stream before the direct handoff and confirms the result diverges
from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.matmul.matmul_576x192_x128 import app as mm
from ipu_apps.kernels.reshape.unfold_16x16x192 import app as uf

from fixture_kernel_support import (
    POISON, kernel_inst, poison, run_matmul_direct_xmem_handoff,
)

# Structural precondition: unfold's per-stream channel count must equal the
# matmul's K (unfold's output feeds the matmul's contraction dimension), and
# both must agree on the row/lane geometry before any byte handoff.
assert uf.C == uf.N_OUT == mm.K == 192
assert uf.LANES == mm.LANES == 128
assert uf.N_TOK == 64
assert mm.N_TOK == 64
_STRIPE_H = uf.H // uf.N_STRIPES     # 8 spatial rows per stripe
_ROW_BYTES = uf.LANES * 4

_STREAM = 0   # TL: phase (0, 0) -- arbitrary but fixed single-stream choice


def _run_unfold_capture_stream_xmem(x: np.ndarray, tmp_path: Path, tag: str) -> bytes:
    """Run the real unfold_16x16x192 kernel; return stream _STREAM's raw XMEM
    output bytes (N_OUT * ROW_BYTES = 192 * 512 B), read directly via
    state.xmem -- no file round-trip. x is [C, H, W].
    """
    src = np.zeros((uf.N_STRIPES * uf.C, uf.LANES), dtype=np.float32)
    for stripe in range(uf.N_STRIPES):
        r0 = stripe * _STRIPE_H
        for ch in range(uf.C):
            block = x[ch, r0 : r0 + _STRIPE_H, :]
            src[stripe * uf.C + ch, : _STRIPE_H * uf.W] = block.reshape(-1)
    input_path = tmp_path / f"uf_x_{tag}.bin"
    input_path.write_bytes(src.tobytes())

    app = uf.Unfold16x16x192App(inst_path=kernel_inst("unfold_16x16x192"), input_path=input_path)
    state = app.make_state()
    poison(state, uf.DST_BASE, uf.N_STREAMS * uf.N_OUT)
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    stream_base = uf.DST_BASE + _STREAM * uf.N_OUT * _ROW_BYTES
    raw = bytes(state.xmem.read_address(stream_base, uf.N_OUT * _ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(uf.N_OUT, uf.LANES)
    assert not np.all(rows == POISON, axis=1).any(), "unfold left poisoned rows untouched"
    # NOTE: the module docstring's "256 B of stale r_acc garbage" is stale
    # content relative to whatever ACC.STRIDE's r_acc buffer held BEFORE this
    # store (it writes only its computed out_indices slots, leaving the rest
    # of r_acc whatever it was). Since r_acc starts life zeroed and nothing
    # else writes its upper half before this kernel's first store, that stale
    # content reads back as 0.0 here -- NOT proof the tail is zero-padded by
    # design. Do not assert a specific tail value; the seam question is
    # whether whatever IS there can reach the matmul's valid output --
    # answered structurally by the per-lane-independence argument, and
    # directly by test_unfold_tail_lanes_are_structurally_inert_when_poisoned.
    return raw


def _run_matmul_576x192_direct_xmem_handoff(uf_raw: bytes, W, tmp_path: Path, tag: str) -> np.ndarray:
    """Feed uf_raw verbatim into matmul_576x192_x128's DATA region -- the
    unfold tail lanes ride along untouched. Returns the cropped [N_OUT, N_TOK]
    result."""
    rows = run_matmul_direct_xmem_handoff(mm, mm.MatMul576x192x128App, data_raw=uf_raw, W=W,
                                          tmp_path=tmp_path, tag=tag)
    return rows[:, :mm.N_TOK]


def _unfold_stream_reference(x: np.ndarray) -> np.ndarray:
    """Independent reference for stream _STREAM (TL, phase (0,0)): the
    standard stride-2 space-to-depth decimation, matching
    unfold_16x16x192's module docstring exactly (stream s takes every other
    spatial row/col at phase (s//2, s%2)) -- not the kernel's own
    implementation.
    """
    r_ph, c_ph = _STREAM // 2, _STREAM % 2
    return x[:, r_ph::2, c_ph::2].reshape(uf.C, uf.N_TOK).astype(np.float64)  # [C, N_TOK]


def _inputs(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(uf.C, uf.H, uf.W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(mm.N_OUT, mm.K)).astype(np.float32)
    return x, W


def test_unfold_16x16x192_output_is_not_byte_compatible_with_matmul_576x192_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: corrupt one channel's VALID lanes (0:N_TOK) in
    the captured unfold stream before handing it to the matmul and confirm
    the result diverges from the reference. (Corrupting the stale tail
    deliberately would prove nothing, since the whole point under test is
    that the tail is inert.)
    """
    x, W = _inputs(0x4A11)

    uf_raw = bytearray(_run_unfold_capture_stream_xmem(x, tmp_path, tag="mut"))
    off = 9 * _ROW_BYTES
    uf_raw[off : off + uf.N_TOK * 4] = np.full(uf.N_TOK, 999.0, dtype=np.float32).tobytes()

    got = _run_matmul_576x192_direct_xmem_handoff(bytes(uf_raw), W, tmp_path, tag="mut")
    expected = W.astype(np.float64) @ _unfold_stream_reference(x)   # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_unfold_tail_lanes_are_structurally_inert_when_poisoned(tmp_path: Path) -> None:
    """Direct test of the inertness claim itself: rather than relying on
    whatever the emulator happens to leave in the tail lanes, fill lanes
    [N_TOK:LANES] of EVERY row with a large, distinctive, non-zero marker
    (999.0) before the handoff -- the worst case the docstring's "stale
    garbage" language warns about -- and confirm the matmul result is still
    determined only by lanes [0:N_TOK], matching the reference.
    """
    x, W = _inputs(0x4A13)

    uf_raw = _run_unfold_capture_stream_xmem(x, tmp_path, tag="poison_tail")
    rows = np.frombuffer(uf_raw, dtype=np.float32).reshape(uf.N_OUT, uf.LANES).copy()
    rows[:, uf.N_TOK:] = 999.0

    got = _run_matmul_576x192_direct_xmem_handoff(rows.tobytes(), W, tmp_path, tag="poison_tail")
    expected = W.astype(np.float64) @ _unfold_stream_reference(x)

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
    """With the corruption removed, does the verbatim byte handoff (no
    _load_data, no file round-trip, stale r_acc tail lanes included as-is)
    produce the correct matmul result end to end?
    """
    x, W = _inputs(0x4A12)

    uf_raw = _run_unfold_capture_stream_xmem(x, tmp_path, tag="clean")
    got = _run_matmul_576x192_direct_xmem_handoff(uf_raw, W, tmp_path, tag="clean")
    expected = W.astype(np.float64) @ _unfold_stream_reference(x)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam unfold_16x16x192(direct XMEM, stream {_STREAM})->matmul_576x192_x128 "
          f"max abs error = {max_err:.3e}")

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
