"""Seam investigation: does unfold_8x8x240's raw per-stream XMEM output
already satisfy matmul_720x240_x128's raw XMEM DATA-region input contract,
with NO file round-trip and NO ``_load_data()`` re-packing step at all?
(L5 QKV; unfold's C=240 output channels match matmul_720x240_x128's K=240.)

Same idea as the L4 counterpart (unfold_16x16x192 -> matmul_576x192_x128),
but the "garbage tail" premise differs structurally for THIS unfold kernel:
unfold_8x8x240's own module docstring and its `test_output_shape_and_stale_lanes`
test establish that lanes N_TOK..127 of every output row are exactly ZERO, not
stale r_acc garbage -- because `ACC.STRIDE 16 ...` only ever writes 32
elements into r_acc slot 0 (lanes 0..31, of which 16..31 are the decimated
ZERO input-padding), and lanes 32..127 are never written by this kernel and
stay 0.0 from reset. (unfold_16x16x192 uses the older STR_ACC_REG-full-register
path and genuinely leaves stale garbage; unfold_8x8x240 uses the newer
ACTIVATE.QUANTIZE + STR_POST_AAQ_REG path and does not.) This test still
checks the seam exactly like the L4 counterpart -- verbatim byte handoff, no
_load_data, no repacking -- and separately records whether the tail bytes it
captured are actually the zero the docstring predicts or something else.

matmul_720x240_x128's contraction step (matmul_720x240_x128.asm) is the same
per-lane-independent family as matmul_576x192_x128:
    MULT.RC.VE r0[fixed_idx] x r_cyclic[:] ; ACC.ADD[.FIRST]
Each of the 128 SIMD lanes (the token axis) accumulates independently; the
K-dimension chunk loop (lr6=126, lr11=110) walks CHANNELS/ROWS, not lanes.
So whatever sits in lanes N_TOK..127 (zero, per the above) can only ever
contaminate output lanes N_TOK..127, cropped away here, never lanes 0..15.

Method: run the REAL unfold_8x8x240 kernel (using pack_input_rows to build
valid packed input per its own input contract -- _ROW_PACK_ORDER correctness
is out of scope here, this test only exercises the output->matmul-input
seam), capture one stream's raw XMEM output bytes directly via
state.xmem.read_address, feed them verbatim into matmul_720x240_x128's DATA
region (bypassing _load_data -- the matmul harness runs its own setup() with
_load_data disabled), and compare against an independent reference
computed straight from the unfold definition (stride-2 decimation of the
UNPACKED [C,H,W] array) and the matmul definition (C = W @ D) -- never from
either kernel's own golden/internals.

Mutation-first: the control test corrupts one channel's valid lanes in the
captured unfold stream before the direct handoff and confirms the result
diverges from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.matmul.matmul_720x240_x128 import app as mm
from ipu_apps.kernels.reshape.unfold_8x8x240 import app as uf

from fixture_kernel_support import (
    POISON, kernel_inst, poison, run_matmul_direct_xmem_handoff,
)

assert uf.C == uf.N_OUT == mm.K == 240
assert uf.LANES == mm.LANES == 128
assert uf.N_TOK == 16
assert mm.N_TOK == 16
_ROW_BYTES = uf.LANES * 4

_STREAM = 0   # phase (0, 0) -- arbitrary but fixed single-stream choice


def _run_unfold_capture_stream_xmem(x: np.ndarray, tmp_path: Path, tag: str) -> bytes:
    """Run the real unfold_8x8x240 kernel; return stream _STREAM's raw XMEM
    output bytes (N_OUT * ROW_BYTES = 240 * 512 B), read directly via
    state.xmem -- no file round-trip. x is [C, H, W]; packed into the
    kernel's required input layout via pack_input_rows (its own documented
    input contract, per the module docstring -- not re-derived here).
    """
    input_path = tmp_path / f"uf_x_{tag}.bin"
    input_path.write_bytes(np.ascontiguousarray(uf.pack_input_rows(x), dtype=np.float32).tobytes())

    app = uf.Unfold8x8x240App(inst_path=kernel_inst("unfold_8x8x240"), input_path=input_path)
    state = app.make_state()
    poison(state, uf.DST_BASE, uf.N_STREAMS * uf.N_OUT)
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    stream_base = uf.DST_BASE + _STREAM * uf.N_OUT * _ROW_BYTES
    raw = bytes(state.xmem.read_address(stream_base, uf.N_OUT * _ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(uf.N_OUT, uf.LANES)
    assert not np.all(rows == POISON, axis=1).any(), "unfold left poisoned rows untouched"
    return raw


def _run_matmul_720x240_direct_xmem_handoff(uf_raw: bytes, W, tmp_path: Path, tag: str) -> np.ndarray:
    """Feed uf_raw verbatim into matmul_720x240_x128's DATA region. Returns
    the cropped [N_OUT, N_TOK] result, read directly from XMEM at full
    ROW_BYTES width (the harness runs with no output_path)."""
    rows = run_matmul_direct_xmem_handoff(mm, mm.MatMul720x240x128App, data_raw=uf_raw, W=W,
                                          tmp_path=tmp_path, tag=tag, output_file=False)
    return rows[:, :mm.N_TOK]


def _unfold_stream_reference(x: np.ndarray) -> np.ndarray:
    """Independent reference for stream _STREAM (phase (0,0)): the standard
    stride-2 space-to-depth decimation, matching unfold_8x8x240's module
    docstring exactly -- computed directly from the UNPACKED [C, H, W] array,
    not from pack_input_rows or the kernel's own implementation.
    """
    r_ph, c_ph = _STREAM // 2, _STREAM % 2
    return x[:, r_ph::2, c_ph::2].reshape(uf.C, uf.N_TOK).astype(np.float64)  # [C, N_TOK]


def _inputs(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(uf.C, uf.H, uf.W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(mm.N_OUT, mm.K)).astype(np.float32)
    return x, W


def test_unfold_8x8x240_output_is_not_byte_compatible_with_matmul_720x240_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: corrupt one channel's VALID lanes (0:N_TOK) in
    the captured unfold stream before handing it to the matmul and confirm
    the result diverges from the reference.
    """
    x, W = _inputs(0x5E15)

    uf_raw = bytearray(_run_unfold_capture_stream_xmem(x, tmp_path, tag="mut"))
    off = 13 * _ROW_BYTES
    uf_raw[off : off + uf.N_TOK * 4] = np.full(uf.N_TOK, 999.0, dtype=np.float32).tobytes()

    got = _run_matmul_720x240_direct_xmem_handoff(bytes(uf_raw), W, tmp_path, tag="mut")
    expected = W.astype(np.float64) @ _unfold_stream_reference(x)   # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_unfold_8x8x240_feeds_matmul_720x240_via_direct_xmem_no_file_staging_l5(
    tmp_path: Path,
) -> None:
    """With the corruption removed, does the verbatim byte handoff (no
    _load_data, no file round-trip, tail lanes included as-is) produce the
    correct matmul result end to end? Also records whether the captured tail
    bytes really are the zero the module docstring predicts for THIS kernel
    (as opposed to unfold_16x16x192's genuine stale r_acc garbage).
    """
    x, W = _inputs(0x5E16)

    uf_raw = _run_unfold_capture_stream_xmem(x, tmp_path, tag="clean")

    tail = np.frombuffer(uf_raw, dtype=np.float32).reshape(uf.N_OUT, uf.LANES)[:, uf.N_TOK:]
    print(
        f"unfold_8x8x240 stream {_STREAM} tail lanes [{uf.N_TOK}:{uf.LANES}] "
        f"all zero = {bool(np.all(tail == 0.0))} (docstring predicts True for this "
        "kernel, unlike unfold_16x16x192's genuine stale r_acc garbage)"
    )

    got = _run_matmul_720x240_direct_xmem_handoff(uf_raw, W, tmp_path, tag="clean")
    expected = W.astype(np.float64) @ _unfold_stream_reference(x)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam unfold_8x8x240(direct XMEM, stream {_STREAM})->matmul_720x240_x128 "
          f"max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "unfold_8x8x240's raw per-stream XMEM output, handed to "
            "matmul_720x240_x128 verbatim with NO file staging and NO "
            "_load_data repack, does not match an independent reference"
        ),
    )
