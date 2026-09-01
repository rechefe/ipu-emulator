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
region (bypassing _load_data), and compare against an independent reference
computed straight from the unfold definition (stride-2 decimation of the
UNPACKED [C,H,W] array) and the matmul definition (C = W @ D) -- never from
either kernel's own golden/internals.

Mutation-first: the control test corrupts one channel's valid lanes in the
captured unfold stream before the direct handoff and confirms the result
diverges from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_8x8x240 import (
    Unfold8x8x240App, H as UF_H, W as UF_W, C as UF_C,
    N_STREAMS as UF_N_STREAMS, N_OUT as UF_N_OUT, N_TOK as UF_N_TOK, LANES as UF_LANES,
    DST_BASE as UF_DST_BASE, pack_input_rows,
)
from ipu_apps.matmuls.matmul_720x240_x128 import (
    MatMul720x240x128App, K as MM_K, N_OUT as MM_N_OUT, N_TOK as MM_N_TOK,
    LANES as MM_LANES, ROW_BYTES as MM_ROW_BYTES,
    DATA_BASE as MM_DATA_BASE, OUTPUT_BASE as MM_OUTPUT_BASE,
    DATA_STRIDE_ROWS as MM_DATA_STRIDE_ROWS, OUTPUT_STRIDE_ROWS as MM_OUTPUT_STRIDE_ROWS,
    W_STRIDE_ROWS as MM_W_STRIDE_ROWS, WEIGHTS_BASE_ROW as MM_WEIGHTS_BASE_ROW,
    OUTPUT_BASE_ROW as MM_OUTPUT_BASE_ROW,
)

_UF_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/unfold_8x8x240/unfold_8x8x240.asm"
)
_uf_tmpdir = tempfile.TemporaryDirectory()
_UF_INST_BIN = Path(_uf_tmpdir.name) / "unfold_8x8x240.bin"
assemble_to_bin_file(_UF_ASM_PATH.read_text(encoding="utf-8"), str(_UF_INST_BIN))

_MM_INST_BIN = Path(os.environ["MATMUL_720X240_X128_INST_BIN"])

_POISON = 1e3

assert UF_C == UF_N_OUT == MM_K == 240
assert UF_LANES == MM_LANES == 128
assert UF_N_TOK == 16
assert MM_N_TOK == 16

_STREAM = 0   # phase (0, 0) -- arbitrary but fixed single-stream choice


def _run_unfold_capture_stream_xmem(
    x: np.ndarray, tmp_path: Path, tag: str,
) -> bytes:
    """Run the real unfold_8x8x240 kernel; return stream _STREAM's raw XMEM
    output bytes (N_OUT * ROW_BYTES = 240 * 512 B), read directly via
    state.xmem -- no file round-trip. x is [C, H, W]; packed into the
    kernel's required input layout via pack_input_rows (its own documented
    input contract, per the module docstring -- not re-derived here).
    """
    src = pack_input_rows(x)
    input_path = tmp_path / f"uf_x_{tag}.bin"
    input_path.write_bytes(np.ascontiguousarray(src, dtype=np.float32).tobytes())

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    row_bytes = UF_LANES * 4
    state.xmem.write_address(
        UF_DST_BASE,
        bytearray(np.full(UF_N_STREAMS * UF_N_OUT * UF_LANES, _POISON, dtype=np.float32).tobytes()),
    )
    app = Unfold8x8x240App(inst_path=_UF_INST_BIN, input_path=input_path)
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    stream_base = UF_DST_BASE + _STREAM * UF_N_OUT * row_bytes
    raw = bytes(state.xmem.read_address(stream_base, UF_N_OUT * row_bytes))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(UF_N_OUT, UF_LANES)
    assert not np.all(rows == _POISON, axis=1).any(), "unfold left poisoned rows untouched"
    return raw


def _run_matmul_720x240_direct_xmem_handoff(
    uf_raw: bytes, W: np.ndarray, tmp_path: Path, tag: str,
) -> np.ndarray:
    """Feed uf_raw straight into matmul_720x240_x128's DATA region via
    state.xmem.write_address -- _load_data() is NEVER called. Only weights go
    through the normal file-staged _load_weights path. Returns the cropped
    [N_OUT, N_TOK] result, read directly from XMEM at full ROW_BYTES width
    (teardown()'s output-file crop is a separately-known issue, out of scope,
    and never exercised here).
    """
    assert len(uf_raw) == UF_N_OUT * UF_LANES * 4 == MM_K * MM_ROW_BYTES, (
        "byte-length mismatch between unfold's raw per-stream output region "
        "and matmul's raw DATA region -- direct handoff is not even "
        "byte-shape compatible"
    )

    weights_path = tmp_path / f"mm_w_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())

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

    # THE ACTUAL SEAM UNDER TEST: raw bytes, verbatim, no repacking.
    state.xmem.write_address(MM_DATA_BASE, bytearray(uf_raw))

    app = MatMul720x240x128App(
        inst_path=_MM_INST_BIN,
        input_path="/dev/null",   # placeholder; _load_data is bypassed below
        weights_path=weights_path,
        output_path=None,          # avoid exercising the cropping teardown()
    )
    from ipu_apps.matmuls.matmul_720x240_x128 import _load_weights as mm_load_weights

    def setup_no_load_data(state: "IpuState") -> None:
        mm_load_weights(state, app.weights_path)
        state.regfile.set_cr(0, 0)
        state.regfile.set_cr(9, MM_WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, MM_WEIGHTS_BASE_ROW + 1)
        state.regfile.set_cr(5, MM_OUTPUT_BASE_ROW)
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
        state.regfile.set_lr(11, 110)
        state.regfile.set_lr(12, MM_W_STRIDE_ROWS)

    app.setup = setup_no_load_data
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(MM_OUTPUT_BASE, MM_N_OUT * MM_LANES * 4))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(MM_N_OUT, MM_LANES)
    assert not np.all(rows == _POISON, axis=1).any(), "matmul left poisoned output rows untouched"
    return rows[:, :MM_N_TOK]


def _unfold_stream_reference(x: np.ndarray) -> np.ndarray:
    """Independent reference for stream _STREAM (phase (0,0)): the standard
    stride-2 space-to-depth decimation, matching unfold_8x8x240's module
    docstring exactly -- computed directly from the UNPACKED [C, H, W] array,
    not from pack_input_rows or the kernel's own implementation.
    """
    r_ph, c_ph = _STREAM // 2, _STREAM % 2
    return x[:, r_ph::2, c_ph::2].reshape(UF_C, UF_N_TOK).astype(np.float64)  # [C, N_TOK]


def test_unfold_8x8x240_output_is_not_byte_compatible_with_matmul_720x240_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    channel's VALID lanes (0:N_TOK) in the captured unfold stream before
    handing it to the matmul and confirm the result diverges from the
    reference.
    """
    rng = np.random.RandomState(0x5E15)
    x = rng.uniform(-1.0, 1.0, size=(UF_C, UF_H, UF_W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    uf_raw = bytearray(_run_unfold_capture_stream_xmem(x, tmp_path, tag="mut"))
    row_bytes = UF_LANES * 4
    ch = 13
    corrupt_valid = np.full(UF_N_TOK, 999.0, dtype=np.float32).tobytes()
    off = ch * row_bytes
    uf_raw[off : off + UF_N_TOK * 4] = corrupt_valid

    got = _run_matmul_720x240_direct_xmem_handoff(bytes(uf_raw), W, tmp_path, tag="mut")

    d_expected = _unfold_stream_reference(x)          # [K, N_TOK]
    expected = (W.astype(np.float64) @ d_expected)     # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_unfold_8x8x240_feeds_matmul_720x240_via_direct_xmem_no_file_staging_l5(
    tmp_path: Path,
) -> None:
    """The real question: with the corruption removed, does the verbatim
    byte handoff (no _load_data, no file round-trip, tail lanes included
    as-is) produce the correct matmul result end to end? Also records
    whether the captured tail bytes really are the zero the module docstring
    predicts for THIS kernel (as opposed to unfold_16x16x192's genuine stale
    r_acc garbage).
    """
    rng = np.random.RandomState(0x5E16)
    x = rng.uniform(-1.0, 1.0, size=(UF_C, UF_H, UF_W)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    uf_raw = _run_unfold_capture_stream_xmem(x, tmp_path, tag="clean")

    tail = np.frombuffer(uf_raw, dtype=np.float32).reshape(UF_N_OUT, UF_LANES)[:, UF_N_TOK:]
    tail_is_zero = bool(np.all(tail == 0.0))
    print(
        f"unfold_8x8x240 stream {_STREAM} tail lanes [{UF_N_TOK}:{UF_LANES}] "
        f"all zero = {tail_is_zero} (docstring predicts True for this kernel, "
        "unlike unfold_16x16x192's genuine stale r_acc garbage)"
    )

    got = _run_matmul_720x240_direct_xmem_handoff(uf_raw, W, tmp_path, tag="clean")

    d_expected = _unfold_stream_reference(x)
    expected = (W.astype(np.float64) @ d_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam unfold_8x8x240(direct XMEM, stream {_STREAM})->matmul_720x240_x128 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "unfold_8x8x240's raw per-stream XMEM output, handed to "
            "matmul_720x240_x128 verbatim with NO file staging and NO "
            "_load_data repack, does not match an independent reference"
        ),
    )
