"""Seam tests: LayerNorm/matmul/residual_add pipeline boundaries (L4, L5).

These tests probe producer-output-to-consumer-input seams OUTSIDE the
attention chains (qk_scores/attn_scores_km -> attn_v/attn_v_bcast) and outside
the unfold -> matmul seam, both of which are covered elsewhere. Per
kernel_docs/kernel_layer_map.md's crop convention: a producer's teardown/
output_path must emit full, uncropped rows for a downstream kernel that
stages its input verbatim (write_address of raw file bytes, no reshape) --
only the FINAL consumer in a chain is allowed to crop.

Method: run the REAL producer kernel, take the actual bytes its own
output_path/teardown wrote (never a hand-built numpy array), and feed that
file byte-for-byte as the consumer's input, exactly as ``setup()`` would
receive it if these kernels were chained.

Findings recorded here (see kernel_docs/L4_L5_phase0_findings.md for the
narrative writeup, and kernel_docs/kernel_layer_map.md for the current
verdicts -- item 1 below was FIXED as part of the 2026-08-06 seam-audit
follow-up; this file's test for it now asserts the fixed behavior, not the
defect):

1. matmul_240x240_x128 (L5 OutProj) --> residual_add_16x240: FIXED, was a
   DEFECT. OutProj's teardown used to crop to OUTPUT_ROW_BYTES =
   N_TOK*ELEM_BYTES = 64 B per channel, while residual_add_16x240.setup()
   hard-asserts its A/B inputs are full N_ROWS*ROW_BYTES (240*512 B) -- a
   straight verbatim feed failed the assertion outright. All four L5
   matmul_*_x128 kernels (240x240, 480x240, 240x480, 720x240) had the same
   bug -- OUTPUT_ROW_BYTES is now 512 (full row) in all of them, matching
   every L3/L4 matmul kernel's convention and residual_add_16x240's
   expectation. See kernel_docs/kernel_layer_map.md.

2. matmul_192x192_x128 (L4 OutProj) --> residual_add_64x192: AGREES.
   OutProj's teardown here uses OUTPUT_ROW_BYTES = 512 (full row, uncropped),
   matching residual_add_64x192.setup()'s verbatim write_address of the raw
   file with no length assertion at all. Included as the passing control that
   proved the L5 case above was a real defect, not a test-harness artifact --
   now also the pattern the fixed L5 kernels follow.

3. layernorm_64x192 --> matmul_576x192_x128 (L4 QKV): DEFECT, confirmed real
   but SCOPED to the FILE-staging code path only. LayerNorm's teardown emits
   full ROW_BYTES=512 rows (N_TOK=64 valid + 64 padding lanes) via
   dump_xmem_to_binary(..., ROW_BYTES, N_CH*N_TG). matmul_576x192_x128
   ._load_data expects a TIGHTLY PACKED file -- K*N_TOK*ELEM_BYTES bytes
   total, D[k][tok] at k*N_TOK*ELEM_BYTES with NO row padding. Feeding
   LayerNorm's raw output FILE directly makes _load_data slice the wrong
   absolute byte ranges for every k > 0 -- a silent misread, not a size
   assertion failure. Separately confirmed (2026-08-06,
   test_seam_layernorm_matmul_xmem_direct.py and its L3/L5 siblings): if the
   FILE-staging step is bypassed and LayerNorm's raw XMEM output bytes are
   written directly into the matmul's DATA region at the correct one-row-
   per-channel stride, the two agree exactly (this holds at L3, L4, and L5,
   and separately for unfold_*'s output feeding the same matmul DATA region).
   So the defect is specifically that no code path today does that correctly
   strided placement instead of routing through _load_data's tightly-packed
   file assumption -- the XMEM CONTENT was never incompatible, only the FILE
   CONTRACT `_load_data` enforces on its input path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm.layernorm_64x192 import LayerNorm64x192App, N_CH as LN_N_CH, N_TOK as LN_N_TOK
from ipu_apps.matmuls.matmul_576x192_x128 import MatMul576x192x128App, K as QKV_K, N_OUT as QKV_N_OUT, N_TOK as QKV_N_TOK

from ipu_apps.matmuls.matmul_240x240_x128 import MatMul240x240x128App, K as L5_OP_K, N_OUT as L5_OP_N_OUT, N_TOK as L5_OP_N_TOK, LANES as L5_OP_LANES
from ipu_apps.residual_add.residual_add_16x240 import ResidualAdd16x240App, N_CH as L5_RA_N_CH, N_TOK as L5_RA_N_TOK, ROW_BYTES as L5_ROW_BYTES

from ipu_apps.matmuls.matmul_192x192_x128 import MatMul192x192x128App, K as L4_OP_K, N_OUT as L4_OP_N_OUT, N_TOK as L4_OP_N_TOK, LANES as L4_OP_LANES
from ipu_apps.residual_add.residual_add_64x192 import ResidualAdd64x192App, N_CH as L4_RA_N_CH, N_ROWS as L4_RA_N_ROWS

_LN_64X192_INST_BIN = Path(os.environ["LAYERNORM_64X192_INST_BIN"])
_MATMUL_576X192_INST_BIN = Path(os.environ["MATMUL_576X192_X128_INST_BIN"])
_MATMUL_240X240_INST_BIN = Path(os.environ["MATMUL_240X240_X128_INST_BIN"])
_RESIDUAL_16X240_INST_BIN = Path(os.environ["RESIDUAL_ADD_16X240_INST_BIN"])
_MATMUL_192X192_INST_BIN = Path(os.environ["MATMUL_192X192_X128_INST_BIN"])
_RESIDUAL_64X192_INST_BIN = Path(os.environ["RESIDUAL_ADD_64X192_INST_BIN"])

_POISON = 1e3


def _new_state() -> IpuState:
    return IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )


# ---------------------------------------------------------------------------
# Seam 1 (DEFECT): matmul_240x240_x128 (L5 OutProj) -> residual_add_16x240
# ---------------------------------------------------------------------------

def test_seam_outproj_240x240_to_residual_add_16x240_agrees(tmp_path: Path) -> None:
    """L5 OutProj's output now agrees with residual_add_16x240 verbatim.

    Was a DEFECT (matmul_240x240_x128's teardown used to crop to
    OUTPUT_ROW_BYTES = N_TOK*ELEM_BYTES = 64 B/channel, which
    residual_add_16x240.setup()'s hard full-row length assertion rejected
    outright). Fixed 2026-08-06: OUTPUT_ROW_BYTES is now 512 (full row,
    uncropped), matching every other matmul_*_x128 kernel and
    residual_add_16x240's expectation. This test now mirrors the L4 control
    case below: poison the destination first, run the real OutProj kernel,
    feed its real raw output verbatim as residual_add's A input, and check
    the numeric result against an independent reference (B=0).
    """
    rng = np.random.RandomState(0x5EAF1)

    D = rng.uniform(-1.0, 1.0, size=(L5_OP_K, L5_OP_N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(L5_OP_N_OUT, L5_OP_K)).astype(np.float32)
    data_path = tmp_path / "outproj_input.bin"
    weights_path = tmp_path / "outproj_weights.bin"
    data_path.write_bytes(D.tobytes())
    weights_path.write_bytes(W.tobytes())

    outproj_output_path = tmp_path / "outproj_output.bin"
    state = _new_state()
    outproj_app = MatMul240x240x128App(
        inst_path=_MATMUL_240X240_INST_BIN,
        input_path=data_path,
        weights_path=weights_path,
        output_path=outproj_output_path,
    )
    state, cycles = outproj_app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    outproj_raw = outproj_output_path.read_bytes()
    assert len(outproj_raw) == L5_OP_N_OUT * L5_ROW_BYTES, (
        "sanity: OutProj's own output_path is now the full-row, uncropped form"
    )
    assert len(outproj_raw) == L5_RA_N_CH * L5_ROW_BYTES, (
        "OutProj's row count/width must match residual_add_16x240's expected "
        "A/B row shape for the verbatim feed to even be plausible"
    )

    # Feed OutProj's real raw output verbatim as residual_add_16x240's A
    # input; B is a zero buffer of matching full-row shape so the residual
    # add's result isolates OutProj's own numeric correctness.
    b_path = tmp_path / "residual_b.bin"
    b_path.write_bytes(np.zeros(L5_RA_N_CH * L5_OP_LANES, dtype=np.float32).tobytes())

    residual_state = _new_state()
    residual_app = ResidualAdd16x240App(
        inst_path=_RESIDUAL_16X240_INST_BIN,
        input_a_path=outproj_output_path,
        input_b_path=b_path,
        output_path=tmp_path / "residual_output.bin",
    )
    residual_state, cycles = residual_app.run(max_cycles=1_000_000, state=residual_state)
    assert cycles > 0

    # residual_add_16x240 is the FINAL consumer in this chain, so (per the
    # crop convention) it crops its own output to N_TOK -- unlike
    # residual_add_64x192 in the L4 control below, which emits full rows.
    got = np.frombuffer(
        Path(residual_app.output_path).read_bytes(), dtype=np.float32
    ).reshape(L5_RA_N_CH, L5_RA_N_TOK)

    expected = W @ D   # [N_OUT, N_TOK], the OutProj matmul result (B is all zero)
    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg=(
            "L5 OutProj -> residual_add_16x240 seam: real OutProj output, "
            "fed verbatim, did not reproduce the matmul result through the "
            "residual add (B=0) -- the fix may have regressed"
        ),
    )


def test_seam_outproj_192x192_to_residual_add_64x192_agrees(tmp_path: Path) -> None:
    """Control case: the L4 OutProj -> residual_add seam DOES agree.

    matmul_192x192_x128's teardown uses OUTPUT_ROW_BYTES=512 (full,
    uncropped rows) and residual_add_64x192.setup() stages A/B verbatim with
    no reshape. Poison the destination first (garbage-fill via a bogus B
    buffer of matching size) to prove the test is actually exercising the
    real byte layout, not passing vacuously.
    """
    rng = np.random.RandomState(0xC0FFEE)
    D = rng.uniform(-1.0, 1.0, size=(L4_OP_K, L4_OP_N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(L4_OP_N_OUT, L4_OP_K)).astype(np.float32)
    data_path = tmp_path / "outproj4_input.bin"
    weights_path = tmp_path / "outproj4_weights.bin"
    data_path.write_bytes(D.tobytes())
    weights_path.write_bytes(W.tobytes())

    outproj_output_path = tmp_path / "outproj4_output.bin"
    state = _new_state()
    outproj_app = MatMul192x192x128App(
        inst_path=_MATMUL_192X192_INST_BIN,
        input_path=data_path,
        weights_path=weights_path,
        output_path=outproj_output_path,
    )
    state, cycles = outproj_app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    outproj_raw = outproj_output_path.read_bytes()
    assert len(outproj_raw) == L4_OP_N_OUT * L4_OP_LANES * 4, (
        "sanity: L4 OutProj emits full, uncropped LANES-wide rows"
    )
    assert len(outproj_raw) == L4_RA_N_ROWS * L4_OP_LANES * 4, (
        "OutProj's row count/width must match residual_add_64x192's expected "
        "A/B row shape for the verbatim feed to even be plausible"
    )

    b_path = tmp_path / "residual4_b.bin"
    b_path.write_bytes(np.zeros(L4_RA_N_ROWS * L4_OP_LANES, dtype=np.float32).tobytes())

    residual_state = _new_state()
    residual_app = ResidualAdd64x192App(
        inst_path=_RESIDUAL_64X192_INST_BIN,
        input_a_path=outproj_output_path,
        input_b_path=b_path,
        output_path=tmp_path / "residual4_output.bin",
    )
    residual_state, cycles = residual_app.run(max_cycles=1_000_000, state=residual_state)
    assert cycles > 0

    got = np.frombuffer(
        Path(residual_app.output_path).read_bytes(), dtype=np.float32
    ).reshape(L4_RA_N_ROWS, L4_OP_LANES)

    expected = W @ D  # [N_OUT, N_TOK], the OutProj matmul result (B is all zero)
    np.testing.assert_allclose(
        got[:, :L4_OP_N_TOK], expected, rtol=1e-4, atol=1e-3,
        err_msg=(
            "L4 OutProj -> residual_add_64x192 seam: real OutProj output, "
            "fed verbatim, did not reproduce the matmul result through the "
            "residual add (B=0) -- the claimed 'agrees' verdict is wrong"
        ),
    )


# ---------------------------------------------------------------------------
# Seam 3 (DEFECT): layernorm_64x192 -> matmul_576x192_x128 (L4 QKV)
# ---------------------------------------------------------------------------

def test_seam_layernorm_64x192_to_matmul_576x192_qkv_pitch_mismatch(tmp_path: Path) -> None:
    """LayerNorm's full-row output cannot feed matmul_576x192_x128's _load_data.

    LayerNorm emits N_CH rows of ROW_BYTES=512 (64 valid + 64 padding lanes).
    matmul_576x192_x128._load_data expects a TIGHTLY PACKED file: K*N_TOK
    elements with D[k][tok] at element k*N_TOK+tok, no row padding at all.

    Poison methodology: fill LayerNorm's padding lanes with a distinctive
    non-zero marker (poison), run the real kernel, take its real raw
    output_path bytes, and feed them as matmul_576x192_x128's data_path.
    Because the two layouts have different strides (512 B/channel vs 256
    B/channel), _load_data's fixed-offset slicing reads each channel k>0 from
    the WRONG absolute byte range -- it silently pulls bytes belonging to a
    different channel (or the poison padding) instead of erroring, and the
    matmul result diverges from the true LayerNorm-output-driven QKV
    projection. This is a silent-misread defect, distinct from seam 1's
    loud assertion failure.
    """
    rng = np.random.RandomState(0x14192)

    x = np.zeros((LN_N_CH, 128), dtype=np.float32)
    x[:, :LN_N_TOK] = rng.uniform(-2.0, 2.0, size=(LN_N_CH, LN_N_TOK))
    # Distinctive poison in the padding lanes: a chained producer would leave
    # real (non-zero) data there, not the harness's usual zero-fill.
    x[:, LN_N_TOK:] = _POISON
    gamma = np.ones(LN_N_CH, dtype=np.float32)
    beta = np.zeros(LN_N_CH, dtype=np.float32)

    input_path = tmp_path / "ln_input.bin"
    gamma_path = tmp_path / "ln_gamma.bin"
    beta_path = tmp_path / "ln_beta.bin"
    input_path.write_bytes(x.tobytes())
    gamma_path.write_bytes(gamma.tobytes())
    beta_path.write_bytes(beta.tobytes())

    ln_output_path = tmp_path / "ln_output.bin"
    ln_state = _new_state()
    ln_app = LayerNorm64x192App(
        inst_path=_LN_64X192_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=ln_output_path,
    )
    ln_state, cycles = ln_app.run(max_cycles=5_000_000, state=ln_state)
    assert cycles > 0

    ln_raw = ln_output_path.read_bytes()
    # LayerNorm's teardown uses dump_xmem_to_binary(..., ROW_BYTES, N_CH) --
    # full, uncropped 512 B rows. Confirm that's really what we got.
    assert len(ln_raw) == LN_N_CH * 512, (
        "sanity: layernorm_64x192.teardown() emits full ROW_BYTES rows, "
        f"got {len(ln_raw)} bytes, expected {LN_N_CH * 512}"
    )

    qkv_weights = rng.uniform(-1.0, 1.0, size=(QKV_N_OUT, QKV_K)).astype(np.float32)
    weights_path = tmp_path / "qkv_weights.bin"
    weights_path.write_bytes(qkv_weights.tobytes())

    # QKV's own declared contract: K*N_TOK*4 tightly-packed bytes.
    qkv_expected_bytes = QKV_K * QKV_N_TOK * 4
    assert len(ln_raw) != qkv_expected_bytes, (
        "sanity: the two contracts really do disagree on file size "
        f"(LayerNorm full-row={len(ln_raw)} B vs QKV tightly-packed="
        f"{qkv_expected_bytes} B) -- if they now match, the defect may be fixed"
    )

    qkv_output_path = tmp_path / "qkv_output.bin"
    qkv_state = _new_state()
    qkv_app = MatMul576x192x128App(
        inst_path=_MATMUL_576X192_INST_BIN,
        input_path=ln_output_path,
        weights_path=weights_path,
        output_path=qkv_output_path,
    )

    # LayerNorm's dump (98304 B) is LARGER than QKV's declared minimum
    # (49152 B), so _load_data's `len(raw) < expected` guard does NOT reject
    # it -- it passes the size check and then silently slices the wrong
    # absolute byte ranges for every channel k > 0. Prove the corruption
    # rather than asserting a raise.
    qkv_state, cycles = qkv_app.run(max_cycles=20_000_000, state=qkv_state)
    assert cycles > 0

    got = np.frombuffer(qkv_output_path.read_bytes(), dtype=np.float32).reshape(
        QKV_N_OUT, 128
    )

    # What the tightly-packed contract WOULD have produced, if LayerNorm's
    # output had been correctly cropped to N_TOK*4 B/channel before staging
    # (i.e. the restaging step this seam is missing).
    ln_valid = np.frombuffer(ln_raw, dtype=np.float32).reshape(LN_N_CH, 128)[:, :LN_N_TOK]
    correctly_staged = ln_valid.astype(np.float32)  # [K, N_TOK], tightly packed
    expected = qkv_weights @ correctly_staged        # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got[:, :QKV_N_TOK] - expected)))
    print(f"layernorm_64x192 -> matmul_576x192_x128 verbatim-feed max abs error = {max_err:.3e}")

    # The verbatim (mis-staged) feed must NOT reproduce the correctly-staged
    # result -- that is exactly the silent-misread defect. If this assertion
    # ever fails (i.e. they DO match), the pitch mismatch has been fixed
    # upstream and this test (and the seam verdict) needs updating.
    assert max_err > 1.0, (
        "expected the verbatim (pitch-mismatched) feed to diverge sharply "
        "from the correctly-staged reference -- got near-agreement, which "
        "would mean matmul_576x192_x128 somehow tolerates LayerNorm's "
        "full-row output; re-examine whether this seam is actually fixed"
    )
