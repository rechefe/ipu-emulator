"""Seam tests: LayerNorm/matmul/residual_add pipeline boundaries (L4, L5).

These tests probe producer-output-to-consumer-input seams OUTSIDE the
attention chains (qk_scores/attn_scores_km -> attn_v/attn_v_bcast) and outside
the unfold -> matmul seam, both of which are covered elsewhere. Per the
crop rule in docs/content/kernels/mobilevit.md: a producer's teardown/
output_path must emit full, uncropped rows for a downstream kernel that
stages its input verbatim (write_address of raw file bytes, no reshape) --
only the FINAL consumer in a chain is allowed to crop.

Method: run the REAL producer kernel, take the actual bytes its own
output_path/teardown wrote (never a hand-built numpy array), and feed that
file byte-for-byte as the consumer's input, exactly as ``setup()`` would
receive it if these kernels were chained.

Findings recorded here (item 1's test asserts the fixed behavior, not the
defect):

1. matmul_240x240_x128 (L5 OutProj) --> residual_add_16x240: FIXED, was a
   DEFECT. OutProj's teardown used to crop to OUTPUT_ROW_BYTES =
   N_TOK*ELEM_BYTES = 64 B per channel, while residual_add_16x240.setup()
   hard-asserts its A/B inputs are full N_ROWS*ROW_BYTES (240*512 B) -- a
   straight verbatim feed failed the assertion outright. All four L5
   matmul_*_x128 kernels (240x240, 480x240, 240x480, 720x240) had the same
   bug -- OUTPUT_ROW_BYTES is now 512 (full row) in all of them, matching
   every L3/L4 matmul kernel's convention and residual_add_16x240's
   expectation.

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

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.elementwise.residual_add_16x240 import app as ra5
from ipu_apps.kernels.elementwise.residual_add_64x192 import app as ra4
from ipu_apps.kernels.matmul.matmul_192x192_x128 import app as op4
from ipu_apps.kernels.matmul.matmul_240x240_x128 import app as op5
from ipu_apps.kernels.matmul.matmul_576x192_x128 import app as qkv4
from ipu_apps.kernels.normalize.layernorm_64x192 import app as ln4

from fixture_kernel_support import POISON, kernel_inst


# ---------------------------------------------------------------------------
# Seam 1 (was a DEFECT, fixed): matmul_240x240_x128 (L5 OutProj) -> residual_add_16x240
# ---------------------------------------------------------------------------

def test_seam_outproj_240x240_to_residual_add_16x240_agrees(tmp_path: Path) -> None:
    """L5 OutProj's output now agrees with residual_add_16x240 verbatim.

    Was a DEFECT (matmul_240x240_x128's teardown used to crop to
    OUTPUT_ROW_BYTES = N_TOK*ELEM_BYTES = 64 B/channel, which
    residual_add_16x240.setup()'s hard full-row length assertion rejected
    outright). Fixed 2026-08-06: OUTPUT_ROW_BYTES is now 512 (full row,
    uncropped), matching every other matmul_*_x128 kernel and
    residual_add_16x240's expectation. Run the real OutProj kernel, feed its
    real raw output verbatim as residual_add's A input, and check the numeric
    result against an independent reference (B=0).
    """
    rng = np.random.RandomState(0x5EAF1)

    D = rng.uniform(-1.0, 1.0, size=(op5.K, op5.N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(op5.N_OUT, op5.K)).astype(np.float32)
    data_path = tmp_path / "outproj_input.bin"
    weights_path = tmp_path / "outproj_weights.bin"
    data_path.write_bytes(D.tobytes())
    weights_path.write_bytes(W.tobytes())

    outproj_output_path = tmp_path / "outproj_output.bin"
    outproj_app = op5.MatMul240x240x128App(
        inst_path=kernel_inst("matmul_240x240_x128"),
        input_path=data_path,
        weights_path=weights_path,
        output_path=outproj_output_path,
    )
    _, cycles = outproj_app.run(max_cycles=5_000_000)
    assert cycles > 0

    outproj_raw = outproj_output_path.read_bytes()
    assert len(outproj_raw) == op5.N_OUT * ra5.ROW_BYTES, (
        "sanity: OutProj's own output_path is now the full-row, uncropped form"
    )
    assert len(outproj_raw) == ra5.N_CH * ra5.ROW_BYTES, (
        "OutProj's row count/width must match residual_add_16x240's expected "
        "A/B row shape for the verbatim feed to even be plausible"
    )

    # Feed OutProj's real raw output verbatim as residual_add_16x240's A
    # input; B is a zero buffer of matching full-row shape so the residual
    # add's result isolates OutProj's own numeric correctness.
    b_path = tmp_path / "residual_b.bin"
    b_path.write_bytes(np.zeros(ra5.N_CH * op5.LANES, dtype=np.float32).tobytes())

    residual_app = ra5.ResidualAdd16x240App(
        inst_path=kernel_inst("residual_add_16x240"),
        input_a_path=outproj_output_path,
        input_b_path=b_path,
        output_path=tmp_path / "residual_output.bin",
    )
    _, cycles = residual_app.run(max_cycles=1_000_000)
    assert cycles > 0

    # residual_add_16x240 is the FINAL consumer in this chain, so (per the
    # crop convention) it crops its own output to N_TOK -- unlike
    # residual_add_64x192 in the L4 control below, which emits full rows.
    got = np.frombuffer(
        Path(residual_app.output_path).read_bytes(), dtype=np.float32
    ).reshape(ra5.N_CH, ra5.N_TOK)

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
    no reshape.
    """
    rng = np.random.RandomState(0xC0FFEE)
    D = rng.uniform(-1.0, 1.0, size=(op4.K, op4.N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(op4.N_OUT, op4.K)).astype(np.float32)
    data_path = tmp_path / "outproj4_input.bin"
    weights_path = tmp_path / "outproj4_weights.bin"
    data_path.write_bytes(D.tobytes())
    weights_path.write_bytes(W.tobytes())

    outproj_output_path = tmp_path / "outproj4_output.bin"
    outproj_app = op4.MatMul192x192x128App(
        inst_path=kernel_inst("matmul_192x192_x128"),
        input_path=data_path,
        weights_path=weights_path,
        output_path=outproj_output_path,
    )
    _, cycles = outproj_app.run(max_cycles=20_000_000)
    assert cycles > 0

    outproj_raw = outproj_output_path.read_bytes()
    assert len(outproj_raw) == op4.N_OUT * op4.LANES * 4, (
        "sanity: L4 OutProj emits full, uncropped LANES-wide rows"
    )
    assert len(outproj_raw) == ra4.N_ROWS * op4.LANES * 4, (
        "OutProj's row count/width must match residual_add_64x192's expected "
        "A/B row shape for the verbatim feed to even be plausible"
    )

    b_path = tmp_path / "residual4_b.bin"
    b_path.write_bytes(np.zeros(ra4.N_ROWS * op4.LANES, dtype=np.float32).tobytes())

    residual_app = ra4.ResidualAdd64x192App(
        inst_path=kernel_inst("residual_add_64x192"),
        input_a_path=outproj_output_path,
        input_b_path=b_path,
        output_path=tmp_path / "residual4_output.bin",
    )
    _, cycles = residual_app.run(max_cycles=1_000_000)
    assert cycles > 0

    got = np.frombuffer(
        Path(residual_app.output_path).read_bytes(), dtype=np.float32
    ).reshape(ra4.N_ROWS, op4.LANES)

    expected = W @ D  # [N_OUT, N_TOK], the OutProj matmul result (B is all zero)
    np.testing.assert_allclose(
        got[:, :op4.N_TOK], expected, rtol=1e-4, atol=1e-3,
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

    x = np.zeros((ln4.N_CH, 128), dtype=np.float32)
    x[:, :ln4.N_TOK] = rng.uniform(-2.0, 2.0, size=(ln4.N_CH, ln4.N_TOK))
    # Distinctive poison in the padding lanes: a chained producer would leave
    # real (non-zero) data there, not the harness's usual zero-fill.
    x[:, ln4.N_TOK:] = POISON
    gamma = np.ones(ln4.N_CH, dtype=np.float32)
    beta = np.zeros(ln4.N_CH, dtype=np.float32)

    input_path = tmp_path / "ln_input.bin"
    gamma_path = tmp_path / "ln_gamma.bin"
    beta_path = tmp_path / "ln_beta.bin"
    input_path.write_bytes(x.tobytes())
    gamma_path.write_bytes(gamma.tobytes())
    beta_path.write_bytes(beta.tobytes())

    ln_output_path = tmp_path / "ln_output.bin"
    ln_app = ln4.LayerNorm64x192App(
        inst_path=kernel_inst("layernorm_64x192"),
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=ln_output_path,
    )
    _, cycles = ln_app.run(max_cycles=5_000_000)
    assert cycles > 0

    ln_raw = ln_output_path.read_bytes()
    # LayerNorm's teardown uses dump_xmem_to_binary(..., ROW_BYTES, N_CH) --
    # full, uncropped 512 B rows. Confirm that's really what we got.
    assert len(ln_raw) == ln4.N_CH * 512, (
        "sanity: layernorm_64x192.teardown() emits full ROW_BYTES rows, "
        f"got {len(ln_raw)} bytes, expected {ln4.N_CH * 512}"
    )

    qkv_weights = rng.uniform(-1.0, 1.0, size=(qkv4.N_OUT, qkv4.K)).astype(np.float32)
    weights_path = tmp_path / "qkv_weights.bin"
    weights_path.write_bytes(qkv_weights.tobytes())

    # QKV's own declared contract: K*N_TOK*4 tightly-packed bytes.
    qkv_expected_bytes = qkv4.K * qkv4.N_TOK * 4
    assert len(ln_raw) != qkv_expected_bytes, (
        "sanity: the two contracts really do disagree on file size "
        f"(LayerNorm full-row={len(ln_raw)} B vs QKV tightly-packed="
        f"{qkv_expected_bytes} B) -- if they now match, the defect may be fixed"
    )

    qkv_output_path = tmp_path / "qkv_output.bin"
    qkv_app = qkv4.MatMul576x192x128App(
        inst_path=kernel_inst("matmul_576x192_x128"),
        input_path=ln_output_path,
        weights_path=weights_path,
        output_path=qkv_output_path,
    )

    # LayerNorm's dump (98304 B) is LARGER than QKV's declared minimum
    # (49152 B), so _load_data's `len(raw) < expected` guard does NOT reject
    # it -- it passes the size check and then silently slices the wrong
    # absolute byte ranges for every channel k > 0. Prove the corruption
    # rather than asserting a raise.
    _, cycles = qkv_app.run(max_cycles=20_000_000)
    assert cycles > 0

    got = np.frombuffer(qkv_output_path.read_bytes(), dtype=np.float32).reshape(qkv4.N_OUT, 128)

    # What the tightly-packed contract WOULD have produced, if LayerNorm's
    # output had been correctly cropped to N_TOK*4 B/channel before staging
    # (i.e. the restaging step this seam is missing).
    ln_valid = np.frombuffer(ln_raw, dtype=np.float32).reshape(ln4.N_CH, 128)[:, :ln4.N_TOK]
    expected = qkv_weights @ ln_valid.astype(np.float32)   # [N_OUT, N_TOK]

    max_err = float(np.max(np.abs(got[:, :qkv4.N_TOK] - expected)))
    print(f"layernorm_64x192 -> matmul_576x192_x128 verbatim-feed max abs error = {max_err:.3e}")

    # The verbatim (mis-staged) feed must NOT reproduce the correctly-staged
    # result -- that is exactly the silent-misread defect. If this assertion
    # ever fails (i.e. they DO match), the pitch mismatch has been fixed
    # in the producer and this test (and the seam verdict) needs updating.
    assert max_err > 1.0, (
        "expected the verbatim (pitch-mismatched) feed to diverge sharply "
        "from the correctly-staged reference -- got near-agreement, which "
        "would mean matmul_576x192_x128 somehow tolerates LayerNorm's "
        "full-row output; re-examine whether this seam is actually fixed"
    )
