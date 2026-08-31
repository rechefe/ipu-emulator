"""Seam investigation: does layernorm_256x144's raw XMEM output already
satisfy matmul_144x144_x128's raw XMEM input contract, with NO file
round-trip and NO ``_load_data()`` re-packing step at all? (L3 OutProj.)

This is a near-direct port of ``test_seam_layernorm_matmul_xmem_direct.py``
(the L4 reference, LayerNorm 64x192 -> matmul_192x192_x128) to the L3 shape,
which differs in one structural way: layernorm_256x144 has TWO token groups
(N_TG=2) interleaved per channel, output row order ``(ch*N_TG + tg)``. That
matters here because matmul_144x144_x128's own DATA region is laid out
one-row-per-(k, tg) with EXACTLY the same ``(k*N_TG + tg)`` row order
(``DATA_ROWS = K * N_TG``, see matmul_144x144_x128/__init__.py) -- so the two
kernels already agree on row order without any repacking, PROVIDED both use
N_CH == K == 144 and N_TG == 2 (verified by the asserts below, not assumed).

Method: run the REAL layernorm_256x144 kernel on a fresh state, capture its
raw OUTPUT_BASE bytes directly via state.xmem.read_address. On a SEPARATE
fresh state, poison the matmul's DATA region, write the captured LayerNorm
bytes into it verbatim, skip _load_data entirely, load weights normally, run
the matmul, and compare against an independent reference computed from
gamma/beta/x directly (LayerNorm formula, mu/sigma over the CHANNEL axis per
token) times W (matmul formula C = W @ D) -- the reference never touches
either kernel's own golden/internals.

Mutation-first: the control test corrupts one (channel, token-group) row in
the captured LayerNorm output before the direct handoff and confirms the
result diverges from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm.layernorm_256x144 import (
    LayerNorm256x144App, N_CH as LN_N_CH, N_TG as LN_N_TG, N_TPG as LN_N_TPG,
    ROW_BYTES as LN_ROW_BYTES, OUTPUT_BASE as LN_OUTPUT_BASE,
)
from ipu_apps.matmuls.matmul_144x144_x128 import (
    MatMul144x144x128App, K as MM_K, N_OUT as MM_N_OUT, N_TG as MM_N_TG,
    N_TOK as MM_N_TOK, LANES as MM_LANES, ROW_BYTES as MM_ROW_BYTES,
    DATA_BASE as MM_DATA_BASE, OUTPUT_BASE as MM_OUTPUT_BASE,
    OUTPUT_ROW_BYTES as MM_OUTPUT_ROW_BYTES, DATA_STRIDE_ROWS as MM_DATA_STRIDE_ROWS,
    OUTPUT_STRIDE_ROWS as MM_OUTPUT_STRIDE_ROWS, W_STRIDE_ROWS as MM_W_STRIDE_ROWS,
    WEIGHTS_BASE_ROW as MM_WEIGHTS_BASE_ROW, OUTPUT_BASE_ROW as MM_OUTPUT_BASE_ROW,
)

_LN_INST_BIN = Path(os.environ["LAYERNORM_256X144_INST_BIN"])
_MM_INST_BIN = Path(os.environ["MATMUL_144X144_X128_INST_BIN"])

_POISON = 1e3

# Structural precondition: both kernels must agree on N_CH==K and N_TG before
# any byte handoff makes sense.
assert LN_N_CH == MM_K == 144
assert LN_N_TG == MM_N_TG == 2
assert LN_N_TPG == MM_N_TOK == 128


def _run_layernorm_256x144_capture_xmem(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, tmp_path: Path, tag: str,
) -> bytes:
    """Run the real layernorm_256x144 kernel; return its raw XMEM output bytes
    (N_CH * N_TG * ROW_BYTES), read directly via state.xmem -- no file
    round-trip.

    layernorm_256x144.setup() writes its input_path's bytes to XMEM VERBATIM
    (no row-expansion/padding logic of its own), so the input FILE itself
    must already be pre-padded to N_CH * N_TG * ROW_BYTES: one zero-padded
    128-lane row per (ch, tg), in (ch*N_TG + tg) order. x here is
    [N_CH, N_TG, N_TPG]; it is padded out to [N_CH, N_TG, LANES] before
    writing (N_TPG == LANES == 128 here so no padding is actually needed,
    but the shape is kept explicit to mirror the L4 reference).
    """
    lanes = LN_ROW_BYTES // 4
    x_padded = np.zeros((LN_N_CH, LN_N_TG, lanes), dtype=np.float32)
    x_padded[:, :, :LN_N_TPG] = x
    input_path = tmp_path / f"ln_x_{tag}.bin"
    gamma_path = tmp_path / f"ln_g_{tag}.bin"
    beta_path = tmp_path / f"ln_b_{tag}.bin"
    input_path.write_bytes(x_padded.tobytes())
    gamma_path.write_bytes(gamma.astype(np.float32).tobytes())
    beta_path.write_bytes(beta.astype(np.float32).tobytes())

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    state.xmem.write_address(
        LN_OUTPUT_BASE,
        bytearray(np.full(LN_N_CH * LN_N_TG * lanes, _POISON, dtype=np.float32).tobytes()),
    )
    app = LayerNorm256x144App(
        inst_path=_LN_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(LN_OUTPUT_BASE, LN_N_CH * LN_N_TG * LN_ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(LN_N_CH * LN_N_TG, lanes)
    assert not np.all(rows == _POISON, axis=1).any(), "layernorm left poisoned rows untouched"
    return raw


def _run_matmul_144x144_direct_xmem_handoff(
    ln_raw: bytes, W: np.ndarray, tmp_path: Path, tag: str,
) -> np.ndarray:
    """Feed ln_raw straight into matmul_144x144_x128's DATA region via
    state.xmem.write_address -- _load_data() is NEVER called. Only weights go
    through the normal file-staged _load_weights path (weights are not under
    test here). Returns the [N_OUT, N_TG, N_TOK] result (tg-split, cropped).
    """
    assert len(ln_raw) == LN_N_CH * LN_N_TG * LN_ROW_BYTES == MM_K * MM_N_TG * MM_ROW_BYTES, (
        "byte-length mismatch between LayerNorm's raw output region and "
        "matmul's raw DATA region -- direct handoff is not even byte-shape "
        "compatible"
    )

    weights_path = tmp_path / f"mm_w_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    output_path = tmp_path / f"mm_out_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Poison the whole data+output region first so a silently-skipped write
    # anywhere in the handoff shows up as 1e3, not as unrelated garbage.
    state.xmem.write_address(
        MM_DATA_BASE,
        bytearray(np.full(MM_K * MM_N_TG * MM_LANES, _POISON, dtype=np.float32).tobytes()),
    )
    state.xmem.write_address(
        MM_OUTPUT_BASE,
        bytearray(np.full(MM_N_OUT * MM_N_TG * MM_LANES, _POISON, dtype=np.float32).tobytes()),
    )

    # THE ACTUAL SEAM UNDER TEST: raw bytes, verbatim, no repacking.
    state.xmem.write_address(MM_DATA_BASE, bytearray(ln_raw))

    app = MatMul144x144x128App(
        inst_path=_MM_INST_BIN,
        input_path="/dev/null",   # placeholder; _load_data is bypassed below
        weights_path=weights_path,
        output_path=output_path,
    )
    from ipu_apps.matmuls.matmul_144x144_x128 import _load_weights as mm_load_weights

    def setup_no_load_data(state: "IpuState") -> None:
        mm_load_weights(state, app.weights_path)
        state.regfile.set_cr(0, 0)
        state.regfile.set_cr(9, MM_WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, MM_WEIGHTS_BASE_ROW + 1)
        state.regfile.set_cr(3, MM_OUTPUT_BASE_ROW)
        state.regfile.set_cr(4, MM_OUTPUT_BASE_ROW + MM_N_OUT * MM_OUTPUT_STRIDE_ROWS)
        state.regfile.set_cr(5, -MM_DATA_STRIDE_ROWS)
        state.regfile.set_cr(6, -(MM_DATA_STRIDE_ROWS // MM_N_TG))
        state.regfile.set_cr(7, -1)
        state.regfile.set_cr(8, 127)
        state.regfile.set_lr(0, 0)
        state.regfile.set_lr(2, MM_DATA_STRIDE_ROWS)
        state.regfile.set_lr(3, MM_OUTPUT_STRIDE_ROWS)
        state.regfile.set_lr(6, 126)
        state.regfile.set_lr(7, 0)
        state.regfile.set_lr(8, 0)
        state.regfile.set_lr(9, 0)
        state.regfile.set_lr(10, MM_N_OUT)
        state.regfile.set_lr(11, 142)
        state.regfile.set_lr(12, MM_W_STRIDE_ROWS)

    app.setup = setup_no_load_data
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(
        MM_OUTPUT_BASE, MM_N_OUT * MM_N_TG * MM_LANES * 4
    ))
    # Output row order is tg-major: row (tg, j) at OUTPUT_BASE + tg*N_OUT*512
    # + j*512 (see matmul_144x144_x128.asm header) -- NOT (j*N_TG+tg) like the
    # DATA region. Reshape accordingly before transposing to [N_OUT, N_TG, ...].
    rows = np.frombuffer(raw, dtype=np.float32).reshape(MM_N_TG, MM_N_OUT, MM_LANES)
    assert not np.all(rows == _POISON, axis=2).any(), "matmul left poisoned output rows untouched"
    return rows[:, :, :MM_N_TOK].transpose(1, 0, 2)   # [N_OUT, N_TG, N_TOK]


def _layernorm_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Independent float64 reference: mu/sigma reduced over the CHANNEL axis,
    per token -- matches layernorm_256x144's docstring formula exactly
    (output[ch, tg, i] = gamma[ch] * (x[ch,tg,i] - mu[tg,i]) / sigma[tg,i] +
    beta[ch]), not its internal implementation. x is [N_CH, N_TG, N_TPG].
    """
    xf = x.astype(np.float64)
    mu = xf.mean(axis=0, keepdims=True)                       # [1, N_TG, N_TPG]
    var = ((xf - mu) ** 2).mean(axis=0, keepdims=True)
    sigma = np.sqrt(var)
    return (
        gamma.astype(np.float64)[:, None, None] * (xf - mu) / sigma
        + beta.astype(np.float64)[:, None, None]
    )


def test_layernorm_256x144_output_is_not_byte_compatible_with_matmul_144x144_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    (channel, tg) row in the captured LayerNorm output before handing it to
    the matmul and confirm the result diverges from the reference.
    """
    rng = np.random.RandomState(0x1EC3)
    x = rng.uniform(-2.0, 2.0, size=(LN_N_CH, LN_N_TG, LN_N_TPG)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    ln_raw = bytearray(
        _run_layernorm_256x144_capture_xmem(x, gamma, beta, tmp_path, tag="mut")
    )
    # Corrupt channel 5, tg 1's row (row index 5*N_TG+1 = 11) -- still
    # zero-padding structure preserved, just wrong values.
    row_idx = 5 * LN_N_TG + 1
    corrupt_row = np.full(MM_LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[row_idx * LN_ROW_BYTES : (row_idx + 1) * LN_ROW_BYTES] = corrupt_row

    got = _run_matmul_144x144_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")

    ln_expected = _layernorm_reference(x, gamma, beta)   # [N_CH, N_TG, N_TPG]
    # C[j, tg, t] = sum_k W[j,k] * D[k, tg, t]
    expected = np.einsum("jk,ktp->jtp", W.astype(np.float64), ln_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_layernorm_256x144_feeds_matmul_144x144_via_direct_xmem_no_file_staging_l3(
    tmp_path: Path,
) -> None:
    """The real question: with the corruption removed, does the verbatim
    byte handoff (no _load_data, no file round-trip) produce the correct
    matmul result end to end?
    """
    rng = np.random.RandomState(0x1EC4)
    x = rng.uniform(-2.0, 2.0, size=(LN_N_CH, LN_N_TG, LN_N_TPG)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    ln_raw = _run_layernorm_256x144_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_144x144_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")

    ln_expected = _layernorm_reference(x, gamma, beta)
    expected = np.einsum("jk,ktp->jtp", W.astype(np.float64), ln_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam layernorm_256x144(direct XMEM)->matmul_144x144_x128 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "layernorm_256x144's raw XMEM output, handed to matmul_144x144_x128 "
            "verbatim with NO file staging and NO _load_data repack, does not "
            "match an independent reference -- the seam needs real restaging"
        ),
    )
