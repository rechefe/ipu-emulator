"""Seam investigation: does layernorm_256x144's raw XMEM output already
satisfy matmul_144x144_x128's raw XMEM input contract, with NO file
round-trip and NO ``_load_data()`` re-packing step at all? (L3 OutProj.)

This is a near-direct port of ``test_seam_layernorm_matmul_xmem_direct.py``
(the L4 reference, LayerNorm 64x192 -> matmul_192x192_x128) to the L3 shape,
which differs in one structural way: layernorm_256x144 has TWO token groups
(N_TG=2) interleaved per channel, output row order ``(ch*N_TG + tg)``. That
matters here because matmul_144x144_x128's own DATA region is laid out
one-row-per-(k, tg) with EXACTLY the same ``(k*N_TG + tg)`` row order
(``DATA_ROWS = K * N_TG``, see kernels/matmul/matmul_144x144_x128/app.py) --
so the two kernels already agree on row order without any repacking,
PROVIDED both use N_CH == K == 144 and N_TG == 2 (verified by the asserts
below, not assumed).

Method: run the REAL layernorm_256x144 kernel on a fresh state, capture its
raw OUTPUT_BASE bytes directly via state.xmem.read_address. On a SEPARATE
fresh state, poison the matmul's DATA region, write the captured LayerNorm
bytes into it verbatim, run the matmul harness's setup() with _load_data
disabled (weights load normally), and compare against an independent
reference computed from gamma/beta/x directly (LayerNorm formula, mu/sigma
over the CHANNEL axis per token) times W (matmul formula C = W @ D) -- the
reference never touches either kernel's own golden/internals.

Mutation-first: the control test corrupts one (channel, token-group) row in
the captured LayerNorm output before the direct handoff and confirms the
result diverges from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.matmul.matmul_144x144_x128 import app as mm
from ipu_apps.kernels.normalize.layernorm_256x144 import app as ln

from fixture_kernel_support import run_layernorm_capture_xmem, run_matmul_direct_xmem_handoff

# Structural precondition: both kernels must agree on N_CH==K and N_TG before
# any byte handoff makes sense.
assert ln.N_CH == mm.K == 144
assert ln.N_TG == mm.N_TG == 2
assert ln.N_TPG == mm.N_TOK == 128
LANES = ln.ROW_BYTES // 4


def _run_layernorm_256x144_capture_xmem(x, gamma, beta, tmp_path: Path, tag: str) -> bytes:
    """x is [N_CH, N_TG, N_TPG]; the input file is one zero-padded 128-lane
    row per (ch, tg), in (ch*N_TG + tg) order (N_TPG == LANES == 128 here so
    no padding is actually needed, but the shape is kept explicit)."""
    x_padded = np.zeros((ln.N_CH, ln.N_TG, LANES), dtype=np.float32)
    x_padded[:, :, :ln.N_TPG] = x
    return run_layernorm_capture_xmem(ln, ln.LayerNorm256x144App,
                                      x_rows=x_padded.reshape(ln.N_CH * ln.N_TG, LANES),
                                      gamma=gamma, beta=beta, tmp_path=tmp_path, tag=tag)


def _run_matmul_144x144_direct_xmem_handoff(ln_raw: bytes, W, tmp_path: Path, tag: str) -> np.ndarray:
    """Returns the [N_OUT, N_TG, N_TOK] result (tg-split, cropped)."""
    rows = run_matmul_direct_xmem_handoff(mm, mm.MatMul144x144x128App, data_raw=ln_raw, W=W,
                                          tmp_path=tmp_path, tag=tag)
    # Output row order is tg-major: row (tg, j) at OUTPUT_BASE + tg*N_OUT*512
    # + j*512 (see matmul_144x144_x128.asm header) -- NOT (j*N_TG+tg) like the
    # DATA region. Reshape accordingly before transposing to [N_OUT, N_TG, ...].
    rows = rows.reshape(mm.N_TG, mm.N_OUT, LANES)
    return rows[:, :, :mm.N_TOK].transpose(1, 0, 2)


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


def _inputs(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, size=(ln.N_CH, ln.N_TG, ln.N_TPG)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=ln.N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=ln.N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(mm.N_OUT, mm.K)).astype(np.float32)
    return x, gamma, beta, W


def _expected(x, gamma, beta, W):
    # C[j, tg, t] = sum_k W[j,k] * D[k, tg, t]
    return np.einsum("jk,ktp->jtp", W.astype(np.float64), _layernorm_reference(x, gamma, beta))


def test_layernorm_256x144_output_is_not_byte_compatible_with_matmul_144x144_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: corrupt one (channel, tg) row in the captured
    LayerNorm output before handing it to the matmul and confirm the result
    diverges from the reference.
    """
    x, gamma, beta, W = _inputs(0x1EC3)

    ln_raw = bytearray(_run_layernorm_256x144_capture_xmem(x, gamma, beta, tmp_path, tag="mut"))
    # Corrupt channel 5, tg 1's row (row index 5*N_TG+1 = 11) -- still
    # zero-padding structure preserved, just wrong values.
    row_idx = 5 * ln.N_TG + 1
    corrupt_row = np.full(LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[row_idx * ln.ROW_BYTES : (row_idx + 1) * ln.ROW_BYTES] = corrupt_row

    got = _run_matmul_144x144_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")

    max_err = float(np.max(np.abs(got.astype(np.float64) - _expected(x, gamma, beta, W))))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_layernorm_256x144_feeds_matmul_144x144_via_direct_xmem_no_file_staging_l3(
    tmp_path: Path,
) -> None:
    """With the corruption removed, does the verbatim byte handoff (no
    _load_data, no file round-trip) produce the correct matmul result?
    """
    x, gamma, beta, W = _inputs(0x1EC4)

    ln_raw = _run_layernorm_256x144_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_144x144_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")
    expected = _expected(x, gamma, beta, W)

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
