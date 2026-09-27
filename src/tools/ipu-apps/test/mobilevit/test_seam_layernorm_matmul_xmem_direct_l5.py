"""Seam investigation: does layernorm_16x240's raw XMEM output already
satisfy matmul_240x240_x128's raw XMEM input contract, with NO file
round-trip and NO ``_load_data()`` re-packing step at all? (L5 OutProj.)

Near-direct port of ``test_seam_layernorm_matmul_xmem_direct.py`` (the L4
reference, LayerNorm 64x192 -> matmul_192x192_x128) to the L5 shape: same
single-token-group (N_TG=1) structure, different constants (N_CH=K=240,
N_TOK=16). layernorm_16x240's output layout is simply ``OUTPUT_BASE + ch``
(one row per channel, N_TOK=16 valid lanes + zero pad) and
matmul_240x240_x128's DATA region is ``DATA_BASE + k`` (one row per channel,
N_TOK=16 valid lanes + zero pad) -- structurally identical to L4.

The matmul's OUTPUT region is read back via state.xmem.read_address at full
ROW_BYTES width, never through teardown()/an output file (the harness runs
with no output_path), exactly like the L4 reference test.

Method: run the REAL layernorm_16x240 kernel on a fresh state, capture its
raw OUTPUT_BASE bytes directly via state.xmem.read_address. On a SEPARATE
fresh state, poison the matmul's DATA region, write the captured LayerNorm
bytes into it verbatim, run the matmul harness's setup() with _load_data
disabled (weights load normally), and compare against an independent
reference computed from gamma/beta/x directly (LayerNorm formula) times W
(matmul formula) -- the reference never touches either kernel's own
golden/internals.

Mutation-first: the control test corrupts one channel's row in the captured
LayerNorm output before the direct handoff and confirms the result diverges
from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.matmul.matmul_240x240_x128 import app as mm
from ipu_apps.kernels.normalize.layernorm_16x240 import app as ln

from fixture_kernel_support import run_layernorm_capture_xmem, run_matmul_direct_xmem_handoff

assert ln.N_CH == mm.K == 240
assert ln.N_TOK == mm.N_TOK == 16
LANES = ln.ROW_BYTES // 4


def _run_layernorm_16x240_capture_xmem(x, gamma, beta, tmp_path: Path, tag: str) -> bytes:
    x_padded = np.zeros((ln.N_CH, LANES), dtype=np.float32)
    x_padded[:, :ln.N_TOK] = x
    return run_layernorm_capture_xmem(ln, ln.LayerNorm16x240App, x_rows=x_padded,
                                      gamma=gamma, beta=beta, tmp_path=tmp_path, tag=tag)


def _run_matmul_240x240_direct_xmem_handoff(ln_raw: bytes, W, tmp_path: Path, tag: str) -> np.ndarray:
    """Returns the cropped [N_OUT, N_TOK] result."""
    rows = run_matmul_direct_xmem_handoff(mm, mm.MatMul240x240x128App, data_raw=ln_raw, W=W,
                                          tmp_path=tmp_path, tag=tag, output_file=False)
    return rows[:, :mm.N_TOK]


def _layernorm_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Independent float64 reference: mu/sigma reduced over the CHANNEL axis,
    per token -- matches layernorm_16x240's docstring formula exactly, not
    its internal implementation.
    """
    xf = x.astype(np.float64)
    mu = xf.mean(axis=0, keepdims=True)                       # [1, N_TOK]
    var = ((xf - mu) ** 2).mean(axis=0, keepdims=True)
    sigma = np.sqrt(var)
    return gamma.astype(np.float64)[:, None] * (xf - mu) / sigma + beta.astype(np.float64)[:, None]


def _inputs(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, size=(ln.N_CH, ln.N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=ln.N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=ln.N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(mm.N_OUT, mm.K)).astype(np.float32)
    return x, gamma, beta, W


def test_layernorm_16x240_output_is_not_byte_compatible_with_matmul_240x240_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: corrupt one channel's row in the captured
    LayerNorm output before handing it to the matmul and confirm the result
    diverges from the reference.
    """
    x, gamma, beta, W = _inputs(0x5EA5)

    ln_raw = bytearray(_run_layernorm_16x240_capture_xmem(x, gamma, beta, tmp_path, tag="mut"))
    corrupt_row = np.full(LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[7 * ln.ROW_BYTES : 8 * ln.ROW_BYTES] = corrupt_row

    got = _run_matmul_240x240_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")
    expected = W.astype(np.float64) @ _layernorm_reference(x, gamma, beta)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_layernorm_16x240_feeds_matmul_240x240_via_direct_xmem_no_file_staging_l5(
    tmp_path: Path,
) -> None:
    """With the corruption removed, does the verbatim byte handoff (no
    _load_data, no file round-trip) produce the correct matmul result?
    """
    x, gamma, beta, W = _inputs(0x5EA6)

    ln_raw = _run_layernorm_16x240_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_240x240_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")
    expected = W.astype(np.float64) @ _layernorm_reference(x, gamma, beta)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam layernorm_16x240(direct XMEM)->matmul_240x240_x128 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "layernorm_16x240's raw XMEM output, handed to matmul_240x240_x128 "
            "verbatim with NO file staging and NO _load_data repack, does not "
            "match an independent reference -- the seam needs real restaging"
        ),
    )
