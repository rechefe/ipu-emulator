"""Seam investigation: does layernorm_*'s raw XMEM output already satisfy
matmul_*_x128's raw XMEM input contract, with NO file round-trip and NO
_load_data() re-packing step at all?

Prior seam-audit context: matmul_*_x128's `_load_data` reads a *tightly
packed disk file* (K*N_TOK contiguous FP32 elements) and row-expands it into
XMEM -- one zero-padded 512 B row per channel. That is a property of the
FILE format, not of XMEM itself. layernorm_*'s `teardown()` already writes
one zero-padded 512 B row per channel directly to XMEM (see
kernels/normalize/layernorm_64x192/app.py: DATA_BASE + ch, N_TOK valid lanes
+ zero pad).

So the file-packing step in matmul's `_load_data` may be pure ceremony: if
LayerNorm's raw XMEM output bytes, copied verbatim into the matmul's DATA
region, already produce the right answer with `_load_data` never called,
then the "seam" is not a missing bridge -- it's a harness artifact of the
test path (LayerNorm writes XMEM directly; matmul's test harness insists on
going through a file even when nothing about the ISA requires it).

Relationship to test_seam_pipeline_boundaries.py's confirmed "DEFECT" verdict
(layernorm_64x192 -> matmul_576x192_x128, seam 3): that test is NOT
contradicted by this one -- it exercises a different code path. It feeds
LayerNorm's full-row *file* into `_load_data`, which parses any input file
as tightly-packed (256 B/channel) regardless of its true layout, so a
512 B/channel file is misread there. This test skips `_load_data` and the
file boundary entirely, writing LayerNorm's raw *XMEM* bytes straight into
the matmul's DATA region at the correct one-row-per-channel stride. Both are
true at once: the FILE contract (`_load_data`) is incompatible with
LayerNorm's full-row file output, while the XMEM CONTENT itself (once placed
at the right row stride) is exactly what the matmul expects.

Method: run the REAL layernorm_* kernel on a fresh state, capture its raw
OUTPUT_BASE bytes directly via state.xmem.read_address (not via
dump_xmem_to_binary + a file re-read). On a SEPARATE fresh state, poison the
matmul's DATA region, write the captured LayerNorm bytes into it verbatim
(byte length must match exactly -- N_CH*ROW_BYTES on one side must equal
K*ROW_BYTES on the other), run the matmul harness's own setup() with
`_load_data` disabled (weights and CR/LR programming unchanged), and compare
against an independent reference computed from gamma/beta/x directly
(LayerNorm formula) times W (matmul formula) -- so the reference never
touches either kernel's own golden/internals.

Mutation-first: each test first proves a byte-for-byte MISMATCH is caught
(corrupt one channel's row before the direct handoff, assert the corrupted
run disagrees with the reference) before trusting the "clean" run's PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernels.matmul.matmul_192x192_x128 import app as mm4
from ipu_apps.kernels.normalize.layernorm_64x192 import app as ln4

from fixture_kernel_support import run_layernorm_capture_xmem, run_matmul_direct_xmem_handoff

assert ln4.N_CH == mm4.K and ln4.N_TOK == mm4.N_TOK and ln4.ROW_BYTES == mm4.ROW_BYTES


def _run_layernorm_64x192_capture_xmem(x, gamma, beta, tmp_path: Path, tag: str) -> bytes:
    """layernorm_64x192.setup() writes its input file to XMEM verbatim, so the
    file is pre-padded to one zero-padded 128-lane row per channel."""
    x_padded = np.zeros((ln4.N_CH, ln4.LANES), dtype=np.float32)
    x_padded[:, :ln4.N_TOK] = x
    return run_layernorm_capture_xmem(ln4, ln4.LayerNorm64x192App, x_rows=x_padded,
                                      gamma=gamma, beta=beta, tmp_path=tmp_path, tag=tag)


def _run_matmul_192x192_direct_xmem_handoff(ln_raw: bytes, W, tmp_path: Path, tag: str) -> np.ndarray:
    """Returns the cropped [N_OUT, N_TOK] result."""
    rows = run_matmul_direct_xmem_handoff(mm4, mm4.MatMul192x192x128App, data_raw=ln_raw, W=W,
                                          tmp_path=tmp_path, tag=tag)
    return rows[:, :mm4.N_TOK]


def _layernorm_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Independent float64 reference: mu/sigma reduced over the CHANNEL axis,
    per token -- matches layernorm_64x192's docstring formula exactly, not
    its internal implementation.
    """
    xf = x.astype(np.float64)
    mu = xf.mean(axis=0, keepdims=True)                       # [1, N_TOK]
    var = ((xf - mu) ** 2).mean(axis=0, keepdims=True)
    sigma = np.sqrt(var)
    return gamma.astype(np.float64)[:, None] * (xf - mu) / sigma + beta.astype(np.float64)[:, None]


def _inputs(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, size=(ln4.N_CH, ln4.N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=ln4.N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=ln4.N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(mm4.N_OUT, mm4.K)).astype(np.float32)
    return x, gamma, beta, W


def test_layernorm_output_is_not_byte_compatible_with_matmul_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    channel's row in the captured LayerNorm output before handing it to the
    matmul and confirm the result diverges from the reference.
    """
    x, gamma, beta, W = _inputs(0x1EAF)

    ln_raw = bytearray(_run_layernorm_64x192_capture_xmem(x, gamma, beta, tmp_path, tag="mut"))
    # Corrupt channel 5's row (still zero-padding structure preserved, just
    # wrong values) -- a plausible-looking but wrong handoff.
    corrupt_row = np.full(mm4.LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[5 * ln4.ROW_BYTES : 6 * ln4.ROW_BYTES] = corrupt_row

    got = _run_matmul_192x192_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")
    expected = W.astype(np.float64) @ _layernorm_reference(x, gamma, beta)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_layernorm_output_feeds_matmul_via_direct_xmem_no_file_staging_l4(
    tmp_path: Path,
) -> None:
    """The real question: with the corruption removed, does the verbatim
    byte handoff (no _load_data, no file round-trip) produce the correct
    matmul result end to end?
    """
    x, gamma, beta, W = _inputs(0x1EB0)

    ln_raw = _run_layernorm_64x192_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_192x192_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")
    expected = W.astype(np.float64) @ _layernorm_reference(x, gamma, beta)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    print(f"seam layernorm_64x192(direct XMEM)->matmul_192x192_x128 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=2e-3, atol=2e-2,
        err_msg=(
            "layernorm_64x192's raw XMEM output, handed to matmul_192x192_x128 "
            "verbatim with NO file staging and NO _load_data repack, does not "
            "match an independent reference -- the seam needs real restaging"
        ),
    )
