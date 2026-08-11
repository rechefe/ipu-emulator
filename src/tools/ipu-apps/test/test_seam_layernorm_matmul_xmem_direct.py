"""Seam investigation: does layernorm_*'s raw XMEM output already satisfy
matmul_*_x128's raw XMEM input contract, with NO file round-trip and NO
_load_data() re-packing step at all?

Prior seam-audit context: matmul_*_x128's `_load_data` reads a *tightly
packed disk file* (K*N_TOK contiguous FP32 elements) and row-expands it into
XMEM -- one zero-padded 512 B row per channel. That is a property of the
FILE format, not of XMEM itself. layernorm_*'s `teardown()` already writes
one zero-padded 512 B row per channel directly to XMEM (see
layernorm_64x192/__init__.py DATA_BASE + ch, N_TOK valid lanes + zero pad).

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
at the right row stride) is exactly what the matmul expects. That means the
"defect" is real only insofar as no code path today does the direct-XMEM
placement (or an equivalent correctly-strided file read) instead of routing
through `_load_data`'s tightly-packed assumption -- see the report for what
this implies about where the fix belongs.

Method: run the REAL layernorm_* kernel on a fresh state, capture its raw
OUTPUT_BASE bytes directly via state.xmem.read_address (not via
dump_xmem_to_binary + a file re-read). On a SEPARATE fresh state, poison the
matmul's DATA region, write the captured LayerNorm bytes into it verbatim
(byte length must match exactly -- N_CH*ROW_BYTES on one side must equal
K*ROW_BYTES on the other), skip _load_data entirely, load weights normally,
run the matmul, and compare against an independent reference computed from
gamma/beta/x directly (LayerNorm formula) times W (matmul formula) -- so the
reference never touches either kernel's own golden/internals.

Mutation-first: each test first proves a byte-for-byte MISMATCH is caught
(corrupt one channel's row before the direct handoff, assert the corrupted
run disagrees with the reference) before trusting the "clean" run's PASS.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm_64x192 import (
    LayerNorm64x192App, N_CH as LN4_N_CH, N_TOK as LN4_N_TOK,
    ROW_BYTES as LN4_ROW_BYTES, OUTPUT_BASE as LN4_OUTPUT_BASE,
)
from ipu_apps.matmul_192x192_x128 import (
    MatMul192x192x128App, K as MM4_K, N_OUT as MM4_N_OUT, N_TOK as MM4_N_TOK,
    LANES as MM4_LANES, ROW_BYTES as MM4_ROW_BYTES,
    DATA_BASE as MM4_DATA_BASE, OUTPUT_BASE as MM4_OUTPUT_BASE,
    OUTPUT_ROW_BYTES as MM4_OUTPUT_ROW_BYTES,
)

_LN4_INST_BIN = Path(os.environ["LAYERNORM_64X192_INST_BIN"])
_MM4_INST_BIN = Path(os.environ["MATMUL_192X192_X128_INST_BIN"])

_POISON = 1e3


def _run_layernorm_64x192_capture_xmem(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, tmp_path: Path, tag: str,
) -> bytes:
    """Run the real layernorm_64x192 kernel; return its raw XMEM output bytes
    (N_CH * ROW_BYTES), read directly via state.xmem -- no file round-trip.

    layernorm_64x192.setup() writes its input_path's bytes to XMEM VERBATIM
    (no row-expansion/padding logic of its own -- unlike matmul's
    _load_data), so the input FILE itself must already be pre-padded to
    N_CH * ROW_BYTES (one zero-padded 128-lane row per channel), exactly as
    test_layernorm_64x192_wide.py does. x here is [N_CH, N_TOK]; it is
    padded out to [N_CH, LANES] before writing.
    """
    x_padded = np.zeros((LN4_N_CH, LN4_ROW_BYTES // 4), dtype=np.float32)
    x_padded[:, :LN4_N_TOK] = x
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
        LN4_OUTPUT_BASE,
        bytearray(np.full(LN4_N_CH * (LN4_ROW_BYTES // 4), _POISON, dtype=np.float32).tobytes()),
    )
    app = LayerNorm64x192App(
        inst_path=_LN4_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(LN4_OUTPUT_BASE, LN4_N_CH * LN4_ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(LN4_N_CH, LN4_ROW_BYTES // 4)
    assert not np.all(rows == _POISON, axis=1).any(), "layernorm left poisoned rows untouched"
    return raw


def _run_matmul_192x192_direct_xmem_handoff(
    ln_raw: bytes, W: np.ndarray, tmp_path: Path, tag: str,
) -> np.ndarray:
    """Feed ln_raw straight into matmul_192x192_x128's DATA region via
    state.xmem.write_address -- _load_data() is NEVER called. Only weights
    go through the normal file-staged _load_weights path (weights are not
    under test here). Returns the cropped [N_OUT, N_TOK] result.
    """
    assert len(ln_raw) == LN4_N_CH * LN4_ROW_BYTES == MM4_K * MM4_ROW_BYTES, (
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
        MM4_DATA_BASE,
        bytearray(np.full(MM4_K * MM4_LANES, _POISON, dtype=np.float32).tobytes()),
    )
    state.xmem.write_address(
        MM4_OUTPUT_BASE,
        bytearray(np.full(MM4_N_OUT * MM4_LANES, _POISON, dtype=np.float32).tobytes()),
    )

    # THE ACTUAL SEAM UNDER TEST: raw bytes, verbatim, no repacking.
    state.xmem.write_address(MM4_DATA_BASE, bytearray(ln_raw))

    app = MatMul192x192x128App(
        inst_path=_MM4_INST_BIN,
        input_path="/dev/null",   # placeholder; _load_data is bypassed below
        weights_path=weights_path,
        output_path=output_path,
    )
    # Bypass _load_data (which would overwrite our direct handoff by reading
    # input_path) -- monkeypatch setup to skip straight to weights + CR/LR.
    from ipu_apps.matmul_192x192_x128 import _load_weights as mm4_load_weights

    def setup_no_load_data(state: "IpuState") -> None:
        mm4_load_weights(state, app.weights_path)
        state.regfile.set_cr(0, 0)
        state.regfile.set_cr(9, MM4_K * 1)  # WEIGHTS_BASE_ROW = DATA_ROWS (K rows, stride 1)
        from ipu_apps.matmul_192x192_x128 import (
            WEIGHTS_BASE_ROW, OUTPUT_BASE_ROW, DATA_STRIDE_ROWS,
            OUTPUT_STRIDE_ROWS, W_STRIDE_ROWS, N_OUT as _N_OUT,
        )
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, -DATA_STRIDE_ROWS)
        state.regfile.set_cr(8, -1)
        state.regfile.set_lr(0, 0)
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)
        state.regfile.set_lr(6, 126)
        state.regfile.set_lr(7, 0)
        state.regfile.set_lr(8, 0)
        state.regfile.set_lr(9, 0)
        state.regfile.set_lr(10, _N_OUT)
        state.regfile.set_lr(11, (MM4_K - MM4_LANES) - 2)
        state.regfile.set_lr(12, W_STRIDE_ROWS)

    app.setup = setup_no_load_data
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(MM4_OUTPUT_BASE, MM4_N_OUT * MM4_LANES * 4))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(MM4_N_OUT, MM4_LANES)
    assert not np.all(rows == _POISON, axis=1).any(), "matmul left poisoned output rows untouched"
    return rows[:, :MM4_N_TOK]


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


def test_layernorm_output_is_not_byte_compatible_with_matmul_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    channel's row in the captured LayerNorm output before handing it to the
    matmul and confirm the result diverges from the reference.
    """
    rng = np.random.RandomState(0x1EAF)
    x = rng.uniform(-2.0, 2.0, size=(LN4_N_CH, LN4_N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN4_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN4_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM4_N_OUT, MM4_K)).astype(np.float32)

    ln_raw = bytearray(
        _run_layernorm_64x192_capture_xmem(x, gamma, beta, tmp_path, tag="mut")
    )
    # Corrupt channel 5's row (still zero-padding structure preserved, just
    # wrong values) -- a plausible-looking but wrong handoff.
    corrupt_row = np.full(MM4_LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[5 * LN4_ROW_BYTES : 6 * LN4_ROW_BYTES] = corrupt_row

    got = _run_matmul_192x192_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")

    ln_expected = _layernorm_reference(x, gamma, beta)
    expected = (W.astype(np.float64) @ ln_expected)

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
    rng = np.random.RandomState(0x1EB0)
    x = rng.uniform(-2.0, 2.0, size=(LN4_N_CH, LN4_N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN4_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN4_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM4_N_OUT, MM4_K)).astype(np.float32)

    ln_raw = _run_layernorm_64x192_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_192x192_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")

    ln_expected = _layernorm_reference(x, gamma, beta)
    expected = (W.astype(np.float64) @ ln_expected)

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
