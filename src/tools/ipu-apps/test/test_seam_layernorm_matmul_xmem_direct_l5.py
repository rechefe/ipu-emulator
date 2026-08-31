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

IMPORTANT: matmul_240x240_x128 crops its OUTPUT to N_TOK*4 bytes/row in
teardown() (a separately-known, out-of-scope issue) -- this test reads the
OUTPUT region back via state.xmem.read_address directly at full ROW_BYTES
width, never through teardown()/an output file, exactly like the L4
reference test does, so that crop never enters this test.

Method: run the REAL layernorm_16x240 kernel on a fresh state, capture its
raw OUTPUT_BASE bytes directly via state.xmem.read_address. On a SEPARATE
fresh state, poison the matmul's DATA region, write the captured LayerNorm
bytes into it verbatim, skip _load_data entirely, load weights normally, run
the matmul, and compare against an independent reference computed from
gamma/beta/x directly (LayerNorm formula) times W (matmul formula) -- the
reference never touches either kernel's own golden/internals.

Mutation-first: the control test corrupts one channel's row in the captured
LayerNorm output before the direct handoff and confirms the result diverges
from the reference, before trusting the "clean" test's PASS.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm.layernorm_16x240 import (
    LayerNorm16x240App, N_CH as LN_N_CH, N_TOK as LN_N_TOK,
    ROW_BYTES as LN_ROW_BYTES, OUTPUT_BASE as LN_OUTPUT_BASE,
)
from ipu_apps.matmuls.matmul_240x240_x128 import (
    MatMul240x240x128App, K as MM_K, N_OUT as MM_N_OUT, N_TOK as MM_N_TOK,
    LANES as MM_LANES, ROW_BYTES as MM_ROW_BYTES,
    DATA_BASE as MM_DATA_BASE, OUTPUT_BASE as MM_OUTPUT_BASE,
    DATA_STRIDE_ROWS as MM_DATA_STRIDE_ROWS, OUTPUT_STRIDE_ROWS as MM_OUTPUT_STRIDE_ROWS,
    W_STRIDE_ROWS as MM_W_STRIDE_ROWS, WEIGHTS_BASE_ROW as MM_WEIGHTS_BASE_ROW,
    OUTPUT_BASE_ROW as MM_OUTPUT_BASE_ROW,
)

_LN_INST_BIN = Path(os.environ["LAYERNORM_16X240_INST_BIN"])
_MM_INST_BIN = Path(os.environ["MATMUL_240X240_X128_INST_BIN"])

_POISON = 1e3

assert LN_N_CH == MM_K == 240
assert LN_N_TOK == MM_N_TOK == 16


def _run_layernorm_16x240_capture_xmem(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, tmp_path: Path, tag: str,
) -> bytes:
    """Run the real layernorm_16x240 kernel; return its raw XMEM output bytes
    (N_CH * ROW_BYTES), read directly via state.xmem -- no file round-trip.

    layernorm_16x240.setup() writes its input_path's bytes to XMEM VERBATIM
    (no row-expansion/padding logic of its own), so the input FILE itself
    must already be pre-padded to N_CH * ROW_BYTES (one zero-padded 128-lane
    row per channel), exactly as test_layernorm_16x240_wide.py does. x here
    is [N_CH, N_TOK]; it is padded out to [N_CH, LANES] before writing.
    """
    x_padded = np.zeros((LN_N_CH, LN_ROW_BYTES // 4), dtype=np.float32)
    x_padded[:, :LN_N_TOK] = x
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
        bytearray(np.full(LN_N_CH * (LN_ROW_BYTES // 4), _POISON, dtype=np.float32).tobytes()),
    )
    app = LayerNorm16x240App(
        inst_path=_LN_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(LN_OUTPUT_BASE, LN_N_CH * LN_ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(LN_N_CH, LN_ROW_BYTES // 4)
    assert not np.all(rows == _POISON, axis=1).any(), "layernorm left poisoned rows untouched"
    return raw


def _run_matmul_240x240_direct_xmem_handoff(
    ln_raw: bytes, W: np.ndarray, tmp_path: Path, tag: str,
) -> np.ndarray:
    """Feed ln_raw straight into matmul_240x240_x128's DATA region via
    state.xmem.write_address -- _load_data() is NEVER called. Only weights go
    through the normal file-staged _load_weights path. Returns the cropped
    [N_OUT, N_TOK] result, read directly from XMEM at full ROW_BYTES width
    (teardown()'s output-file crop is never exercised here).
    """
    assert len(ln_raw) == LN_N_CH * LN_ROW_BYTES == MM_K * MM_ROW_BYTES, (
        "byte-length mismatch between LayerNorm's raw output region and "
        "matmul's raw DATA region -- direct handoff is not even byte-shape "
        "compatible"
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
    state.xmem.write_address(MM_DATA_BASE, bytearray(ln_raw))

    app = MatMul240x240x128App(
        inst_path=_MM_INST_BIN,
        input_path="/dev/null",   # placeholder; _load_data is bypassed below
        weights_path=weights_path,
        output_path=None,          # avoid exercising the cropping teardown()
    )
    from ipu_apps.matmuls.matmul_240x240_x128 import _load_weights as mm_load_weights

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


def test_layernorm_16x240_output_is_not_byte_compatible_with_matmul_240x240_input_when_corrupted(
    tmp_path: Path,
) -> None:
    """Mutation-first control: prove the direct-handoff test actually detects
    a mismatch before trusting the "they agree" result below. Corrupt one
    channel's row in the captured LayerNorm output before handing it to the
    matmul and confirm the result diverges from the reference.
    """
    rng = np.random.RandomState(0x5EA5)
    x = rng.uniform(-2.0, 2.0, size=(LN_N_CH, LN_N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    ln_raw = bytearray(
        _run_layernorm_16x240_capture_xmem(x, gamma, beta, tmp_path, tag="mut")
    )
    corrupt_row = np.full(MM_LANES, 999.0, dtype=np.float32).tobytes()
    ln_raw[7 * LN_ROW_BYTES : 8 * LN_ROW_BYTES] = corrupt_row

    got = _run_matmul_240x240_direct_xmem_handoff(bytes(ln_raw), W, tmp_path, tag="mut")

    ln_expected = _layernorm_reference(x, gamma, beta)
    expected = (W.astype(np.float64) @ ln_expected)

    max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
    assert max_err > 1.0, (
        f"corrupted-row control did not diverge (max_err={max_err:.3e}) -- "
        "the direct-XMEM-handoff test is not actually sensitive to the seam"
    )


def test_layernorm_16x240_feeds_matmul_240x240_via_direct_xmem_no_file_staging_l5(
    tmp_path: Path,
) -> None:
    """The real question: with the corruption removed, does the verbatim
    byte handoff (no _load_data, no file round-trip) produce the correct
    matmul result end to end?
    """
    rng = np.random.RandomState(0x5EA6)
    x = rng.uniform(-2.0, 2.0, size=(LN_N_CH, LN_N_TOK)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=LN_N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=LN_N_CH).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(MM_N_OUT, MM_K)).astype(np.float32)

    ln_raw = _run_layernorm_16x240_capture_xmem(x, gamma, beta, tmp_path, tag="clean")
    got = _run_matmul_240x240_direct_xmem_handoff(ln_raw, W, tmp_path, tag="clean")

    ln_expected = _layernorm_reference(x, gamma, beta)
    expected = (W.astype(np.float64) @ ln_expected)

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
