"""Wide-vector FP32 end-to-end test for matmul_240x240_x128.

Runs the REAL kernel binary against a numpy reference, in wide-vector debug
mode. No checked-in golden is involved: FP32 inputs are generated here and the
expected result is computed directly.

N_TOK=16 < LANES, so each output channel occupies a whole 512 B XMEM row of
which only the first N_TOK*4 bytes are valid; the harness teardown crops them,
leaving a densely packed N_OUT x N_TOK output file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_240x240_x128 import MatMul240x240x128App, K, N_OUT, N_TOK

_INST_BIN = Path(os.environ["MATMUL_240X240_X128_INST_BIN"])

# PRE-EXISTING .asm DEFECT (not introduced by the wide conversion).
#
# In this kernel's .asm the LDR_CYCLIC_MULT_REG and the MULT.RC.VE that
# consumes it share ONE VLIW bundle:
#
#     LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
#     MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD; BLT lr5 lr6 k_chunk0;;
#
# MULT reads r_cyclic from the start-of-cycle SNAPSHOT (issue #157), so it
# multiplies the chunk loaded the PREVIOUS cycle: the kernel pairs W[j,k] with
# D[k-1] and drops one k off each end. matmul_144x144_x128's .asm carries the
# fix (prime the first chunk, then bias k_index down by one with a SUB); these
# single-token-group kernels never received it.
#
# Confirmed pre-existing: the pre-conversion NARROW harness at the base commit,
# run against this same .asm, also fails its own checked-in INT8 golden with
# exactly this one-k lag. The golden itself is a true reference (it matches an
# exact int32 numpy contraction), so the .asm regressed, not the golden.
#
# Fixing it means editing .asm, which is out of scope for this conversion. The
# test is written against the CORRECT reference and marked xfail so it starts
# passing the moment the .asm is fixed.
_ASM_LAG_BUG = pytest.mark.xfail(
    reason="pre-existing .asm bug: MULT consumes the same-bundle r_cyclic load, "
           "so the contraction lags one k (also fails in narrow mode)",
    strict=True,
)


def _write_fp32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.astype(np.float32).tobytes())


@_ASM_LAG_BUG
def test_matmul_240x240_x128_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x240240)

    # D is channel-major [K, N_TOK]; W is output-major [N_OUT, K].
    D = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    data_path = tmp_path / "input_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    _write_fp32(data_path, D)
    _write_fp32(weights_path, W)

    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Paths are passed as plain strings: the harness must coerce them itself
    # (a converted __init__ that lost its Path(...) coercion fails here).
    app = MatMul240x240x128App(
        inst_path=_INST_BIN,
        input_path=str(data_path),
        weights_path=str(weights_path),
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = W @ D                       # C[j, t] = sum_k W[j,k] * D[k,t]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_OUT * N_TOK, (
        f"output has {raw.size} floats, expected {N_OUT * N_TOK}"
    )
    got = raw.reshape(N_OUT, N_TOK)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_wide_memory_map_is_exercised_end_to_end(tmp_path: Path) -> None:
    """Verify the wide memory map by checking the lagged result the .asm does compute.

    This is the real check on the conversion. It asserts against
    ``W @ roll(D, 1)`` -- the one-k-lagged contraction the .asm's bundling
    actually produces (see _ASM_LAG_BUG). Matching that to FP32 tolerance means
    D, W and C were all staged and addressed correctly at 4 bytes/element:
    every weight chunk and every data row landed where the kernel reads it, and
    the output rows were cropped correctly. Only the pipeline phase is wrong.

    When the .asm is fixed this test SHOULD start failing while the xfail test
    above starts passing -- that pairing is the signal to delete this one.
    """
    rng = np.random.RandomState(0x240240)

    D = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    data_path = tmp_path / "input_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    _write_fp32(data_path, D)
    _write_fp32(weights_path, W)

    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = MatMul240x240x128App(
        inst_path=_INST_BIN,
        input_path=str(data_path),
        weights_path=str(weights_path),
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    # Pairing derived empirically with one-hot probes: W[:, k] multiplies
    # D[k-1] uniformly (no anomaly at the chunk seam), W[:, 0] contributes
    # nothing, and D[K-1] is never consumed.
    lagged = W[:, 1:] @ D[:-1]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_OUT * N_TOK
    got = raw.reshape(N_OUT, N_TOK)

    np.testing.assert_allclose(got, lagged, rtol=1e-4, atol=1e-3)
