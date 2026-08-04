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

# NOTE on the pipeline shape this kernel relies on (fixed 2026-08-04).
#
# MULT.RC.VE reads r_cyclic from the start-of-cycle SNAPSHOT (issue #157), so it
# cannot consume a chunk that LDR_CYCLIC_MULT_REG loads in its OWN bundle. The
# .asm therefore primes k=0's row one bundle ahead and biases chunk0's
# fixed_idx startup down by one (SUB lr5 lr5 cr1) to match. Without that bias
# the kernel pairs W[j,k] with D[k-1] -- the one-k lag this test used to xfail
# on. chunk1 must NOT re-prime: its first row is already in flight from
# chunk0's trailing prefetch.
#
def _write_fp32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.astype(np.float32).tobytes())


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
