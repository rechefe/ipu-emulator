"""qk_scores_16x60: every case in cases.py, plus the stored-extent probe."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.cases import run_poked
from ipu_apps.kernels.attention.qk_scores_16x60.app import (
    D, ELEM_BYTES, K_BASE, K_STRIDE_ROWS, LANES, N, N_TG, OUTPUT_ROW_BYTES, ROW_BYTES, S_BASE,
)

test_case = case_tests(__package__)


def test_padding_lanes_not_stored(tmp_path):
    """valid_elements=N gates both MULT.RC.VE's mask and ACTIVATE.QUANTIZE's
    store window. There is no AGG in this kernel -- lanes are independent, so
    asserting on the VALID lanes (as attn_v_64x48's padding probe does) proves
    nothing here: garbage in lanes N:128 stays in lanes N:128 and never
    reaches lanes 0:N regardless of gating.

    What the gate actually controls is the STORED EXTENT. Stage non-zero
    garbage in K's padding lanes (columns N..LANES-1 of every channel row --
    a real producer in a chained pipeline would leave another stream's data
    there, not zeros) and assert every stored row is zero past byte
    N*ELEM_BYTES.
    """
    garbage = bytearray(np.full(LANES - N, 1e3, dtype=np.float32).tobytes())

    def poke(state):
        for c in range(D):
            state.xmem.write_address(K_BASE + c * K_STRIDE_ROWS * ROW_BYTES + N * ELEM_BYTES, garbage)

    state = run_poked("qk_scores_16x60", tmp_path, poke, seed=0x5C1)
    for r in range(N * N_TG):
        row = state.xmem.read_address(S_BASE + r * OUTPUT_ROW_BYTES, ROW_BYTES)
        tail = np.frombuffer(bytes(row[N * ELEM_BYTES:]), dtype=np.float32)
        assert np.all(tail == 0.0), (
            f"row {r}: padding lanes {N}:{LANES} were stored non-zero -- "
            f"valid_elements is not gating the ACTIVATE.QUANTIZE store window"
        )
