"""attn_scores_km_16x60: every case in cases.py, plus the stored-extent probe."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.attn_scores_km_16x60.app import (
    D, ELEM_BYTES, LANES, N_TG, N_TOK, OUTPUT_ROW_BYTES, Q_CHAN_ROWS, QBASE, ROW_BYTES, SBASE,
)
from ipu_apps.kernels.attention.cases import run_poked

test_case = case_tests(__package__)


def test_padding_lanes_not_stored(tmp_path):
    """valid_elements=N_TOK gates both MULT.RC.VE's mask (over Q's channel
    columns, the R_CYCLIC operand) and ACTIVATE.QUANTIZE's store window. No
    AGG here -- lanes are independent, so asserting on the VALID lanes (as
    attn_v_64x48's padding probe does) proves nothing: garbage in lanes
    N_TOK:128 stays in lanes N_TOK:128 and never reaches lanes 0:N_TOK
    regardless of gating.

    What the gate actually controls is the STORED EXTENT. Stage non-zero
    garbage in Q's padding lanes (columns N_TOK..LANES-1 of every channel row
    -- a real producer in a chained pipeline would leave another stream's
    data there, not zeros) and assert every stored row is zero past byte
    N_TOK*ELEM_BYTES.
    """
    garbage = bytearray(np.full(LANES - N_TOK, 1e3, dtype=np.float32).tobytes())

    def poke(state):
        for c in range(D):
            state.xmem.write_address(
                QBASE + c * Q_CHAN_ROWS * ROW_BYTES + N_TOK * ELEM_BYTES, garbage)

    state = run_poked("attn_scores_km_16x60", tmp_path, poke, seed=0xD61, head=1)
    for r in range(N_TOK * N_TG):
        row = state.xmem.read_address(SBASE + r * OUTPUT_ROW_BYTES, ROW_BYTES)
        tail = np.frombuffer(bytes(row[N_TOK * ELEM_BYTES:]), dtype=np.float32)
        assert np.all(tail == 0.0), (
            f"row {r}: padding lanes {N_TOK}:{LANES} were stored non-zero -- "
            f"valid_elements is not gating the ACTIVATE.QUANTIZE store window"
        )
