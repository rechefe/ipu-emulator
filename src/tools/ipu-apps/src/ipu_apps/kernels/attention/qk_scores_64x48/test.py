"""qk_scores_64x48: every case in cases.py, plus the stored-extent probe."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.cases import run_poked
from ipu_apps.kernels.attention.qk_scores_64x48.app import (
    ACC_STORE_ROWS, D, ELEM_BYTES, K_BASE, LANES, N, N_BLOCK, N_TG, ROW_BYTES, S_BASE,
)

test_case = case_tests(__package__)


def test_padding_lanes_not_stored(tmp_path):
    """valid_elements=N gates both MULT.RC.VE's mask and ACTIVATE.QUANTIZE's
    store window (no AGG here -- lanes are independent, so a garbage-padding
    probe on the VALID lanes proves nothing: garbage in lanes N:128 stays in
    lanes N:128 and never reaches lanes 0:N regardless of gating).

    What the gate actually controls is the STORED EXTENT: with it in place,
    ACTIVATE.QUANTIZE hard-zeros post_aaq_reg lanes N:128 before
    STR_POST_AAQ_REG writes the whole 512 B row, so those bytes are zero no
    matter what garbage sits in K's padding lanes. Stage non-zero garbage in
    K's lanes N:128 (a real producer in a chained pipeline would leave
    another stream's data there) and assert the RAW stored row -- read
    directly from XMEM, bypassing teardown's crop -- is zero past byte
    N*ELEM_BYTES.
    """
    garbage = bytearray(np.full(LANES - N, 1e3, dtype=np.float32).tobytes())

    def poke(state):
        # Overwrite the padding lanes (columns N..LANES-1) of every K row.
        for row in range(N_BLOCK * D):
            state.xmem.write_address(K_BASE + row * ROW_BYTES + N * ELEM_BYTES, garbage)

    state = run_poked("qk_scores_64x48", tmp_path, poke, seed=0x4C1)
    stride = ACC_STORE_ROWS * ROW_BYTES
    for r in range(N_BLOCK * N * N_TG):
        row = state.xmem.read_address(S_BASE + r * stride, ROW_BYTES)
        tail = np.frombuffer(bytes(row[N * ELEM_BYTES:]), dtype=np.float32)
        assert np.all(tail == 0.0), (
            f"row {r}: padding lanes N:{LANES} were stored non-zero -- "
            f"valid_elements is not gating the ACTIVATE.QUANTIZE store window"
        )
