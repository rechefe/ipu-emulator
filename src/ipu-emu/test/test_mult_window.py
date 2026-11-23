"""The MULT R_CYCLIC window decoded in float32 equals the float64 window.

``_r_cyclic_window32`` decodes only the one or two ring rows a window covers;
``_r_cyclic_window`` decodes the whole ring and indexes it. For every start
element of the ring, in NX8 (each row with its own metadata) and FP32, the
two must hold the same values.
"""

from __future__ import annotations

import numpy as np
import pytest
from emu_support import E4, ElementFormat, assemble_program, new_state

from ipu_emu.emulator import load_program
from ipu_emu.geometry import vector_registers
from ipu_emu.ipu import Ipu
from ipu_emu.slots import mult


@pytest.mark.parametrize("fmt", [ElementFormat.NX8, ElementFormat.FP32], ids=["nx8", "fp32"])
def test_every_window_start_decodes_the_same_values(fmt):
    rng = np.random.default_rng(5)
    state = new_state(fmt=fmt)
    load_program(state, assemble_program("MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ;;\nHALT ;;"))
    ring = vector_registers(state).r_cyclic
    size = len(state.regfile.view(ring))
    if fmt is ElementFormat.FP32:
        values = rng.standard_normal(size // 4).astype("<f4")
        values[[3, 200, 511]] = (np.nan, np.inf, -0.0)
        state.regfile.mutable(ring)[:] = values.tobytes()
    else:
        state.regfile.mutable(ring)[:] = bytes(rng.integers(0, 256, size, dtype=np.uint8))
        meta = bytes(rng.integers(0, 256, len(state.regfile.view("r_cyclic_meta")), dtype=np.uint8))
        state.regfile.mutable("r_cyclic_meta")[:] = meta
    ipu = Ipu(state)
    try:
        ipu.cycle._take_snapshot()
        schedule = ipu.cycle._fetch(0)[1]
        ipu.view.plan = next(op.plan for op in schedule.slot_ops if op.slot == "mult")
        nx8 = None if fmt is ElementFormat.FP32 else E4
        for start in range(512):
            narrow = mult._r_cyclic_window32(ipu, ipu.codec, start, nx8)
            wide = mult._r_cyclic_window(ipu, ipu.codec, start, nx8)
            assert narrow.dtype == np.float32
            assert narrow.astype(np.float64).tobytes() == wide.tobytes(), start
    finally:
        ipu.close()
