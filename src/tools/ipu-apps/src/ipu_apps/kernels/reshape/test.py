"""Reshape family tests that span kernels: fold(unfold(x)) == x.

Each fold kernel's own cases check it against hand-built streams; the
strongest test is the round trip. Build a random spatial tensor, run it
through the REAL unfold kernel, feed unfold's output into the matching fold,
and assert we get the original tensor back (within FP32 tolerance; the x1.0
pass-through multiply is exact).

The poisoned variant starts fold with r_acc full of a non-zero sentinel.
ACC.RESHAPE (unlike ACC.STRIDE) only updates the R_ACC indexes it is told to
write and leaves the rest alone, so each fold kernel relies on its
ACC.RESHAPE calls partitioning every valid r_acc lane exactly once with no
gaps -- if that claim were wrong, stale sentinel bytes would leak into the
output here.

unfold_8x8x240's input goes through its ``pack_input_rows`` contract, and
fold_8x8x240 consumes unfold's RAW uncropped rows (the ``.rows.bin`` sibling
of its output), not the cropped ``[N_STREAMS, N_OUT, N_TOK]`` convenience
array. Unlike L3/L4, fold_8x8x240's OUTPUT is naive row-major ``[H, W]``
(lane = row*8 + col), not a mirror of unfold's permuted input packing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_apps.kernel_registry import create_harness, kernel_spec
from ipu_apps.kernel_registry.cases import assemble_kernel
from ipu_apps.kernels.reshape.unfold_cases import (
    MAX_CYCLES, check_striped, spatial_tensor, stripe_input,
)
from ipu_apps.kernels.reshape.unfold_8x8x240.app import pack_input_rows

# (unfold kernel, fold kernel, (H, W, C), N_STRIPES of the striped layout,
#  seed for the plain round trip, seed for the poisoned one).
LAYERS = {
    "l3": ("unfold_32x32x144", "fold_32x32x144", (32, 32, 144), 8, 0x032, 0x033),
    "l4": ("unfold_16x16x192", "fold_16x16x192", (16, 16, 192), 2, 0x016, 0x017),
    "l5": ("unfold_8x8x240", "fold_8x8x240", (8, 8, 240), 1, 0x058, 0x059),
}


def _poison(app):
    """Wrap ``app.setup`` so r_acc starts as 0xAA bytes instead of zeros."""
    setup = app.setup

    def poisoned(state):
        setup(state)
        state.regfile.set_r_acc_bytes(bytearray(b"\xAA" * 512))

    app.setup = poisoned


def _run(kernel, shape, workspace, input_path, *, poisoned=False):
    inst = assemble_kernel(kernel, workspace)
    output_path = workspace / f"{kernel}.out.bin"
    app = create_harness(kernel, params={"shape": shape},
                         bindings={"inst_path": inst, "input_path": input_path,
                                   "output_path": output_path})
    if poisoned:
        _poison(app)
    state, _ = app.run(max_cycles=MAX_CYCLES)
    assert state.is_halted, f"{kernel} did not halt within {MAX_CYCLES} cycles"
    return output_path


@pytest.mark.parametrize("poisoned", [False, True], ids=["plain", "poisoned_r_acc"])
@pytest.mark.parametrize("layer", list(LAYERS))
def test_fold_inverts_unfold(layer, poisoned, tmp_path):
    unfold, fold, (h, w, c), n_stripes, seed, poisoned_seed = LAYERS[layer]
    x = spatial_tensor(poisoned_seed if poisoned else seed, c, h, w)

    unfold_input = tmp_path / "unfold_input.bin"
    packed = pack_input_rows(x) if unfold == "unfold_8x8x240" else stripe_input(x, n_stripes)
    unfold_input.write_bytes(packed.tobytes())
    streams = _run(unfold, (h, w, c), tmp_path, unfold_input)
    if unfold == "unfold_8x8x240":
        # fold_8x8x240 reads the whole rows unfold stored, not the crop.
        streams = streams.with_suffix(".rows.bin")

    folded = _run(fold, (h, w, c), tmp_path, streams, poisoned=poisoned)
    check_striped(folded, x, n_stripes)


def test_fold_and_unfold_share_their_shapes():
    """Every fold kernel inverts exactly one unfold kernel's (H, W, C)."""
    for unfold, fold, shape, *_ in LAYERS.values():
        assert kernel_spec(unfold).check(shape=shape).ok
        assert kernel_spec(fold).check(shape=shape).ok
