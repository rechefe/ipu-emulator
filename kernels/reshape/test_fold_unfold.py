"""The reshape tests beyond each kernel's cases (E07-05): fold(unfold(x)) == x, and
unfold_8x8x240's row contract.

Each kernel's own cases check it against the shared reference; the strongest
test is the round trip. A seeded ``[c, h, w]`` tensor runs through the real
unfold, whose output region is the matching fold's input on the same model
(``ipu_harness.chain``), and fold must give the tensor back exactly (the x1.0
pass-through multiply is exact).

The poisoned variant starts fold with R_ACC full of a non-zero sentinel.
ACC.RESHAPE (unlike ACC.STRIDE) writes only the R_ACC lanes it is told to
and leaves the rest, so each fold relies on its ACC.RESHAPE calls covering
every valid R_ACC lane exactly once: if not, sentinel bytes would leak into
the output here.

unfold_8x8x240's input goes through its ``paired_rows`` pack transform, and
fold_8x8x240's output is the plain one-stripe layout, not unfold's paired
rows; both unpack to the ``[c, h, w]`` tensor, which is what is compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

#: Each layer's (unfold kernel, fold kernel, seed of the plain round trip,
#: seed of the poisoned one): the old family test's.
LAYERS = {
    "l3": ("unfold_32x32x144", "fold_32x32x144", 0x032, 0x033),
    "l4": ("unfold_16x16x192", "fold_16x16x192", 0x016, 0x017),
    "l5": ("unfold_8x8x240", "fold_8x8x240", 0x058, 0x059),
}

#: The sentinel R_ACC holds when a poisoned fold starts.
SENTINEL = b"\xaa" * REGISTER_DEFINITIONS["r_acc"]["size_bytes"]


def _shape(kernel: str) -> tuple[int, int, int]:
    params = default_registry().kernel(kernel).params
    return params["c"].default, params["h"].default, params["w"].default


@pytest.mark.parametrize("poisoned", [False, True], ids=["plain", "poisoned_r_acc"])
@pytest.mark.parametrize("layer", list(LAYERS))
def test_fold_inverts_unfold(layer, poisoned):
    unfold, fold, seed, poisoned_seed = LAYERS[layer]
    c, h, w = _shape(unfold)
    rng = np.random.RandomState(poisoned_seed if poisoned else seed)
    x = rng.uniform(-1.0, 1.0, size=(c, h, w)).astype(np.float32)

    chain = Chain(ElementFormat.FP32)
    phases = chain.run(unfold, {}, {"x": x}).resident["y"]
    if poisoned:
        chain.model.write_register("r_acc", SENTINEL)
    step = chain.run(fold, {}, {"x": phases})
    assert step.result.halted and step.result.fault is None, step.result.describe()
    np.testing.assert_array_equal(step.result.outputs["y"], x.astype(np.float64))


def test_fold_and_unfold_share_their_shapes():
    """Every fold kernel inverts exactly one unfold kernel's (c, h, w)."""
    for unfold, fold, *_ in LAYERS.values():
        assert _shape(unfold) == _shape(fold)
        verdict = default_registry().resolve("fold", **dict(zip("chw", _shape(fold), strict=True)))
        assert verdict.kernel_name == fold, verdict.reason


def test_unfold_8x8x240_stores_one_channel_per_row_and_a_zero_tail():
    """The row contract of unfold_8x8x240's output, read from the raw rows: one
    channel per whole row, its 16 tokens in lanes 0..15, and the rest zero.

    Lanes 16..31 take ACC.STRIDE's decimation of x's zero padding (lanes
    64..127 of each source row) and lanes 32..127 of R_ACC are never written
    by ACC.STRIDE, so the kernel zeroes R_ACC itself once: the tail is exactly
    0.0 on a fresh model too.
    """
    c, h, w = _shape("unfold_8x8x240")
    x = np.random.RandomState(0x058).uniform(-1.0, 1.0, size=(c, h, w)).astype(np.float32)
    step = Chain(ElementFormat.FP32).run("unfold_8x8x240", {}, {"x": x})
    rows = np.frombuffer(step.result.raw_outputs["y"], dtype="<f4").reshape(4, c, 128)
    for s in range(4):
        i, j = divmod(s, 2)
        np.testing.assert_array_equal(rows[s, :, :16], x[:, i::2, j::2].reshape(c, 16))
        np.testing.assert_array_equal(rows[s, :, 16:], 0.0)


#: Per unfold kernel: the first output lane ACC.STRIDE never writes, and
#: unfold_32x32x144 (None) as the control that writes all 128.
UNFOLD_TAILS = {
    "unfold_8x8x240": 32,
    "unfold_16x16x192": 64,
    "unfold_32x32x144": None,
}


@pytest.mark.parametrize("kernel", list(UNFOLD_TAILS))
def test_unfold_zeroes_lanes_acc_stride_never_writes(kernel):
    """A kernel starts with whatever R_ACC the previous one left, so the lanes
    ACC.STRIDE never writes must come out 0.0 from the kernel itself, not from
    a clean register file.

    R_ACC is filled with a finite non-zero sentinel (0x55 bytes, about 1.4e13
    as FP32) before the run. unfold_32x32x144 writes every lane, so its output
    must not depend on R_ACC at all.
    """
    c, h, w = _shape(kernel)
    x = np.random.RandomState(0x5A).uniform(-1.0, 1.0, size=(c, h, w)).astype(np.float32)
    chain = Chain(ElementFormat.FP32)
    chain.model.write_register("r_acc", b"\x55" * REGISTER_DEFINITIONS["r_acc"]["size_bytes"])
    step = chain.run(kernel, {}, {"x": x})
    assert step.result.halted and step.result.fault is None, step.result.describe()
    rows = np.frombuffer(step.result.raw_outputs["y"], dtype="<f4").reshape(4, -1, 128)
    tail = UNFOLD_TAILS[kernel]
    if tail is not None:
        np.testing.assert_array_equal(rows[:, :, tail:], 0.0)
    clean = Chain(ElementFormat.FP32).run(kernel, {}, {"x": x})
    assert rows.tobytes() == clean.result.raw_outputs["y"], "a dirty R_ACC changed the output"
