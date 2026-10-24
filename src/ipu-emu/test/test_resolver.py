"""The operand resolver seam: one interface, two adapters (E04-10).

The faithful and packed resolvers must return equal operand dicts, and raise
the same error, for every plan and encoding against the same register files.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from ipu_emu.cycle import resolver_for
from ipu_emu.errors import RegisterKindError
from ipu_emu.ipu import plans_for
from ipu_emu.ipu_state import IpuState
from ipu_emu.regfile import RegFile
from ipu_emu.resolver import (
    READ_LR,
    READ_MULT_STAGE,
    FaithfulResolver,
    PackedResolver,
)
from ipu_isa.formats import ElementFormat
from ipu_isa.timing import SLOT_EXECUTION_ORDER

ENCODINGS_PER_PLAN = 200


def _plans_with_reads():
    plans = {}
    for slot in SLOT_EXECUTION_ORDER:
        for _field, table in plans_for(slot):
            for plan in table:
                if plan.reads:
                    plans[(str(slot), plan.inst_name)] = plan
    return sorted(plans.items())


PLANS = _plans_with_reads()


def _randomise(regfile: RegFile, rng: random.Random) -> None:
    for name in regfile.register_names:
        buf = regfile.mutable(name)
        buf[:] = rng.randbytes(len(buf))


def _encoding(plan, rng: random.Random) -> dict[str, int]:
    """Random raw fields. One read operand in ten ranges over one bit more than
    its width, so reserved and out-of-bank encodings are drawn too."""
    widths = {read.name: read.op_type.bits or 8 for read in plan.reads}
    return {
        name: rng.randrange(1 << (widths.get(name, 8) + (rng.random() < 0.1)))
        for name, _key in plan.fields
    }


def _outcome(resolver, plan, encoded, snapshot):
    try:
        result = resolver.resolve(plan, encoded, snapshot)
    except Exception as exc:  # noqa: BLE001 - the error itself is compared
        return ("raised", type(exc), str(exc))
    return ("ok", {name: (type(value), value) for name, value in result.items()})


def test_a_plan_reads_a_mult_stage_register():
    assert any(read.code == READ_MULT_STAGE for _key, plan in PLANS for read in plan.reads)


@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
@pytest.mark.parametrize(("key", "plan"), PLANS, ids=[f"{s}.{n}" for (s, n), _ in PLANS])
def test_both_resolvers_return_equal_operands(key, plan, wide):
    rng = random.Random(f"{key}-{wide}")
    state = IpuState(fmt=ElementFormat.FP32 if wide else ElementFormat.NX8)
    _randomise(state.regfile, rng)
    snapshot = state.regfile.snapshot()
    _randomise(snapshot, rng)  # cycle-start and live values differ
    faithful, packed = FaithfulResolver(state), PackedResolver(state)
    resolved = 0
    for _ in range(ENCODINGS_PER_PLAN):
        encoded = _encoding(plan, rng)
        before = dict(encoded)
        expected = _outcome(faithful, plan, encoded, snapshot)
        assert _outcome(packed, plan, encoded, snapshot) == expected, encoded
        assert encoded == before  # neither adapter mutates the encoding
        resolved += expected[0] == "ok"
    assert resolved > ENCODINGS_PER_PLAN // 4  # most draws are valid encodings


def _lr_plan():
    return next(plan for (slot, _name), plan in PLANS if any(r.code == READ_LR for r in plan.reads))


def test_packed_resolver_sees_later_writes_and_a_replaced_register_file():
    plan = _lr_plan()
    read = next(r for r in plan.reads if r.code == READ_LR)
    encoded = {name: 0 for name, _key in plan.fields} | {read.name: 5}
    state = IpuState()
    snapshot = state.regfile.snapshot()
    packed = PackedResolver(state)
    source = (lambda: snapshot) if read.from_snapshot else (lambda: state.regfile)
    for value in (7, 9):
        source().set_lr(5, value)
        assert packed.resolve(plan, encoded, snapshot)[read.name] == value
    state.regfile = RegFile()
    if read.from_snapshot:
        snapshot = state.regfile.snapshot()
    source().set_lr(5, 11)
    assert packed.resolve(plan, encoded, snapshot)[read.name] == 11


def test_resolver_for_picks_packed_only_when_prototyping():
    regfile = RegFile()
    assert type(resolver_for(SimpleNamespace(prototyping=True, regfile=regfile))) is PackedResolver
    assert (
        type(resolver_for(SimpleNamespace(prototyping=False, regfile=regfile))) is FaithfulResolver
    )


def test_packed_bank_is_read_only_uint32_words():
    regfile = RegFile()
    regfile.set_lr(3, 0x12345678)
    bank = regfile.packed_bank("lr")
    assert bank.readonly
    assert bytes(bank[12:16]) == (0x12345678).to_bytes(4, "little")
    with pytest.raises(TypeError):
        bank[0] = 1


def test_packed_bank_rejects_a_vector_register():
    with pytest.raises(RegisterKindError):
        RegFile().packed_bank("r_acc")
