"""The ISA view as a JSON snapshot, ``test/fixtures/spec_snapshot.json`` (E03-09).

``test_view`` checks the view reproduces the snapshot; adding or changing an
instruction changes it, and ``//src/tools/ipu-isa:update_isa_baselines`` (run
by ``tools/update_baselines.sh``) rewrites it.
"""

from __future__ import annotations

import json

from ipu_isa.view import Isa

SNAPSHOT_PATH = "src/tools/ipu-isa/test/fixtures/spec_snapshot.json"


def _operands(operands) -> list[dict]:
    return [
        {"name": op.name, "type": op.type.name, "read": op.read.value if op.read else None}
        for op in operands
    ]


def records(view: Isa, slots=None) -> dict:
    """Every instruction (slots in ``slots``, default spec order) and pseudo-instruction."""
    order = view.spec_slot_order if slots is None else slots
    return {
        "instructions": [
            {
                "slot": i.slot.value,
                "mnemonic": i.mnemonic,
                "opcode": i.opcode,
                "operands": _operands(i.operands),
                "tag": i.tag.value,
                "execute_fn": i.execute_fn,
            }
            for slot in order
            for i in view.instructions(slot)
        ],
        "pseudo_instructions": [
            {
                "mnemonic": p.mnemonic,
                "operands": _operands(p.operands),
                "expands_to": {"slot": p.expands_to.slot.value, "mnemonic": p.expands_to.mnemonic},
                "args": list(p.args),
                "tag": p.tag.value,
            }
            for p in view.pseudos()
        ],
    }


def render(view: Isa) -> str:
    """The snapshot file's text for ``view``."""
    return json.dumps(records(view), indent=1) + "\n"
