"""The dummy instruction of the "one edit" test (E03-24).

Adding an instruction takes one entry in ``instruction_spec.py``, one execute
handler and one test (CLAUDE.md, "Adding an Instruction"). ``TESTNEG`` is that
entry and that handler, written as text so the end-to-end test can insert them
into copies of the real sources, and as data for the in-process tests.
"""

from __future__ import annotations

from typing import Any

from ipu_isa.instruction_spec import InstructionDoc
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS
from ipu_isa.view import Isa, authoring_copies, load_isa

MNEMONIC = "TESTNEG"
SLOT = "lr"

#: The spec entry, as it is written into instruction_spec.py's "lr" dict.
ENTRY_SOURCE = """\
        "TESTNEG": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "src", "type": "LrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Test negate",
                summary="Negate a local register (a test instruction).",
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src: Source local register (LR0–LR15)",
                ],
                operation="dest = -src",
                example="TESTNEG LR1 LR0;;",
            ),
            "execute_fn": "execute_testneg",
        },
"""

#: The handler, as it is written into the LR slot's handler module.
HANDLER_SOURCE = '''\
def execute_testneg(ipu: Ipu, *, dest: int, src: int) -> None:
    """Execute TESTNEG: uint32 ``dest = -src``."""
    ipu.state.regfile.set_lr(dest, (-src) & 0xFFFFFFFF)


'''


def entry() -> dict[str, Any]:
    """The spec entry as data (``ENTRY_SOURCE`` evaluated)."""
    source = "{" + ENTRY_SOURCE.strip().rstrip(",") + "}"
    return eval(source, {"InstructionDoc": InstructionDoc})[MNEMONIC]  # our own constant literal


def view_with_dummy() -> Isa:
    """The repository ISA with ``TESTNEG`` appended to the LR slot."""
    spec, pseudo = authoring_copies()
    spec[SLOT][MNEMONIC] = entry()
    return load_isa(spec, pseudo, SLOTS, OPERAND_TYPES)
