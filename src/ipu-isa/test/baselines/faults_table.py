"""The F-1 table of ``docs/content/requirements/faults.md``, from ``ipu_isa.faults``.

The requirements page is hand-written, but its table of reserved encodings is
derived: adding an instruction changes an opcode count. ``f1_table`` renders
the table from :func:`reserved_encodings`; ``test_faults_docs`` checks the page
holds exactly that table, and ``//src/ipu-isa:update_isa_baselines``
(run by ``tools/update_baselines.sh``) rewrites it.
"""

from __future__ import annotations

from ipu_isa.faults import ReservedOperand, reserved_encodings
from ipu_isa.operand_types import OperandKind
from ipu_isa.view import Isa

FAULTS_PAGE = "docs/content/requirements/faults.md"
MARKER = "Reserved encodings for F-1, from the instruction reference:"

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


def count_word(count: int) -> str:
    """``count`` in words, as the page writes it (``seven``); digits past eight."""
    return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _users(view: Isa, slot, operand_type: str) -> list[str]:
    return [
        instruction.mnemonic
        for instruction in view.instructions(slot)
        if any(op.type.name == operand_type for op in instruction.operands)
    ]


def _operand_sentences(view: Isa, slot, operands: tuple[ReservedOperand, ...]) -> list[str]:
    sentences = []
    above: dict[int, list[str]] = {}  # enum threshold -> operand names, merged as one sentence
    for reserved in operands:
        operand_type = view.operand_types[reserved.operand_type]
        members = operand_type.domain.members
        if operand_type.kind is OperandKind.REGISTER:
            valid = [
                f"`{m.upper()}`" for code, m in enumerate(members) if code not in reserved.codes
            ]
            sentences.append(f"`{operand_type.name}` values other than {_join(valid)}")
        elif operand_type.kind is OperandKind.ENUM:
            names = above.setdefault(min(reserved.codes) - 1, [])
            names.extend(f"`{name}`" for name in reserved.operands)
        else:
            low, high = operand_type.domain.value_range or (0, 0)
            users = _join([f"`{m}`" for m in _users(view, slot, operand_type.name)])
            for name in reserved.operands:
                sentences.append(f"`{name}` outside {low} to {high} in {users}")
    for threshold, names in above.items():
        sentences.append(f"Values above {threshold} of {_join(names)}")
    return sentences


def f1_table(view: Isa) -> str:
    """The F-1 table: one row per hardware slot with a reserved encoding, in layout order."""
    lines = ["| Slot | Reserved encodings |", "|---|---|"]
    for slot, entry in reserved_encodings(view).items():
        if entry.empty:
            continue
        sentences = []
        if entry.opcodes:
            count = len(entry.opcodes)
            noun = "opcode" if count == 1 else "opcodes"
            sentences.append(
                f"{count_word(count).capitalize()} {noun} of the {entry.opcode_bits}-bit field"
            )
        sentences += _operand_sentences(view, slot, entry.operands)
        lines.append(f"| `{slot.value.upper()}` | {'. '.join(sentences)}. |")
    return "\n".join(lines) + "\n"


def table_span(page: str) -> tuple[int, int]:
    """Where the F-1 table sits in ``page``: [start, end) offsets of its lines."""
    start = page.index(MARKER)
    start = page.index("\n|", start) + 1
    end = page.index("\n\n", start) + 1
    return start, end


def with_table(page: str, view: Isa) -> str:
    """``page`` with its F-1 table replaced by ``f1_table(view)``."""
    start, end = table_span(page)
    return page[:start] + f1_table(view) + page[end:]
