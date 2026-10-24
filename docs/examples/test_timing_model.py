"""The examples of the timing model page (docs/content/explanation/timing-model.md).

Every ``asm`` block of the page is an example: this test sets up a state from its
``# given:`` comments, runs it to ``HALT``, and checks each ``# expect:`` comment,
so the value the page claims a slot sees is the value the emulator gives. Each
rule section of the page holds at least one example, and the page includes the
table that ``ipu_gen.docs_pages`` generates from the timing contract.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest
from emu_support import ElementFormat, run

from docs.doc_blocks import Block, fenced_blocks
from ipu_emu.ipu_state import IpuState

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "content" / "explanation" / "timing-model.md"

#: XMEM row and register widths, in elements. The examples run in the FP32
#: mode (ADR 0007): an element is its FP32 value, four bytes, with no metadata.
ROW = 128
_F32 = 4

_GIVEN_CR = re.compile(r"^CR(?P<n>\d+) = (?P<v>-?\d+)$")
_GIVEN_ROW = re.compile(r"^XMEM\[(?P<row>\d+)\] = (?P<v>-?\d+)$")
_EXPECT = re.compile(
    r"^(?P<reg>LR\d+|R[01]|MULT_RES|R_ACC|XMEM\[\d+\])"
    r"(?:\[(?P<index>\d+)\])? == (?P<v>-?\d+)$"
)


def _comments(block: Block, kind: str) -> list[str]:
    prefix = f"# {kind}:"
    return [
        line.strip()[len(prefix) :].strip()
        for line in block.body.splitlines()
        if line.strip().startswith(prefix)
    ]


def _observed(state: IpuState, register: str, index: int | None) -> float:
    regfile = state.regfile
    if register.startswith("LR"):
        return regfile.get_lr(int(register[2:]))
    if register in ("R0", "R1"):
        offset = (ROW * int(register[1]) + index) * _F32
        return struct.unpack_from("<f", regfile.view("r_wide_debug"), offset)[0]
    if register in ("MULT_RES", "R_ACC"):
        return struct.unpack_from("<f", regfile.view(register.lower()), _F32 * index)[0]
    row = int(register[len("XMEM[") : -1])
    data = state.xmem.read_address((row * ROW + index) * _F32, _F32)
    return struct.unpack("<f", bytes(data))[0]


def _run_example(block: Block) -> IpuState:
    cr: dict[int, int] = {}
    rows: dict[int, int] = {}
    for given in _comments(block, "given"):
        if m := _GIVEN_CR.match(given):
            cr[int(m["n"])] = int(m["v"]) & 0xFFFFFFFF
        elif m := _GIVEN_ROW.match(given):
            rows[int(m["row"])] = int(m["v"])
        else:
            pytest.fail(f"{block.path}:{block.line}: unknown given {given!r}")

    def setup(state: IpuState) -> None:
        for row, value in rows.items():
            state.xmem.write_address(row * ROW * _F32, struct.pack(f"<{ROW}f", *[value] * ROW))

    return run(block.body, cr=cr, setup=setup, fmt=ElementFormat.FP32)


def _examples() -> list[Block]:
    return fenced_blocks(PAGE, "asm")


@pytest.mark.parametrize("block", _examples(), ids=lambda b: f"line-{b.line}")
def test_each_example_shows_the_values_the_page_claims(block: Block):
    expects = _comments(block, "expect")
    assert expects, f"{PAGE.name}:{block.line}: an example needs an '# expect:' comment"
    state = _run_example(block)
    assert state.is_halted
    for expect in expects:
        m = _EXPECT.match(expect)
        assert m, f"{PAGE.name}:{block.line}: cannot read expect {expect!r}"
        index = int(m["index"]) if m["index"] is not None else None
        assert _observed(state, m["reg"], index) == int(m["v"]), (
            f"{PAGE.name}:{block.line}: {expect}"
        )


#: A rule heading: ``### ... { #rule-1 }`` or ``{ #slot-order }``.
_SECTION = re.compile(r"^#{2,3} .*\{ #(?P<anchor>[\w-]+) \}$", re.M)


@pytest.mark.parametrize("anchor", ["rule-1", "rule-2", "rule-3", "rule-4"])
def test_every_rule_has_a_tested_example(anchor):
    text = PAGE.read_text(encoding="utf-8")
    sections = {
        m["anchor"]: text[m.end() : nxt.start() if nxt else len(text)]
        for m, nxt in zip(
            list(_SECTION.finditer(text)),
            [*list(_SECTION.finditer(text))[1:], None],
            strict=True,
        )
    }
    assert "```asm" in sections[anchor], f"{anchor} has no example"


def test_the_page_includes_the_generated_table():
    assert '--8<-- "generated/timing-contract-table.md"' in PAGE.read_text(encoding="utf-8")
