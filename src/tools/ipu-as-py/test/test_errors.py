"""Every assembler input error is a located AssemblerError subclass (E05-04)."""

from __future__ import annotations

import lark
import pytest
from ipu_as.errors import (
    AssemblerError,
    EncodeError,
    LabelError,
    OperandError,
    ParseError,
    SlotError,
    SourceLocation,
    TemplateError,
)
from ipu_as.session import AssemblySession, SymbolTable
from ipu_isa.errors import IpuError


def _encode(source: str) -> list[int]:
    """Assemble through the session, which raises instead of printing and exiting."""
    return list(AssemblySession(source).assemble().words)


CASES = [
    # (id, source, error class, location (line, column), token, message start)
    (
        "number-out-of-range",
        "INC LR0 99999;;",
        OperandError,
        (1, 9),
        "99999",
        "Invalid token value - 99999 in token LrIncDecImmediate\nIn Line 1, Column 9",
    ),
    (
        "enum-not-an-option",
        "ACTIVATE.QUANTIZE nosuch CR15;;",
        OperandError,
        (1, 19),
        "nosuch",
        "Invalid token value - nosuch in token ActivationFnField\nIn Line 1, Column 19",
    ),
    (
        "label-not-defined",
        "B nowhere;;",
        LabelError,
        (1, 3),
        "nowhere",
        "Invalid token value - nowhere",
    ),
    (
        "relative-label-out-of-range",
        "B +5000;;",
        OperandError,
        (1, 3),
        "+5000",
        "Invalid token value - +5000",
    ),
    (
        "label-defined-twice",
        "a:\nNOP;;\na:\nNOP;;",
        LabelError,
        (3, 1),
        "a",
        "Label 'a' is defined for the second time at Line 3, Column 1.",
    ),
    (
        "nop-all-slots-filled",
        ";".join(["NOP"] * 20) + ";;",
        SlotError,
        (1, 45),
        "NOP",
        "NOP: all slots are already filled\nAt: Line 1, Column 45",
    ),
    (
        "too-many-per-slot",
        "ACC.ADD; ACC.ADD;;",
        SlotError,
        (1, 10),
        "ACC.ADD",
        "Too many instructions of type AccInst (max 1)\nAt: Line 1, Column 10",
    ),
    (
        "operand-count",
        "ACC.ADD LR0;;",
        OperandError,
        (1, 1),
        "ACC.ADD",
        "Instruction ACC.ADD expects 0 operands, got 1, in Line 1, Column 1.",
    ),
    (
        "unknown-mnemonic",
        "FOO;;",
        SlotError,
        (1, 1),
        "FOO",
        "Opcode 'FOO' not found in any Inst subclass.",
    ),
    (
        "pseudo-operand-count",
        "BGT LR0;;",
        OperandError,
        (1, 1),
        "BGT",
        "Pseudo-instruction 'BGT' expects 3 operand(s)",
    ),
]


@pytest.mark.parametrize(
    ("source", "error_class", "where", "token", "message"),
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_error_site_raises_located_subclass(source, error_class, where, token, message) -> None:
    with pytest.raises(error_class) as info:
        _encode(source)
    error = info.value
    assert type(error) is error_class
    assert str(error).startswith(message)
    assert error.token == token
    line, column = where
    assert error.location.line == line
    if column is not None:
        assert error.location.column == column
    assert error.stage == "assemble"
    assert isinstance(error, ValueError)
    assert isinstance(error, IpuError)


def test_undefined_label_lookup_raises_label_error() -> None:
    token = lark.Token("LABEL", "missing", line=4, column=7)
    with pytest.raises(LabelError) as info:
        SymbolTable().address(token)
    assert str(info.value) == "Label 'missing' referenced in Line 4, Column 7 not defined."
    assert info.value.location == SourceLocation(4, 7)
    assert info.value.token == "missing"


def test_opcode_class_lookup_raises_slot_error() -> None:
    from ipu_as.opcodes import Opcode

    token = lark.Token("OPCODE", "FOO", line=2, column=3)
    with pytest.raises(SlotError) as info:
        Opcode.find_opcode_class(token)
    assert str(info.value) == "Opcode 'FOO' not found in any Opcode subclass"
    assert info.value.location == SourceLocation(2, 3)


def test_source_location_text_is_the_assembler_message_text() -> None:
    assert str(SourceLocation(12, 5, end_column=9)) == "Line 12, Column 5"


@pytest.mark.parametrize(
    ("error_class", "stage", "base"),
    [
        (TemplateError, "template", AssemblerError),
        (ParseError, "parse", AssemblerError),
        (EncodeError, "assemble", AssemblerError),
        (OperandError, "assemble", EncodeError),
        (SlotError, "assemble", EncodeError),
        (LabelError, "assemble", EncodeError),
    ],
)
def test_hierarchy_and_stages(error_class, stage, base) -> None:
    assert issubclass(error_class, base)
    assert error_class.stage == stage
    error = error_class("message")
    assert error.location is None and error.token is None
    assert isinstance(error, IpuError) and isinstance(error, ValueError)
