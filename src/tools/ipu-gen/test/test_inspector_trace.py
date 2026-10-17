"""ipu_gen.inspector_trace: a trace and its context -> the Cycle Inspector's JSON (E08-16).

A three-word program with one loop, its trace written by hand: what the page
gets is checked against hand-written expectations, so the test needs no
emulator.
"""

from __future__ import annotations

import jsonschema
import pytest

from ipu_as import session
from ipu_gen import inspector_trace
from ipu_gen.inspector_trace import TraceError, convert
from ipu_isa.codec.word import WORD_LAYOUT
from ipu_isa.trace_schema import isa_digest
from ipu_isa.view import isa

SOURCE = """\
    SET LR0 CR0;;
loop:
    ADD LR0 LR0 CR1;;
    BLT.S LR0 CR2 loop;;
"""

WORDS = session.assemble(SOURCE).words

HEADER = {
    "schema_version": "1.0.0",
    "isa_digest": isa_digest(),
    "format": "int8",
    "program_digest": "0" * 64,
    "emulator_version": "0.1.0",
}


def _record(cycle, pc_before, pc_after, lr0=None):
    writes = [] if lr0 is None else [{"register": "lr", "index": 0, "value": lr0}]
    return {
        "cycle": cycle,
        "pc_before": pc_before,
        "pc_after": pc_after,
        "slots": [],
        "reg_writes": writes,
        "xmem_writes": [],
        "outcome": "retired",
        "fault": None,
        "arith_events": [],
    }


# CR2 = 2: SET, then ADD/BLT.S twice (LR0 1, then 2), leaving past the BLT.
RECORDS = [
    _record(0, 0, 1, lr0=0),
    _record(1, 1, 2, lr0=1),
    _record(2, 2, 1),
    _record(3, 1, 2, lr0=2),
    _record(4, 2, 3),
]


def _context(labels=None, cycles=None):
    return {
        "context_version": 1,
        "kernel": "tiny",
        "case": "default",
        "options": {},
        "source_path": "tiny.asm",
        "source": SOURCE,
        "labels": {"loop": 1} if labels is None else labels,
        "program": [f"{word:x}" for word in WORDS],
        "init": {"lr": "00" * 64, "cr": "00" * 64},
        "cycles": [{"lanes": 0, "xmem_reads": []} for _ in RECORDS] if cycles is None else cycles,
    }


def test_program_is_decoded_with_the_source_labels():
    program = convert(HEADER, RECORDS, _context())["program"]
    assert program == [
        {
            "label": None,
            "word": f"{WORDS[0]:x}",
            "slots": [{"slot": "lr", "instance": 0, "op": "SET", "args": ["LR0", "CR0"]}],
        },
        {
            "label": "loop",
            "word": f"{WORDS[1]:x}",
            "slots": [{"slot": "lr", "instance": 0, "op": "ADD", "args": ["LR0", "LR0", "CR1"]}],
        },
        {
            "label": None,
            "word": f"{WORDS[2]:x}",
            "slots": [
                {"slot": "cond", "instance": 0, "op": "BLT.S", "args": ["LR0", "CR2", "loop"]}
            ],
        },
    ]


def test_without_source_labels_branch_targets_get_generated_ones():
    program = convert(HEADER, RECORDS, _context(labels={}))["program"]
    assert [word["label"] for word in program] == [None, "w0001", None]
    assert program[2]["slots"][0]["args"][-1] == "w0001"


def test_layout_is_every_field_msb_first():
    layout = convert(HEADER, RECORDS, _context())["layout"]
    assert layout[0] == [WORD_LAYOUT[-1].name, WORD_LAYOUT[-1].bits, WORD_LAYOUT[-1].slot.value]
    assert layout[-1] == [WORD_LAYOUT[0].name, WORD_LAYOUT[0].bits, WORD_LAYOUT[0].slot.value]
    assert sum(width for _, width, _ in layout) == isa().layout.word_bits()


def test_cycles_hold_each_word_pc_and_register_deltas():
    cycles = convert(HEADER, RECORDS, _context())["cycles"]
    assert [c["pc"] for c in cycles] == [0, 1, 2, 1, 2]
    assert [c["d"] for c in cycles] == [
        {"lr": {"0": 0}},
        {"lr": {"0": 1}},
        {},
        {"lr": {"0": 2}},
        {},
    ]
    assert all(c["m"] == [] and "L" not in c and "o" not in c for c in cycles)


def test_vector_writes_memory_and_lanes():
    record = dict(
        _record(0, 0, 1),
        reg_writes=[{"register": "r_acc", "offset": 8, "data": "0102"}],
        xmem_writes=[{"address": 256, "data": "aabbcc"}],
        outcome="halt",
    )
    facts = {"lanes": 64, "xmem_reads": [[128, 128]]}
    (cycle,) = convert(HEADER, [record], _context(cycles=[facts]))["cycles"]
    assert cycle == {
        "pc": 0,
        "d": {"r_acc": [[8, "0102"]]},
        "m": [["R", 128, 128], ["W", 256, 3]],
        "L": 64,
        "o": "halt",
    }


def test_max_cycles_truncates():
    document = convert(HEADER, RECORDS, _context(), max_cycles=2)
    assert len(document["cycles"]) == 2
    assert document["total_cycles"] == 5
    assert document["truncated"] is True
    assert convert(HEADER, RECORDS, _context())["truncated"] is False


def test_the_document_validates_against_its_schema():
    document = convert(HEADER, RECORDS, _context())
    jsonschema.validate(document, inspector_trace.schema())
    assert document["schemaVersion"] == inspector_trace.SCHEMA_VERSION
    assert document["meta"]["format"] == "int8"
    assert document["meta"]["wide"] is False
    assert document["meta"]["lut"] is None


def test_an_fp8_format_carries_its_value_table():
    meta = convert(dict(HEADER, format="e4"), RECORDS, _context())["meta"]
    lut = meta["lut"]
    assert len(lut) == 256
    assert lut[0] == 0.0
    assert lut[0x38] == 1.0  # E4M3: exponent 7 = bias, mantissa 0
    assert lut[0x7F] == "nan"


@pytest.mark.parametrize(
    ("header", "context", "message"),
    [
        (dict(HEADER, schema_version="2.0.0"), _context(), "schema"),
        (dict(HEADER, isa_digest="0" * 64), _context(), "another ISA"),
        (HEADER, dict(_context(), context_version=2), "context version"),
        (HEADER, _context(cycles=[]), "context cycles"),
    ],
)
def test_an_unknown_trace_or_context_is_refused(header, context, message):
    with pytest.raises(TraceError, match=message):
        convert(header, RECORDS, context)
