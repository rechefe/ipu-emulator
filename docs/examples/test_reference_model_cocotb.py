"""The example testbench of How-to: use the reference model from cocotb
(docs/content/how-to/reference-model-cocotb.md).

The page includes the sections between the ``--8<--`` markers below. They are
the testbench an RTL engineer writes: set up the model, compare it with the RTL
word by word (lockstep), at the end of a run, or through a trace file.

No simulator runs here. The DUT is a second :class:`ReferenceModel` behind the
same coroutines a cocotb monitor would offer (``load_rtl``, ``rtl_retire``,
``rtl_read_register``, ``rtl_read_xmem``), and ``cocotb.test`` is a stub, so
asyncio runs the cocotb test function as cocotb would. A DUT with a planted
bug checks that each comparison fails, and names the field that differs.
"""

from __future__ import annotations

import asyncio
import re
import types

import pytest

from ipu_as.lark_tree import assemble_to_bin_file

# The page shows the imports below: the testbench's own.
# isort: split
# --8<-- [start:imports]
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ipu_emu.reference import ReferenceModel, RunState
from ipu_emu.trace import EMULATOR_VERSION, TraceWriter
from ipu_isa.formats import ElementFormat
from ipu_isa.trace_schema import SCHEMA_VERSION, isa_digest

# --8<-- [end:imports]

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "content" / "how-to" / "reference-model-cocotb.md"

#: The program the testbench runs: ST.AAQ writes POST_AAQ_REG (512 bytes, four
#: XMEM rows) to rows 0, 4 and 8, then the program halts.
PROGRAM = """\
SET LR1 CR2 ;;
loop:
ST.AAQ LR0 CR3 ;;
INC LR0 4 ;;
BNE LR0 LR1 loop ;;
HALT ;;
"""

# --8<-- [start:prepare]
FMT = ElementFormat.INT8
ROW = 128  # XMEM bytes per row


def prepare(target: ReferenceModel, program: bytes) -> None:
    """Reset *target*, load *program* and write the starting state."""
    target.reset(FMT)
    target.load_program(program)  # a program file's bytes
    target.write_register("cr", 12, index=2)  # CR2: stop after row 8
    target.write_register("cr", 0, index=3)  # CR3: base row 0
    target.write_register("post_aaq_reg", bytes(range(256)) * 2)
    target.write_xmem(0, b"\xff" * (16 * ROW))  # rows 0-15


# --8<-- [end:prepare]

# -- the fake DUT ---------------------------------------------------------------
# What the RTL testbench provides. Here a second model plays the RTL; the
# planted bug, when set, changes the DUT's starting state after `prepare`.


class FakeDut:
    def __init__(self, bug: Callable[[ReferenceModel], None] | None = None) -> None:
        self.model = ReferenceModel()
        self.bug = bug


async def load_rtl(dut: FakeDut, program: bytes) -> None:
    prepare(dut.model, program)
    if dut.bug is not None:
        dut.bug(dut.model)
    await asyncio.sleep(0)


async def rtl_retire(dut: FakeDut) -> dict[str, Any]:
    await asyncio.sleep(0)  # a real monitor awaits clock edges until a word retires
    return dut.model.step().to_json()


async def rtl_run(dut: FakeDut) -> int:
    await asyncio.sleep(0)
    return dut.model.run().cycles


async def rtl_read_register(dut: FakeDut, bank: str, index: int) -> int | bytes:
    await asyncio.sleep(0)
    return dut.model.read_register(bank, index)


async def rtl_read_xmem(dut: FakeDut, address: int, size: int) -> bytes:
    await asyncio.sleep(0)
    return dut.model.read_xmem(address, size)


# No simulator: `@cocotb.test()` leaves the coroutine as it is, and asyncio runs it.
cocotb = types.SimpleNamespace(test=lambda: lambda function: function)


# --8<-- [start:lockstep]
#: The commit record's fields. Leave out the ones your monitor cannot see.
FIELDS = (
    "cycle",
    "pc_before",
    "pc_after",
    "slots",
    "reg_writes",
    "xmem_writes",
    "outcome",
    "fault",
    "arith_events",
)


async def lockstep(
    model: ReferenceModel,
    retire: Callable[[], Awaitable[dict[str, Any]]],
    fields: tuple[str, ...] = FIELDS,
    max_words: int = 100_000,
) -> int:
    """Step *model* once per word the RTL retires and compare the two records.

    Args:
        model: The reference model, prepared like the RTL.
        retire: Awaits the RTL's next retired word, as a commit record dict.
        fields: The record fields to compare.
        max_words: Fail if no HALT or fault comes within this many words.

    Returns:
        The number of words compared.
    """
    for _ in range(max_words):
        expected = model.step().to_json()
        actual = await retire()
        for field in fields:
            if actual.get(field) != expected[field]:
                raise AssertionError(
                    f"word {expected['cycle']} (PC {expected['pc_before']}) {field}: "
                    f"model {expected[field]!r}, RTL {actual.get(field)!r}"
                )
        if expected["outcome"] != "retired":  # "halt" or "fault": the run is over
            return expected["cycle"] + 1
    raise AssertionError(f"no HALT within {max_words} words")


# --8<-- [end:lockstep]


# --8<-- [start:cocotb]
@cocotb.test()
async def kernel_lockstep(dut):
    program = Path("prog.bin").read_bytes()  # ipu-as assemble --output prog.bin
    model = ReferenceModel()
    prepare(model, program)
    await load_rtl(dut, program)  # your testbench: the same program and state
    words = await lockstep(model, lambda: rtl_retire(dut))
    assert model.status.state is RunState.HALTED, f"after {words} words: {model.status}"


# --8<-- [end:cocotb]


# --8<-- [start:end-of-run]
async def compare_final_state(model, dut, xmem_rows: range) -> None:
    """Run both to the end, then compare the LRs, the CRs and XMEM rows."""
    result = model.run()
    words = await rtl_run(dut)  # your testbench: start the RTL, await done
    assert words == result.cycles, f"RTL retired {words} words, model {result.cycles}"
    for bank in ("lr", "cr"):
        for index in range(16):
            want = model.read_register(bank, index)
            got = await rtl_read_register(dut, bank, index)
            assert got == want, f"{bank.upper()}{index}: model {want:#x}, RTL {got:#x}"
    for row in xmem_rows:
        want = model.read_xmem(row * ROW, ROW)
        got = await rtl_read_xmem(dut, row * ROW, ROW)
        assert got == want, f"XMEM row {row}: model {want.hex()}, RTL {got.hex()}"


@cocotb.test()
async def kernel_end_of_run(dut):
    program = Path("prog.bin").read_bytes()
    model = ReferenceModel()
    prepare(model, program)
    await load_rtl(dut, program)
    await compare_final_state(model, dut, range(16))


# --8<-- [end:end-of-run]


# --8<-- [start:trace-write]
def write_model_trace(program: bytes, path: Path) -> None:
    """Run the model and write its trace: a header line, then one record per word."""
    model = ReferenceModel()
    prepare(model, program)
    header = {
        "schema_version": SCHEMA_VERSION,
        "isa_digest": isa_digest(),
        "format": FMT.value,
        "program_digest": hashlib.sha256(program).hexdigest(),
        "emulator_version": EMULATOR_VERSION,
    }
    with TraceWriter(path, header) as trace:
        model.run(trace=trace)


# --8<-- [end:trace-write]


# --8<-- [start:trace-compare]
#: Header fields that say what the run was; `emulator_version` names the producer.
HEADER_FIELDS = ("schema_version", "isa_digest", "format", "program_digest")


def first_trace_difference(model_trace: Path, rtl_trace: Path) -> str | None:
    """The first difference between two traces, or None when they agree."""
    model_header, *model_records = map(json.loads, model_trace.read_text().splitlines())
    rtl_header, *rtl_records = map(json.loads, rtl_trace.read_text().splitlines())
    for field in HEADER_FIELDS:
        if rtl_header.get(field) != model_header[field]:
            return f"header {field}: model {model_header[field]!r}, RTL {rtl_header.get(field)!r}"
    for want, got in zip(model_records, rtl_records, strict=False):
        if got != want:
            return f"record {want['cycle']}: model {want}, RTL {got}"
    if len(rtl_records) != len(model_records):
        return f"model {len(model_records)} records, RTL {len(rtl_records)}"
    return None


# --8<-- [end:trace-compare]


# -- the tests --------------------------------------------------------------------


@pytest.fixture
def program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    """``prog.bin`` in the current directory, as the cocotb tests read it."""
    (tmp_path / "prog.asm").write_text(PROGRAM, encoding="utf-8")
    assemble_to_bin_file(PROGRAM, str(tmp_path / "prog.bin"))
    monkeypatch.chdir(tmp_path)
    return (tmp_path / "prog.bin").read_bytes()


def _cr2_is_8(model: ReferenceModel) -> None:
    model.write_register("cr", 8, index=2)  # the RTL stops one row early


def _lr5_is_7(model: ReferenceModel) -> None:
    model.write_register("lr", 7, index=5)  # a register the program never writes


def _row_13_is_zero(model: ReferenceModel) -> None:
    model.write_xmem(13 * ROW, bytes(ROW))  # a row the program never writes


def test_the_program_writes_xmem_and_branches(program):
    model = ReferenceModel()
    prepare(model, program)
    records = [model.step().to_json()]
    while records[-1]["outcome"] == "retired":
        records.append(model.step().to_json())
    assert records[-1]["outcome"] == "halt"
    assert sum(bool(r["xmem_writes"]) for r in records) == 3
    assert model.read_xmem(8 * ROW, 4) == bytes(range(4))


def test_lockstep_matches_a_model_dut(program):
    asyncio.run(kernel_lockstep(FakeDut()))


def test_lockstep_names_the_first_difference(program):
    with pytest.raises(
        AssertionError,
        match=r"^word 0 \(PC 0\) reg_writes: model \[\{'register': 'lr', 'index': 1, 'value': 12\}\], RTL .*'value': 8",
    ):
        asyncio.run(kernel_lockstep(FakeDut(_cr2_is_8)))


def test_lockstep_stops_at_halt(program):
    model, dut = ReferenceModel(), FakeDut()
    prepare(model, program)
    asyncio.run(load_rtl(dut, program))
    assert asyncio.run(lockstep(model, lambda: rtl_retire(dut))) == 11
    assert model.status.state is RunState.HALTED


def test_end_of_run_matches_a_model_dut(program):
    asyncio.run(kernel_end_of_run(FakeDut()))


def test_end_of_run_counts_the_words(program):
    with pytest.raises(AssertionError, match=r"^RTL retired 8 words, model 11"):
        asyncio.run(kernel_end_of_run(FakeDut(_cr2_is_8)))


def test_end_of_run_names_the_register(program):
    with pytest.raises(AssertionError, match=r"^LR5: model 0x0, RTL 0x7"):
        asyncio.run(kernel_end_of_run(FakeDut(_lr5_is_7)))


def test_end_of_run_names_the_xmem_row(program):
    with pytest.raises(AssertionError, match=r"^XMEM row 13: "):
        asyncio.run(kernel_end_of_run(FakeDut(_row_13_is_zero)))


def test_traces_of_two_models_agree(program, tmp_path):
    write_model_trace(program, tmp_path / "model.jsonl")
    write_model_trace(program, tmp_path / "rtl.jsonl")
    assert first_trace_difference(tmp_path / "model.jsonl", tmp_path / "rtl.jsonl") is None


def test_a_trace_difference_names_the_record(program, tmp_path):
    write_model_trace(program, tmp_path / "model.jsonl")
    lines = (tmp_path / "model.jsonl").read_text().splitlines()
    record = json.loads(lines[3])
    record["pc_after"] += 1
    lines[3] = json.dumps(record)
    (tmp_path / "rtl.jsonl").write_text("\n".join(lines) + "\n")
    difference = first_trace_difference(tmp_path / "model.jsonl", tmp_path / "rtl.jsonl")
    assert difference is not None and difference.startswith("record 2: ")


#: `--8<-- "docs/examples/test_reference_model_cocotb.py:section"` in the page.
_INCLUDE = re.compile(
    r'^--8<-- "docs/examples/test_reference_model_cocotb\.py:(?P<section>[\w-]+)"$', re.M
)


def test_the_page_includes_every_section():
    text = Path(__file__).read_text(encoding="utf-8")
    sections = set(re.findall(r"# --8<-- \[start:([\w-]+)\]", text))
    included = set(_INCLUDE.findall(PAGE.read_text(encoding="utf-8")))
    assert included == sections
