"""The conformance suite (E04-31): every case's files, run through the
reference model, give its expected trace record by record.

The runner reads only the case files (``conformance.load``, ``run_case``);
the other tests check the files are complete, cover the ISA and are what the
generator writes from the catalog (``conformance_cases``).
"""

from __future__ import annotations

import dataclasses
import json
import time

import conformance
import pytest
from conformance_cases import all_cases, files_of
from jsonschema import Draft202012Validator

from ipu_emu.trace import read_trace
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.tags import Tag
from ipu_isa.view import isa

SUITE = conformance.SUITE
CASES = conformance.cases(SUITE)
#: The spec's limit for the whole suite.
SUITE_SECONDS = 30


def test_every_case_gives_its_expected_trace(tmp_path):
    started = time.monotonic()
    failures = []
    for case in CASES:
        trace = tmp_path / f"{case.name}.jsonl"
        conformance.run_case(case, trace)
        difference = conformance.first_difference(
            read_trace(case / "expected.trace.jsonl"), read_trace(trace)
        )
        if difference is not None:
            failures.append(f"{case.name}: {difference}")
    elapsed = time.monotonic() - started
    assert not failures, f"{len(failures)} of {len(CASES)} cases differ:\n" + "\n".join(failures)
    assert elapsed < SUITE_SECONDS, f"the suite took {elapsed:.1f} s"


def test_a_corrupted_trace_names_the_cycle_and_field():
    case = SUITE / "insn_add_nx8"
    header, records = read_trace(case / "expected.trace.jsonl")
    write = records[1].reg_writes[0]
    corrupted = list(records)
    corrupted[1] = dataclasses.replace(
        records[1], reg_writes=(dataclasses.replace(write, value=write.value + 1),)
    )
    message = conformance.first_difference((header, records), (header, corrupted))
    assert message == (
        f"cycle 1 (record 1): record.reg_writes[0].value: expected {write.value}, "
        f"got {write.value + 1}"
    )


@pytest.mark.parametrize("field", ["isa_digest", "program_digest"])
def test_a_header_from_another_isa_or_program_differs(field):
    header, records = read_trace(SUITE / "insn_add_nx8" / "expected.trace.jsonl")
    other = {**header, field: "0" * 64}
    message = conformance.first_difference((header, records), (other, records))
    assert message == f"header.{field}: expected {header[field]!r}, got {'0' * 64!r}"


def test_the_emulator_version_is_not_compared():
    header, records = read_trace(SUITE / "insn_add_nx8" / "expected.trace.jsonl")
    producer = {**header, "emulator_version": "9.9.9"}
    assert conformance.first_difference((header, records), (producer, records)) is None


def test_every_case_holds_the_four_files_and_no_python():
    schema = Draft202012Validator(
        json.loads((SUITE / conformance.SETUP_SCHEMA).read_text(encoding="utf-8"))
    )
    for case in CASES:
        assert sorted(p.name for p in case.iterdir()) == sorted(conformance.FILES), case.name
        schema.validate(json.loads((case / "setup.json").read_text(encoding="utf-8")))
    loose = sorted(p.name for p in SUITE.iterdir() if p.is_file())
    assert loose == ["README.md", conformance.SETUP_SCHEMA]


def test_every_hardware_instruction_runs_in_a_case():
    """Each HARDWARE instruction retires in some expected trace (not only
    sits in a program behind a branch)."""
    view = isa()
    hardware = {
        (slot.value, instruction.mnemonic)
        for slot in view.spec_slot_order
        for instruction in view.instructions(slot)
        if instruction.tag is Tag.HARDWARE and instruction.mnemonic != "NOP"
    }
    ran = set()
    for case in CASES:
        _, records = read_trace(case / "expected.trace.jsonl")
        ran.update((op.slot, op.mnemonic) for record in records for op in record.slots)
    missing = sorted(f"{slot}:{mnemonic}" for slot, mnemonic in hardware - ran)
    assert not missing, f"no case runs {missing}"


def test_the_readme_and_schema_name_the_isas_registers_and_formats():
    schema = json.loads((SUITE / conformance.SETUP_SCHEMA).read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["format"]["enum"] == [fmt.value for fmt in ElementFormat]
    names = properties["registers"]["items"]["properties"]["name"]["enum"]
    assert names == list(REGISTER_DEFINITIONS)
    readme = (SUITE / "README.md").read_text(encoding="utf-8")
    for name, definition in REGISTER_DEFINITIONS.items():
        row = next(line for line in readme.splitlines() if line.startswith(f"| `{name}` |"))
        count, size = (cell.strip() for cell in row.split("|")[2:4])
        assert count.split()[0] == str(definition["count"]), name
        assert size == str(definition["size_bytes"]), name


def test_the_files_are_what_the_generator_writes():
    """A case changed in the catalog without a regeneration."""
    cases = all_cases()
    assert [case.name for case in CASES] == [case.name for case in cases], (
        f"cases added or removed: run {conformance.UPDATE_COMMAND}"
    )
    for case in cases:
        for name, data in files_of(case).items():
            assert (SUITE / case.name / name).read_bytes() == data, (
                f"{case.name}/{name} is stale: run {conformance.UPDATE_COMMAND}"
            )
