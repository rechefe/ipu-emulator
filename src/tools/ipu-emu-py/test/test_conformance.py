"""The conformance suite (E04-31): every case's files, run through the
reference model, give its expected trace record by record.

The runner reads only the case files (``conformance.load``, ``run_case``);
the other tests check the files are complete, cover the ISA and are what the
generator writes.
"""

from __future__ import annotations

import dataclasses
import json
import time

import conformance
import pytest
from conformance_cases import all_cases
from jsonschema import Draft202012Validator

from ipu_isa.tags import Tag
from ipu_isa.view import isa

SUITE = conformance.SUITE
CASES = conformance.cases(SUITE)
#: The spec's limit for the whole suite.
SUITE_SECONDS = 30


# The probes include ST.ACC, the emulator-only store: its warning is expected.
@pytest.mark.filterwarnings("ignore::ipu_emu.errors.DebugOnlyInstructionWarning")
def test_every_case_gives_its_expected_trace(tmp_path):
    started = time.monotonic()
    failures = []
    for case in CASES:
        trace = tmp_path / f"{case.name}.jsonl"
        conformance.run_case(case, trace)
        difference = conformance.first_difference(
            conformance.read(case / "expected.trace.jsonl"), conformance.read(trace)
        )
        if difference is not None:
            failures.append(f"{case.name}: {difference}")
    elapsed = time.monotonic() - started
    assert not failures, f"{len(failures)} of {len(CASES)} cases differ:\n" + "\n".join(failures)
    assert elapsed < SUITE_SECONDS, f"the suite took {elapsed:.1f} s"


def test_a_corrupted_trace_names_the_cycle_and_field():
    case = SUITE / "insn_add_int8"
    header, records = conformance.read(case / "expected.trace.jsonl")
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


def test_a_header_from_another_isa_or_program_differs():
    header, records = conformance.read(SUITE / "insn_add_int8" / "expected.trace.jsonl")
    other = {**header, "program_digest": "0" * 64, "emulator_version": "9.9.9"}
    message = conformance.first_difference((header, records), (other, records))
    assert message is not None and message.startswith("header.program_digest")
    producer = {**header, "emulator_version": "9.9.9"}
    assert conformance.first_difference((header, records), (producer, records)) is None


def test_every_case_holds_the_four_files_and_no_python():
    schema = Draft202012Validator(json.loads((SUITE / conformance.SETUP_SCHEMA).read_text()))
    for case in CASES:
        assert sorted(p.name for p in case.iterdir()) == sorted(conformance.FILES), case.name
        schema.validate(json.loads((case / "setup.json").read_text(encoding="utf-8")))
    loose = sorted(p.name for p in SUITE.iterdir() if p.is_file())
    assert loose == ["README.md", conformance.SETUP_SCHEMA]


def test_every_hardware_instruction_is_in_a_case():
    view = isa()
    hardware = {
        (slot, instruction.mnemonic)
        for slot in view.spec_slot_order
        for instruction in view.instructions(slot)
        if instruction.tag is Tag.HARDWARE and instruction.mnemonic != "NOP"
    }
    found = set()
    for case in CASES:
        found.update(conformance.slot_ops((case / "program.bin").read_bytes()))
    missing = sorted(f"{slot.value}:{mnemonic}" for slot, mnemonic in hardware - found)
    assert not missing, f"no case runs {missing}"


def test_the_suite_holds_the_generators_cases():
    """A case added to or removed from the catalog without a regeneration."""
    assert [case.name for case in CASES] == [case.name for case in all_cases()], (
        "run bazel run //src/tools/ipu-emu-py:regen_conformance"
    )
