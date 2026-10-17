"""Rewrite the conformance suite from its cases (E04-31).

    bazel run //src/tools/ipu-emu-py:regen_conformance

For each case of ``conformance_cases.all_cases()`` it writes ``program.asm``,
``program.bin`` and ``setup.json``, then runs the case from those files
through the reference model (``conformance.run_case``, as the runner does)
and writes ``expected.trace.jsonl``. Folders of cases no longer listed are
removed. Review the diff like a golden's: a behaviour-change PR regenerates
only the traces it changes and says why.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import conformance
from conformance_cases import Case, all_cases

from ipu_emu.ipu_state import IpuState
from ipu_isa.codec.word import pack_program
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.tags import Tag
from ipu_isa.view import isa

UPDATE_COMMAND = "bazel run //src/tools/ipu-emu-py:regen_conformance"
_WIDE_FORMATS = ("fp32", "int32")


def _registers(state: IpuState, fresh: IpuState) -> list[dict[str, Any]]:
    """Every register that differs from its reset value, in the order of
    ``ipu_isa.registers``."""
    writes = []
    for name, definition in REGISTER_DEFINITIONS.items():
        size = definition["size_bytes"]
        have, reset = state.regfile.view(name), fresh.regfile.view(name)
        for index in range(definition["count"]):
            data = bytes(have[index * size : (index + 1) * size])
            if data == bytes(reset[index * size : (index + 1) * size]):
                continue
            if definition["vector"]:
                writes.append({"name": name, "index": index, "hex": data.hex()})
            else:
                writes.append(
                    {"name": name, "index": index, "value": int.from_bytes(data, "little")}
                )
    return writes


def _xmem(state: IpuState) -> list[dict[str, Any]]:
    """The XMEM bytes that are not zero, one region per run (zero runs of
    fewer than 64 bytes stay inside a region)."""
    regions = []
    for base, page in state.xmem.touched_pages():
        data = bytes(page)
        start = None
        zeros = 0
        for offset, byte in enumerate(data):
            if byte:
                if start is None:
                    start = offset
                zeros = 0
            elif start is not None:
                zeros += 1
                if zeros == 64:
                    end = offset - zeros + 1
                    regions.append({"address": base + start, "hex": data[start:end].hex()})
                    start, zeros = None, 0
        if start is not None:
            end = len(data) - zeros
            regions.append({"address": base + start, "hex": data[start:end].hex()})
    return regions


def _emulator_only(case: Case, data: bytes) -> bool:
    """True when the hardware cannot run the case: a wide format, or an
    EMULATOR slot's op other than NOP."""
    if case.fmt.value in _WIDE_FORMATS:
        return True
    emulator = set(isa().layout.slots_with_tag(Tag.EMULATOR))
    return any(slot in emulator and op != "NOP" for slot, op in conformance.slot_ops(data))


def setup_of(case: Case, data: bytes) -> dict[str, Any]:
    """The ``setup.json`` of *case*: its starting state, as differences from reset."""
    state, fresh = IpuState(fmt=case.fmt), IpuState(fmt=case.fmt)
    case.prepare(state)
    return {
        "description": case.description,
        "format": case.fmt.value,
        "emulator_only": _emulator_only(case, data),
        "max_cycles": case.max_cycles,
        "fault_enables": dict(case.fault_enables),
        "registers": _registers(state, fresh),
        "xmem": _xmem(state),
    }


def write_case(suite: Path, case: Case) -> None:
    folder = suite / case.name
    folder.mkdir(parents=True, exist_ok=True)
    words = case.words()
    edited = (
        "\n# program.bin edits this source: see the case's description.\n" if case.edit else "\n"
    )
    source = f"# {case.name}: {case.description}{edited}{case.asm.strip()}\n"
    (folder / "program.asm").write_text(source, encoding="utf-8")
    data = pack_program(words)
    (folder / "program.bin").write_bytes(data)
    setup = setup_of(case, data)
    (folder / "setup.json").write_text(json.dumps(setup, indent=1) + "\n", encoding="utf-8")
    conformance.run_case(folder, folder / "expected.trace.jsonl")


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {UPDATE_COMMAND}", file=sys.stderr)
        return 2
    suite = Path(workspace) / conformance.SUITE
    cases = all_cases()
    names = {case.name for case in cases}
    for folder in conformance.cases(suite) if suite.is_dir() else []:
        if folder.name not in names:
            shutil.rmtree(folder)
    for case in cases:
        try:
            write_case(suite, case)
        except Exception as error:
            raise RuntimeError(f"case {case.name}: {error}") from error
    print(f"Wrote {len(cases)} cases to {conformance.SUITE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
