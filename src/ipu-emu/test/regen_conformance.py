"""Rewrite the conformance suite from its cases (E04-31).

    bazel run //src/ipu-emu:regen_conformance

For each case of ``conformance_cases.all_cases()`` it writes ``program.asm``,
``program.bin`` and ``setup.json``, then runs the case from those files
through the reference model (``conformance.run_case``, as the runner does)
and writes ``expected.trace.jsonl``. Folders of cases no longer listed are
removed. Review the diff like a golden's: a behaviour-change PR regenerates
only the traces it changes and says why.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

import conformance
from conformance_cases import Case, all_cases, files_of

from ipu_emu.errors import DebugOnlyInstructionWarning


def write_case(suite: Path, case: Case) -> None:
    """Write *case*'s input files into its folder of *suite*, then run them
    and write the expected trace.

    Args:
        suite: The suite folder.
        case: The case.
    """
    folder = suite / case.name
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files_of(case).items():
        (folder / name).write_bytes(data)
    conformance.run_case(folder, folder / "expected.trace.jsonl")


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {conformance.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    # The probes include ST.ACC, the emulator-only store: its warning is expected.
    warnings.filterwarnings("ignore", category=DebugOnlyInstructionWarning)
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
