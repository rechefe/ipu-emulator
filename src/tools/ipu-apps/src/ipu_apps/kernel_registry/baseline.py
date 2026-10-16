"""Per-case kernel baselines: what each case ran and what it produced (E02-07).

A kernel case's own check compares with a tolerance, so a change that moves a
result by one ulp passes it. The baseline makes every case also prove
"bit-identical to the recorded run": for each case, ``<kernel folder>/baseline.json``
records

* for every program the case runs (one per stage; most kernels have one stage),
  the digest of the program as loaded, the CR and LR values right after the
  harness's ``setup``, and the digest of XMEM at that point;
* the digest of the case's output file, if it has one;
* the digest of every register and XMEM after the run;
* the cycle count.

``case_tests`` compares each run with its record after the case's own check
passes. ``bazel run //src/tools/ipu-apps:update_kernel_baselines -- [<kernel> ...]``
rewrites the files after an intended change.

The capture hooks ``ipu_emu.emulator.run_until_complete``, which ``run_test``
calls right after ``setup``; every stage of every kernel reaches the emulator
through ``run_test``, so no harness needs to change.

Fields that differ between faithful and prototyping mode would be listed in
``MODE_DEPENDENT_FIELDS``; none do (run statistics are not recorded), so a
flagged build checks the same records (E02-08).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

import ipu_emu.emulator as emulator
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.ipu_config import CR_REGISTER_NAME
from ipu_emu.ipu_state import IpuState

from ipu_apps.kernel_registry.registry import kernel_spec

#: The one command that rewrites baselines from the current tree.
UPDATE_COMMAND = "bazel run //src/tools/ipu-apps:update_kernel_baselines -- <kernel> ..."

#: Set by the update command to the workspace root: ``case_tests`` then writes
#: records there instead of comparing.
UPDATE_ENV = "IPU_UPDATE_KERNEL_BASELINES"

BASELINE_FILE = "baseline.json"

#: Record fields allowed to differ between faithful and prototyping mode.
MODE_DEPENDENT_FIELDS: frozenset[str] = frozenset()

_PAGE = 64 * 1024
_REGISTER_COUNT = 16


@dataclass(frozen=True)
class StageRecord:
    """One program run: what was loaded and set up before it started."""

    program_sha256: str
    cr: list[int]
    lr: list[int]
    xmem_setup_sha256: str


@dataclass(frozen=True)
class CaseRecord:
    stages: list[StageRecord]
    output_sha256: str | None
    state_sha256: str
    cycles: int

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> CaseRecord:
        return CaseRecord(
            stages=[StageRecord(**stage) for stage in data["stages"]],
            output_sha256=data["output_sha256"],
            state_sha256=data["state_sha256"],
            cycles=data["cycles"],
        )


def _xmem_digest(state: IpuState, digest) -> None:
    """Feed every non-zero 64 KiB XMEM page, with its index, into ``digest``."""
    # XMem has no public buffer view; reading through read_address would copy
    # all 512 MiB.
    data = np.frombuffer(state.xmem._data, dtype=np.uint8).reshape(-1, _PAGE)
    for page in np.flatnonzero(data.any(axis=1)):
        digest.update(int(page).to_bytes(4, "little"))
        digest.update(data[page].tobytes())


def xmem_sha256(state: IpuState) -> str:
    digest = hashlib.sha256()
    _xmem_digest(state, digest)
    return digest.hexdigest()


def program_sha256(state: IpuState) -> str:
    """Digest of instruction memory as loaded (decoded fields, sorted by name)."""
    canonical = [sorted(word.items()) if word is not None else None for word in state.inst_mem]
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def state_sha256(state: IpuState) -> str:
    """Digest of every register and every non-zero XMEM page."""
    digest = hashlib.sha256()
    for descriptor in REGFILE_SCHEMA:
        digest.update(descriptor.name.encode())
        digest.update(state.regfile.raw_readonly(descriptor.name))
    _xmem_digest(state, digest)
    return digest.hexdigest()


def _stage(state: IpuState) -> StageRecord:
    return StageRecord(
        program_sha256=program_sha256(state),
        cr=[state.regfile.get_cr(i) for i in range(_REGISTER_COUNT)],
        lr=[state.regfile.get_lr(i) for i in range(_REGISTER_COUNT)],
        xmem_setup_sha256=xmem_sha256(state),
    )


class Recorder:
    """Captures one case run: ``watch()`` around the run, then ``finish``."""

    def __init__(self) -> None:
        self.stages: list[StageRecord] = []
        self.record: CaseRecord | None = None

    @contextmanager
    def watch(self):
        original = emulator.run_until_complete

        def run_until_complete(state, *args, **kwargs):
            self.stages.append(_stage(state))
            return original(state, *args, **kwargs)

        emulator.run_until_complete = run_until_complete
        try:
            yield self
        finally:
            emulator.run_until_complete = original

    def finish(self, state: IpuState, cycles: int, output_path=None) -> CaseRecord:
        output = None
        if output_path is not None and Path(output_path).is_file():
            output = hashlib.sha256(Path(output_path).read_bytes()).hexdigest()
        self.record = CaseRecord(list(self.stages), output, state_sha256(state), cycles)
        return self.record


def differences(expected: CaseRecord, actual: CaseRecord) -> list[str]:
    """One line per field that differs; CR/LR changes name each register."""
    lines = []
    if len(expected.stages) != len(actual.stages):
        lines.append(f"stages: {len(expected.stages)} -> {len(actual.stages)}")
    for index, (old, new) in enumerate(zip(expected.stages, actual.stages)):
        where = f"stage {index} " if len(actual.stages) > 1 else ""
        for name in ("program_sha256", "xmem_setup_sha256"):
            if getattr(old, name) != getattr(new, name):
                lines.append(f"{where}{name}: {getattr(old, name)} -> {getattr(new, name)}")
        for bank, prefix in (("cr", CR_REGISTER_NAME.upper()), ("lr", "LR")):
            for reg, (a, b) in enumerate(zip(getattr(old, bank), getattr(new, bank))):
                if a != b:
                    lines.append(f"{where}{prefix}{reg}: {a} -> {b}")
    for name in ("output_sha256", "state_sha256", "cycles"):
        if getattr(expected, name) != getattr(actual, name):
            lines.append(f"{name}: {getattr(expected, name)} -> {getattr(actual, name)}")
    return lines


def _resource_path(kernel: str) -> Path:
    return Path(str(files(kernel_spec(kernel).resource_package).joinpath(BASELINE_FILE)))


def load(kernel: str) -> dict[str, CaseRecord]:
    """The recorded cases of ``kernel``; empty if it has no baseline file."""
    path = _resource_path(kernel)
    if not path.is_file():
        return {}
    return {case: CaseRecord.from_json(data) for case, data in json.loads(path.read_text()).items()}


def check(kernel: str, case_id: str, actual: CaseRecord) -> None:
    """Fail unless ``actual`` matches the recorded run of ``case_id``."""
    expected = load(kernel).get(case_id)
    if expected is None:
        raise AssertionError(
            f"{kernel}: case {case_id} has no baseline record; run {UPDATE_COMMAND}"
        )
    changed = differences(expected, actual)
    if changed:
        raise AssertionError(
            f"{kernel}: case {case_id} differs from its baseline:\n  "
            + "\n  ".join(changed)
            + f"\nIf this change is intended, run {UPDATE_COMMAND}"
        )


def update_root() -> Path | None:
    """The workspace to write baselines into, when the update command runs."""
    root = os.environ.get(UPDATE_ENV)
    return Path(root) if root else None


def source_path(root: Path, kernel: str) -> Path:
    package = kernel_spec(kernel).resource_package
    return root / "src/tools/ipu-apps/src" / package.replace(".", "/") / BASELINE_FILE


def write(path: Path, records: dict[str, CaseRecord]) -> None:
    """Indented JSON, with each CR/LR list on one line so a review diff shows
    the register bank that changed."""
    data = {case: record.to_json() for case, record in records.items()}
    text = json.dumps(data, indent=1, sort_keys=True)
    text = re.sub(
        r"\[\s+(-?\d+(?:,\s+-?\d+)*)\s+\]",
        lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]",
        text,
    )
    path.write_text(text + "\n")
