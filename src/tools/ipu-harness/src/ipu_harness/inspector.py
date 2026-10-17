"""Record a kernel case for the Cycle Inspector (E08-16, E07-26).

    python -m ipu_harness.cli KERNEL --case NAME --trace T.jsonl --context C.json

``--context`` makes the run go through an :class:`InspectorRecorder`, the
runner's executor (``RunOptions.executor``): the case is staged, run and
compared exactly as its test runs it, while the recorder steps the model and
keeps what the page needs. Two files come out:

* the trace (``--trace``): JSON Lines, the header, then one commit record per
  retired word (``ipu_isa/schemas``). This is the contract.
* the context (``--context``): what a viewer needs that the trace does not
  carry: the kernel's source text and labels, the loaded program's words,
  every register's value when the program started, and per record the
  multiplier lanes it used and the XMEM ranges it read
  (``ReferenceModel.observe_xmem_reads``).

``ipu-gen inspector-trace`` turns the two into the page's JSON
(``docs/inspector.bzl``'s ``kernel_trace``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ipu_emu.reference import ReferenceModel, TraceWriter
from ipu_harness.cases import Case
from ipu_harness.manifest import KernelManifest
from ipu_harness.runner import RunContext
from ipu_isa.registers import REGISTER_DEFINITIONS

__all__ = ["CONTEXT_VERSION", "InspectorRecorder"]

#: The context file's layout version; ``ipu_gen.inspector_trace`` checks it.
CONTEXT_VERSION = 1

#: The outcomes after which the model takes no more steps.
_LAST = ("halt", "fault")


@dataclass
class InspectorRecorder:
    """An executor that steps the loaded case, writing its trace as it goes.

    Pass it as ``RunOptions(executor=recorder)`` for one run, then take
    :meth:`context`.

    Attributes:
        trace_path: Where the trace goes.
    """

    trace_path: Path
    _start: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _labels: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _cycles: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __call__(self, context: RunContext) -> None:
        """Step ``context.model`` to a HALT, a fault or the cycle limit."""
        if self._start is not None:
            raise RuntimeError("an InspectorRecorder records one run")
        model = context.model
        self._start = {
            "program": [f"{word:x}" for word in context.program.words],
            "init": _registers(model),
        }
        self._labels = dict(context.program.symbols)
        reads: list[list[int]] = []
        stop = model.observe_xmem_reads(lambda address, size: reads.append([address, size]))
        try:
            with TraceWriter(self.trace_path, dict(context.trace_header)) as trace:
                while not context.cycle_limit or len(self._cycles) < context.cycle_limit:
                    lanes = model.stats().mult_lane_ops
                    record = model.step()
                    trace.write(record)
                    self._cycles.append(
                        {"lanes": model.stats().mult_lane_ops - lanes, "xmem_reads": reads[:]}
                    )
                    reads.clear()
                    if record.outcome in _LAST:
                        return
        finally:
            stop()

    def context(self, manifest: KernelManifest, case: Case) -> dict[str, Any]:
        """The context document of the run this recorder stepped.

        Args:
            manifest: The kernel that ran.
            case: The case that ran.

        Raises:
            RuntimeError: The recorder has not run.
        """
        if self._start is None:
            raise RuntimeError(f"{manifest.name}/{case.name}: the recorder ran no program")
        return {
            "context_version": CONTEXT_VERSION,
            "kernel": manifest.name,
            "case": case.name,
            "options": {},  # a case runs as kernel.toml declares it: no overrides
            "source_path": _source_path(manifest),
            "source": manifest.asm.read_text(encoding="utf-8"),
            "labels": self._labels,
            **self._start,
            "cycles": self._cycles,
        }


def _registers(model: ReferenceModel) -> dict[str, str]:
    """Every register's bytes, bank by bank, in hex (``lr`` and ``cr`` as
    little-endian 32-bit words, as the register file holds them)."""
    banks = {}
    for name, definition in REGISTER_DEFINITIONS.items():
        data = bytearray()
        for index in range(definition["count"]):
            value = model.read_register(name, index)
            if isinstance(value, int):
                value = value.to_bytes(definition["size_bytes"], "little")
            data += value
        banks[name] = data.hex()
    return banks


def _source_path(manifest: KernelManifest) -> str:
    """The program's path from the folder that holds the kernel tree:
    ``kernels/<family>/<name>/<name>.asm``."""
    tree_parent = manifest.directory.parent.parent.parent
    return manifest.asm.relative_to(tree_parent).as_posix()
