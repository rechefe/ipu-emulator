"""Record one registered kernel case as an E04-29 trace, for the Cycle Inspector (E08-16).

    bazel run //src/tools/ipu-apps:trace_case -- identity --trace "$PWD/t.jsonl" --context "$PWD/c.json"

The case runs exactly as its test runs it (``run_case``: the harness prepares
the inputs, runs the program and checks the output). While it runs, the
emulator's commit collector records one commit record per word and the
E04-29 trace writer writes them, after the trace header.

Two files come out:

* ``--trace``: the trace, JSON Lines: the header, then one commit record per
  retired word (``ipu_isa/schemas``). This is the contract.
* ``--context``: what a viewer needs that the trace does not carry: the
  kernel's source text and labels, the loaded program's words, every
  register's value when the program started, and per record the multiplier
  lanes it used and the XMEM ranges it read.

Until the E06-11 runner's trace option lands, this is the case's runner. It
records from the harness's own run instead of re-creating the case in
``ipu_emu.reference.ReferenceModel``: the harness prepares an ``IpuState``
(inputs, element format, emulator-only activation settings), and only a run
on that state is the run the case's test checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.programs import assemble_program
from ipu_apps.kernel_registry.registry import kernel_spec
from ipu_emu import emulator
from ipu_emu.commit import CommitCollector
from ipu_emu.ipu_state import IpuState
from ipu_emu.trace import TraceWriter, trace_header
from ipu_isa.codec.word import DecodedWord
from ipu_isa.registers import REGISTER_DEFINITIONS

__all__ = ["CONTEXT_VERSION", "CaseTrace", "main", "record_case"]

#: The context file's layout version; ``ipu_gen.inspector_trace`` checks it.
CONTEXT_VERSION = 1


class _CycleFacts:
    """Per word: the multiplier lanes it retired and the XMEM ranges it read.

    A cycle observer on the state, like the commit collector, so it sees the
    same words: one entry per word that ends.
    """

    def __init__(self, state: IpuState) -> None:
        self._state = state
        self._lanes_before = 0
        self._reads: list[list[int]] = []
        self.entries: list[dict[str, Any]] = []
        self._detach: list[Callable[[], None]] = [
            state.add_cycle_observer(self),
            state.xmem.add_observer(self._memory_event),
        ]

    def close(self) -> None:
        for detach in self._detach:
            detach()
        self._detach = []

    def begin(self, pc: int) -> None:
        self._lanes_before = self._state.stats.mult_lane_ops
        self._reads = []

    def issued(self, facts: object) -> None:
        pass

    def retired(self, facts: object, result: object) -> None:
        pass

    def end(self) -> None:
        lanes = self._state.stats.mult_lane_ops - self._lanes_before
        self.entries.append({"lanes": lanes, "xmem_reads": self._reads})
        self._reads = []

    def _memory_event(self, kind: str, address: int, size: int, data: object) -> None:
        if kind == "read" and size:
            self._reads.append([address, size])


class CaseTrace:
    """Records the one program a case runs: its trace and its context.

    Use :meth:`watch` around the case's run.

    Args:
        trace_path: Where the trace goes.
    """

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.start: dict[str, Any] | None = None
        self.facts: list[dict[str, Any]] = []
        self.records = 0

    @contextmanager
    def watch(self) -> Iterator[CaseTrace]:
        """Trace the program that reaches ``ipu_emu.emulator.run_until_complete``
        while the block runs (by patching it, as ``baseline.Recorder`` does, so
        no harness changes)."""
        original = emulator.run_until_complete

        def run_until_complete(state: IpuState, *args: Any, **kwargs: Any) -> int:
            if self.start is not None:
                raise RuntimeError("a traced case runs one program; this one runs several")
            self.start = _start(state)
            collector = CommitCollector(state)
            cycle_facts = _CycleFacts(state)
            try:
                return original(state, *args, **kwargs)
            finally:
                cycle_facts.close()
                records = collector.close()
                with TraceWriter(self.trace_path, trace_header(state)) as trace:
                    trace.write_all(records)
                self.records = len(records)
                self.facts = cycle_facts.entries

        emulator.run_until_complete = run_until_complete
        try:
            yield self
        finally:
            emulator.run_until_complete = original


def _start(state: IpuState) -> dict[str, Any]:
    """The loaded program's words and every register's bytes, as the run starts."""
    words = [
        0 if word is None else DecodedWord(word).to_word()
        for word in state.inst_mem[: state.program_length]
    ]
    return {
        "program": [f"{word:x}" for word in words],
        "init": {name: bytes(state.regfile.view(name)).hex() for name in REGISTER_DEFINITIONS},
    }


def _labels(kernel: str, program: list[str]) -> dict[str, int]:
    """The kernel source's labels, when the source alone assembles to ``program``.

    A kernel whose program depends on its case's parameters assembles to other
    words without them: it gets no labels, and a viewer names the branch
    targets itself.
    """
    spec = kernel_spec(kernel)
    result = assemble_program(files(spec.resource_package).joinpath(spec.asm))
    if [f"{word:x}" for word in result.words] != program:
        return {}
    return dict(result.symbols)


def record_case(
    kernel: str,
    case: str,
    trace_path: Path,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``kernel``'s ``case`` and write its trace to ``trace_path``.

    Args:
        kernel: The registered kernel.
        case: One of its cases.
        trace_path: Where the trace goes.
        options: The case's option overrides.

    Returns:
        The context document (see the module docstring).

    Raises:
        ValueError: An unknown kernel, case or option, or the case's output
            check failed.
        RuntimeError: The case ran no program, or more than one.
    """
    cases = load_cases(kernel)
    if case not in cases:
        raise ValueError(f"{kernel} has no case {case!r}; cases: {sorted(cases)}")
    recorder = CaseTrace(trace_path)
    with recorder.watch():
        run_case(kernel, cases[case], options=options)
    if recorder.start is None:
        raise RuntimeError(f"{kernel}/{case} ran no program through run_until_complete")
    if len(recorder.facts) != recorder.records:
        raise RuntimeError(
            f"{recorder.records} commit records but {len(recorder.facts)} cycle facts"
            " (a faulting word ends no cycle)"
        )
    spec = kernel_spec(kernel)
    package_path = spec.resource_package.replace(".", "/")
    return {
        "context_version": CONTEXT_VERSION,
        "kernel": kernel,
        "case": case,
        "options": dict(options or {}),
        "source_path": f"{package_path}/{spec.asm}",
        "source": files(spec.resource_package).joinpath(spec.asm).read_text(encoding="utf-8"),
        "labels": _labels(kernel, recorder.start["program"]),
        **recorder.start,
        "cycles": recorder.facts,
    }


def _option(text: str) -> tuple[str, Any]:
    name, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kernel")
    parser.add_argument("--case", default="default")
    parser.add_argument(
        "--option",
        action="append",
        type=_option,
        default=[],
        metavar="NAME=VALUE",
        help="a case option override; VALUE is JSON when it parses as JSON",
    )
    parser.add_argument("--trace", type=Path, required=True, help="the trace (JSON Lines)")
    parser.add_argument("--context", type=Path, required=True, help="the context (JSON)")
    args = parser.parse_args(argv)
    context = record_case(args.kernel, args.case, args.trace, dict(args.option))
    args.context.write_text(json.dumps(context, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
