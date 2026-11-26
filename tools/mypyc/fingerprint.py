"""Fingerprint emulator runs in kernelgen's fast-path mode, for the mypyc build check.

    python fingerprint.py KERNEL:CASE[:FMT] ...

Prints one line per case: ``<spec> passed=<bool> cycles=<n> sha=<hex>``. The hash
covers the final register file, every XMEM page and metadata word, the PC, the
halt record, the run statistics and the arithmetic flags, so two emulator builds
that print the same lines ended every case in the same state.

Fast-path mode is what kernelgen's ``emurun.fast_path()`` switches on at run
time (reproduced here, not imported): prototyping mode allowed by default, and
no commit collector during an untraced run.
"""

from __future__ import annotations

import hashlib
import sys


def fast_path() -> None:
    import ipu_emu.prototyping as prototyping

    prototyping._BUILD_DEFAULT = True
    from ipu_emu.reference import ReferenceModel

    plain = ReferenceModel.run

    def run(self, max_cycles=None, *, trace=None):
        if trace is not None:
            return plain(self, max_cycles, trace=trace)
        self._collector.close()
        try:
            return plain(self, max_cycles, trace=None)
        finally:
            self._new_collector()

    ReferenceModel.run = run


def fingerprint(spec: str) -> str:
    from ipu_harness.assembly import AssemblyCache
    from ipu_harness.cases import expand
    from ipu_harness.registry import default_registry
    from ipu_harness.runner import RunOptions, run_kernel
    from ipu_harness.variants import variant

    kernel, case_name, *rest = spec.split(":")
    manifest = default_registry().kernel(kernel)
    case = next(c for c in expand(manifest) if c.name == case_name)
    element = variant(rest[0]) if rest else case.formats[0]
    out = {}

    def execute(context):
        context.model.run(context.cycle_limit)
        out["model"] = context.model

    result = run_kernel(
        manifest, case, element, RunOptions(executor=execute), cache=AssemblyCache()
    )
    state = out["model"]._state
    h = hashlib.sha256()
    h.update(repr(state.regfile.to_dict()).encode())
    for page in sorted(state.xmem._pages):
        h.update(page.to_bytes(8, "little"))
        h.update(bytes(state.xmem._pages[page]))
    h.update(repr(sorted(state.xmem._metadata.items())).encode())
    h.update(
        repr(
            (state.program_counter, state.halt, state.stats, sorted(map(str, state.arith_flags)))
        ).encode()
    )
    return f"{spec} passed={result.passed} cycles={result.cycles} sha={h.hexdigest()}"


def main(argv: list[str]) -> int:
    fast_path()
    for spec in argv:
        print(fingerprint(spec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
