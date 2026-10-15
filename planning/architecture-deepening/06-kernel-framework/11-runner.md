# Runner and CLI: run a case end to end, report cycles/MAC, trace

Part of [Epic: Kernel framework](00-epic.md) · ID E06-11 · Size M · Depends on: E06-05, E06-06, E06-07, E06-08, E06-09, E06-10, E04 (stable reference-model API, trace writer) · Behaviour change: none

## Goal

One function runs one case in one format, from the manifest to a verdict.
`bazel run` uses it through a thin CLI. It is the deep module of the framework:
callers pass a kernel, a case and a format, and everything in between is hidden.

## Background

- Today's run path is split across several modules:
  - `kernel_registry/cases.py:109-159` `run_case` (prepare, guard, assemble,
    `create_harness`, `app.run`, export, check);
  - `kernel_registry/base.py:68-124` `IpuApp.run` (debug, profiling, activation
    knobs);
  - `kernel_registry/registry.py:83-101` `create_state` (maps
    `ExecutionConfig` onto three wide-mode flags);
  - `kernel_registry/runner.py` (argparse front end).
- `runner.py:62-63` catches `TypeError`, `AttributeError`, `KeyError` and more,
  then exits 1 with one line. Harness bugs look like user errors.
- `ipu_emu/stats.py:76` already computes `effective_mac_utilization`, and
  `format_summary` prints it.
- Q80: run to `HALT`, unpack, reference, compare, report cycles/MAC
  utilisation, optional trace. Q59: two error kinds, hardware faults (F-1..F-9)
  and configuration errors.

## Implementation

Module `ipu_harness/runner.py`:

```python
@dataclass(frozen=True)
class RunOptions:
    max_cycles: int | None = None
    inputs: Mapping[str, Path] = {}          # --input x=path.npy
    output_dir: Path | None = None           # --output: write <name>.npy per output (even on failure)
    trace: Path | None = None                # --trace: E04 trace writer (JSON schema from E04)
    debug: DebugLaunch | None = None         # E06-12
    alias_profile: bool = False              # E06-12

@dataclass(frozen=True)
class RunResult:
    kernel: str
    case: str
    format: ElementFormat
    cycles: int
    mac_utilization: float
    outputs: Mapping[str, np.ndarray]
    raw_outputs: Mapping[str, bytes]          # output regions' XMEM bytes before unpack (E07 proofs)
    comparisons: tuple[Comparison, ...]       # empty if no golden (bazel run only)
    fault: HardwareFault | None               # E04 fault record (ID + PC), if the run faulted
    @property
    def passed(self) -> bool: ...

def run_kernel(manifest: KernelManifest, case: Case, fmt: ElementFormat,
               options: RunOptions = RunOptions(), *, session: AssemblySession | None = None) -> RunResult:
    plan = plan_case(manifest, case.values, fmt)                       # E06-03, once
    inputs = inputs_for(case, plan, overrides=options.inputs)         # E06-08
    placement = place(plan, row_counts(plan))                         # E06-04
    program = assemble(plan, session or default_session())           # E06-07
    model = reference_model(format=fmt)                               # E04 stable API: reset(format)
    model.load_program(program)
    for name, a in inputs.items(): model.write_xmem_rows(placement.region(name).base_row, pack(...))
    for w in cr_writes(plan, placement): model.set_reg(f"CR{w.cr}", w.value)
    model.run(max_cycles=...)                                         # to HALT, or fault / limit
    outputs = {o: unpack(...) for o in plan.outputs}
    comparisons = compare_all(case, plan, inputs, outputs)            # E06-09/10
    return RunResult(...)
```

CLI (`python -m ipu_harness.cli <kernel_dir> ...`, wrapped by the `ipu_kernel()`
target for `bazel run`):

```
--case NAME            (default: default; sweep points by full name, e.g. 'sweep[rows=7]')
--format FMT           (default: first in [kernel] formats; 'all' runs every format)
--input NAME=PATH.npy  (repeatable)
--output DIR           write outputs as .npy
--trace PATH           write a per-cycle trace
--max-cycles N
--list-cases
--debug                (E06-12)
--profile-aliases      (E06-12)
```

Output, one line per run, then the stats summary:
`scale/one_row/e4: PASS  cycles=9  MAC util=…%  y: max|err|=0 (FP8_ULP)`.

Error handling:
- `ConfigurationError` and `OSError` exit 1 with a one-line message.
- A hardware fault exits 1 with the fault ID, the PC and the source line.
- Anything else is a bug: it prints a traceback and exits 2.

## Tests

- [ ] `run_kernel` on `scale` returns `passed` for every case and format, with `cycles > 0` and `0 < mac_utilization <= 1`.
- [ ] `--input x=...` overrides the generator, and `--output` writes `y.npy` even when the comparison fails.
- [ ] `--trace` writes a file that validates against the E04 trace schema.
- [ ] Not reaching `HALT` within `max_cycles` gives a clear error (the E04 cycle-limit fault).
- [ ] A planted `AttributeError` in a generator exits 2 with a traceback.
- [ ] The runner does not import `torch` when the case has an expected file and no reference.

## Acceptance Criteria

- [ ] `run_kernel` is the only execution path for kernels in `ipu_harness`. It uses only the E04 stable API, with no `IpuState` internals and no wide-mode flags.
- [ ] `bazel test //...` is green.

## Open question

- **Benchmarks.** 19 kernels have a `benchmark.py` with `CONFIGS` and a shared
  runner that writes `results.md` (`kernel_registry/benchmarking.py`). Proposal:
  a `[benchmarks]` table of sweep points in `kernel.toml`, and
  `bazel run //kernels/<family>/<name>:benchmark`. Q71/E01 say benchmarks are
  tracked without a gate. Confirm the format before E07 migrates the 19 files.
