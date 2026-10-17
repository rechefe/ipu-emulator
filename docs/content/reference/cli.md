# Command-line tools

Every command-line tool in the repository, with its subcommands and options.
The option tables are generated at docs build time from each tool's own
option declarations (`docs/cli_reference_page.py`), so they list exactly what
the tool parses. Run any of them through `bazel run`; everything after `--`
goes to the tool. Relative paths are read from the directory you run `bazel`
in. Every tool also takes `--help`.

## Assembler: `ipu-as`

`ipu-as` assembles, disassembles, checks and formats IPU assembly. A source
can be a Jinja2 template: `--define` gives the values its `ipu:params` header
declares, and `-I` adds include directories after the source's own one.

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin" --define rows=8
```

`assemble` writes `bin` for an `--output` ending in `.bin` and `mem`
otherwise, unless `--format` says which. `--target hardware` assembles the
hardware slots only, and an emulator-only instruction such as `BREAK` is then an
error. `check` exits 0 for a clean source and 1 otherwise; `format` exits 1
under `--check` when the source would change and 3 when it cannot format it.

--8<-- "generated/cli/ipu-as.md"

## Kernel runner: kernel targets

Each kernel registered under `src/tools/ipu-apps` is a Bazel target that runs
one of its cases end to end and checks the output. The target passes the
kernel's name itself; the options below and the case options of the next
table come after `--`.

```bash
bazel run //src/tools/ipu-apps:softmax_rows -- --rows 8 --profile-aliases
```

A run that passes prints `KERNEL/CASE: PASS (N cycles)` and then the
[run statistics](#run-statistics). `bazel test` on the same label runs the
kernel's test suite.

--8<-- "generated/cli/kernel-targets.md"

### Case options

Each case declares its own options, one per default value it sets. A case
option overrides that value for one run; a boolean option is a pair of
switches. `--list-cases` lists a kernel's cases, and `--help` with `--case`
lists that case's options.

--8<-- "generated/cli/case-options.md"

### Run statistics

Every run prints a summary of `RunStats`, the counters the emulator keeps for
one run (`state.stats` in Python). For example:

--8<-- "generated/cli/run-summary.md"

| Counter | What it counts |
|---|---|
| Total cycles | Instruction words retired. |
| Mult active, Acc active | Cycles whose MULT or ACC slot was not a no-op, and their fraction of the total cycles. |
| XMEM reads | `LD.MULT`, `LD.MULT.CYCLIC` and `LD.MASK` executed. |
| XMEM writes | `ST.ACC` and `ST.AAQ` executed. |
| XMEM accesses | Reads plus writes. |
| Mult idle | Cycles whose MULT slot was a no-op. |
| Identity mult | Active multiply cycles that multiplied by a constant 1 to move data, and their fraction of the active cycles. |
| Lane ops | Multiplier lanes that passed the mask and lay within `valid_elements`, summed over every multiply; occupancy is that over the total cycles times 128 lanes. |
| Effective MAC | Lane ops less the identity multiplies' lanes, over the run's peak: the multiplier work that was real arithmetic. |

The counters describe one run: each run resets them before its first cycle,
so a second program on the same state reports only itself (the emulator's
`run_until_complete` and `run_with_debug` take `reset_stats=False` to add to
the earlier ones). The total cycles are set
even when the run raises: a fault, the cycle limit, or an emulator error. Runs
add up: `stage1_stats + state.stats`.

A run that reaches its cycle limit faults F-9, with the PC of the word that
would have run next. The default limit is `ipu_emu.emulator.DEFAULT_MAX_CYCLES`
(1 000 000); `--max-cycles N` sets it for one run, and 0 removes it.

A run in which every multiply used all 128 lanes adds a note: occupancy is
overstated if the kernel pads a narrower row without declaring
`CR15.valid_elements`. A run that executed an [ISA idiom](../isa-alias-catalogue.md)
adds a `--- ISA alias hits ---` block, with the count for each idiom, and
`--profile-aliases` (or `--alias-report FILE`) adds the full idiom
measurements.

## Kernel runner: `run_kernel`

`run_kernel` runs a case of a kernel described by a `kernel.toml`, in one
element format or all of them, and prints one report per run. A kernel under
`kernels/` has its own target, `bazel run //kernels/<family>/<name>` (or
`//kernels/<family>:<entry>` for a family entry), which takes the same options
with the kernel filled in; `run_kernel` takes the kernel's folder, for a kernel
outside the tree.

```bash
bazel run //src/tools/ipu-harness:run_kernel -- path/to/kernel --case default --format all
```

Each run prints `KERNEL/CASE/FORMAT: PASS  cycles=N  MAC util=P%`, then the
fault, if the run faulted, and each output's comparison with its expected
values. It exits 0 when every run passed; 1 when a run failed or faulted, or
the request was wrong (one line, no traceback); and 2 for a bug in the
runner or in the kernel's Python (with the traceback).

--8<-- "generated/cli/run-kernel.md"

## Debugger

The debugger is the kernel runner in a terminal user interface: add
`--config=debug` to `bazel run` on any kernel target. It stops at cycle 0,
before the first word runs, and takes the same options as an ordinary run.

```bash
bazel run --config=debug //src/tools/ipu-apps:identity -- --case single_row
```

`--config=debug` sets `IPU_DEBUG_TUI=1` (`.bazelrc`). An interactive terminal
is required. [Debug a kernel](../how-to/debug-a-kernel.md) explains the panes and the tab
editor. The controls, from the debugger's own key table:

--8<-- "generated/cli/debugger-keys.md"

## Trace recorder: `trace_case`

A kernel-tree kernel is recorded by the runner itself: `run_kernel` with
`--trace` and `--context` (above) writes the trace and the
[Cycle Inspector](../debugging-inspector.md)'s context. `trace_case` does the
same for a kernel still in `ipu_apps`, until its family migrates (E07).

`trace_case` runs one registered case and records its trace (JSON Lines, the
[trace schema](reference-model-api.md)) and the context the
[Cycle Inspector](../debugging-inspector.md) needs beside it.

```bash
bazel run //src/tools/ipu-apps:trace_case -- identity --case single_row --trace "$PWD/trace.jsonl" --context "$PWD/context.json"
```

--8<-- "generated/cli/trace-case.md"

## Kernel query: `query`

`query` asks the kernel registry which kernel handles a computation, and why.
With no operation it prints the coverage report.

```bash
bazel run //src/tools/ipu-apps:query -- maxpool2d shape=2,8,8 kernel_size=7 stride=1 padding=3
```

`NAME=VALUE` parameters go to the registry as given: a value with a comma is
a tuple of integers, an integer is an `int`, `true` and `false` are booleans,
and anything else is a string. It exits 0 when the query is covered (for a
sweep, every value) and 1 when it is not.

--8<-- "generated/cli/query.md"
