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
bazel run //src/ipu-as:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin" --define rows=8
```

`assemble` writes `bin` for an `--output` ending in `.bin` and `mem`
otherwise, unless `--format` says which. `--target hardware` assembles the
hardware slots only, and an emulator-only instruction such as `BREAK` is then an
error. `check` exits 0 for a clean source and 1 otherwise; `format` exits 1
under `--check` when the source would change and 3 when it cannot format it.

--8<-- "generated/cli/ipu-as.md"

## Kernel runner: kernel targets and `run_kernel`

Each kernel under `kernels/` is a Bazel target that runs one of its cases end
to end, in one element format or all of them, and checks every output against
its golden: `bazel run //kernels/<family>/<name>`, or
`//kernels/<family>:<entry>` for a family's registry entry. The target passes
the kernel itself; the options below come after `--`. `run_kernel` takes the
kernel's folder instead, for a kernel outside the tree.

```bash
bazel run //kernels/softmax/softmax_rows -- --case constant_row --profile-aliases
bazel run //src/ipu-harness:run_kernel -- path/to/kernel --case default --format all
```

Each run prints `KERNEL/CASE/FORMAT: PASS  cycles=N  MAC util=P%`, then the
fault, if the run faulted, and each output's comparison with its expected
values. It exits 0 when every run passed; 1 when a run failed or faulted, or
the request was wrong (one line, no traceback); and 2 for a bug in the
runner or in the kernel's Python (with the traceback). `bazel test` on the
same label runs every case in every format.

--8<-- "generated/cli/run-kernel.md"

### Kernel cases

A case is a named set of parameter values in the kernel's `kernel.toml`
(`[cases]`), run in the formats it lists; a tuple of values is a sweep, one run
per point (`--case 'sweep[rows=7]'`). `--list-cases` lists a kernel's cases.
To run other values, add a case ([Write a kernel](../how-to/write-a-kernel.md)).

--8<-- "generated/cli/kernel-cases.md"

### Run statistics

`MAC util` is the effective MAC utilization of the run, from the counters the
reference model keeps for one run (`ReferenceModel.stats()`, the
[reference-model API](reference-model-api.md)); `--benchmark` reports them for
each `[benchmark]` point:

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

The counters describe one run: loading a program resets them, so a second
program on the same model reports only itself.

A run that reaches its cycle limit faults F-9, with the PC of the word that
would have run next. The default limit is `ipu_emu.emulator.DEFAULT_MAX_CYCLES`
(1 000 000); `--max-cycles N` sets it for one run, and 0 removes it. A run that
executed an [ISA idiom](../explanation/isa-idioms.md) counts it:
`--profile-aliases` prints the idioms' measurements, and `--alias-report FILE`
writes them as JSON.

## Debugger

The debugger is the kernel runner in a terminal user interface: add
`--config=debug` to `bazel run` on any kernel target. It stops at cycle 0,
before the first word runs, and takes the same options as an ordinary run.

```bash
bazel run --config=debug //kernels/reshape/identity -- --case single_row
```

`--config=debug` sets `IPU_DEBUG_TUI=1` (`.bazelrc`). An interactive terminal
is required. [Debug a kernel](../how-to/debug-a-kernel.md) explains the panes and the tab
editor. The controls, from the debugger's own key table:

--8<-- "generated/cli/debugger-keys.md"

## Trace recorder

The kernel runner records a case itself: `--trace` writes its trace (JSON
Lines, the [trace schema](reference-model-api.md)), and `--context` adds the
context the [Cycle Inspector](../how-to/step-through-a-kernel-run.md) needs beside it.

```bash
bazel run //kernels/reshape/identity -- --case single_row --trace "$PWD/trace.jsonl" --context "$PWD/context.json"
```

## Kernel query: `query`

`query` asks the kernel registry which kernel computes an operation, and why.
With no operation it prints the coverage report.

```bash
bazel run //kernels:query -- maxpool2d channels=2 height=8 width=8 kernel_size=7 stride=1 padding=3
```

`NAME=VALUE` parameters go to the registry as given: a value with a comma is
a tuple of integers, an integer is an `int`, a number with a point a `float`,
`true` and `false` are booleans, and anything else is a string; `format=e4`
limits the answer to one element format. It exits 0 when the query is covered
(for a sweep, every value) and 1 when it is not.

--8<-- "generated/cli/query.md"
