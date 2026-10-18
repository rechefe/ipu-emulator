# Architecture

The toolchain is eight Python packages under `src/tools/`, one directory each.
Each package has one responsibility, and dependencies point one way: towards
`ipu_isa`, the instruction set, which imports no other package. Terms such as
instruction word, slot, commit record and harness are defined in the
[glossary](../reference/glossary.md); the decisions behind the design are the
[architecture decision records](https://github.com/rechefe/ipu-emulator/tree/master/docs/adr)
in `docs/adr/`; the rules for writing code in these packages are the
[code conventions](../code-conventions.md).

## Packages

An arrow points from a package to a package it imports. The diagram leaves
out an arrow that another path already implies: `ipu_harness` imports
`ipu_isa` directly, and the path through `ipu_as` stands for it.

--8<-- "generated/architecture/dependencies.md"

--8<-- "generated/architecture/packages.md"

The diagram and the table are generated at docs build time
(`docs/architecture_page.py`) from `tools/import_graph.toml` and the first
paragraph of each package's `__init__.py` docstring. A renamed or added
package changes this page with no doc edit.

`tools/import_graph.toml` is the one list of allowed edges.
`//tools:test_import_graph` reads every source of every package and fails on a
first-party import its package's row does not list, including an import inside
a function. Each package's Bazel `deps` enforce the same edges at build time:
an import outside them fails in the sandbox.

Two edges are absent by design. Nothing imports `ipu_gen`: it reads the ISA
and the assembler's parser and writes files. `ipu_harness` never imports `ipu_kernels`: the registry, pack
and reference lookups load kernel-side modules by a name read from registry
data, as plugins.

## One source of truth

`src/tools/ipu-isa/src/ipu_isa/instruction_spec.py` defines every instruction:
its slot, operands, timing declaration, documentation and handler name. An
instruction's opcode is its position in its slot's table, never a number
written by hand. Every other package reads the ISA through the typed view
`ipu_isa.view.isa()`; only `ipu_isa`'s loader reads the authoring tables
([ADR 0001](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0001-isa-package-is-the-single-source-of-truth.md)).

| Derived artifact | Built by | From the ISA it takes |
|---|---|---|
| Word layout, slot codec, disassembly | `ipu_isa.codec` | Field widths, opcodes, operand types |
| Assembler grammar and encoder | `ipu_as` | Mnemonics, operand names and types |
| Emulator dispatch | `ipu_emu` (`plans_for` in `ipu.py`) | One plan per opcode: its `execute_<name>` handler, operand fields, register reads and fault checks |
| SystemVerilog package | `ipu_gen.sv_package` | The word layout and every `*_WIDTH` |
| ISA reference pages and layout SVGs | `ipu_gen.docs_pages`, `ipu_gen.layout_svg` | Instruction docs, operand types, registers |
| VS Code language data | `ipu_gen.vscode_data` | Mnemonics and register names |

The "one edit" test proves it. It adds a dummy instruction, `TESTNEG`, as one
spec entry and one handler, and checks that every artifact follows:
`//src/tools/ipu-isa:test_one_edit` for the ISA's own artifacts,
`src/tools/ipu-gen/test/test_one_edit_generators.py` for the generators, and
`src/tools/ipu-emu/test/test_one_edit_end_to_end.py` for an assembled program
running on the emulator. Its list of manual edits beyond the entry and the
handler, `KNOWN_MANUAL_EDITS`, is empty.
[How-to: add an instruction](../how-to/add-an-instruction.md) walks through
the same edit.

## Life of a program

```mermaid
flowchart LR
    src[".asm source"] --> session["AssemblySession<br/>(ipu_as)"]
    session --> codec["word codec<br/>(ipu_isa.codec)"]
    codec --> load["load_program<br/>(ipu_emu.reference)"]
    load --> cycle["Cycle: one word<br/>(ipu_emu.cycle)"]
    cycle --> record["commit record"]
    record --> compare["compare with golden<br/>(ipu_harness)"]
```

1. **Assemble.** `ipu_as.session.AssemblySession` renders the source's Jinja2
   layer once, parses it, resolves labels and encodes each instruction word.
   Every failure is an `AssemblerError` with a source location.
2. **Encode.** The word codec in `ipu_isa.codec.word` packs the words into a
   program file. The emulator decodes with the same codec, so the assembler
   and the emulator never hold two copies of the layout.
3. **Load.** `ReferenceModel.load_program` (`ipu_emu.reference`) writes the
   words into instruction memory. The harness also writes the inputs to XMEM
   and sets the CRs its ABI names.
4. **Plan the word.** At each pc, `schedule_word` (`ipu_emu.cycle`) turns the
   word into a `Schedule`: the issued instruction of each slot, in slot
   execution order, with any F-1 or F-3 fault found before a slot runs. The
   schedule is a pure function of the word, so it is built once per pc and
   cached until that instruction memory entry is written.
5. **Run one cycle.** `Cycle` (`ipu_emu.cycle`) is the one module that executes
   an instruction word. It takes the cycle-start register values, evaluates
   `BREAK`, runs the LR sub-slots and every other slot in order, and runs the
   fault checks after the LR slot. Each slot's instruction runs through its
   handler in `ipu_emu.slots` (`execute_<name>`, keyword arguments named after
   the spec's operands). A handler reads registers through `CycleView`, which
   returns the cycle-start or the forwarded value as the
   [timing contract](timing-model.md) declares.
6. **Commit.** The word retires as a commit record: the pc before and after,
   the instruction retired in each slot, the register and XMEM writes, and any
   halt or fault. `ReferenceModel.step` returns it; `run` writes each one to a
   trace when given a `TraceWriter`.
7. **Compare.** The harness (`ipu_harness.runner.run_kernel`) runs the program
   to `HALT`, reads the output regions back from XMEM, unpacks them, and
   compares each with its golden at the kernel's tolerance tier
   (`ipu_harness.compare`). A faulting run returns its fault instead of a
   comparison.

## Faithful and prototyping execution

By default the emulator models what the hardware does, slot by slot: every
slot is dispatched even when it holds a `NOP`, every multiply applies its
element mask, and every cycle reads the full cycle-start register values.
That fidelity is what makes the emulator answerable about the chip.

Prototyping mode trades some of it for speed on long runs, where the question
is "what does this kernel compute" rather than "what does the chip do". It is
an adapter over the faithful path, not a second emulator: the same `Cycle`
runs both, and prototyping mode changes four things on it, all listed in
`ipu_emu.prototyping`.

| Shortcut | Why it is safe | Why it is not faithful |
|---|---|---|
| Schedules leave out slots holding an idle `NOP` (`skip_idle_nops`) | Their handlers do nothing | The machine still issues the slot |
| Read operands go through the packed resolver (`ipu_emu.resolver.PackedResolver`) | The values and invalid-index exceptions match the generated accessors | It bypasses the normal register-access machinery |
| A `MULT` with no mask shift and an all-ones mask skips the mask pipeline | It deactivates no lane, so `MULT_RES` is unchanged | The mask is applied on every multiply in hardware |
| A full-row FP32 `ACTIVATE.QUANTIZE` without output quantization runs as one batched row | It keeps the scalar arithmetic, the FP32 store boundary, exceptions and partial writes | The hardware activation pipeline works lane by lane |

The cond slot's `NOP` is never skipped, because `execute_cond_nop` advances
the pc.

### What it guarantees

The state after every cycle is bit-identical to the faithful path: every
register, all of XMEM, the cycle count and the run statistics.
`src/tools/ipu-emu/test/test_prototyping_equivalence.py` runs each program
in lockstep, once per mode, and compares every register, all 512 MiB of XMEM,
cycle outcomes, the cycle-start register values and run statistics after
every cycle. It also runs every register-timing probe
(`src/tools/ipu-emu/test/timing_probes.py`, one minimal program per row of
the timing tables) in every element format the probe covers.

At kernel level, each case's baseline record (`baseline.json` beside each
kernel's `test.py`) is recorded in a default build and checked unchanged in a
flagged one. Each case's program, setup, output, final state and cycle count
must be identical in both modes.

Prototyping mode does not guarantee that the work the emulator did resembles
the work the chip would do. Trust the numbers a prototyping run produces; do
not reason from it about what a cycle costs the hardware.

### Enabling it

```bash
bazel test --config=proto //...
```

`--config=proto` (`.bazelrc`) is the short form of `--define=ipu_proto=1`.
The Bazel flag is mandatory: a default build and a plain checkout cannot
activate the shortcuts, and requesting them there raises `ValueError`.

| Level | How | Use |
|---|---|---|
| Build gate | `--config=proto` (`--define=ipu_proto=1`) | Required; prototyping mode is the default in that build |
| Per state | `IpuState(prototyping=True/False)` | Select a mode inside a flagged build |
| Per process | `IPU_EMU_PROTOTYPING=1/0` | Select the default inside a flagged build |

`IPU_EMU_PROTOTYPING` is read case-insensitively, with surrounding whitespace
ignored. `0`, `false`, `no`, `off` and the empty string mean off, in any
build. `1`, `true`, `yes` and `on` mean on. Any other value raises
`ValueError`.

CI runs the default tier in both modes on every pull request. The `proto` job
fails if any case of `test_prototyping_equivalence` is skipped. The nightly
workflow runs the heavy tier with the flag too.

### What it is worth

The representative workload is FP32 `softmax_rows` with 128 rows. Assembly and
input preparation are outside the timed region. Each version received two
independent warm-ups followed by 14 measured runs in total:

| Version | Median time | Throughput | Gain vs master |
|---|---:|---:|---:|
| `master` (`ea7613e`) | 0.7197 s | 3,228 cycles/s | baseline |
| `performance-fix` (`b072bb3`) | 0.1145 s | 20,280 cycles/s | 6.28x |
| Prototyping mode | 0.07118 s | 32,636 cycles/s | 10.11x |

That is 1.61x over `performance-fix`, or 37.9% less wall time. All versions
executed 2,323 cycles and produced the same output SHA-256:

`16c5777b09191f41d5f965097b0a29372cb42eb1d608d1450045989bcd6549ae`

These figures describe one machine and workload, not a universal speedup.
Kernels with partial activation rows or other activation functions take the
scalar activation path; other instruction mixes gain in other proportions.

## Stable and internal code

Only the reference model API is stable: the names of `ipu_emu.reference`, the
commit records its `step()` returns and the commit record and trace schemas.
The [Reference model API](../reference/reference-model-api.md) page lists
them and says what a breaking change is. Everything else in `ipu_emu`, and
every other package's Python API, is internal and can change in any release
([ADR 0004](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0004-stable-reference-model-api.md)).
The harness drives the emulator through the reference model API only, like an
RTL testbench.

## Where to change what

| Task | Package | Start at |
|---|---|---|
| Add or change an instruction, operand type or register | `ipu_isa`, then a handler in `ipu_emu` | [How-to: add an instruction](../how-to/add-an-instruction.md) |
| Change the instruction word layout or the program file format | `ipu_isa` (`codec`) | `CLAUDE.md`, "Generated SystemVerilog Package": widths are wire format |
| Change the assembly syntax, diagnostics or formatter | `ipu_as` | `src/tools/ipu-as/src/ipu_as/asm_grammar.lark`, `session.py` |
| Change what an instruction does in a cycle | `ipu_emu` (`slots/`, `cycle.py`) | [Timing model](timing-model.md) |
| Change the stable API for RTL verification | `ipu_emu` (`reference.py`) | [Reference model API](../reference/reference-model-api.md) |
| Change the debugger | `ipu_debug` | [How-to: debug a kernel](../how-to/debug-a-kernel.md) |
| Change a generated artifact (SV package, ISA pages, VS Code data) | `ipu_gen` | `src/tools/ipu-gen/src/ipu_gen/cli.py` |
| Change how kernels are declared, placed, packed, run or compared | `ipu_harness` | [ADR 0006](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0006-kernels-are-assembly-plus-declaration.md) |
| Share a reference or pack transform between kernels | `ipu_kernels` | `src/tools/ipu-kernels/src/ipu_kernels/__init__.py` |
| Add a package or an edge between packages | `tools/import_graph.toml` | Its header lists every registration a new package needs |
