# IPU Emulator — agent index

ipu-emulator defines the instruction set of the IPU, a 128-lane VLIW vector processor for neural-network inference. It assembles programs for it, emulates it in Python, and holds the kernels that run on it; Bazel builds and tests everything.

`docs/` is canonical. This file only says where each topic is written; follow the link instead of relying on memory of the code.

## Rules

The rules agents must never break are in [CLAUDE.md](CLAUDE.md), under "Key Constraints" and "Generated SystemVerilog Package". They are explained in:

- Opcodes from position, no duplicated metadata, handler argument names: [Add an instruction](docs/content/how-to/add-an-instruction.md) and [ADR 0001](docs/adr/0001-isa-package-is-the-single-source-of-truth.md).
- Docs conventions and SystemVerilog package widths: CLAUDE.md is their only home.

## Where each topic is documented

| Task or topic | Page |
|---|---|
| Build and run tests | [CLAUDE.md](CLAUDE.md), "Build & Test" |
| Packages and where the code lives | [README.md](README.md), "Python Packages"; kernels: [adding-applications.md](docs/content/adding-applications.md) |
| Write a kernel | [building-applications.md](docs/content/building-applications.md) |
| Register a kernel (folder contract, Bazel targets) | [adding-applications.md](docs/content/adding-applications.md) |
| Kernel catalogue | [kernels/index.md](docs/content/kernels/index.md) |
| Add an instruction or pseudo-instruction | [Add an instruction](docs/content/how-to/add-an-instruction.md) |
| Debug a kernel | [debugging.md](docs/content/debugging.md) |
| Instructions, slots and operands | generated [Instructions](https://rechefe.github.io/ipu-emulator/dev/reference/isa/instructions/) and [Operand types](https://rechefe.github.io/ipu-emulator/dev/reference/isa/operand-types/) |
| Assembly syntax, registers, instruction words | generated [Assembly syntax](https://rechefe.github.io/ipu-emulator/dev/reference/isa/assembly-syntax/) |
| Bit layout of the instruction word | generated [Instruction format](https://rechefe.github.io/ipu-emulator/dev/reference/isa/instruction-format/) |
| Pseudo-instructions | generated [Programmer's guide](https://rechefe.github.io/ipu-emulator/dev/reference/isa/programmer-guide/) |
| Registers | generated [Registers](https://rechefe.github.io/ipu-emulator/dev/reference/isa/registers/) |
| CR constants, dstructure | generated [Configuration registers](https://rechefe.github.io/ipu-emulator/dev/reference/isa/configuration-registers/) |
| Element formats | [element-formats.md](docs/content/explanation/element-formats.md) |
| Command-line tools | [reference/cli.md](docs/content/reference/cli.md) (stub, E09-15); meanwhile `bazel run //src/tools/ipu-as-py:ipu-as -- --help` |
| Reference-model API | [reference/reference-model-api.md](docs/content/reference/reference-model-api.md); the decision: [ADR 0004](docs/adr/0004-stable-reference-model-api.md) |
| Driving the reference model from a cocotb testbench | [how-to/reference-model-cocotb.md](docs/content/how-to/reference-model-cocotb.md) |
| Conformance suite (cases as data, running them in an RTL testbench) | [src/tools/ipu-isa/conformance/README.md](src/tools/ipu-isa/conformance/README.md) |
| Architecture | [explanation/architecture.md](docs/content/explanation/architecture.md) (stub, E09-17); meanwhile [docs/adr/](docs/adr/) |
| Timing model (cycle-start and forwarded values) | [explanation/timing-model.md](docs/content/explanation/timing-model.md); decision in [ADR 0002](docs/adr/0002-timing-contract.md) |
| Prototyping mode | [prototyping-mode.md](docs/content/prototyping-mode.md) |
| ISA idioms and kernel coverage | [isa-alias-catalogue.md](docs/content/isa-alias-catalogue.md), [app-coverage.md](docs/content/app-coverage.md) |
| Hardware specs and requirements | [specs/](docs/content/specs/), [requirements/index.md](docs/content/requirements/index.md) |
| Open questions | [requirements/open-questions.md](docs/content/requirements/open-questions.md) |
| Terms | [GLOSSARY.md](GLOSSARY.md) |
| Design decisions | [docs/adr/](docs/adr/) |
| Code conventions | [code-conventions.md](docs/content/code-conventions.md) |
| Benchmarks (speed, memory, CI time) | [tools/bench/README.md](tools/bench/README.md) |
| Open or review a PR | [code-review.md](docs/content/code-review.md), [PR template](.github/pull_request_template.md) |
