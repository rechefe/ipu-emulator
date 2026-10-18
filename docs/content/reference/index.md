# Reference

These pages are for looking things up: instructions, operands, commands and terms.

| Page | What it covers |
|---|---|
| [Instructions](isa/instructions.md) | Every instruction, by slot, generated from `instruction_spec.py`. |
| [Operand types](isa/operand-types.md) | The operand types the instruction reference uses. |
| [Assembly syntax](isa/assembly-syntax.md) | Instruction words, separators, labels, registers and Jinja2 preprocessing. |
| [Instruction format](isa/instruction-format.md) | The bit layout of the instruction word, and the SystemVerilog package download. |
| [Programmer's guide](isa/programmer-guide.md) | The pseudo-instructions the assembler expands, such as `BGT.S` and `BZ`. |
| [Registers](isa/registers.md) | Every register, its size and its assembly names. |
| [Configuration registers](isa/configuration-registers.md) | The CR constants, the 32-bit scalar rule and the dstructure fields. |
| [`kernel.toml`](kernel-toml.md) | Every key of a kernel's `kernel.toml` and a family's `registry.toml`, generated from the loader. |
| [Command-line tools](cli.md) | Every command-line tool (`ipu-as`, the kernel runner and the debugger), with its subcommands and options. |
| [Reference model API](reference-model-api.md) | The stable emulator API and the trace schema. |
| [Kernels](../kernels/index.md) | The kernel catalogue, by family. |
| [Code conventions](code-conventions.md) | How code in this repository is written, and what checks each rule. |
| [Code review](code-review.md) | What a PR author states and runs, and what reviewers check beyond CI. |
| [Glossary](glossary.md) | The terms these docs use. |
