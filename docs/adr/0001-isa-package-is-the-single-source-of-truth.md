# The ISA package is the single source of truth

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`._

Everything about the instruction set (instructions, registers, operand types, slots, element formats, timing rules, fault list, binary encoding) is defined once, as data, in the `ipu_isa` package. The assembler, emulator, SystemVerilog package, docs and VS Code data are all derived from it, and adding an instruction means editing the ISA package, writing one handler and one test. Authors keep writing nested dicts, because they are easy to edit; a strict schema check rejects unknown keys and consumers read a frozen typed view. `ipu_isa` depends on nothing in this repo; in particular it never imports the assembler.

When the emulator, the spec text and the RTL documents disagree, the declared data in `ipu_isa` wins and the other party has a bug.
