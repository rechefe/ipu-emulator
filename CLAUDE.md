# CLAUDE.md — Project Directives for Claude Code

## Build & Test

```bash
bazel test //...                                          # run all tests
bazel test //src/tools/ipu-emu-py:test_execute           # emulator tests only
bazel test //src/tools/ipu-as-py:test_assemble           # assembler tests only
bazel test //src/tools/ipu-apps:test_fully_connected     # app tests only
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin
```

Use `bazel`, not `pip install` or `python` directly.

## Key Constraints

- **Never assign opcodes manually.** They are derived from position in `instruction_spec.py`.
- **Never duplicate instruction metadata.** `instruction_spec.py` is the single source of truth for both the assembler and emulator.
- **Operand names in `execute_*` handlers must exactly match** the `"name"` fields in `instruction_spec.py`.
- **`cr15` is reserved** — never use it for application data.

## Adding an Instruction (quick checklist)

1. Add entry to `instruction_spec.py` (no opcode assignment — position determines it).
2. Add `execute_<name>(self, *, ...)` handler in `ipu.py` with keyword-only args matching operand names.
3. Write a test in `ipu-emu-py/test/` using `_run()`.
4. Run `bazel test //...`.

## Project Knowledge

See [SKILL.md](SKILL.md) for full architecture details, ISA reference, register file layout, and code examples.
