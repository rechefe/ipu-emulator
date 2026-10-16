# Disassembler in `ipu_isa`; emulator sources stop importing the assembler

Part of [Epic E03: ISA package](00-epic.md) · ID E03-18 · Size S · Depends on: E03-17 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Disassembly (word to text) is derived from the codec and the derived syntax, so it belongs in `ipu_isa`. Move it there, byte-identical to today's text, and switch the emulator's debug modules to it. After this issue no file under `src/tools/ipu-emu-py/src/` imports `ipu_as` (Q69).

## Background

- `CompoundInst.decode(word)` (`ipu_as/compound_inst.py:124-133`) joins per-slot `Inst.decode` text with `";\n\t\t\t"` and ends with `";;"`. Each token class renders its own operand (`ipu_token.py:78`, `:130`, `:198`; `reg.py:37`; `immediate.py:68`, `:125`, `:205`, `:257`).
- Emulator callers: `debug_cli.disassemble_at` (`ipu_emu/debug_cli.py:361-380`, function-level import of `CompoundInst`) and `debug_pipeline.stage_operations` (`debug_pipeline.py:4`, `:15-25`), which splits the text on `;` and re-binds tokens to operands in `semantic_operations` (`:35-55`).
- E02-02 pins the disassembly text of every word in the corpus.
- E04-18 removes the `ipu_as` Bazel dep from `ipu_emu_lib` and E08 moves the debuggers to `ipu_debug`; both need this issue first.

## Implementation

1. `ipu_isa/codec/disasm.py`:
   ```python
   def operand_text(op: Operand, value: int) -> str          # register and enum names from the registries, numbers as today
   def slot_text(decoded: DecodedSlot) -> str                 # "MNEMONIC op op ..." in today's spelling and case
   def disassemble(word: int, *, isa: Isa | None = None) -> str   # today's CompoundInst.decode format
   def slot_operations(word: int) -> list[tuple[str, str]]    # (instance label, text), for the pipeline view
   ```
2. `CompoundInst.decode` delegates to `disassemble` (kept until E05 retires `CompoundInst`).
3. `debug_cli.disassemble_at` and `debug_pipeline.stage_operations` call `disassemble` / `slot_operations` on `DecodedWord.to_word()`. Delete the text splitting and token re-binding in `semantic_operations`; keep its output identical.
4. Delete the temporary `ipu_emu` → `ipu_as` edge from `tools/import_graph.toml`, so `//tools:test_import_graph` fails if any file under `src/tools/ipu-emu-py/src/` imports `ipu_as`. (Tests may still use the assembler.)

## Tests

- [ ] E02-02 disassembly snapshots unchanged, now produced by `ipu_isa.disassemble`.
- [ ] `test_debug_cli.py`, `test_debug_tui.py` and the pipeline-view tests pass unchanged.

## Acceptance Criteria

- [ ] `grep -rn "ipu_as" src/tools/ipu-emu-py/src` finds nothing.
- [ ] Disassembly text is produced in one place.
