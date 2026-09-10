# Debug (GDB), run stats, tracing, and user documentation

Part of the [ETISS integration epic](00-epic.md). Depends on #5.

## Goal

Reach feature parity with the Python emulator's tooling where it matters
(stepping, inspecting registers, saving state, run statistics) and document
the ETISS backend for users.

## Implementation

- **GDB server.** Enable ETISS's `gdbserver` plugin from the runner
  (`--gdb-port`); provide `xml/` target description generated from
  `registers.py` (LR/CR/PC as 32-bit, vector registers as byte arrays) so
  `gdb` can `info registers` and `x/128xb &R0`. `BREAK`/`BKPT` map to the
  GDB breakpoint return code when `break_mode=stop`.
- **State dump.** `--dump-state state.json` writes the same schema as
  `debug_cli.state_to_json_dict` (+ XMEM side-car), so `debug_cli`'s
  `save`/`load` tooling can inspect ETISS runs.
- **Trace.** `--trace` enables `PrintInstruction` with the generated ASM
  printer (one disassembled VLIW per line, matching `CompoundInst.decode`
  formatting) — the ETISS analogue of `disassemble_current`.
- **Stats.** `stats.json` → `RunStats`; `format_summary()` prints identical
  text for both backends.
- **Docs.**
  - `docs/content/setup.md`: Boost/CMake prerequisites, `--build_tag_filters=-etiss`.
  - `docs/content/debugging.md`: GDB workflow, trace, state dumps.
  - `docs/content/adding-instruction.md`: step 2b "add the C handler in
    `IPUFuncs.c`" and the link error you get if you forget.
  - `docs/content/specs/etiss-integration.md`: replace "proposal" status with
    the as-built description and measured performance.
  - `SKILL.md` / `CLAUDE.md`: new package layout and test targets.

## Tests

- [ ] `sh_test`: start the runner with `--gdb-port`, connect with `gdb`
      batch mode, set a breakpoint on a word, `continue`, read `LR0`.
- [ ] `--dump-state` output loads with the existing debug-CLI JSON reader.
- [ ] Stats summary identical for both backends on the FC kernel.

## Acceptance Criteria

- [ ] Interactive stepping and register inspection work through GDB.
- [ ] State dumps and stats are interchangeable between backends.
- [ ] Docs updated; `bazel build //... --build_tag_filters=docs` passes.
