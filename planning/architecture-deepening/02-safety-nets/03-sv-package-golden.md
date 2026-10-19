# SV package golden with a pinned localparam table

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-03 · Size S · Depends on: E01 (toolchain pin) · Behaviour change: none · `*_WIDTH` localparams: unchanged (this issue only records them)

## Goal

A downstream RTL repo vendors `ipu_instr_pkg.sv` verbatim (CLAUDE.md). Lock its content with a golden test, and make every `*_WIDTH` localparam an explicit, named expectation, so a change fails with a message such as `IPU_COMPOUND_INST_WIDTH: 186 -> 155`. Add the base-commit localparam diff that CLAUDE.md asks reviewers to run, as one command.

## Background

- `gen_codegen.build_codegen_context(source_commit)` and `render_template("ipu_instr_pkg.sv.j2")` (`ipu_as/gen_codegen.py:379-406`) render the package. `generate_sv_package` stamps the commit from `bazel-out/stable-status.txt` (`docs/BUILD.bazel:43-51`, `stamp = 1`), so the file differs on every commit even when no type changes.
- Existing tests check structure only: `test_gen_codegen.py` (for example `IPU_COMPOUND_INST_WIDTH = {CompoundInst.bits()}`, which moves with the code) and `test_sv_package_compiles.py` (slang elaboration). Neither pins a value.
- Localparams at HEAD 694a8a8 (`bazel-bin/docs/ipu_instr_pkg.sv`):

  | Slot | `*_SLOT_U_WIDTH` | `*_SLOT_T_WIDTH` |
  |---|---|---|
  | LOAD | 12 | 14 |
  | STORE | 8 | 9 |
  | ACC_STORE | 8 | 9 |
  | LR | 16 | 20 |
  | MULT | 20 | 23 |
  | ACC | 13 | 17 |
  | AAQ | 8 | 9 |
  | COND | 20 | 23 |
  | BREAK | 20 | 22 |

  `IPU_COMPOUND_INST_WIDTH = 186`.

## Implementation

1. Add `src/tools/ipu-as-py/test/golden/ipu_instr_pkg.sv`: the package rendered with `source_commit="unknown"`.
2. Add `src/tools/ipu-as-py/test/test_sv_package_golden.py`:
   - `test_package_text_matches_golden`: render with `source_commit="unknown"` and compare with the golden file. On failure print a unified diff.
   - `test_localparams_match_table`: parse every `localparam int unsigned NAME = N;` line and compare with an explicit dict in the test file (the table above). The failure lists each changed name with old and new value, and each added or removed name.
   - `test_golden_localparams_match_table`: the same dict against the golden file, so the two cannot drift apart.
3. Add `py_binary(name = "update_sv_golden")`: rewrites the golden file. It does not touch the dict; a width change must be an explicit edit to the test, which makes it visible in review.
4. Add `tools/sv_localparam_diff.sh <base-ref>`: builds `//docs:generate_instruction_format_artifacts` at `<base-ref>` (in a `git worktree`) and at the working tree, extracts the `localparam` lines and prints a diff. Exit code 0 when identical, 1 otherwise. Reference it from CLAUDE.md's "Generated SystemVerilog Package" section and from the PR template (E02-09).

## Tests

- [ ] The three tests pass on the base commit.
- [ ] Changing `_SLOT_TARGET_BITS["lr"]` from 20 to 21 in a scratch branch fails `test_localparams_match_table` with a message naming `LR_SLOT_T_WIDTH` and `IPU_COMPOUND_INST_WIDTH` (manual check, recorded in the PR).
- [ ] `tools/sv_localparam_diff.sh HEAD` prints nothing and exits 0.

## Acceptance Criteria

- [ ] Any change to the rendered package fails a default-tier test with a readable diff.
- [ ] Every `*_WIDTH` localparam is pinned by name in one dict.
- [ ] The base-commit diff is one command, linked from CLAUDE.md.

## Risks

- None for behaviour. Later issues that change the package on purpose (E03 tags and SV hardware set, E03 mnemonic rename, E03 header hash) regenerate the golden file, and only the hardware-set issue edits the localparam dict.
