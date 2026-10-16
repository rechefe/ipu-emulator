# CLAUDE.md — Project Directives for Claude Code

## Build & Test

```bash
bazel test //...                                          # run the default test tier
bazel test --config=full //...                            # include the heavy (nightly) tier
bazel test //src/tools/ipu-emu-py:test_execute           # emulator tests only
bazel test //src/tools/ipu-as-py:test_assemble           # assembler tests only
bazel test //src/tools/ipu-apps:test_fully_connected     # app tests only
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin"  # format from the suffix; --define KEY=VALUE, -I DIR
bazel test --config=proto //...                          # prototyping mode (= --define=ipu_proto=1); CI runs both modes
bazel build --config=docs //docs:build_docs              # docs site, offline (bazel-bin/docs/site.tar)
bazel test //tools:lint                                  # ruff lint + format check (ruff.toml; manual until E01-10)
bazel test --test_tag_filters=typecheck //...            # every mypy_test (tools/typecheck.bzl, mypy.ini)
bazel run //tools:ruff -- check --fix                    # auto-fix lint with the locked ruff
bazel run //tools:ruff -- format                         # format with the locked ruff
tools/update_baselines.sh [--with-kernels]               # regenerate every baseline, list the changed files
```

Baselines: a refactor PR changes no baseline file; a behaviour-change PR regenerates only the baselines it changes and says why each changed ([Code conventions, Baselines](docs/content/code-conventions.md#baselines)).

Use `bazel`, not `pip install` or `python` directly.

## Key Constraints

- **Never assign opcodes manually.** They are derived from position in `instruction_spec.py`.
- **Never duplicate instruction metadata.** `instruction_spec.py` is the single source of truth for both the assembler and emulator.
- **Operand names in `execute_*` handlers must exactly match** the `"name"` fields in `instruction_spec.py`.
- **`cr15` conventionally holds dstructure configuration** (`valid_elements`, `partition`, `pad_mode`), but any `CR0`–`CR15` is a valid ISA operand — `cr15` is not blocked as a general-purpose register.
- **Docs and written examples:** instruction mnemonics are **upper case**; hardware register names (`R0`, `R1`, `LR0`–`LR15`, `CR0`–`CR15`) are **upper case**; abstract operand placeholder tokens (`dest`, `src_a`) are **lower case** (the assembler still accepts any case for mnemonics). Operands in assembly commands are separated by **spaces**. Pseudocode data-size comments use **elements**, not bytes. Every fenced `asm` block must assemble (`//docs:test_doc_asm_blocks`); mark a deliberately incomplete one with `<!-- asm-fragment: reason -->` on the line before its fence.

## Adding an Instruction (quick checklist)

1. Add entry to `instruction_spec.py` (no opcode assignment — position determines it).
2. Add `execute_<name>(self, *, ...)` handler in `ipu.py` with keyword-only args matching operand names.
3. Write a test in `ipu-emu-py/test/` using `_run()`.
4. Run `bazel test //...`.
5. Check the generated SystemVerilog package (below).

The "one edit" tests prove these steps are enough: `//src/tools/ipu-apps:test_one_edit_end_to_end` adds a dummy instruction with one spec entry and one handler and runs it; `//src/tools/ipu-isa:test_one_edit` and `//src/tools/ipu-gen:test_one_edit_generators` check every derived artifact. Their `KNOWN_MANUAL_EDITS` list (empty) names any other edit still needed.

## Generated SystemVerilog Package

A downstream RTL repo vendors `ipu_instr_pkg.sv` verbatim. After changing `instruction_spec.py`, `union_layout.py`, `ipu_gen/sv_package.py` or `ipu_instr_pkg.sv.j2`:

```bash
bazel test //src/tools/ipu-gen:test_sv_package_compiles     # slang elaborates it with zero diagnostics
bazel test //src/tools/ipu-as-py:test_sv_package_golden     # text golden + every *_WIDTH pinned by name
bazel build //docs:generate_instruction_format_artifacts    # writes bazel-bin/docs/ipu_instr_pkg.sv
tools/sv_localparam_diff.sh <base-ref>                      # diff localparam lines vs the base commit
```

`sv_localparam_diff.sh` prints nothing and exits 0 when the localparams are identical. After an intended change to the package, regenerate the golden with `bazel run //src/tools/ipu-as-py:update_sv_golden`; a width change also needs a hand edit of `EXPECTED_LOCALPARAMS` in `test_sv_package_golden.py`. Every `*_WIDTH` is wire format: keep each one identical, or call the change out in the PR as breaking. Operand `"name"` fields become SV field names (`<name>_<column>`), so renaming an operand renames a downstream field.

## Project Knowledge

[SKILL.md](SKILL.md) is the index: for each task or topic it names the page that covers it.

Read [Code conventions](docs/content/code-conventions.md) before writing or reviewing code. Terms: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).
Before opening or reviewing a PR, read [Code review](docs/content/code-review.md); the PR template is [.github/pull_request_template.md](.github/pull_request_template.md).

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues (rechefe/ipu-emulator). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
