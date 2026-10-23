# CLI: drop the `--format` prompt, add `--define` and `-I`

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-10 · Size S · Depends on: E05-06, E05-09 · Behaviour change: yes (`--format` defaults from the output extension instead of prompting; new options)

## Goal

`ipu-as assemble` and `ipu-as check` work non-interactively and accept template values from the command line (L8). The interface keeps `--input` and `--output`, drops the `--format` prompt, and adds a repeatable `--define KEY=VALUE` and `-I DIR`.

## Background

- `cli.py:16-20` (`assemble`) and `cli.py:36-40` (`disassemble`) declare `--format` with `prompt="Choose a file format"`. Without the flag, and with stdin not a TTY, the command aborts. CLAUDE.md's own example (`assemble --input prog.asm --output prog.bin`) omits `--format`.
- There is no `--define`. SKILL.md documents `--define ROWS=8` with a positional input. Neither the option nor the positional argument exists.
- `check` (`cli.py:62-122`) has no way to pass defines, so the editor cannot check a parametrised template with real values.

## Implementation

1. `--format` without a prompt:
   - the default is inferred from the `--output` suffix (`.bin` → `bin`, else `mem`);
   - an explicit `--format` wins;
   - `disassemble` infers from the `--input` suffix the same way.
2. Add `--define KEY=VALUE` (repeatable) to `assemble` and `check`. Values pass to `render` as strings; `ParamSpec.bind` coerces them (E05-09). A `KEY` without `=` is a usage error.
3. Add `-I DIR` / `--include-dir DIR` (repeatable) to `assemble` and `check`. The directory of `--input` is always on the search path first.
4. Add `--params FILE` only if the open question in E05-09 resolves towards `kernel.toml`-only declarations. Otherwise leave it out.
5. Update the docs that show the command: the CLAUDE.md "Build & Test" block, `SKILL.md:174-177` and `docs/content/building-applications.md`. Each example must run as written. E09's `test_doc_commands` checks this.

## Tests

- [ ] `CliRunner` with stdin closed: `assemble --input x.asm --output x.bin` writes a bin byte-identical to `--format bin`.
- [ ] `assemble ... --define cpr=3 --define rows=8` on a template with a header produces the same bytes as rendering with those defaults.
- [ ] An undeclared `--define` exits 1 with `Error rendering template: undeclared define 'x'`.
- [ ] `check --json --define …` uses the define (a template that only parses with it reports no diagnostics).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No `prompt=` in `cli.py`.
- [ ] Documented commands run unmodified in a non-interactive shell.
