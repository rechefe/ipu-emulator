# Reference: command-line tools

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-15 · Size S · Depends on: E09-01, E09-06, E05 (CLI: no prompt, `--define`, `--target hardware`, structured errors), E06 (kernel runner options), E08 (debugger CLI) · Behaviour change: none

## Goal

Every command-line tool has a reference page generated from its declared
options, so the page cannot drift from the code.

## Background

- `ipu-as` (`ipu_as/cli.py`) has `assemble`, `disassemble`, `sv-package` and
  `check`. No page lists their options; `setup.md:157-162` shows only `--help`.
- E05 changes the CLI: no prompt for `--format`, `--define`, `--target
  hardware` (Q42), located errors.
- E06's kernel runner takes `--case`, `--format`, `--debug`, `--input
  x=path.npy` and others (Q80, Q82). Today `bazel run <kernel> -- --help` is the
  only reference (`debugging.md:42`).
- The debugger CLI comes from E08.
- E09-01's command test already introspects the click definitions.

## Implementation

1. Generate `reference/cli.md` at docs build time from the click (or argparse)
   definitions of each tool: command, options, types, choices, defaults, help
   text. Use `mkdocs-click` if E01's lockfile can add it; otherwise a small
   generator in `ipu_gen` (E08) that walks `click.Command.params`. Open
   question: which one; prefer the generator if it is under 100 lines, to avoid
   a dependency.
2. Each tool gets a hand-written two-line intro and one example, checked by
   E09-01's command test.
3. Add the page to the `generated_pages` map (E09-04).
4. Run statistics (cycles, MAC utilisation, alias hits) that the runner prints
   are explained here, moved from `building-applications.md:294-329`.

## Tests

- [ ] Adding an option to a click command shows up on the page with no doc
      edit (generator unit test).
- [ ] `bazel test //docs:test_doc_commands` passes.

## Acceptance Criteria

- [ ] Every user-facing CLI (assembler, kernel runner, debugger) has a
      generated reference section.
- [ ] No hand-written option list remains in `docs/content`.
