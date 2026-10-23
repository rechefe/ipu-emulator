# CLAUDE.md and SKILL.md: an `ipu-as assemble` command that runs as written

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-12 · Size S · Depends on: none · Behaviour change: none (docs only)

## Goal

An agent or a contributor who copies the assemble command from `CLAUDE.md` or `SKILL.md` gets a binary file, without an interactive prompt.

## Background

Verified at 694a8a8:

- `CLAUDE.md:10` and `SKILL.md:174`:
  ```bash
  bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin
  ```
- `cli.py:13-21` declares `--format` with `prompt="Choose a file format"`, choices `mem` and `bin`, default `mem`. Without `--format` the command prompts. In a non-interactive shell (an agent, CI) it reads EOF and aborts; interactively, pressing Enter writes the `mem` text format into a file named `prog.bin`.
- `bazel run` starts the binary in its runfiles directory, not the caller's directory, and `cli.py` does not read `BUILD_WORKING_DIRECTORY`. `--input prog.asm` (`click.Path(exists=True)`) is therefore resolved against the runfiles tree and fails with "does not exist" unless the path is absolute.

## Implementation

1. `CLAUDE.md:10` and `SKILL.md:174` become:
   ```bash
   bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin" --format bin
   ```
2. Keep the change to these two lines. The rest of the `SKILL.md` drift (the `--define` line at :177, the project name, the operand list, the `ADD LR0, LR0, 1` example) is E09-02. `CLAUDE.md` follows the `mattpocock-skills:writing-for-agents` guidance; no other wording changes.

## Tests

- [ ] Run the new command from the repository root on a small `.asm` file with stdin closed (`</dev/null`). It exits 0 and writes a binary `prog.bin`. Paste the output in the PR.
- [ ] E09-01's `test_doc_commands` (when it lands) passes on both files without an `xfail` for these lines.

## Acceptance Criteria

- [ ] Both files show the same command, with `--format bin` and absolute paths.

## Risks

- E05-10 drops the `--format` prompt. If E05 also resolves paths against `BUILD_WORKING_DIRECTORY`, `"$PWD/"` becomes unnecessary; the E05 issue that changes the CLI updates these lines.
- E09-01 also plans to add `--format bin` to `CLAUDE.md`. Whichever lands second drops that part.
