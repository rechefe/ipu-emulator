# Test every `asm` block and every documented `ipu-as` command

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-01 · Size M · Depends on: none · Behaviour change: none

## Goal

Every fenced `asm` block in the hand-written docs assembles, or is explicitly
marked as a fragment. Every documented `ipu-as` command uses only options the
CLI declares. Two Bazel tests keep it that way, and they run in the default
`bazel test //...`.

## Background

- Hand-written docs hold 25 fenced `asm` blocks: `docs/content/isa-alias-catalogue.md`
  (16), `building-applications.md` (3), `debugging.md` (3),
  `ipu-configuration.md` (2) and `SKILL.md` (1). `CLAUDE.md` and `README.md`
  have none today.
- Checked at `694a8a8` with `ipu_as.diagnostics.check` (`diagnostics.py:468`,
  returns an empty list when the source assembles), 8 blocks fail:

  | Block | Error | Fix |
  |---|---|---|
  | `building-applications.md:50` | `unexpected ','` | spaces as separators |
  | `building-applications.md:69` | `unexpected ','` | spaces as separators |
  | `building-applications.md:338` | `Opcode 'ACC.FIRST' not found` | use the real mnemonic (`ACC.ADD.FIRST`, check intent) |
  | `debugging.md:79` | `Opcode 'break_nop' not found` | the break-slot NOP is not written in source; change the block to `text` or remove it |
  | `ipu-configuration.md:81` | `unexpected ','` | spaces as separators |
  | `ipu-configuration.md:98` | `unexpected ','` | spaces as separators |
  | `isa-alias-catalogue.md:315` | `unexpected '.'` (a `...` line) | mark as fragment |
  | `SKILL.md:69` | `unexpected ','` | spaces, `INC LR0 1` instead of `ADD LR0, LR0, 1` (`src_b` is `LcrIdx`), add the `next:` label |

- `cli.py:13-21`: `assemble` declares `--input`, `--output` (both required)
  and `--format` with `prompt="Choose a file format"`, choices `mem` and `bin`.
  A documented command without `--format` blocks on the prompt.
- `CLAUDE.md:10`, `SKILL.md:174` omit `--format`. `SKILL.md:177` uses
  `--define` and a positional argument. `adding-instruction.md:230` uses
  `--format hex -`. None of these run.

## Implementation

1. **Fragment marker.** Use an HTML comment on the line before the fence:

   ~~~markdown
   <!-- asm-fragment: shows only the inner loop -->
   ```asm
   ...
   ```
   ~~~

   The comment does not render and does not change syntax highlighting. A
   reason after the colon is required, so a reader of the source knows why the
   block is not checked. Placeholder syntax lines (`ADD dest src_a src_b`)
   use a `text` fence instead, not `asm`.
2. **Shared extractor.** `docs/doc_blocks.py`: a small module with one function,
   `fenced_blocks(path, lang) -> list[Block]`, where `Block` holds `path`,
   `line`, `body` and `fragment_reason`. It handles indented fences (blocks
   inside list items) by stripping the fence's indentation.
3. **`//docs:test_doc_asm_blocks`** (`py_test` with pytest).
   - Data: `glob(["content/**/*.md"])` plus `//:SKILL.md`, `//:CLAUDE.md`,
     `//:README.md`. Add `exports_files(["SKILL.md", "CLAUDE.md", "README.md"])`
     to the root `BUILD.bazel`.
   - Generated pages are not inputs. They are checked by their generators'
     own tests.
   - One parametrised case per block, with id `path:line`.
   - Assert `diagnostics.check(body)` has no `error` diagnostics. Print the
     first diagnostic with the block's `path:line` plus the diagnostic line.
   - Fragments are reported as skipped with their reason, so they stay visible
     in the test log.
   - Also fail if a fragment marker has no reason, or is not followed by an
     `asm` fence.
4. **`//docs:test_doc_commands`** (`py_test`).
   - Find every line in the same files that runs `ipu-as` (`ipu-as -- <sub>`
     or `ipu-as <sub>`), joining `\` continuations.
   - Look up the subcommand in `ipu_as.cli.cli` with click introspection
     (`cli.commands[name].params`).
   - Assert: the subcommand exists; every `--option` used is declared; every
     required option is present; every option with a `prompt` is given (a
     documented command must never prompt); choice values are valid.
   - When E05 removes the prompt and adds `--define`, this test needs no edit:
     it reads the declared parameters.
5. **Fix the 8 failing blocks** as in the table above. Verify each operand order
   against `instruction_spec.py`. Do not rewrite the surrounding prose; the
   page rewrites (E09-09 to E09-19) do that.
6. **Fix the command lines** found by step 4: add `--format bin` to
   `CLAUDE.md:10` and `SKILL.md:174`. `SKILL.md:177` and
   `adding-instruction.md:230` are fixed in E09-02; until then mark them so
   the test can land green. Open question: use an `xfail` list in the test with
   an issue link, or land E09-01 and E09-02 as one PR. Prefer one PR if small.
7. Neither test is tagged `docs`, so both run in CI's default set.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks //docs:test_doc_commands` passes.
- [ ] Adding a comma to any checked block fails the test and names `file:line`.
- [ ] Removing `--format bin` from `CLAUDE.md:10` fails `test_doc_commands`.
- [ ] A fragment marker without a reason fails the test.
- [ ] Unit test for `fenced_blocks`: indented fence, two blocks in one file,
      fence with no closing line (reported as an error, not silently skipped).

## Acceptance Criteria

- [ ] Every `asm` block in `docs/content/**`, `SKILL.md`, `CLAUDE.md` and
      `README.md` assembles, or carries an `asm-fragment` marker with a reason.
- [ ] Every documented `ipu-as` command uses only declared options and never
      triggers a prompt.
- [ ] Both tests run under plain `bazel test //...`.
- [ ] `CLAUDE.md` "Docs and written examples" bullet mentions the fragment
      marker in one sentence (edit with the `mattpocock-skills:writing-for-agents`
      skill).

## Risks

- E03's mnemonic rename (Q45) will rename text inside doc blocks. If the rename
  script misses a doc, this test fails, which is the intended guard. E03 should
  run this test in the rename PR.
- When `ipu_as` moves (E03 package split, E05), the test's `deps` label
  changes. It is a one-line update.
