# Extension tests: negative agreement corpus, no fixed sleeps

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-27 · Size S · Depends on: E08-21 · Behaviour change: none

## Goal

The extension's tests catch disagreements in both directions and run deterministically. They stay a JavaScript project run by the `vscode-extension.yml` workflow, outside Bazel (D9). E01 owns the workflow's path triggers.

## Background

- `test/agreement.js` checks only positive cases: the JS scan agrees with the parser on valid code. `test/language.js:267-340` checks one direction.
- The corpus is a single file, `test/corpus/edge_cases.asm`.
- Integration tests use fixed sleeps:
  - `test/integration/suite.js:193` (`sleep(500)`);
  - `test/integration/layouts.js:55, 78` (`sleep(3000)`, `sleep(2000)`).

  They are slow, and flaky on a loaded runner.
- `vscode-ipu-asm/BUILD.bazel:20-32` defines an `agreement_test_srcs` filegroup with no consumers (plan 08, issue 8.5). Per D9 no JS test runs from Bazel, so it can be deleted.

## Implementation

1. Add `test/corpus/negative/*.asm`, one file per known divergence class:
   - signed hex, binary and octal;
   - an over-full word;
   - multi-form mnemonic collisions;
   - unbalanced Jinja;
   - unknown mnemonics;
   - a label that shadows a register name.

   Expected verdicts come from `ipu-as check --json` in the workflow (generated, not hand-written).
2. `agreement.js` checks both directions: JS accepts ⇔ Python accepts, per file, with a reviewed list of allowed gaps.
3. Extend `mangle.js` fuzzing so mangled inputs are judged by both sides with the same verdict rule.
4. Replace every `sleep(n)` in `test/integration/*.js` with the harness's `until(predicate, timeout)` polling.
5. Delete the unused `agreement_test_srcs` filegroup.

## Tests

- [ ] The negative corpus runs in the workflow and is green with the allowed-gaps file reviewed in the PR.
- [ ] `grep -rn "sleep(" vscode-ipu-asm/test/integration` is empty.
- [ ] The extension CI workflow passes three runs in a row on the PR.

## Acceptance Criteria

- [ ] Agreement is tested in both directions, with a negative corpus.
- [ ] No fixed sleeps in the integration tests.
