# Ruff fix-up: format and lint the repository, record the complexity baseline, enable the test

Part of [Epic E01: Foundation](00-epic.md) · ID E01-10 · Size M · Depends on: E01-09 · Behaviour change: none

## Goal

Every Python file passes `ruff check` and `ruff format --check` under the repository configuration, the ruff targets join the default tier, and the formatting commit does not pollute `git blame`.

## Background

From E01-09's measurements at 694a8a8: 409 of 597 files need reformatting; 572 lint findings for `E4,E7,E9,F,I,UP,B`, of which 517 are auto-fixable; 77 complexity findings (`C901` 35, `PLR0912` 22, `PLR0915` 20) at ruff's default thresholds. The remaining manual findings are mostly unused variables (16), `B` rules and a few `E402` / `F402`.

The complexity offenders are the large modules the later epics split: `ipu.py` handlers (E04), `debug_tui.py` (E08), `gen_docs.py` (E08 `ipu_gen`), kernel harnesses (E06/E07).

## Decisions applied

- Q65: ruff everywhere as a Bazel test.
- Q66: complexity limits; existing offenders are recorded and removed as each area is refactored, not rewritten here.
- D3: this is a refactor and must not change behaviour. Proof is the full test suite plus the E02 golden harnesses when they exist (binary digests of every `.asm`, kernel output digests).

## Implementation

Land as three commits in one PR, each green on its own:

1. **Format.** `bazel run //tools:ruff -- format` over the repository. Nothing else in this commit. Add its hash to a new `.git-blame-ignore-revs` (GitHub reads it automatically; document `git config blame.ignoreRevsFile .git-blame-ignore-revs` in the conventions page).
2. **Safe auto-fixes.** `bazel run //tools:ruff -- check --fix` (safe fixes only; no `--unsafe-fixes`). Review the diff for import reordering that changes import side effects (for example a module that must import before another registers kernels); pin such imports with `# isort: skip` and a comment saying why.
3. **Manual fixes and baseline.**
   - Fix the remaining non-complexity findings by hand (unused variables, bugbear findings). If a finding is intentional, use a line-level `# noqa: <code>` with a reason.
   - For complexity rules (`C901`, `PLR0912`, `PLR0913`, `PLR0915`) only: generate `[lint.per-file-ignores]` entries for the files that violate them today, one file per line, grouped by package with a comment naming the epic that will remove them (E04, E06, E07, E08).
   - Remove `tags = ["manual"]` from the 7 `ruff_test` targets.

Merge it when few PRs are open (announce it a day ahead); the format commit touches 409 files and conflicts with most in-flight branches. Give authors the recipe: rebase, then run `bazel run //tools:ruff -- format` on their branch before resolving conflicts.

## Tests

- [ ] `bazel test --test_tag_filters=lint //...` passes.
- [ ] `bazel test --config=full //...` pass and fail counts equal the run before the PR.
- [ ] If E02 has landed: the `.asm` binary digest and kernel output digest goldens are unchanged. If not, assemble every `.asm` before and after and compare the binaries (`sha256sum`), and record that in the PR.
- [ ] `git blame` on a reformatted line shows the original author when `blame.ignoreRevsFile` is set.

## Acceptance Criteria

- [ ] The ruff targets run in the default tier and pass.
- [ ] The only `per-file-ignores` entries are complexity rules, each group annotated with the epic that removes it.
- [ ] No behaviour change: test results and binary digests are identical.

## Risks

- Import reordering can change registration order in `ipu_apps` (kernel discovery by import). The kernel manifest (`ipu_kernel_targets`) and `test_kernel_registry` catch a changed order; check them explicitly.
