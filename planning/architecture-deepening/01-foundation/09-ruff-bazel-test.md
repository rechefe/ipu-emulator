# Ruff lint and format as a Bazel test over the whole repository

Part of [Epic E01: Foundation](00-epic.md) · ID E01-09 · Size M · Depends on: E01-02 · Behaviour change: none

## Goal

One ruff configuration covers every Python file in the repository, and `bazel test` checks both lint and formatting with the ruff version pinned in the lock. This issue adds the configuration and the test targets; E01-10 makes the code pass and switches the targets on.

## Background

Measured at 694a8a8 with ruff 0.16 (isolated from any user config):

- No ruff configuration exists (no `ruff.toml`, no `[tool.ruff]` in any `pyproject.toml`). The dev container installs the ruff VS Code extension (`devcontainer.json:13`), so editors already format with personal settings.
- Python files: about 61 000 lines in 597 files, under 7 Bazel packages (`//`, `//docs`, `//vscode-ipu-asm`, and the four `src/tools/*` packages).
- `ruff check --select E4,E7,E9,F` (ruff's defaults): 57 findings (30 unused imports, 16 unused variables, ...).
- Adding `I` (import order), `UP` (pyupgrade) and `B` (bugbear): 572 findings, 517 auto-fixable.
- `ruff format --check`: 409 of 597 files would be reformatted.
- Complexity at ruff's default thresholds: 35 `C901` (complex structure), 22 `PLR0912` (too many branches), 20 `PLR0915` (too many statements). 17 files exceed 600 lines (largest: `debug_tui.py` 3 732, `test_execute.py` 3 215, `test_debug_tui.py` 3 139, `ipu.py` 2 351).

## Decisions applied

- Q65: ruff (lint and format) everywhere, as a Bazel test.
- Q66: function size and complexity limits come from ruff's complexity rules (about 60-line functions); the conventions page (E01-12) states them.

## Implementation

1. Add `ruff` to `tools/requirements-tools.in` and relock (E01-02's refresh target).
2. Root `ruff.toml`:
   ```toml
   target-version = "py310"
   line-length = 100

   [lint]
   select = ["E4", "E7", "E9", "F", "I", "UP", "B", "C901", "PLR0912", "PLR0913", "PLR0915"]

   [lint.mccabe]
   max-complexity = 10

   [lint.pylint]
   max-branches = 12
   max-statements = 50
   max-args = 6

   [lint.per-file-ignores]
   # Baseline: existing offenders, removed as each refactor epic reaches the file.
   # Do not add entries; fix the code instead.
   ```
   `PLR0913` (too many arguments) supports the "no flag arguments" convention. `D` (docstrings, Google convention) is not enabled repository-wide yet; E01-11 and the E03 package enable it per package.
3. `tools/lint.bzl` with a `ruff_test(name, srcs)` macro that creates one `py_test` running `python -m ruff check --no-cache <srcs>` and `python -m ruff format --check --no-cache <srcs>` with `ruff.toml` as data. The ruff executable comes from the locked wheel (`requirement("ruff")`), not from `PATH`.
4. In each of the 7 `BUILD.bazel` files: `ruff_test(name = "ruff", srcs = glob(["**/*.py"]), tags = ["manual"])`. A package's `**` glob stops at subpackages, so the 7 targets together cover every Python file exactly once. `tags = ["manual"]` keeps them out of `//...` until E01-10 makes them pass.
5. `tools/check_lint_coverage.sh`, run as a CI step (it needs `git`, so it runs outside the Bazel sandbox): compares `git ls-files '*.py'` with the union of `srcs` of every `ruff_test` (`bazel query 'labels(srcs, attr(name, "^ruff$", //...))'`) and fails on any file not covered. This keeps "whole repository" true when someone adds a package.
6. `CLAUDE.md`: one line under Build & Test: `bazel test //:ruff //docs:ruff ...` or the shorter `bazel test --test_tag_filters=lint //...` if the macro also adds a `lint` tag (add it). Local auto-fix: `bazel run //tools:ruff -- check --fix` and `bazel run //tools:ruff -- format` (a `py_binary` alias over the same wheel, run in `BUILD_WORKSPACE_DIRECTORY`).
7. `.vscode/settings.json`: point the ruff extension at the repository config (it reads `ruff.toml` automatically; set `"ruff.importStrategy": "fromEnvironment"` only if the extension's bundled version differs from the lock in a way that matters).

## Tests

- [ ] `bazel test --test_tag_filters=lint //...` runs 7 targets (and fails today, as expected).
- [ ] `tools/check_lint_coverage.sh` passes; adding a scratch package with a `.py` file and no `ruff_test` fails it.
- [ ] `bazel run //tools:ruff -- --version` prints the locked version.

## Acceptance Criteria

- [ ] One `ruff.toml` at the root; no per-package ruff configuration.
- [ ] Every Python file in git is covered by exactly one `ruff_test`.
- [ ] Ruff runs from the lock, not from the host.

Open question: line length. The proposal is 100 (ruff's default is 88). The number changes how many files E01-10 reformats; pick it before E01-10 starts.

## Risks

- Complexity thresholds that are too strict flood the baseline. The values above follow Q66's "about 60-line functions"; E01-10 records the baseline size, and E01-12 documents the rule for removing entries.
