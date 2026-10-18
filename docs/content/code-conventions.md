# Code conventions

These rules say how code in this repository is written. Reviewers cite a rule by its section instead of a preference; people and agents read the same page.

Each rule names what checks it. **Review** means no tool checks it: the reviewer owns it ([Code review](code-review.md)). The ISA rules (opcodes from position, one source of instruction metadata, handler argument names) and the docs conventions are in `CLAUDE.md`, "Key Constraints", and in [ADR 0001](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0001-isa-package-is-the-single-source-of-truth.md); this page does not repeat them.

## Modules

| Rule | Why | Checked by |
|---|---|---|
| Prefer deep modules: a small interface over a lot of behaviour. | A caller learns little and gets a lot; the module can change inside without breaking callers. | Review |
| One responsibility per package. | A package with one job has one reason to change. | Review |
| Dependencies point one way and are declared in the target's Bazel `deps`. | A cycle makes packages impossible to test, build or replace alone. | Bazel `deps` (an import outside them fails in the sandbox); `//tools:test_import_graph` checks package edges |
| No function-level imports to dodge an import cycle. | A lazy import hides a cycle instead of removing it. Move the shared code to the lower package instead. | Review |
| Files stay under about 600 lines. A file over the limit is split along a seam, not by moving code into a `utils` module. | Long files hold several responsibilities; a `utils` module is a second place with no responsibility at all. | Review |

## Functions

| Rule | Why | Checked by |
|---|---|---|
| About 60 lines at most. | A function that fits on one screen can be read and tested as one unit. | Ruff, with the thresholds in `ruff.toml`: `C901` (complexity above 10), `PLR0912` (more than 12 branches), `PLR0915` (more than 50 statements), `PLR0913` (more than 6 arguments) |
| No flag arguments (`run(x, fast=True)`). Write two functions, or pass a strategy object. | A flag means the function does two things; every caller pays for both. | `PLR0913` caps the argument count; the flag itself is Review |

`ruff.toml` is the only home of these thresholds. A PR that changes one updates this table.

Files that broke a complexity rule before the rules existed are listed under `[lint.per-file-ignores]` in `ruff.toml`, grouped by the epic that removes them. Do not add entries; fix the code instead.

## Access

| Rule | Why | Checked by |
|---|---|---|
| Do not use another module's `_private` names. Add a public function to the owning module instead. | A private name carries no promise; its owner changes it without looking for callers. | Review (the import-graph test covers package edges, not names) |

## Types and docstrings

| Rule | Why | Checked by |
|---|---|---|
| Type hints on every public function. | Types document the interface and let mypy find mistakes before a test runs. | `mypy_test` targets (`tools/typecheck.bzl`) where a package has one; Review elsewhere |
| A package with a `mypy_test` stays clean; one declared `strict = True` stays clean in strict mode. Strict mode comes first to `ipu_isa` and the reference-model API. | A type check that is allowed to fail stops being read. | `bazel test --test_tag_filters=typecheck //...` |
| Google-style docstrings (`Args:`, `Returns:`, `Raises:`) on public modules, classes and functions. | One format reads the same everywhere and renders in the docs. | Review |

## Errors

| Rule | Why | Checked by |
|---|---|---|
| Two kinds of error. A hardware fault (F-1 to F-9, see [Faults](requirements/faults.md)) carries a fault ID and has an enable bit. A configuration error is raised at setup time only, never while a program runs. Both derive from `IpuError`. | Callers tell a faulting program from a wrong setup by type alone. | Review |
| No `assert` for input validation. Raise an exception. | `python -O` removes every `assert`. | Review |
| No bare `except Exception: pass`. | It hides the fault the test or the user needs to see. | Review |

## Tests

| Rule | Why | Checked by |
|---|---|---|
| Test through public interfaces, not private helpers. | A test of a private helper breaks on every refactor and proves nothing about the interface. | Review |
| Expected values come from an independent source: a hand-written table, a reference implementation, `ml_dtypes`. Never from the code under test. | A value computed by the code under test only proves the code agrees with itself. | Review |
| Put a test in the default tier unless it repeats coverage a default-tier test already gives, or takes more than about 2 minutes on CI. Then it goes in the heavy tier (`size = "enormous"`). | The default tier runs on every pull request and must stay fast; the heavy tier runs nightly. | Review |

The two tiers:

| Tier | Holds | Command |
|---|---|---|
| Default | Every test except size `enormous`. CI runs it on every pull request. | `bazel test //...` |
| Heavy | Size `enormous` tests. Runs nightly. | `bazel test --config=full //...` |

Run the heavy tier before review when a change touches code that only heavy-tier tests exercise.

## Changes

| Rule | Why | Checked by |
|---|---|---|
| A refactor PR changes no output, and its description says how that is proven (golden tests, or the full suite with digests). | A reviewer cannot check by reading that a large refactor kept behaviour. | Golden tests; the PR template asks for the proof |
| A behaviour change is its own PR, labelled `behaviour change`. | Mixed into a refactor, a behaviour change goes unnoticed. | Review; the PR template asks for the label |

The [PR template](https://github.com/rechefe/ipu-emulator/blob/master/.github/pull_request_template.md) asks for both; [Code review](code-review.md) says what the author and the reviewer check.

## Baselines

A baseline is a checked-in file that pins today's output. Its test fails on any change. The baselines are how a refactor PR proves it changes no output. A package keeps its baselines in `test/baselines/`, with the scripts that rewrite them beside them:

| Baseline | Location | Update target |
|---|---|---|
| Assembled-program digests | `src/tools/ipu-as-py/test/baselines/asm_digests.json` | `//src/tools/ipu-as-py:update_asm_digests` |
| Field names, emulator field maps, enum descriptors | `src/tools/ipu-as-py/test/baselines/*.json` | `//src/tools/ipu-as-py:update_decode_snapshots` |
| Disassembly text | `src/tools/ipu-as-py/test/baselines/disassembly/` | `//src/tools/ipu-as-py:update_disassembly_snapshots` |
| SV package golden | `src/tools/ipu-as-py/test/baselines/ipu_instr_pkg.sv`, plus the `EXPECTED_LOCALPARAMS` table in `test_sv_package_golden.py` (edited by hand) | `//src/tools/ipu-as-py:update_sv_golden` |
| Generated artifacts | `src/tools/ipu-as-py/test/baselines/generated/` | `//src/tools/ipu-as-py:update_generated_artifacts` |
| The ISA view as JSON, and the F-1 table of the faults requirements page | `src/tools/ipu-isa/test/baselines/spec_snapshot.json`, `docs/content/requirements/faults.md` | `//src/tools/ipu-isa:update_isa_baselines` |
| Conformance suite (programs, setups, expected traces) | `src/tools/ipu-isa/conformance/` | `//src/tools/ipu-emu-py:update_conformance` (also `:regen_conformance`) |
| The check command's JSON output (v1), the VS Code extension's contract | `src/tools/ipu-as-py/test/baselines/check_json_v1.json` | none: a difference is a contract change; the test writes the new file to its undeclared outputs to copy over |
| The assemble command's error output over the check corpus | `src/tools/ipu-as-py/test/baselines/cli_assemble_errors.json` | none: the test writes the new file to its undeclared outputs to copy over |
| Formatter output (the extension's corpus in full, every kernel as digests) | `src/tools/ipu-as-py/test/baselines/format/` | `node vscode-ipu-asm/test/record-format.js` |
| Debugger TUI screens | `src/tools/ipu-debug/test/baselines/tui/` | none: recorded once, before E08-08; edited by hand |

`tools/update_baselines.sh` runs every update target and then lists the baseline files that changed. A kernel of `kernels/` has no baseline file: its tests hold every case to its golden (`kernel.toml`, `[golden]`), and its `:assemble_<name>` writes its program. The timing tests have no baseline file: their expected values are code, changed by hand in the behaviour PR that changes the timing rule.

| Rule | Why | Checked by |
|---|---|---|
| A PR labelled refactor modifies no baseline file. | An unchanged baseline is the proof that the refactor changes no output. Regenerating a baseline to make a refactor pass hides the change it was there to catch. | Review; the PR template asks for it |
| A PR labelled `behaviour change` regenerates only the baselines it means to change. Its description says why each one changed, in one sentence. | The reviewer reads each baseline diff as the effect of the change. | Review; the PR template asks for the list |
| Every `*_WIDTH` localparam of the SV package stays identical to the base, or the PR calls the change out as breaking. `tools/sv_localparam_diff.sh <base-ref>` prints nothing when they are identical. | Each width is wire format for the downstream RTL repository (`CLAUDE.md`, "Generated SystemVerilog Package"). | `//src/tools/ipu-as-py:test_sv_package_golden`; the diff script; Review |
| A new update target is added to `tools/update_baselines.sh`. | One command must regenerate every baseline. | `//tools:test_update_baselines` |

## Words

| Rule | Why | Checked by |
|---|---|---|
| Use the terms in the [Glossary](reference/glossary.md) (`GLOSSARY.md`). Avoid the alternatives it lists, in new code and in docs. | One name per concept; a different word makes a reader look for a different thing. | `//tools:test_glossary_adoption` for the hand-written docs (avoided terms outside code; counts in `tools/glossary_baseline.json` may only go down); review for code |
| Architecture decisions live in [`docs/adr/`](https://github.com/rechefe/ipu-emulator/tree/master/docs/adr). Code does not contradict an accepted ADR; a change of decision is a new ADR. | Decisions stay findable and are not re-argued in every PR. | Review |

## Tooling

| Task | Command |
|---|---|
| Format | `bazel run //tools:ruff -- format` |
| Fix lint automatically | `bazel run //tools:ruff -- check --fix` |
| Check lint and format | `bazel test --test_tag_filters=lint //...` (or `bazel test //tools:lint`) |
| Type-check | `bazel test --test_tag_filters=typecheck //...` |
| Lower the glossary baseline after removing avoided terms from the docs | `bazel run //tools:glossary_baseline` |
| Refresh the dependency lock after editing a `pyproject.toml` | `bazel run //tools:update_requirements_lock` |

A commit that only reformats code goes into `.git-blame-ignore-revs`, with a comment naming the change; `git config blame.ignoreRevsFile .git-blame-ignore-revs` makes `git blame` skip it. Runtime dependencies are also listed in the package's `deps.bzl`; `//tools:test_deps_consistency` fails if the two disagree.

## Performance

| Rule | Why | Checked by |
|---|---|---|
| Pure Python and numpy. Speed comes from vectorising and from better data structures, not from compiled extensions. | Every contributor can read, debug and build the code. | Review |
| A PR that claims a speed change pastes the `bazel run //tools/bench:bench -- --quick` table from before and after the change; a PR that changes only the emulator may paste the `bazel run //src/tools/ipu-emu-py:bench_emulator -- --quick` table instead (the PR template's Performance section). | A speed-up without numbers cannot be checked or kept. The nightly history (`tools/bench/README.md`) shows slow drift. | Review |
