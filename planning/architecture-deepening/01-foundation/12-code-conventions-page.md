# "Code conventions" docs page, linked from CLAUDE.md

Part of [Epic E01: Foundation](00-epic.md) · ID E01-12 · Size S · Depends on: E01-04, E01-09, E01-11 · Behaviour change: none (docs only)

## Goal

One page states how code in this repository is written, so reviewers cite a rule instead of a preference and an agent reads the same rules a person does. `CLAUDE.md` links to it in one line.

## Background

Verified at 694a8a8:

- No written code conventions exist. `CLAUDE.md` holds ISA-specific rules (opcodes from position, operand names match handlers, doc naming conventions) and `SKILL.md` repeats some of them. `docs/content/adding-instruction.md` is the only contributor guide in the docs nav (`docs/mkdocs.yml:86-87`, "Contributing").
- The code shows why the user asked for it ("tons of code lines"): 17 Python files over 600 lines, the largest `debug_tui.py` (3 732 lines, one class with 108 methods), `ipu.py` (2 351 lines, `Ipu` with 101 methods); 22 functions over 80 lines; 77 complexity findings (E01-09).
- `GLOSSARY.md` (repository root) and `docs/adr/0001`-`0006` now exist and define the project's vocabulary and architecture decisions. Nothing points contributors at them yet.

## Decisions applied

- Q66: the page content: deep modules; about 600-line files and about 60-line functions, enforced through ruff complexity rules; type hints; Google-style docstrings; no flag arguments; no private cross-module access; tests through public interfaces; two error kinds.
- Q59: the two error kinds are hardware faults (F-1 to F-9, with ID and enable bit) and configuration errors (setup time only).
- Q62: one responsibility per package, one-way dependencies, no lazy imports to dodge cycles.
- Q63: pure Python and numpy.
- Q65: ruff everywhere; strict typing on `ipu_isa` and the reference API first.
- D3: refactor PRs are output-identical and golden-proven; behaviour changes are separate, labelled PRs.
- Q68: names follow `GLOSSARY.md`.

## Implementation

1. Write `docs/content/code-conventions.md` with the `anthropic-skills:technical-writing` skill. Sections, each a short rule plus the reason and how it is checked:
   - **Modules.** Prefer deep modules: a small interface over a lot of behaviour. One responsibility per package; dependencies point one way (Q62) and are declared in Bazel `deps`; no function-level imports to dodge an import cycle. Files stay under about 600 lines; a file over the limit is split along a seam, not by moving code to a `utils` module.
   - **Functions.** About 60 lines at most. Checked by ruff (`C901` complexity 10, `PLR0912` 12 branches, `PLR0915` 50 statements, `PLR0913` 6 arguments). No flag arguments (`run(x, fast=True)`): two functions, or a strategy object.
   - **Access.** No access to another module's `_private` names; add a public function to the owning module instead. (The import-graph test from the package-split work enforces package edges; this rule covers names.)
   - **Types and docs.** Type hints on every public function; Google-style docstrings on public modules, classes and functions; packages with a strict `mypy_test` stay clean (E01-11).
   - **Errors.** Two kinds: hardware faults (F-1 to F-9) and configuration errors raised at setup time only (Q59). No `assert` for input validation (it disappears under `python -O`). No bare `except Exception: pass`.
   - **Tests.** Test through public interfaces, not private helpers. Expected values come from an independent source (a hand-written table, a reference implementation, `ml_dtypes`), never from the code under test. Tiers: default and heavy (`enormous`), with the rule from E01-04 for putting a test in the heavy tier.
   - **Changes.** A refactor PR changes no output and says how that is proven (E02 goldens). A behaviour change is its own PR with the "behaviour change" label (D3). The PR template (E01-13) asks for both.
   - **Words.** Use the terms in `GLOSSARY.md`; avoid the listed alternatives in new code and docs (checked for docs by E01-15). Architecture decisions live in `docs/adr/`.
   - **Tooling.** Format and lint: `bazel run //tools:ruff -- format`, `bazel test --test_tag_filters=lint //...`. Formatting-only commits go into `.git-blame-ignore-revs`. Dependencies: edit `pyproject.toml`, run `bazel run //tools:update_requirements_lock` (E01-02/E01-03).
   - **Performance.** Pure Python and numpy (Q63); a PR that claims a speed-up shows the benchmark numbers (E01-14).
2. Add the page to `docs/mkdocs.yml` under "Contributing" (E09-06 moves it in the new nav).
3. `CLAUDE.md`: add under "Project Knowledge", using the `mattpocock-skills:writing-for-agents` skill for wording:
   ```
   Read [Code conventions](docs/content/code-conventions.md) before writing or reviewing code. Terms: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).
   ```
4. Each rule that a tool checks names the tool. Rules no tool checks are marked "review" so reviewers know they own them.

## Tests

- [ ] `bazel build --config=docs //docs:build_docs` builds the page with no MkDocs warnings.
- [ ] If E09-01 has landed, its doc tests pass on the new page (every `asm` block assembles, every command is real).
- [ ] Every command on the page runs as written (check by hand; record in the PR).

## Acceptance Criteria

- [ ] The page covers every item of Q66 and states how each is checked.
- [ ] `CLAUDE.md` links to the page, `GLOSSARY.md` and `docs/adr/`.
- [ ] No rule contradicts `CLAUDE.md` or an ADR.

## Risks

- The page and the tools drift. Keep thresholds in `ruff.toml` only and quote them from there in the page; when a threshold changes, the same PR updates the page.
