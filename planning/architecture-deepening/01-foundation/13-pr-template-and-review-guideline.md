# Pull request template and code-review guideline

Part of [Epic E01: Foundation](00-epic.md) · ID E01-13 · Size S · Depends on: E01-04, E01-12 · Behaviour change: none

## Goal

Every PR description says what changed, whether behaviour changed, and that the author ran all tests locally. Reviewers have a short written guideline that tells them what to check and what CI already checks.

## Background

Verified at 694a8a8:

- There is no `.github/pull_request_template.md` and no review guideline. PR descriptions vary from one line to long design notes.
- CI (E01-05) runs only the default tier. The heavy tier runs nightly (E01-06), so a PR can break it unnoticed until the next morning unless the author ran it.
- `CLAUDE.md` asks for an SV-package diff of `*_WIDTH` localparams after ISA changes ("Generated SystemVerilog Package"). Nothing reminds the author at PR time.
- D3 splits refactors (output-identical) from behaviour changes; the planning issues mark each one. Nothing carries that into the PR.

## Decisions applied

- Q71: code-review guideline: the PR author states they ran all tests locally.
- D3: refactor PRs are output-identical and golden-proven; behaviour changes are separate, labelled PRs.
- Q46: SV renames and wire-format changes are announced to the downstream RTL repository.

## Implementation

1. `.github/pull_request_template.md`:
   ```markdown
   ## What and why
   <!-- One paragraph. Link the issue (Closes #N). -->

   ## Kind of change
   - [ ] Refactor: no output changes. Proof: <!-- golden test names, or "full suite + digests" -->
   - [ ] Behaviour change (label `behaviour change`): <!-- what changes, for whom -->
   - [ ] Docs / build / CI only

   ## Checks I ran locally
   - [ ] `bazel test --config=full //...` passed (all tests, including the nightly tier)
   - [ ] `bazel test --config=proto //...` passed (if the emulator changed)
   - [ ] `bazel test --test_tag_filters=lint //...` passed
   - [ ] ISA change: `bazel build //docs:generate_instruction_format_artifacts` and the `*_WIDTH` localparams of `ipu_instr_pkg.sv` are identical to the base, or the change is called out below as breaking

   ## Notes for reviewers
   <!-- Anything surprising: performance numbers (E01-14), golden updates, follow-ups. -->
   ```
2. `docs/content/code-review.md` (in "Contributing" next to the conventions page), written with the `anthropic-skills:technical-writing` skill:
   - **Author.** Keep PRs small (one issue). Fill in the template. Run all tests locally, including the heavy tier, and say so; if a run was not possible, say which part was not run and why. Label behaviour changes. Golden updates are behaviour changes and need a sentence of justification each.
   - **Reviewer.** Check against the code conventions (E01-12) and the issue's acceptance criteria. Things CI checks (formatting, lint, default tier, prototyping tier) are not review comments. Things CI does not check: the "review" rules on the conventions page, test independence (expected values not from the code under test), naming against `GLOSSARY.md`, refactor-vs-behaviour split.
   - **Merging.** Required checks green; one approval; squash or rebase per repository setting. A behaviour change that touches the SV package needs the downstream RTL owners informed (Q46).
   - Mention the `mattpocock-skills:code-review` and `mattpocock-skills:pr` skills for agents writing or reviewing PRs.
3. Link the guideline from `CLAUDE.md` (one line next to the conventions link) and from the template's header comment.
4. Labels the template refers to (`behaviour change`, `nightly`) are created by the user with `gh label create` (labels cannot be created from the cloud session). List the commands in the PR description.

## Tests

- [ ] Opening a draft PR pre-fills the template.
- [ ] `bazel build --config=docs //docs:build_docs` builds the guideline page without warnings.

## Acceptance Criteria

- [ ] The template asks for the kind of change, the proof for refactors, and the author's statement that all tests ran locally.
- [ ] The guideline separates what CI checks from what reviewers check.

## Risks

- `bazel test --config=full //...` takes about 30 minutes on a 4-core machine today (E01-04 measurements). Authors may skip it. The nightly run (E01-06) is the backstop; E04's speed-ups shorten the run. If the time stays a problem, the user may relax the rule to "default tier plus the heavy-tier targets the change affects".
