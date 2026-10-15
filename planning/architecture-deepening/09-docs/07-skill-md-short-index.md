# Reduce SKILL.md to a short agent index into docs/

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-07 · Size S · Depends on: E09-01, E09-02, E09-06 · Behaviour change: none

## Goal

`docs/` is the one place where project knowledge is written. `SKILL.md` is a
short index for agents: what the project is, the rules that must never be
broken, and where each topic is documented. Nothing in it is a second copy
that can drift.

## Background

- `SKILL.md` is 238 lines. It repeats the repository layout (`:15-41`), the
  register file table (`:78-90`), the slot table (`:92-104`), the operand-type
  list (`:106-112`), an add-an-instruction walkthrough (`:114-142`), testing
  patterns (`:144-168`), assembler usage (`:170-182`), the application pattern
  (`:184-215`) and debugging (`:217-228`). Each of these also exists, in a
  different version, in `docs/content/` or in generated pages.
- E09-02 found five drifted facts in it. E09-01 found its only `asm` block
  broken.
- Decision L31 (accepted with Q29): `SKILL.md` becomes a short index linking
  `docs/`; `docs/` is canonical.
- `CLAUDE.md` links `SKILL.md` for "full architecture details". That sentence
  changes with this issue.

## Implementation

1. Edit with the `mattpocock-skills:writing-for-agents` skill (it covers
   `SKILL.md` and `CLAUDE.md`).
2. New `SKILL.md`, about 40 to 60 lines:
   - Two sentences: what the project is.
   - The rules agents must never break, stated once, each with a link to the
     page that explains it: opcodes come from position in the ISA definition;
     no duplicated instruction metadata; handler argument names match operand
     names; naming and separator conventions in docs; check the SystemVerilog
     package after ISA changes. (These overlap `CLAUDE.md`. Keep each rule in
     one file and link from the other; decide which in review.)
   - A topic index: one line per task, with the docs page that covers it
     (write a kernel, add an instruction, debug a kernel, run tests, ISA
     reference, CLI reference, reference-model API, architecture, timing model,
     element formats, glossary, ADRs, open questions, code conventions).
   - Link targets point at the source files in `docs/content/` (agents read
     the repo, not the site), and at the site URL for generated pages.
3. Move any fact that exists only in `SKILL.md` into the right docs page
   first. Expected: none, but check each section against its docs page and
   list the result in the PR.
4. Update the `CLAUDE.md` "Project Knowledge" line to describe `SKILL.md` as
   an index.
5. As each later page rewrite (E09-08 to E09-19) moves a page, it updates the
   link here. E09-01's tests do not check links; `mkdocs build` does not see
   `SKILL.md`. Add a small check to `//docs:test_doc_commands` (or a sibling
   test): every relative link in `SKILL.md` and `CLAUDE.md` points at an
   existing file.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks //docs:test_doc_commands` passes.
- [ ] The new link check fails when a linked file is renamed.

## Acceptance Criteria

- [ ] `SKILL.md` has no tables or walkthroughs copied from `docs/`.
- [ ] Every topic in the old `SKILL.md` is reachable from the new one in one
      link.
- [ ] Every relative link in `SKILL.md` and `CLAUDE.md` resolves, under test.
