# Glossary adoption check: no new uses of the terms `GLOSSARY.md` says to avoid

Part of [Epic E01: Foundation](00-epic.md) · ID E01-15 · Size S · Depends on: GLOSSARY.md (written separately at the repository root) · Behaviour change: none

## Goal

New and edited documentation uses the project vocabulary in `GLOSSARY.md`. A test fails when a change adds a use of a term the glossary lists under "Avoid", and existing uses can only go down.

## Background

- `GLOSSARY.md` (repository root, 147 lines) defines the project terms in sections (program and encoding, machine, faults and events, kernels, reference model). 17 entries end with an `_Avoid_:` line listing discouraged synonyms, for example `_Avoid_: VLIW word, compound instruction, bundle, line` under **Instruction word**, and `_Avoid_: dtype, wide mode, wide-vector debug mode, narrow mode` under **Element format**.
- Some avoided terms carry a qualifier: `op (in prose)`, `PC (in code)`, `lane (for slots)`, `word, chunk, line (for XMEM units)`, `harness (for the kernel itself)`. Others are ordinary English words (`line`, `word`, `image`, `buffer`, `exception`).
- Counted at the time of writing in hand-written Markdown (`docs/content`, `README.md`, `SKILL.md`, `CLAUDE.md`), case-insensitive: "DRAM" 56, "VLIW word" 28, "chunk" 20, "bundle" 18, "program counter" 10, "wide mode" 6, "wide-vector debug mode" 5, "simulation-only" 4, "narrow mode" 3, "slot type" 3, "compound instruction" 3. Many are in pages E09 rewrites; some (wide mode) describe features E04 replaces.
- Code is renamed per area as each area is refactored (Q68); this check covers documentation only.

## Decisions applied

- Q68: `GLOSSARY.md` defines the terms (row, instruction word, instruction, pc, ...); code is renamed per area as it is refactored.

## Implementation

1. `tools/glossary_check.py` and a `py_pytest_test` `//:glossary_adoption_test` in the root package, with data: `GLOSSARY.md`, `//docs:doc_sources`, `README.md`, `SKILL.md`, `CLAUDE.md` and the baseline file.
2. Parse `GLOSSARY.md`: each `**Term**:` entry and its `_Avoid_:` list. Split the list on commas. A term with a parenthesised qualifier is checked only if the qualifier is `(in prose)`; other qualifiers (`in code`, `for slots`, `for XMEM units`, ...) need context a regex cannot see and are skipped. Single common English words are skipped through an explicit list in the test (`line`, `word`, `op`, `inst`, `instr`, `app`, `application`, `harness`, `buffer`, `image`, `blob`, `exception`, `trap`, `scenario`, `lane`, `chunk`), each with a one-line reason. The skip list is the only place to tune the check.
3. Count matches per file and term: case-insensitive, whole words, outside fenced code blocks and inline code (code samples may show real identifiers such as `wide_vector_debug`).
4. Compare with `tools/glossary_baseline.json` (`{"docs/content/setup.md": {"VLIW word": 2}, ...}`):
   - a count above the baseline, or a term in a file with no baseline entry, fails the test with `file:line`, the avoided term and the glossary term to use instead;
   - a count below the baseline fails with "lower the baseline: run `bazel run //:glossary_baseline`", so the baseline only ratchets down.
5. `//:glossary_baseline` (`py_binary`) rewrites the baseline from the current tree, for use when lowering it. Its use to raise a count is a review-visible diff.
6. Generated pages (from `gen_docs.py`, `gen_instruction_format_md`) are not in `doc_sources` and are not checked; E08 (`ipu_gen`) owns their wording.
7. The conventions page (E01-12) and the review guideline (E01-13) mention the check and the baseline command.

## Tests

- [ ] The test passes on the tree with the generated baseline.
- [ ] Adding "VLIW word" to a page fails the test and names the glossary term "Instruction word".
- [ ] Removing one counted use fails with the "lower the baseline" message; after `bazel run //:glossary_baseline` it passes.
- [ ] A term inside a fenced code block or inline code is not counted.
- [ ] A change to `GLOSSARY.md` that adds an `_Avoid_` term is picked up without editing the test.

## Acceptance Criteria

- [ ] The check runs in the default tier and reads the avoid lists from `GLOSSARY.md` (no copy of the terms in the test, apart from the skip list).
- [ ] Baseline counts can only decrease without a visible baseline change in the diff.

## Risks

- Hardware documents (`docs/content/specs/`, `docs/content/requirements/`) may use "DRAM" for a real off-chip memory that is not XMEM. If so, the glossary should say which is which; until then those uses sit in the baseline. Flag it to the glossary owner rather than adding a special case.
