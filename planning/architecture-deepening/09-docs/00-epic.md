# Epic E09: Documentation restructure, drift tests and open questions

ID E09 · Folder `09-docs` · Runs alongside the other epics once each page's prerequisite lands.

## Goal

A reader can find what they need by what they are trying to do, every command
and assembly example in the docs works as written, and every known open
emulator or ISA question is recorded in one place.

Specifically:

1. The MkDocs site is organised by reader need: **Getting started**, **How-to**,
   **Reference**, **Explanation**, **Specs and requirements**, **Open questions**.
2. Two tests keep the hand-written docs honest: every `asm` block assembles (or
   is marked as a fragment), and every documented `ipu-as` command uses only
   real options.
3. `docs/content/requirements/open-questions.md` gains an "Emulator and ISA"
   section that lists every open emulator and ISA question, continuing the
   existing OQ numbering.
4. Each page group is rewritten, with the user's
   `anthropic-skills:technical-writing` skill, after the epic that changes that
   area lands. Pages are not rewritten twice.
5. `SKILL.md` becomes a short agent index that links to `docs/`.
6. `.github/CODEOWNERS` is added as the last issue of the whole program.

## Background

Verified at HEAD `694a8a8`:

- **Asm examples are broken and untested.** There are 25 fenced `asm` blocks in
  hand-written docs: 24 under `docs/content/` (isa-alias-catalogue 16,
  building-applications 3, debugging 3, ipu-configuration 2; recounted, the
  plan's "debugging 4" is 3) and 1 in `SKILL.md`. Running each through
  `ipu_as.diagnostics.check` (`diagnostics.py:468`) gives **8 failures**:
  - commas as operand separators: `building-applications.md:50`, `:69`,
    `ipu-configuration.md:81`, `:98`, `SKILL.md:69`;
  - the unknown mnemonic `ACC.FIRST`: `building-applications.md:338`;
  - `break_nop`, which is not an assembler mnemonic: `debugging.md:79`;
  - a `...` placeholder: `isa-alias-catalogue.md:315`.

  No test checks any of them.
- **Documented commands do not run.** `CLAUDE.md:10` and `SKILL.md:174` run
  `ipu-as -- assemble` without `--format`. `cli.py:16-21` declares
  `prompt="Choose a file format"`, so the command hangs or aborts when stdin is
  not a terminal. `SKILL.md:177` uses `--define` and a positional `.asm.j2`,
  and the CLI has neither. `adding-instruction.md:230` pipes to
  `assemble --format hex -`, but there is no `hex` format and `assemble` does
  not read stdin.
- **Stale facts.** `SKILL.md:9` and `docs/content/index.md:28` still say
  "ipu-c-samples". `index.md:3-9` describes a "C-based emulator".
  `adding-instruction.md:209` calls `state.regfile.set_pc`, which does not
  exist. `SKILL.md:108` lists `ReshapeMaskImmediate`, but the type is
  `LrOrReshapeMaskImmediate` (`instruction_spec.py:34`). `README.md:20-22`
  has an empty "Quick Start". `TODO.md` lists finished work (a softmax kernel, a
  CNN kernel and the LR union all exist).
- **History links 404.** `docs/hooks/history_link.py:44-46` builds a
  `commits/master/docs/content/<page>` URL for every page, including the
  generated ones (`instructions.md`, `operand-types.md`, `assembly-syntax.md`,
  `programmer-guide.md`, `instruction-format.md`, `release-log.md`; see
  `docs/BUILD.bazel:23-81`). Those paths do not exist in git.
- **Navigation is by origin, not by need.** `docs/mkdocs.yml` `nav` has
  "Getting Started", "Programming the IPU", "Kernels", "Contributing",
  "Assembly", "Specs", "HW Requirements". There is no reference for the CLI or
  for the emulator as a reference model, and no explanation of the timing model
  or element formats.
- **Open emulator and ISA questions are scattered** across plans, test
  comments (`test_execute.py:3139` pins an `OverflowError`) and the AaQ spec.
  The requirements already have one open-questions page with OQ-1 to OQ-17
  (`docs/content/requirements/open-questions.md`).

Plan 08 issue 8.6 is the evidence source for drift. Plan 07 lists the AaQ and
numeric facts.

## Decisions applied

- **Q26, Q36, Q54**: open questions go into the existing
  `docs/content/requirements/open-questions.md`, in a new "Emulator and ISA"
  section, with OQ numbers continuing from OQ-17. There is no separate page.
- **Q35, Q25, Q27, Q57**: AaQ is a placeholder and out of scope. Every AaQ
  item, including the crashes, is recorded as an open question and not fixed
  here. LayerNorm epsilon is tied to AaQ.
- **Q34**: the FP8 special-value convention is an open question.
- **Q53**: accumulator overflow and NaN follow the proposed arithmetic-events
  requirement (IEEE result, per-event sticky flag, never halt).
- **Q56, Q59**: ETISS is out of scope but kept in mind. F-8 is listed as "not
  modelled".
- **Q37**: restructure by reader need; skeleton and nav first; each page group
  is rewritten with `anthropic-skills:technical-writing` when its area lands.
  Specs and requirements content is unchanged.
- **Q49**: CR scalars are decoded in the active element format, and this must
  be documented (E09-19).
- **Q51, Q55**: the Cycle Inspector docs page is owned by E08. E09 only links
  it.
- **Q66**: the code-conventions page is owned by E01. E09 only links it.
- **Q29 / L31**: `SKILL.md` becomes a short index; `docs/` is canonical.
- **Q45**: mnemonics are renamed by an E03 script. The asm-block test (E09-01)
  catches any doc the script misses.
- **CODEOWNERS** (Round 10 note): deferred to the end of the program, after the
  package split.
- `GLOSSARY.md` and `docs/adr/` are written separately at the repo root. E09
  links them and does not write them.

## Scope

- `docs/mkdocs.yml` nav and section landing pages.
- Hand-written pages under `docs/content/` except `specs/` and
  `requirements/`, whose content is unchanged (the open-questions page gets a
  new section only).
- `README.md`, `SKILL.md`, `TODO.md`, and the command lines in `CLAUDE.md`.
- `docs/hooks/history_link.py`.
- Two new Bazel tests under `//docs`.
- `.github/CODEOWNERS`.

## Out of scope

- Docs build hermeticity, the release log, giscus and the Pages concurrency
  group (plan 08 issue 8.3). Assumed owned by E01.
- The doc generators themselves (`gen_docs.py`, `gen_codegen.py`, SVG and
  VS Code generators) move into `ipu_gen` in E08. E09-14 only states what the
  generated reference pages must show.
- Per-kernel catalogue pages, generated from the kernel `.asm` headers (Q79),
  owned by E06/E07. E09 places them in the nav.
- The Cycle Inspector page (E08) and its phase 2 live mode.
- The code-conventions page (E01), `GLOSSARY.md` and ADRs.
- Any code behaviour change. All E09 issues are documentation, tests of
  documentation, or repository metadata.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|---|---|---|---|---|---|
| E09-01 | [01-docs-asm-block-and-command-tests.md](01-docs-asm-block-and-command-tests.md) | Test every `asm` block and every documented `ipu-as` command | M | none | none |
| E09-02 | [02-drift-fixes-claude-skill-adding-instruction.md](02-drift-fixes-claude-skill-adding-instruction.md) | Fix drift in CLAUDE.md, SKILL.md and the adding-instruction walkthrough | S | E09-01 | none |
| E09-03 | [03-readme-quick-start-index-todo.md](03-readme-quick-start-index-todo.md) | README Quick Start, docs home page facts, remove TODO.md | S | none | none |
| E09-04 | [04-history-link-generated-pages.md](04-history-link-generated-pages.md) | Stop "page history" links 404ing on generated pages | S | none | none |
| E09-05 | [05-open-questions-emulator-and-isa.md](05-open-questions-emulator-and-isa.md) | Add the "Emulator and ISA" section to open-questions.md | S | none | none |
| E09-06 | [06-site-skeleton-and-nav.md](06-site-skeleton-and-nav.md) | Site skeleton: nav by reader need and section landing pages | M | E09-05 | none |
| E09-07 | [07-skill-md-short-index.md](07-skill-md-short-index.md) | Reduce SKILL.md to a short agent index into docs/ | S | E09-01, E09-02, E09-06 | none |
| E09-08 | [08-getting-started.md](08-getting-started.md) | Rewrite Getting started | S | E09-06, E01 toolchain bootstrap, E06 kernel runner | none |
| E09-09 | [09-how-to-write-a-kernel.md](09-how-to-write-a-kernel.md) | How-to: write a kernel | M | E09-06, E06 kernel framework, E07 pilot family | none |
| E09-10 | [10-explanation-isa-idioms-and-kernel-coverage.md](10-explanation-isa-idioms-and-kernel-coverage.md) | Explanation: ISA idioms and kernel coverage | S | E09-06, E09-09, E03 mnemonic rename, E06 family registry | none |
| E09-11 | [11-how-to-add-an-instruction.md](11-how-to-add-an-instruction.md) | How-to: add an instruction | S | E09-02, E09-06, E03 "one edit" test, E04 per-slot handlers | none |
| E09-12 | [12-how-to-debug-a-kernel.md](12-how-to-debug-a-kernel.md) | How-to: debug a kernel | M | E09-06, E08 `ipu_debug`, E06 runner owns debug | none |
| E09-13 | [13-how-to-reference-model-cocotb.md](13-how-to-reference-model-cocotb.md) | How-to: use the reference model from a cocotb testbench | S | E09-06, E04 reference-model API and conformance suite | none |
| E09-14 | [14-reference-isa.md](14-reference-isa.md) | Reference: ISA pages generated from `ipu_isa` | M | E09-04, E09-06, E03 `ipu_isa` + derived syntax + mnemonic rename, E08 `ipu_gen` | none |
| E09-15 | [15-reference-cli.md](15-reference-cli.md) | Reference: command-line tools | S | E09-01, E09-06, E05 CLI, E06 runner, E08 debugger CLI | none |
| E09-16 | [16-reference-model-api.md](16-reference-model-api.md) | Reference: reference-model API and trace schema | S | E09-06, E09-13, E04 reference-model API + JSON schema | none |
| E09-17 | [17-explanation-architecture.md](17-explanation-architecture.md) | Explanation: architecture | M | E09-06, E03 package split, E04 core (single cycle, prototyping adapter), E05 assembly session, E06/E08 packages | none |
| E09-18 | [18-explanation-timing-model.md](18-explanation-timing-model.md) | Explanation: timing model | S | E09-06, E03 timing contract, E04 timing behaviour PRs | none |
| E09-19 | [19-explanation-element-formats.md](19-explanation-element-formats.md) | Explanation: element formats, including CR scalar decoding | M | E09-05, E09-06, E03 element-format enum, E04 format codecs + CR scalar behaviour PR | none |
| E09-20 | [20-codeowners.md](20-codeowners.md) | Add `.github/CODEOWNERS` (last issue of the program) | S | every other issue of E01-E09 | none |

All E09 issues are "Behaviour change: none" for the emulator and assembler.
They change docs, doc tests and repository metadata only.

## Suggested order

```
Phase 0 (now, no prerequisites):
  E09-01 ──▶ E09-02 ──┐
  E09-03              │
  E09-04              │
  E09-05 ──▶ E09-06 ──┴──▶ E09-07

Page rewrites (each waits for its area's epic):
  E09-06 ─┬─▶ E09-08  (after E01 bootstrap, E06 runner)
          ├─▶ E09-09  (after E06 framework, E07 pilot) ──▶ E09-10 (after E03 rename, E06 registry)
          ├─▶ E09-11  (after E03 one-edit test, E04 handlers)
          ├─▶ E09-12  (after E08 ipu_debug)
          ├─▶ E09-13 ─▶ E09-16  (after E04 reference API)
          ├─▶ E09-14  (after E03 ipu_isa, E08 ipu_gen)
          ├─▶ E09-15  (after E05 CLI)
          ├─▶ E09-17  (after E03/E04/E06/E08 package split)
          ├─▶ E09-18  (after E03 timing contract, E04 timing PRs)
          └─▶ E09-19  (after E03 element formats, E04 codecs + CR scalar PR)

Last issue of the whole program:
  everything ──▶ E09-20 (CODEOWNERS)
```

E09-01 lands first because every later page rewrite must keep it green. E09-05
can land at any time; it records the questions, it does not answer them.

## Rules for every page rewrite (E09-08 to E09-19)

- Draft or edit the page with the `anthropic-skills:technical-writing` skill.
- Follow the CLAUDE.md conventions: mnemonics and register names in upper case,
  operand placeholders in lower case, operands separated by spaces, pseudocode
  sizes in elements.
- Every `asm` block assembles under E09-01, or is marked as a fragment with a
  reason.
- Every documented `ipu-as` command passes E09-01's command test.
- Move the page to its new path (for example `how-to/write-a-kernel.md`), delete
  the pages it replaces, and fix every inbound link. `mkdocs build` must show no
  new warnings.
- Use terms from `GLOSSARY.md`. Link an ADR instead of restating a decision.
