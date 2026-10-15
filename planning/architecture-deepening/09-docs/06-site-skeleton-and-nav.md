# Site skeleton: nav by reader need and section landing pages

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-06 · Size M · Depends on: E09-05 · Behaviour change: none

## Goal

The docs site navigation is organised by what the reader is trying to do. Every
later page rewrite (E09-08 to E09-19) has a fixed place to land. No page content
is rewritten in this issue.

## Background

- `docs/mkdocs.yml` `nav` today groups pages by origin: "Getting Started",
  "Programming the IPU", "Kernels", "Contributing", "Assembly", "Specs",
  "HW Requirements", "Release Log". A reader who wants to "debug my kernel"
  or "look up `ACC.STRIDE`" has to know which group the author chose.
- There is no page for the CLI reference, the reference-model API, the timing
  model or the element formats.
- Q37 sets the new top level: Getting started / How-to / Reference /
  Explanation / Specs and requirements / Open questions. Specs and requirements
  content is unchanged.
- Q54: the open-questions page stays at `requirements/open-questions.md`.
- Q55, Q66: the Cycle Inspector page (E08) and the code-conventions page (E01)
  are written by other epics and linked from here.
- `GLOSSARY.md` and `docs/adr/` are at the repo root, outside the MkDocs
  `docs_dir` (`docs/content`).

## Implementation

1. **Nav.** Replace `nav` in `docs/mkdocs.yml`. Existing pages keep their file
   paths in this issue, so no inbound link breaks; each rewrite issue moves its
   page later.

   ```yaml
   nav:
     - Home: index.md
     - Getting started:
         - getting-started/index.md
         - Setup: setup.md
         - Getting Started guide (PDF): https://github.com/rechefe/ipu-emulator/releases/download/guide-latest/getting-started.pdf
     - How-to:
         - how-to/index.md
         - Write a kernel: building-applications.md        # E09-09
         - Register a kernel: adding-applications.md       # E09-09 merges it
         - Add an instruction: adding-instruction.md       # E09-11
         - Debug a kernel: debugging.md                    # E09-12; Cycle Inspector page (E08) goes here
         - Use the reference model from cocotb: how-to/reference-model-cocotb.md   # placeholder, E09-13
     - Reference:
         - reference/index.md
         - ISA:
             - Instructions: instructions.md
             - Operand types: operand-types.md
             - Assembly syntax: assembly-syntax.md
             - Instruction format: instruction-format.md
             - Programmer's guide: programmer-guide.md
             - IPU configuration: ipu-configuration.md     # E09-14 folds it in
         - Command-line tools: reference/cli.md            # placeholder, E09-15
         - Reference model API: reference/reference-model-api.md   # placeholder, E09-16
         - Kernels: [existing kernels/*.md pages]          # E06/E07 generate these
         - Code conventions: <E01 page>                    # link only
         - Glossary: reference/glossary.md
     - Explanation:
         - explanation/index.md
         - Architecture: explanation/architecture.md       # placeholder, E09-17
         - Prototyping mode: prototyping-mode.md           # E09-17 folds it in
         - Timing model: explanation/timing-model.md       # placeholder, E09-18
         - Element formats: wide-vector-debug-mode.md      # E09-19 replaces it
         - ISA idioms: isa-alias-catalogue.md              # E09-10
         - Kernel coverage: app-coverage.md                # E09-10
     - Specs and requirements:
         - Specs: [unchanged list]
         - HW requirements: [unchanged list, without open-questions.md]
     - Open questions: requirements/open-questions.md
     - Release log: release-log.md
   ```

   The exact titles may change in review; the six sections may not.
2. **Section landing pages.** `getting-started/index.md`, `how-to/index.md`,
   `reference/index.md`, `explanation/index.md`. Each is a few lines: who the
   section is for, and a list of its pages with one sentence each. Write them
   with `anthropic-skills:technical-writing`. With `navigation.indexes` enabled
   (add it to `theme.features`), the landing page is the section's own page.
3. **Placeholders.** Each placeholder page has a title, one sentence saying
   what the page will cover, and a link to its E09 issue. No other content.
4. **Glossary.** `reference/glossary.md` renders the root `GLOSSARY.md` with the
   enabled `pymdownx.snippets` extension (`--8<-- "GLOSSARY.md"`). Add
   `//:GLOSSARY.md` to the `build_docs` sources and set the snippets
   `base_path` so the Bazel temp tree finds it. If `GLOSSARY.md` does not exist
   yet when this lands, the page links to the GitHub path instead.
5. **ADRs.** `explanation/index.md` links the `docs/adr/` directory on GitHub.
6. **Home page.** `index.md` becomes a short router: one line per section,
   pointing at the landing pages, plus the badges. Keep it under 40 lines.
7. **Not-in-nav check.** Set `validation.nav.omitted_files: warn` (the locked
   MkDocs is 1.6.1, which supports it) so a page added without a nav entry shows a
   warning.

## Open questions

- Page URLs change when each rewrite moves its file. Add the
  `mkdocs-redirects` plugin (a new docs dependency through E01's lockfile), or
  accept that old URLs break once? Decide before E09-08 lands.
- Kernel catalogue pages: under Reference (proposed here) or a top-level
  "Kernels" section? Q37 does not list one.

## Tests

- [ ] `bazel build //docs:build_docs` succeeds with no new warnings.
- [ ] Every page under `docs/content` is in the nav or explicitly excluded
      (`not_in_nav`).
- [ ] `bazel test //docs:test_doc_asm_blocks` stays green.

## Acceptance Criteria

- [ ] The site's top level is exactly: Home, Getting started, How-to,
      Reference, Explanation, Specs and requirements, Open questions,
      Release log.
- [ ] Each of the four new sections has a landing page.
- [ ] No existing page's content changes, except `index.md`.
- [ ] Every placeholder links to the issue that fills it.
- [ ] Specs and requirements pages are unchanged; only their nav location
      moves (open questions gets its own top-level entry).
