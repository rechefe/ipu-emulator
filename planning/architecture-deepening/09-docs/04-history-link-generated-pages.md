# Stop "page history" links 404ing on generated pages

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-04 · Size S · Depends on: none · Behaviour change: none

## Goal

The "history" link on every docs page goes to a real GitHub history: the page
source for hand-written pages, the generator for generated pages.

## Background

- `docs/hooks/history_link.py:26-47` (`on_page_context`) builds
  `<repo>/commits/<branch>/docs/content/<rel>` for every page.
- Six pages are generated at build time and do not exist under
  `docs/content/` in git (`docs/BUILD.bazel:23-81`):
  `instructions.md`, `operand-types.md`, `assembly-syntax.md`,
  `programmer-guide.md` (from `gen_docs`), `instruction-format.md` (from
  `gen_instruction_format_md`) and `release-log.md` (from
  `update_release_log.py`). Their history links 404.
- The generators move in E08 (`ipu_gen`), and E09-06 changes page paths. The
  mapping must therefore live in one place that those issues update.

## Implementation

1. Add a mapping in `docs/mkdocs.yml` `extra`, next to `docs_git_branch`:

   ```yaml
   extra:
     generated_pages:
       instructions.md: src/tools/ipu-as-py/src/ipu_as/gen_docs.py
       operand-types.md: src/tools/ipu-as-py/src/ipu_as/gen_docs.py
       assembly-syntax.md: src/tools/ipu-as-py/src/ipu_as/gen_docs.py
       programmer-guide.md: src/tools/ipu-as-py/src/ipu_as/gen_docs.py
       instruction-format.md: src/tools/ipu-as-py/src/ipu_as/gen_docs.py
       release-log.md: docs/update_release_log.py
   ```

   All five ISA pages come from `ipu_as/gen_docs.py` today (`docs/gen_docs_wrapper.py`, `docs/gen_instruction_format_md_wrapper.py`).
2. In `history_link.py`, look up the page's `rel` path in `generated_pages`
   first. If found, link the generator's history. If not found and the page is
   not a file under `docs/content` in the repo, set no `history_url`
   (`docs/overrides/main.html:5` already hides the link when it is unset).
3. Add a build-time consistency check in the hook: if a page listed in
   `generated_pages` is a real file in `docs/content`, or a generator path in
   the map does not exist, log a MkDocs warning naming it.

## Tests

- [ ] `docs/hooks/test_history_link.py` (`py_test`): hand-written page →
      `docs/content/<rel>`; mapped generated page → generator path; unmapped
      page outside `docs/content` → no URL.
- [ ] Build the site and check that the history link of `instructions.md`
      and `release-log.md` points to an existing file (manual check, recorded
      in the PR).

## Acceptance Criteria

- [ ] No history link on the built site points to a path that does not exist
      in the repo.
- [ ] The generated-pages map is the only place to update when a generator
      moves (E08) or a page moves (E09-06 onwards). The E08 `ipu_gen` issue and
      E09-14 update it.
