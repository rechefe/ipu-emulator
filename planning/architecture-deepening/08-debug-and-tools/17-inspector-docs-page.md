# Cycle Inspector docs page (phase 1)

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-17 · Size M · Depends on: E08-16 · Behaviour change: none (new docs page)

## Goal

The docs site gains a "Cycle Inspector" page (Q55, phase 1). It is the trace-replay viewer from the artifact, served as a static page from the MkDocs site and fed by the Bazel-generated traces from E08-16. Readers step through real kernels cycle by cycle, with the listing, the VLIW word, the register file and the pipeline view, without installing anything.

## Background

- The artifact page is self-contained HTML, CSS and JS (about 75 KB). It loads Google Fonts and no script libraries, and fetches `data/isa.json` and `data/<id>.json`. Its kernel list (`KERNELS`) is hard-coded in the script.
- `docs/BUILD.bazel:85-150` (`build_docs`) assembles the site from `:doc_sources` and the generated pages into `site.tar`. The genrule is tagged `docs` and `no-sandbox`.
- D9: JavaScript stays out of Bazel. The page is a static asset copied into the site; there is no JS toolchain and no bundling.
- Q37 / E09: the docs restructure puts debugging under a reader-need section. E09 decides where the page sits in the nav; this issue adds it under the current "Getting Started → Debugging" group.

## Implementation

1. Check in the page as `docs/content/inspector/index.html` plus `inspector.js` and `inspector.css`. Take them from the artifact source with these changes:
   - the kernel list is loaded from `data/kernels.json`, generated from `curated.toml` by E08-16's target, instead of being hard-coded;
   - fetch paths are relative to the page;
   - an unknown `schemaVersion` in a trace shows "this trace is newer than the viewer";
   - the "Assembly source" tab shows the kernel's `.asm` from the trace's `source` field.

   Keep the existing light/dark theming and keyboard controls.
2. `build_docs` copies `//docs/inspector:inspector_data` into `content/inspector/data/` in the temp tree (same pattern as the generated pages).
3. Add a nav entry, `Cycle Inspector: inspector/index.html`, and a short Markdown intro page (`docs/content/debugging-inspector.md`): what the page shows, how traces are produced (`kernel_trace`), and how to add a kernel.
4. Licence and privacy note: the page loads Google Fonts. Self-host the two font families under `content/inspector/fonts/`, or fall back to system fonts, so the docs site makes no third-party request. Open question: self-host or system fonts?
5. A docs smoke test (tagged `docs`) extracts `site.tar` and checks that `inspector/index.html`, `data/kernels.json` and every listed trace exist and parse.

## Tests

- [ ] The docs smoke test passes: `bazel test //docs:test_inspector_site` (tag `docs`).
- [ ] Manual check (screenshots in the PR): load the page from `bazel run //docs:serve_docs`, step through fully_connected INT8 and conv_universal, switch themes, and use the phone-width layout.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] The published docs site has the Cycle Inspector page with the curated kernels.
- [ ] Rebuilding the docs after an ISA or kernel change regenerates the traces with no manual step.
- [ ] No trace or ISA JSON is checked in under `docs/`.
