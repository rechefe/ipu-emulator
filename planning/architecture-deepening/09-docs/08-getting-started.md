# Rewrite Getting started

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-08 · Size S · Depends on: E09-06, E01 (pinned toolchain bootstrap and CI config), E06 (kernel runner command) · Behaviour change: none

## Goal

A new user goes from a fresh clone to a passing test run and one kernel run,
in one page, with commands that work on the first try.

## Background

- `docs/content/setup.md` (240 lines) lists GCC and "C/C++ Dependencies"
  (`:118`, `:138`), although the repo has no C or C++ code. It describes Docker,
  a dev container, WSL2 and CLion, and installing the assembler "locally with
  uv" (`:159`).
- `docs/guide/getting-started.tex` renders to a PDF published by
  `.github/workflows/getting-started-guide.yml`; the nav links the PDF.
- E01 adds a pinned toolchain bootstrap (Bazel version, bazelisk) and a CI
  config. E06 changes how a kernel is run
  (`bazel test //kernels/<family>/<name>`, `bazel run ... -- --case X --format e4`,
  decision Q80). Commands written before those land would be rewritten.
- E09-03 put a minimal Quick Start in `README.md`.

## Implementation

1. Write `getting-started/index.md` with `anthropic-skills:technical-writing`.
   Sections:
   - Prerequisites: the minimum (OS, the bootstrap script or Bazelisk). Link the
     E01 bootstrap.
   - Clone and test: `bazel test //...` and how long it takes (number from
     E01's CI timing).
   - Run one kernel: the E06 run command for the pilot family's FP32 case,
     what the output means (pass/fail, cycles, MAC utilisation).
   - Next steps: links to the four how-to pages.
2. Keep platform notes (WSL2, Docker, dev container, VS Code extension) in a
   short "Other environments" section of the same page, or in
   `getting-started/environments.md` if it is long. Delete the GCC, C/C++ and
   CLion content. Whether Docker and the dev container stay is E01's decision;
   follow it.
3. Delete `setup.md`; fix inbound links (`index.md`, `README.md`, `SKILL.md`).
4. Replace the README Quick Start commands with the same three commands and a
   link to this page.

## Open questions

- Does the LaTeX PDF guide stay alongside this page, or is it retired once this
  page exists? If it stays, who keeps it in sync?

## Tests

- [ ] `bazel test //docs:test_doc_commands` passes.
- [ ] Follow the page on a fresh clone in a clean container; record the
      commands and their results in the PR.

## Acceptance Criteria

- [ ] One page takes a reader from clone to a passing kernel run.
- [ ] No mention of GCC, C/C++ or local `uv` installs remains.
- [ ] `README.md` Quick Start and this page use the same commands.
