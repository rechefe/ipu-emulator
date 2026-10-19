# Explanation: architecture

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-17 · Size M · Depends on: E09-06, E03 (`ipu_isa` package), E04 (single cycle module, per-word plans, per-slot handlers, prototyping as adapter, reference-model API), E05 (assembly session), E06 (`ipu_harness`, `ipu_kernels`), E08 (`ipu_debug`, `ipu_gen`) · Behaviour change: none

## Goal

A contributor understands how the toolchain is put together: which packages
exist, what each one owns, which way dependencies point, how a program goes
from source to a run, and where to make a change.

## Background

- No page explains the architecture today. `SKILL.md:15-41` has a file tree
  for the old layout (`ipu-common`, `ipu-as-py`, `ipu-emu-py`, `ipu-apps`).
  `README.md:9-17` lists three packages. `docs/content/index.md:28-40` showed
  a tree that never existed (fixed in E09-03).
- Q69: the packages become `ipu_isa`, `ipu_as`, `ipu_emu`, `ipu_debug`,
  `ipu_gen`, `ipu_harness`, `ipu_kernels`, one directory each under
  `src/tools/`. Q62: one responsibility per package, one-way dependencies,
  enforced by Bazel deps and an import-graph test.
- D2, Q16: `instruction_spec.py` (in `ipu_isa`) is the single source of truth;
  one edit there propagates everywhere.
- Q7, Q72: prototyping mode stays as an opt-in fast path, built as an adapter
  over the faithful path (E04). `prototyping-mode.md` (80 lines) explains it
  today.
- Q11: only the reference-model API is stable; everything else is internal.
- E01 owns the code-conventions page (Q66). `GLOSSARY.md` and `docs/adr/` are
  at the repo root.

## Implementation

1. Write `explanation/architecture.md` with `anthropic-skills:technical-writing`:
   - The packages and what each owns, in one table, with the dependency
     direction as a diagram (mermaid, already enabled in `mkdocs.yml`).
   - The single source of truth: what is derived from `ipu_isa` (assembler
     tables, emulator dispatch, SystemVerilog package, docs, VS Code language
     data), and the "one edit" test that proves it.
   - Life of a program: `.asm` → assembly session → binary codec → emulator
     (per-word plan cached by PC → one cycle → per-slot handlers → commit
     record) → harness comparison.
   - Faithful and prototyping execution: what the adapter changes and what it
     guarantees. Fold `prototyping-mode.md` in and delete it.
   - Stable and internal: the reference-model API is the only stable
     interface (link E09-16).
   - Where to change what: a short table from task to package.
   - Links: `GLOSSARY.md` (terms), `docs/adr/` (decisions), code conventions
     (E01).
2. Use the deep-module vocabulary (module, interface, seam, adapter) as the
   glossary defines it.
3. The package table is checked by a test, or generated: Open question —
   generate it from the import-graph test's package list (E01/E03) so it cannot
   drift?
4. Update `README.md` "Python Packages" to link this page instead of keeping its
   own table.

## Tests

- [ ] `bazel build //docs:build_docs` shows no new warnings; the mermaid
      diagram renders.
- [ ] If the package table is generated: renaming a package changes the page.

## Acceptance Criteria

- [ ] One page explains the packages, their dependencies and the life of a
      program.
- [ ] `prototyping-mode.md` is folded in and deleted.
- [ ] `README.md` and `SKILL.md` link this page instead of listing packages.
