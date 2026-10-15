# How-to "Write a kernel", `kernel.toml` reference, kernel-page data

Part of [Epic: Kernel framework](00-epic.md) · ID E06-21 · Size S · Depends on: E06-13, E06-17, E09 (docs site skeleton) · Behaviour change: none

## Goal

A new author can write, run, test and debug a kernel from one how-to page. A
reference page lists every `kernel.toml` key. Each kernel also gets the data
that the generated kernel page (E08 `ipu_gen` / E09) shows: the design notes
from the `.asm` header (Q79), the ABI table, the parameter domain, the cases
and formats.

## Background

- `docs/content/adding-applications.md` (335 lines) and
  `docs/content/building-applications.md` (533 lines) describe the 5-file
  kernel, `IpuApp` subclassing and `set_cr` calls. Cluster plan 06 notes they
  lead with raw `IpuApp` (`building-applications.md` around line 10 and lines
  149-168).
- 49 kernels keep a CR map in comments. These maps drift, for example
  `softmax/softmax_rows/softmax_rows.asm:32` against `softmax/app.py:166`.
- Q79: design notes only in the `.asm` header, shown on a generated kernel docs
  page. Q37/E09: pages are organised by reader need, and are written with the
  technical-writing skill.

## Implementation

1. **How-to page** (in the E09 "How-to" section): "Write a kernel". It walks
   through `scale` end to end:
   - the folder and its files;
   - `kernel.toml` key by key;
   - `{{ cr.x }}` in the program;
   - `bazel test`, then `bazel run -- --case --format --debug --input --output --trace`;
   - adding a sweep, user data and a shared reference;
   - choosing a tolerance tier;
   - when to reach for `pack.py`, `test_*.py`, `stages.py` or a `registry.toml`
     entry.
   Follow CLAUDE.md conventions: upper-case mnemonics and registers, spaces
   between operands.
2. **Reference page**: the `kernel.toml` schema, generated from the
   `ipu_harness.manifest` module docstring and schema table, so it cannot drift.
3. **Kernel-page data**: `bazel run //kernels:manifest` gains per kernel:
   - `design_notes`: the leading `{#- ... -#}` block of the `.asm`;
   - `abi`: name, CR and kind (region, int, element scalar, derived);
   - `params`, `formats`, `cases` and the tier per format.
   E08/E09 render it. This issue only provides the data and a test that every
   kernel has a non-empty header.
4. Mark the old `adding-applications.md` and `building-applications.md`
   sections as legacy and link the new pages. E07-24 deletes the legacy
   sections.

## Tests

- [ ] The docs asm-block test (E09) assembles the `scale` listing on the how-to page.
- [ ] The reference page is regenerated in the build, and a stale checked-in copy fails the test.
- [ ] Every kernel in `//kernels:manifest` has `design_notes` and an `abi` table.

## Acceptance Criteria

- [ ] A reader can create a new kernel from the how-to page alone (reviewed by someone who has not written a kernel before).
- [ ] `bazel test //...` is green.
