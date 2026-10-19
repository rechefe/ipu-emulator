# How-to: write a kernel

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-09 · Size M · Depends on: E09-06, E06 (`ipu_harness`: kernel.toml, placement, per-format packing, PyTorch reference, generated tests, runner), E07 pilot family (projections in FP32, INT8, E4) · Behaviour change: none

## Goal

One how-to page takes a kernel author from an empty folder to a kernel that
runs and is tested in every element format: write `<name>.asm`, write
`kernel.toml`, optionally a PyTorch reference, run it, test it.

## Background

- Two pages describe the current, older contract:
  - `building-applications.md` (533 lines): `IpuApp` subclasses with `setup`
    and `teardown`, hand-placed XMEM addresses, Bazel targets, reusable cases,
    regression tests, run statistics, and a fully_connected walkthrough
    (`:330-517`). It also holds the `ACTIVATE.QUANTIZE` notes (`:63-118`) and
    dstructure configuration (`:17-58`).
  - `adding-applications.md` (335 lines): registry specs, constructor guards,
    framework layers, `MemoryApp`.
- E06 replaces this contract (decisions Q80 to Q83): a kernel is `<name>.asm`
  plus `kernel.toml` (`[kernel]`, `[params]`, `[inputs]`/`[outputs]`, `[abi]`,
  `[cases]`) plus an optional `reference.py`. The framework places regions,
  packs per format, programs CRs, assembles with case parameters, runs to
  `HALT`, compares against the reference with a format tolerance tier.
  `MemoryApp` and the whole-image path are deleted (Q83). Cases may name `.npy`
  inputs and expected outputs (Q82).
- E07 migrates the projections family first, in FP32, INT8 and E4 (Q50). That
  family is the worked example.
- Q79: kernel design notes live only in the `.asm` header and appear on the
  generated per-kernel page (E06/E07). The how-to links those pages; it does
  not copy them.

## Implementation

1. Write `how-to/write-a-kernel.md` with `anthropic-skills:technical-writing`.
   Sections:
   - What a kernel is (three files) and where it lives.
   - Step 1: the assembly. Template parameters from the case (`{{ cr.x }}` or
     the E05 template API), the ABI (name to CR), ending with `HALT`.
   - Step 2: `kernel.toml`, one table per section, using the pilot kernel as a
     complete, real example (included from the repo with
     `pymdownx.snippets`, so it cannot drift).
   - Step 3: the reference (shared reference by op, or a `reference.py`), and
     how to test against your own `.npy` data instead.
   - Step 4: run and test: the `bazel test` and `bazel run ... -- --case X
     --format e4` commands, and what a failure report looks like.
   - Escape hatches, briefly: `pack.py`, extra `test_*.py`, the stages hook.
     Link the E06 reference for details.
   - Debugging: one paragraph and a link to E09-12.
2. Move what is not a how-to step:
   - `ACTIVATE.QUANTIZE` notes → the ISA reference (E09-14) and OQ-20.
   - dstructure configuration → the ISA reference (E09-14).
   - Run statistics → the CLI or runner reference (E09-15).
3. Delete `building-applications.md` and `adding-applications.md`; fix every
   inbound link (`SKILL.md`, `README.md`, `debugging.md`, kernel pages).
4. Every `asm` block on the page assembles (E09-01), or is included from a
   real kernel file.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks //docs:test_doc_commands` passes.
- [ ] Follow the page on a scratch branch to add a trivial kernel (for example
      an element-wise copy) in FP32 and E4; its generated test passes. Record the
      steps in the PR. Do not merge the scratch kernel.

## Acceptance Criteria

- [ ] One page covers writing, running and testing a kernel.
- [ ] No mention of `IpuApp.setup/teardown`, `MemoryApp` or hand-placed XMEM
      addresses remains in `docs/content`.
- [ ] The `kernel.toml` example is included from a real file, not copied.
