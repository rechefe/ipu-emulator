# Explanation: ISA idioms and kernel coverage

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-10 · Size S · Depends on: E09-06, E09-09, E03 (mnemonic standard rename, Q45), E06 (family `registry.toml` variants, Q76) · Behaviour change: none

## Goal

Two explanation pages for kernel authors: why kernels use certain instruction
sequences in place of operations the ISA lacks, and how the project knows which
kernel covers which computation.

## Background

- `isa-alias-catalogue.md` (413 lines, 16 `asm` blocks) explains the idioms
  ("every `ACC.*` and `AGG.*` op consumes `MULT_RES`", `:6-9`) and documents
  the alias profiler (`--profile-aliases`, `RunStats.alias_hits`, `:13-50`).
  One block fails to assemble (`:315`, a `...` placeholder; handled by E09-01).
- `app-coverage.md` (237 lines) explains the kernel registry: specs, discovery,
  resolution, framework layers, the `query` target (`:171-173`) and a
  "Current coverage" list (`:203-237`) that is maintained by hand.
- E03 renames mnemonics (`LDR_MULT_REG` → `LD.MULT`, `LDR_CYCLIC_MULT_REG` →
  `LD.MULT.CYCLIC`, and so on). The catalogue uses the old names.
- E06 replaces registry specs with a family `registry.toml` for shape variants
  (Q76). The coverage page describes the old mechanism.
- `isa_alias_spec.py` moves from `ipu_common` to `ipu_emu` (Q31).

## Implementation

1. `explanation/isa-idioms.md`, rewritten from `isa-alias-catalogue.md` with
   `anthropic-skills:technical-writing`:
   - Keep the idioms and their reasons. Use the new mnemonics.
   - Keep each idiom's example as a complete, assembling block.
   - Move the profiler how-to (flags, JSON export) to the debug how-to
     (E09-12) or the CLI reference (E09-15); link it from here.
2. `explanation/kernel-coverage.md`, rewritten from `app-coverage.md`:
   - Explain how a computation resolves to a kernel with the E06 mechanism
     (`registry.toml` variants).
   - Replace the hand-kept "Current coverage" list with the generated report
     (E06), included or linked. A hand-kept list is a second copy.
3. Delete the two old pages; fix inbound links.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks` passes with no fragment markers on
      these pages, unless a block is a loop body and says so.

## Acceptance Criteria

- [ ] Both pages use only post-rename mnemonics.
- [ ] No hand-maintained kernel coverage list remains in the docs.
