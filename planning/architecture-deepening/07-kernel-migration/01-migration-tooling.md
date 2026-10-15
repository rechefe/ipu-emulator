# Migration tooling: `compare_legacy`, `compare_setup`, upper-case mnemonics

Part of [Epic: Kernel migration](00-epic.md) · ID E07-01 · Size S · Depends on: E02 (asm binary digests, kernel output digests), E06-13 · Behaviour change: none

## Goal

Give every family migration three tools. Each one turns a step of the per-family
checklist into a command with a pass/fail answer, so that reviewers check a
report instead of a diff of 2,000 lines.

## Background

- D3: refactor PRs must be output-identical, proven by golden harnesses. E02
  provides the asm binary digests (every `.asm`) and the kernel output digests
  (FP32, every case).
- The old harness's state after setup is the hard part to reproduce. Bases, CR
  values, LR presets (234 `set_lr` calls in 27 `app.py`) and the padding of
  XMEM rows all have to match for the binary and output digests to stay equal.
- Lower-case mnemonics are concentrated in convolutions (about 660 lines). The
  assembler accepts any case (CLAUDE.md), so upper-casing should be
  binary-identical.

## Implementation

1. **`compare_setup <old kernel> <new kernel dir> [--case C]`** (a `py_binary`
   in `//kernels/tools`, temporary, deleted in E07-24):
   - Build the old harness through `ipu_apps` `create_harness` and run only
     `setup()` on a fresh FP32 state.
   - Run the new framework up to the point just before `run` (plan, place,
     pack, CR writes).
   - Compare all CRs, all LRs and the XMEM bytes over the union of written
     regions. Report the first differences by register name and by
     region/row/lane.
2. **`compare_legacy <family>`** (same package):
   - For each old kernel in the family, check that the `assemble_<name>` digest
     of the new target equals the E02 asm digest.
   - For each case of each kernel, run the old harness and the new
     `run_kernel` on the same input values in FP32.
   - Check that the old output file still matches its E02 output digest. This
     is a sanity check on the baseline.
   - Check that the new `RunResult.raw_outputs` bytes equal the old harness's
     output-region bytes. Comparing XMEM regions, not files, makes the check
     independent of how each old `teardown` wrote its file (some transpose,
     `convolutions/conv_universal_wide384/app.py:316-327`).
   - Cycles must also be equal.
   - Print a Markdown table (kernel, case, binary ok, output ok, cycles old/new)
     for the PR description.
3. **`tools/upper_mnemonics.py <files...>`**:
   - Rewrite mnemonics to upper case using the assembler's tokenizer. Template
     syntax (`{{ }}`, `{% %}`, `{# #}`), labels and enum operand values (for
     example `identity` in `ACTIVATE.QUANTIZE identity CR15`) are left as they
     are. Raw register names (`lr0`, `cr15`) are upper-cased (`LR0`, `CR15`).
   - `--check` mode exits non-zero if any file would change. E07-24 turns this
     on for all of `kernels/`.

## Tests

- [ ] `compare_setup` reports no difference for an unchanged kernel, and reports `CR7` when a planted wrong `[abi]` value is introduced.
- [ ] `compare_legacy` flags a planted one-instruction change (binary) and a planted one-lane input change (output).
- [ ] `upper_mnemonics.py` on all current kernel `.asm` files leaves every E02 binary digest unchanged (dry run in the test, files not committed).
- [ ] Template blocks and enum values are byte-identical after the rewrite.

## Acceptance Criteria

- [ ] The three tools exist, are documented in the per-family checklist (epic), and are used by E07-02.
- [ ] `bazel test //...` is green.
