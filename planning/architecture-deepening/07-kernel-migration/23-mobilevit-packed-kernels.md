# Promote the 12 MobileViT packed programs to registered kernels

Part of [Epic: Kernel migration](00-epic.md) · ID E07-23 · Size M · Depends on: E07-06, E07-10, E06-17 · Behaviour change: none

## Goal

Turn the 12 `.asm` programs in `src/tools/ipu-apps/test/mobilevit/` into
registered kernels with `kernel.toml`, cases and references. Per L13, an
experiment is promoted if the L4/L5 chains use it and deleted otherwise. All 12
are used, so none is deleted. Today `bazel test //...` never assembles these
programs, and 4 of them do not assemble at all without test-time context.

## Background

- The programs, all under `test/mobilevit/` (2,743 asm lines in total):
  - `asm_packed_{layernorm,pack,unpack,residual_add}_{192x64,240x16}.asm`;
  - `asm_packed_output_linear_{generic,silu}.asm`, used by L5;
  - `asm_packed_output_linear_{generic,silu}_p4.asm`, used by L4.

  Every one of them is used by `test_full_layer_l4_packed.py` or
  `test_full_layer_l5_packed.py`.
- They are rendered with unsandboxed `jinja2.Template(...).render(...)`
  (`test_full_layer_l4_packed.py:190,231,253,281`, the same in l5, and
  `test_packed_layernorm_*.py:106`). The 4 `output_linear` programs need
  `chunk_widths` and fail under the official assembler (cluster plan 05).
- They have no `assemble_*` target. `ipu_app.bzl:185-188` passes them only as
  data to the manual `test_full_*` targets.
- Similarity once comments are stripped:
  - the `residual_add` pair is 1.00;
  - `output_linear` generic/SiLU is 0.99, for both the P4 and non-P4 pairs;
  - the `pack` pair is 0.90, the `layernorm` pair 0.88, and the `unpack` pair
    0.85;
  - P4 versus non-P4 `output_linear` is 0.77.
- The packed tests (`test_packed_layernorm_{192x64,240x16}.py`,
  `test_packed_residual_add_{192x64,240x16}.py`) are effectively these
  kernels' case tests.

## Implementation

- New family `kernels/mobilevit_packed/`. The family name is an open question
  below.
  - `packed_residual_add`: one template plus 2 `registry.toml` entries.
  - `packed_output_linear` and `packed_output_linear_p4`: two templates. Each
    takes `activation` (`none`, `silu`) and `chunk_widths` as declared
    `[params]`, with 2 entries each.
  - `packed_layernorm`, `packed_pack` and `packed_unpack`: a template plus
    entries if the digests stay identical, otherwise 2 folder kernels each.
- Digests: E02 records these programs with the template values the chain tests
  use (cluster plan 05, step 0). Each new `assemble_*` must match.
- Cases: port the 4 `test_packed_*.py` files into `[cases]` plus `gen.py` where
  inputs need structure. References: `ops.layernorm`, `ops.residual_add`,
  `ops.linear`, and a local `reference.py` for pack/unpack (a pure reshape, so
  `EXACT`).
- Delete the 12 `.asm` files and the 4 `test_packed_*.py` files from
  `test/mobilevit/`. The chain tests switch to the new kernels in E07-24.
- Apply the per-family checklist.

## Tests

- [ ] Every promoted kernel has an `assemble_*` target in `bazel test //...` whose digest matches E02.
- [ ] The ported packed cases produce outputs identical to the old `test_packed_*` runs (`compare_legacy` adapted to these four tests).
- [ ] No `jinja2` import remains under `test/mobilevit/` for these programs.

## Acceptance Criteria

- [ ] All 12 programs are registered kernels (or entries), assembled and tested by default.
- [ ] `bazel test //...` is green.

## Open question

- **Family name and placement.** Should these be (a) one family
  `mobilevit_packed`, or (b) spread over the existing families (normalize,
  elementwise, reshape, projections) by operation? (a) keeps the shared packed
  layout transforms in one `pack.py`. (b) matches routing by op. The drafts
  assume (a).
