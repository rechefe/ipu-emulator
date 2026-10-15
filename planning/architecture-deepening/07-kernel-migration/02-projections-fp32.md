# Pilot: projections onto `ipu_harness` (FP32, 12 `registry.toml` entries over 3 templates)

Part of [Epic: Kernel migration](00-epic.md) · ID E07-02 · Size M · Depends on: E07-01, E06-17 (and through it E06-01..E06-16) · Behaviour change: none

## Goal

Migrate the 12 `proj_*_{144,192,240}_p4` kernels to 3 template kernels plus a
12-entry `kernels/projections/registry.toml`. Prove binaries and FP32 outputs
identical to the E02 digests. This is the first real user of every E06 piece
except stages, so gaps found here are fixed in E06 before the other families
start.

## Background

- Files today (`src/tools/ipu-apps/src/ipu_apps/kernels/projections/`): 1,532
  lines of Python and 2,854 lines of assembly. That is the family `app.py`
  (383), `cases.py` (93) and `test.py` (191), plus 12 folders with 5 files each.
- With comments stripped, the 12 `.asm` bodies are 4 distinct programs:

  | Body | Kernels |
  |------|---------|
  | single token group, identity | `proj_{qkv,outproj,ffn2}_{192,240}_p4` (6) |
  | single token group, SiLU | `proj_ffn1_{192,240}_p4` (2) |
  | two token groups, identity | `proj_{qkv,outproj,ffn2}_144_p4` (3) |
  | two token groups, SiLU | `proj_ffn1_144_p4` (1) |

  The two-group bodies use a different LR allocation (cluster plan 05). Merging
  them with the single-group body would change binaries.
- The harness: `ProjectionLayout` (`app.py:92-161`) computes bases and derived
  values. `ProjectionP4App.setup` (`app.py:213-233`) writes 13 CRs:
  `DATA_BASE`, `WEIGHTS_BASE`, `OUTPUT_BASE`, `NEG_ONE = -1`,
  `FULL_BOUND = 126`, `TAIL_BOUND`, `W_STRIDE`, `N_OUT_CR`, `CHUNK_COUNT`,
  `LAST_CHUNK_IDX`, `DATA_STREAM_STR`, `OUT_STREAM_STR`, `STREAM_COUNT = 4`.
- Loaders: input is channel-major per stream (`app.py:186-199`); weights are
  output-major, padded per chunk (`app.py:201-211`). The output is raw XMEM
  rows with the first `n_tok` lanes valid (`app.py:235-245`).
- Known comment drift fixed by moving to `[abi]`: `proj_*_240_p4.asm:41` says
  `K=240 needs ceil(192/128)`, and `proj_ffn2_*:42` gives wrong chunk counts
  (cluster plan 05).
- The family `test.py` covers:
  - routing: each kernel found by exactly `(k, n_out, activation)`; the wrong
    activation refused; FFN1 is exactly the SiLU kernels; other stream counts
    refused;
  - an activation contract: the SPEC's activation equals the `.asm`
    `ACTIVATE.QUANTIZE`;
  - an opt-in slow weight-pointer mutation test (`IPU_RUN_SLOW_TESTS`).

## Implementation

1. **Templates.**
   - `kernels/projections/proj_p4/`: the single-group body, with `activation`
     as a template parameter (`{{ activation }}` in `ACTIVATE.QUANTIZE`).
   - `kernels/projections/proj_p4_2tg/`: the two-group identity body.
   - `kernels/projections/proj_p4_2tg_silu/`: the two-group SiLU body.
   - Each template has a `kernel.toml` with `[params]` `k`, `n_out`, `n_tok`,
     `n_tg`, `activation` (choices `none`, `silu`), plus `n_streams` fixed at 4.
   - Inputs: `input` with shape `[4, "n_tg", "k", "n_tok"]` and
     `pack = "channel_major"`; `weights` with shape `["n_out", "k"]` and
     `pack = "output_major_weights"`.
   - Output: `output` with shape `[4, "n_tg", "n_out", "n_tok"]`,
     `pack = "projection_output"`. This is a family `pack.py` transform with
     grouped channel-major rows and lanes past `n_tok` dropped on unpack.
   - The `[abi]` table gives the 13 CRs above by name. `NEG_ONE`, `FULL_BOUND`,
     `TAIL_BOUND`, `W_STRIDE`, `CHUNK_COUNT`, `LAST_CHUNK_IDX`,
     `DATA_STREAM_STR`, `OUT_STREAM_STR` and `STREAM_COUNT` are `[derived]`
     values (E06-06).
2. **`registry.toml`**: 12 entries with today's names, `params` from today's
   `ProjectionLayout(...)` lines, and a `doc` line ("Layer 4 FFN1", from
   `projection_spec`'s role/layer naming, `app.py:332-348`). The split is:
   8 entries on `proj_p4`, 3 on `proj_p4_2tg`, 1 on `proj_p4_2tg_silu`.
3. **Cases.** `default` keeps today's data so the outputs can be compared bit
   for bit: `RandomState(0xC0FFEE)`, D per stream then W, `uniform(-1, 1)`
   (`cases.py:77-93`). Use a family `gen.py:legacy_uniform` generator. The
   reference is the shared `ops.linear` (matmul plus optional SiLU). The FP32
   tier is chosen from measurement and recorded (today it is `rtol=1e-4, atol=1e-3`).
4. **Design notes.** Merge the `app.py` docstrings (family and per kernel) into
   the template `.asm` headers. Keep the runtime chunk loop rationale already
   in the `.asm`.
5. **Family tests** move to `kernels/projections/test_routing.py`, using
   `Registry` and `//kernels:query`. The activation-contract test is no longer
   needed, because the activation is now a template parameter rendered into the
   program. Delete it, and note this in the PR. The slow mutation test becomes
   a `tier = "nightly"` case set.
6. Delete `src/tools/ipu-apps/src/ipu_apps/kernels/projections/`. Repoint the
   MobileViT chain users (`test/mobilevit/fixture_kernel_support.py`, the L4/L5
   tests) to the new kernels through a temporary adapter. E07-23 removes it.

## Tests

- [ ] `compare_setup` shows identical CRs, LRs and XMEM after setup for all 12 kernels.
- [ ] `compare_legacy projections` shows 12/12 binaries equal to the E02 asm digests, every case's FP32 raw output equal, and cycles equal.
- [ ] `bazel test //kernels/projections/...` passes. The routing tests pass against the new registry.
- [ ] The MobileViT non-manual tests still pass. Run the manual L4/L5 chains by name once (hours), with results in the PR.

## Acceptance Criteria

- [ ] `kernels/projections/` holds 3 template folders, `registry.toml`, `pack.py`, `gen.py` and `test_routing.py`. No `app.py`, `cases.py`, `__init__.py` or per-shape folder remains.
- [ ] Target names (`proj_ffn1_192_p4`, ...) are unchanged under `//kernels/projections:`.
- [ ] Every E06 gap found is filed or fixed in E06 before E07-04 starts. List them in the PR.
- [ ] `bazel test //...` is green.
