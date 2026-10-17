# Attention (2/2): `attn_v` x3 and `attn_v_bcast` x3

Part of [Epic: Kernel migration](00-epic.md) · ID E07-13 · Size M · Depends on: E07-12, E04 (behaviour PR fixing the 3 `attn_v_*` kernels for the timing contract, Q32) · Behaviour change: none

## Goal

Migrate the 6 value kernels and finish the attention family. Do it after E04's
timing-contract behaviour PR has changed the `attn_v_*` programs, so the E02
digests are re-baselined once, in that PR, and this migration compares against
the new baseline.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/attention/`.

- `attn_v_{16x60,64x48,256x36}`: `app.py` files of 130-176 lines, 9-12
  `set_cr`, no LR presets.
- `attn_v_bcast_{36,48,60}`: `app.py` files of 144-172 lines, 8-10 `set_cr`,
  2-4 `set_lr`. `bcast_36`/`bcast_60` are 0.78 similar once comments are
  stripped.
- Q32: three `attn_v_*` kernels read LRs in a way the new timing contract
  changes. Fixing them is a behaviour PR in E04, which updates their E02
  digests. The E04 issue names the three; this issue takes them as given.

## Implementation

- One folder kernel each.
- Golden: the primary is `op = "matmul"` (P V), with the datapath mirror as the
  additional `EXACT` check (as in E07-12).
- `FAMILY_DEPS = {"attention": ["softmax"]}` is declared (E06-18). Any family
  test that runs softmax uses the registry, not a direct import.
- Delete the old `attention/` directory: `app.py`, `cases.py`, `test.py` and
  all kernel folders. (Done in part: the MobileViT tests still import nine of
  the old kernel folders, so they, `app.py` and `cases.py` stay as a shim that
  E07-24 deletes; DELETIONS.md lists its users.)
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` and `compare_legacy` are identical for the 6 kernels, against the post-E04 digests.
- [ ] The import-graph test passes with only the declared `attention -> softmax` edge.

## Acceptance Criteria

- [ ] The attention family is fully migrated.
- [ ] `bazel test //...` is green.
