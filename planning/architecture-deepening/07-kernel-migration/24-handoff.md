# E07-24 hand-off (lane C to lane A, 2026-10-17)

A working note for whoever finishes E07-24. Delete it in the commit that
finishes the issue.

## Done (lane C, on `claude/tender-pascal-5fsay7-lane-c`)

All eight seam tests run on one `ipu_harness.chain.Chain` in
`kernels/mobilevit/` (a family without kernels; decisions row "MobileViT
chain tests (E07-24)"), `//kernels/mobilevit:tests`, sharded 6 ways:

| New file | Replaces (deleted, mapped in DELETIONS.md) |
|---|---|
| `test_seam_qkv_to_scores_km.py` | `test_seam_proj_qkv_to_scores_km_repack.py` |
| `test_seam_layernorm_to_matmul.py` | `test_seam_layernorm_matmul_xmem_direct{,_l3,_l5}.py` |
| `test_seam_unfold_to_matmul.py` | `test_seam_unfold_matmul_xmem_direct_l{4,5}.py` |
| `test_seam_qkv_matmul_to_scores_km.py` | `test_seam_qkv_matmul_scores_km_64x48.py` |
| `test_seam_pipeline_boundaries.py` | `test_seam_pipeline_boundaries.py` (src/tools/ipu-apps) |

Also deleted: `test_kernel_registry_new_families.py`, `fixture_reshape.py`,
the direct-handoff helpers of `fixture_kernel_support.py`, and 10 of the 12
old `matmul_*_x128` folders (`matmul_720x240_x128` and `matmul_384x192_x128`
stay for `tools/bench` until E07-25).

## Left

1. **The unpacked full-layer chains** `src/tools/ipu-apps/test/mobilevit/test_full_layer_l{4,5}.py`
   (922 and 836 lines) with `fixture_full_layer_chain.py` (552) and
   `fixture_host_softmax_fold_stubs.py` (91). Not started; no draft exists.
   Lane B ports the packed ones (`test_full_layer_l{4,5}_packed.py`, E07-23's
   kernels).
2. **Then delete the shims** (E07-24 step 5): the attention shim
   (`src/tools/ipu-apps/src/ipu_apps/kernels/attention/`, nine folders and
   `app.py`, `cases.py`, `__init__.py`; its only users are the four full-layer
   tests and `fixture_full_layer_chain.py`), the MobileViT users of the
   softmax shim (`softmax_rows_partial`, `softmax_columns_packed`: used by
   `fixture_full_layer_chain.real_softmax_*`), and the fixtures
   (`fixture_kernel_support.py`, `fixture_projections.py`,
   `fixture_normalize.py`, `fixture_elementwise.py` once nothing imports
   them). Each deletion needs a DELETIONS.md row and a "Tests from" table.

## How the full-layer port maps onto the seams

Each `_run_layer_*` is a chain of the kernels the seams already chain; reuse
their patterns (see the four new seam files):

| Old helper (`fixture_full_layer_chain.py`) | New step |
|---|---|
| `run_layernorm` | `chain.run("layernorm_64x192", {}, {"x", "gamma", "beta"})`; `x` is pinned at row 0 |
| `run_proj_p4` | `chain.run("proj_qkv_192_p4", {}, {"b": d[:, None], "a": w})` (all 4 streams in one step) |
| `run_qk_scores_query_major_all_blocks`, `run_attn_scores_km_one_head` | `qk_scores_64x48` / `attn_scores_km_64x48`, inputs repacked on the host as in `test_seam_qkv_to_scores_km.py` (`repack`, `key_major`) |
| `run_attn_v_query_major_all_blocks`, `run_attn_v_bcast` | `attn_v_64x48` / `attn_v_bcast_48` (b = P, a = V, both pack `"padded"`); `kernels/attention/test_seams.py` shows the score-to-P hand-off in place (`at` = a slice of a reserved P) |
| `real_softmax_query_major` / `_key_major` | `softmax_rows_partial` / `softmax_columns_packed` from `kernels/softmax` (inputs are the scores' first `n_tok` lanes; `kernels/attention/test_softmax_feed.py` shows the shape refusal) |
| `run_residual_add` | `residual_add_64x192` (`a` pinned at row 0) |
| out-proj / FFN | `proj_outproj_192_p4`, `proj_ffn1_192_p4`, `proj_ffn2_192_p4`, or the single-stream `matmul_*_x128` entries |

## Traps

- **Row 0.** LayerNorm `x`, the matmul `x128` data `b`, the residual add's
  `a`, `qk_scores_16x60`/`256x36` and `attn_scores_km_16x60`/`256x36`'s `b`
  all read their base through CR0, so they are pinned at row 0. A producer's
  output can never sit there while its own row-0 input does: copy the stored
  rows with `chain.write(rows, at=0, pack=...)`. A reservation made while row
  0 is free lands on it: reserve a guard of the pinned input's rows first
  (`test_seam_qkv_to_scores_km.py::_scores`).
- **Packs.** A resident passes only when its pack name is the consumer's:
  scores are `"padded"` (as attn_v's P), the matmul data `"channel_major"`,
  most others `None`. `chain.write(..., pack=...)` relabels host rows.
- **Tolerances on chained data.** The score kernels and `attn_v_bcast_48`
  exceed FP32_ORDER's 1e-6 floor on chained (larger, cancelling) data: hold
  those steps to their datapath mirror (`ipu_kernels.reference.datapath`,
  `acc_fold`/`agg_fold`) bit for bit instead of `check=True`. The coordinator
  logged it as an owner question. LayerNorm, the projections, the matmuls and
  the residual add pass `check=True` on chained data.
- **LayerNorm eps.** The full-layer references use `eps = 1e-5`
  (`test_full_layer_l4.py:122,169,890`); the kernel reference has none (Q57,
  OQ-27). The issue says use `ops.layernorm` (no eps) in the chain reference.
- **Seeding.** `_stable_seed` (zlib) already replaced `hash((tag, "filler"))`;
  the issue's `rng(tag, "filler")` (E06-08) changes the data, so record the
  per-stage errors before and after in the PR.
- **`per_stage_error` tests** only print: E07-24 step 3 makes them assert
  budgets (measured x 2, tier overrides with a `why`).
- **Run time.** Each full-layer test takes about 80-90 minutes; they are
  `manual`/`enormous` and run by name with `--config=full`. Run them once at
  the end.
- **Softmax stubs.** By default the full-layer tests use the host softmax
  stubs (`fixture_host_softmax_fold_stubs.py`); an environment switch
  (`test_full_layer_l4.py:63`) selects the real softmax kernels.
- **DELETIONS.md** has the softmax-shim paragraph three times (a merge
  artifact already in base); clean it up when the shim goes.
