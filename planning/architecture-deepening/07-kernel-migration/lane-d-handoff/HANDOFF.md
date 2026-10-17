# Lane D handoff: convolutions E07-18, E07-19, E07-20

Lane D ended on 2026-10-17 before these three issues were implemented. Lane A
continues them. E07-16 and E07-17 are done on lane D (see DELETIONS.md and
IMPLEMENTATION-DECISIONS.md, "Convolutions (1/4)" and "(2/4)").

`conv-drafts-E07-18-19-20.tar.gz` holds lane D's drafts. Extract it outside the
repo (for example `tar xzf ... -C /tmp/conv-drafts`). Each issue's `NOTES.md` is
its plan:

- the old CR/LR maps, with `app.py` line numbers;
- the regions, the pack transforms and the registry entries;
- the routing table, the cases;
- the "Tests from" tables ready for DELETIONS.md;
- the open questions and the latent bugs.

The `_check/` folders are throwaway proof scripts. They were run against a
scratch copy of the tree, not through Bazel.

## State of each draft

| Issue | Draft state | Proof recorded |
|---|---|---|
| E07-18 pointwise/depthwise twins | `pointwise_conv_tpl/`, `depthwise_conv_tpl/` (kernel.toml, `.asm`, BUILD), `convolutions/*_additions.{py,toml}`. Follows E07-17's conventions: one parameter set (programs read `has_bias`/`apply_relu`), every query param `query_required`. | All 39 cases equal E02 digests. compare_setup is IDENTICAL from poisoned registers; pointwise binds LR0 and LR7 as reset-zero presets. compare_legacy gives 39/39 `yes \| yes` with equal cycles. The routing answers are today's. Goldens and tolerances were NOT run (no torch in the check interpreter). |
| E07-19 wide384, stride-2 stages | `conv_universal_wide384/`, `depthwise_conv_stride2_{narrow,128,16}/` (stage kernels with `stages.py`), `convolutions/pack_additions.py`. | wide384: compare_legacy 4/4 (presets of LR0 and LR2-LR5). Stride-2: 10/10, with the H1 fix, which has now landed (277cc587, `stages.py` keeps the `[abi]` offsets). Routing: 13/13. |
| E07-20 bias without ReLU | `registry.toml` (3 entries), `_alt_templates/` (cross-check only). | CR expressions checked against 39 baseline records. **Stale:** written against f0fc8e2c with the old `bias`/`activation` program params. Rewrite it to the one-parameter-set form: each entry fixes `has_bias = true, apply_relu = false`, and the bias tensor's `when` is `"has_bias or apply_relu"` (the `or` grammar landed in 237122cb). |

## Traps

1. **Zero bias under the seeded generators.** E07-17's spec review found that
   `has_bias = false, apply_relu = true` packed and checked a random bias: the
   harness's default generator fills the whole `filters` tensor, and the pack
   and the reference both read its bias column. Lane D's fix: the pack writes 0
   and the reference skips the bias when `not has_bias`. E07-18's pointwise and
   depthwise packs (`pass0_bias`, `depthwise_filters`) and their references
   need the same treatment, plus a case at `(has_bias, apply_relu) = (F, T)`.
   The proof `gen.py` hides this, because it zeroes `b` itself.
2. **Commit with an explicit pathspec** (`git commit -m ... -- <paths>`). A
   plain `git commit` swept another agent's staged deletion into 277cc587.
3. **`fpb` cannot be `[derived]` from a bool** (`params._derived` takes ints
   only). Each depthwise entry fixes it, and the pack refuses a mismatch
   (E07-18 NOTES, Parameters and OQ4).
4. **Depthwise costs 1,** so that the C = 1 plain 3x3 query still answers
   `conv_universal`. An equal cost is a tie error (E07-18 OQ3).
5. **Frozen old files.** The family `ipu_apps/.../convolutions/app.py`,
   `cases.py` and `test.py`, and the `conv_universal/` folder, are kept until
   the E07-26 trace move. The width table in `test_width_routing_boundaries`
   names `conv_universal_wide384`: edit it, don't delete it, when E07-19
   removes wide384.
6. **Latent bugs, noted and not fixed**, filed as issues:
   - #258: the chunk count (`num_chunks`) rounds down while the pack rounds up
     (height 17, width 16);
   - #259: an oversize file is truncated (the pointwise weight file).

   Do not fix them in a refactor. Keep the programs' digests.
8. **The `conv_universal` Cycle Inspector trace move goes to lane A, at
   E07-25** (lane E has ended). Until then the frozen old folder and the
   family `app.py`, `cases.py` and `test.py` stay.
7. **The proof comes first, then the seeded generators.** Run the E07-01
   checklist from poisoned registers, then switch the cases to the seeded
   generators in a follow-up commit and re-measure the tolerance. E07-17's
   measurements: FP32_LONG, worst error 1.7e-5; FP32_ORDER failed 7 of 22
   cases.
