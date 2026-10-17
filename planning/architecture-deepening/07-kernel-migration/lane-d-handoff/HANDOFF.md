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

All three were re-run at 20:40 UTC on 9df1033f, with that commit's
`kernels/convolutions` copied into a scratch tree. No Bazel was used, and the
goldens were not run (the checking interpreter has no torch).

| Issue | Draft state | Proof recorded |
|---|---|---|
| E07-18 pointwise/depthwise twins | `pointwise_conv_tpl/`, `depthwise_conv_tpl/` (kernel.toml, `.asm`, `reference.py`, BUILD), `convolutions/*_additions.{py,toml}`. Follows E07-17: one parameter set, every query param `query_required`. The zero-bias fix (trap 1) is applied to `pass0_bias` and `depthwise_filters` and to both references. A `relu_no_bias` (F, T) case is on both templates; the pointwise bn entry runs it, and depthwise keeps (F, T) refused. Pointwise uses a `reference.py`, not `op = "conv2d"`: the op golden would pass the random `b`. | E02 digests 39/39. compare_setup 39/39 IDENTICAL; pointwise's LR0/LR7 presets read as "bound reset value". compare_legacy 39/39 `yes \| yes` with equal cycles. Routing unchanged. **To do:** goldens, tolerances, DELETIONS rows, the in-tree commit and the seeded-generator follow-up. |
| E07-19 wide384, stride-2 stages | `conv_universal_wide384/`, `depthwise_conv_stride2_{narrow,128,16}/` (stage kernels with `stages.py`), `convolutions/pack_additions.py`. No monkeypatch: the stages fix is in the tree (277cc587). | compare_legacy 14/14 (wide384 4, stride-2 10, summed cycles equal). wide384 compare_setup 4/4 IDENTICAL. Routing 13/13. **To do:** as for E07-18; it lands after E07-18. |
| E07-20 bias without ReLU | `registry.toml` with 3 entries in the one-parameter-set form: `has_bias = true, apply_relu = false`, all params `query_required`, and depthwise also `fpb = 25`. | `conv_universal_bias` and `depthwise_conv_universal_bias` route (T, F); all 12 cases of each run to HALT, and each renders a new binary. **Blocker (decision needed):** `pointwise_conv_unified_bias` renders the plain program. Its bias region needs `when = "has_bias or apply_relu"`, but CR10's unread `tail_size` binding would then need `not has_bias and not apply_relu`, and the conditions have no `and`. Either add `and`, or drop the unread `tail_size` binding and log the CR10 compare_setup difference. Not checked: goldens, new E02 digest rows, `test_routing.py` edits, the coverage doc. |

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
