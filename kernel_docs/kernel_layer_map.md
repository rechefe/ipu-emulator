# Kernel → layer map

Which MobileViT transformer layer each app in `src/tools/ipu-apps/src/ipu_apps/`
belongs to. The on-disk layout is intentionally **flat** — kernels are *not*
grouped into `L3/`/`L4/`/`L5/` directories, because each kernel is merged to
master individually and a flat `ipu_apps.<kernel>` import path keeps every
cherry-pick self-contained. This file is the grouping.

All apps run in wide-vector FP32 debug mode. `bazel test //src/tools/ipu-apps/... //src/tools/ipu-emu-py/... //src/tools/ipu-as-py/... //src/tools/ipu-common/...`
→ **75/75 targets, 441 test cases** as of 2026-08-06 (includes all 16
`proj_*_p4` kernels — the 8 from 2026-08-06's first pass plus the 4 L3
`proj_*_144_p4` kernels and the fix to a real weight-pointer bug found while
building them — 3 new direct-XMEM seam tests, 1 new QKV→scores repack seam
test, and the L5 matmul output-crop fix, all added since the 2026-08-04
count; `bazel test //...` also builds `//docs:build_docs`, which can fail
locally on a stale read-only `bazel-out` artifact from an earlier build —
unrelated to app/emulator code, safe to ignore or `bazel clean` if it blocks
a full-repo run). Some individual `proj_*_p4` / seam targets run 3–8 minutes
each; a full run under default bazel test-size timeouts with many targets
competing for local resources can spuriously time out one or two of the
slower targets even though they pass cleanly in isolation or with
`--test_timeout` raised — rerun the specific target before treating a
timeout as a real failure.

---

## Layer 3 — d=144, N=256 tokens (2 token groups × 128), h=4, head_dim=36

| Kernel | Role |
|---|---|
| `unfold_32x32x144` | space-to-depth, 4 stride-2 streams |
| `layernorm_256x144` | layer norm |
| `residual_add_256x144` | residual add |
| `qk_scores_256x36` | QKᵀ scores, **query-major** |
| `attn_v_256x36` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_256x36` | QKᵀ scores, **key-major** |
| `attn_v_bcast_36` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_144x144_x128` | OutProj, single stream (K=144→144), **identity** |
| `matmul_288x144_x128` | FFN1, single stream (K=144→288), **silu** |
| `matmul_432x144_x128` | QKV, single stream (K=144→432), **identity** |
| `matmul_144x288_x128` | FFN2, single stream (K=288→144), **identity** |
| `proj_qkv_144_p4` | QKV, all P=4 streams in one invocation (K=144→432), **identity** |
| `proj_outproj_144_p4` | OutProj, all P=4 streams in one invocation (K=144→144), **identity** |
| `proj_ffn1_144_p4` | FFN1, all P=4 streams in one invocation (K=144→288), **silu** |
| `proj_ffn2_144_p4` | FFN2, all P=4 streams in one invocation (K=288→144), **identity** |

## Layer 4 — d=192, N=64 tokens/stream, P=4, h=4, head_dim=48, L=4

| Kernel | Role |
|---|---|
| `unfold_16x16x192` | space-to-depth, 4 stride-2 streams |
| `layernorm_64x192` | layer norm |
| `residual_add_64x192` | residual add |
| `qk_scores_64x48` | QKᵀ scores, **query-major** |
| `attn_v_64x48` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_64x48` | QKᵀ scores, **key-major** |
| `attn_v_bcast_48` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_192x192_x128` | OutProj, single stream (K=192→192), **identity** |
| `matmul_384x192_x128` | FFN1, single stream (K=192→384), **silu** |
| `matmul_576x192_x128` | QKV, single stream (K=192→576), **identity** |
| `matmul_192x384_x128` | FFN2, single stream (K=384→192), **identity** |
| `proj_qkv_192_p4` | QKV, all P=4 streams in one invocation (K=192→576), **identity** |
| `proj_outproj_192_p4` | OutProj, all P=4 streams in one invocation (K=192→192), **identity** |
| `proj_ffn1_192_p4` | FFN1, all P=4 streams in one invocation (K=192→384), **silu** |
| `proj_ffn2_192_p4` | FFN2, all P=4 streams in one invocation (K=384→192), **identity** |

## Layer 5 — d=240, N=16 tokens/stream, P=4, h=4, head_dim=60, L=3

| Kernel | Role |
|---|---|
| `unfold_8x8x240` | space-to-depth, 4 stride-2 streams |
| `layernorm_16x240` | layer norm |
| `residual_add_16x240` | residual add |
| `qk_scores_16x60` | QKᵀ scores, **query-major** |
| `attn_v_16x60` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_16x60` | QKᵀ scores, **key-major** |
| `attn_v_bcast_60` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_240x240_x128` | OutProj, single stream (K=240→240), **identity** |
| `matmul_480x240_x128` | FFN1, single stream (K=240→480), **silu** |
| `matmul_720x240_x128` | QKV, single stream (K=240→720), **identity** |
| `matmul_240x480_x128` | FFN2, single stream (K=480→240), **identity** |
| `proj_qkv_240_p4` | QKV, all P=4 streams in one invocation (K=240→720), **identity** |
| `proj_outproj_240_p4` | OutProj, all P=4 streams in one invocation (K=240→240), **identity** |
| `proj_ffn1_240_p4` | FFN1, all P=4 streams in one invocation (K=240→480), **silu** |
| `proj_ffn2_240_p4` | FFN2, all P=4 streams in one invocation (K=480→240), **identity** |

## Not layer-specific

| Kernel | Why |
|---|---|
| `fully_connected` | FC block, not part of a transformer layer |
| `layernorm_128x16` | 16-channel layer norm, not one of the three layer shapes |
| `matmul_64x64x64` | generic matmul harness |
| `matmul_128x64x64` | generic matmul harness |
| `matmul_128x64x128` | generic matmul harness |
| `matmul_128x128` | generic matmul harness |

---

## Two attention mappings — never mix them

Each layer implements **both** score mappings, and the chains are used end to end
without reshaping in between:

- **query-major:** `qk_scores_*` → `attn_v_*`
- **key-major:** `attn_scores_km_*` → `attn_v_bcast_*`

They produce bit-different results by design and have separate goldens.

**The `attn_v_bcast_*` kernels use `MULT.RC.VE` + `ACC.ADD[.FIRST]` — no AGG.**
Their references are a per-step-rounded float32 left-fold. The AGG-based
`attn_v_*` siblings instead need a per-128-chunk float64 left-fold rounded to
float32 on each `R_ACC` write. Using the AGG recipe for a bcast kernel (or vice
versa) yields a subtly wrong golden that can still pass a loose tolerance.

### Attention kernels expect pre-scaled Q — no kernel applies `1/√head_dim`

None of `qk_scores_*` / `attn_scores_km_*` apply the standard attention scale
`1/√head_dim`, and none should: the design folds the scale into the Q rows of
the QKV projection weight matrix ahead of time, once, rather than paying for a
scale instruction in every score kernel. Because query-major and key-major
compute the same product, a Q pre-scaled once serves both chains.

Expected per-layer scale (apply once, to Q, before either chain runs):

| Layer | head_dim | scale = 1/√head_dim |
|---|---|---|
| L3 | 36 | 0.166667 |
| L4 | 48 | 0.144338 |
| L5 | 60 | 0.129099 |

**This fold currently has no implementation site.** There is no QKV weight
generator anywhere in this repo — every `matmul_*_x128` "projection" kernel
(including the two labeled QKV, `matmul_576x192_x128` / `matmul_720x240_x128`)
generates unscaled, unrelated random weights (`rng.uniform(-1, 1, ...)`,
self-contained per kernel by design), and every `qk_scores_*` /
`attn_scores_km_*` kernel's `gen_debug_data.py` and test file synthesizes Q
directly as unscaled random data with no projection step at all
(`Q = rng.uniform(-1, 1, ...)`, not `Q = X @ W_q`). There is nothing to fold
the scale into yet, and no test currently asserts a Q is scaled. The one prior
trace of this plan is `kernel_docs/L4_L5_phase0_findings.md`'s "zero-cost items
still unasserted" note. When a real QKV weight generator is written, it must
apply this scale to Q and assert it did (so an unscaled generator fails loudly
— an unscaled Q makes L3 scores ~6× too large and the softmax that follows
near one-hot).

## `proj_*_p4` — multi-stream projection kernels (all 3 layers)

The `matmul_*_x128` "projection" kernels each process **one pixel-stream per
invocation** — a real transformer layer needs the same projection applied to
all P=4 streams, which would otherwise cost 4 host round-trips per matmul.
`proj_<role>_<d>_p4` (`role` ∈ {`qkv`, `outproj`, `ffn1`, `ffn2`}, `d` ∈ {144,
192, 240}) wraps the identical arithmetic in one invocation that loops all 4
streams internally, sharing one weight matrix `W` across streams (a real
per-layer property — one set of learned weights, four streams needing the
same projection), the same way `attn_v_256x36` loops its 4 heads internally
instead of being called once per head.

**The 12 single-stream `matmul_*_x128` kernels above are untouched and remain
the general-purpose, individually-tested base** — `proj_*_p4` is built on top
of them, not a replacement.

**L3 was incorrectly excluded from this family until 2026-08-06.** The
earlier reasoning ("L3 has no `P` stream partition... L3's matmuls consume
all N=256 tokens directly with no stream split") conflated two different
claims: L3's single-stream matmuls genuinely never PACK tokens from multiple
streams into one 128-lane row (`L4_L5_phase0_findings.md`'s §4 finding, still
true) — but that says nothing about whether a full L3 layer needs the
projection applied once per stream, which it does, exactly like L4/L5.
`unfold_32x32x144` emits 4 streams at L3 same as the other two layers; nothing
in the repo looped them at the matmul stage before this fix, so a real L3
layer needed 4 host round-trips same as L4/L5 did before `proj_*_192_p4` /
`proj_*_240_p4` existed. `proj_*_144_p4` closes that gap the same way.

Design points, shared by all 16 `proj_*_p4` kernels:
- **Stream loop outermost**, then the output-channel (`j`) loop, then a
  **runtime chunk loop** over the K-dimension contraction. Streams outside
  the whole j-loop means the single-stream kernels' already-debugged
  MULT-snapshot-contract priming/biasing (issue #157) is reused unchanged —
  only a per-stream base-row offset is added before each j-loop.
- **Runtime chunk loop, not per-shape hand-unrolled `k_chunk0/1/2` labels.**
  One chunk-loop body runs `CHUNK_COUNT = ceil(K/128)` times; the harness
  supplies `CHUNK_COUNT` and the last chunk's `TAIL_BOUND` (every non-last
  chunk is a fixed width-128 → bound 126). The same `.asm` control-flow text
  handles K=144/192/240/288/384/480 without new labels per shape.
- FFN1 variants (`proj_ffn1_144_p4`, `proj_ffn1_192_p4`, `proj_ffn1_240_p4`)
  apply `ACTIVATE.QUANTIZE silu`; all others use `identity` — see the FFN
  activation note below.
- Same one-channel-per-row, row-based `.asm` operand, full-row-store
  conventions as every other kernel in this repo.

**L3's one real structural difference: `N_TG=2`.** L3 has N=256 tokens per
stream (two 128-token groups), unlike L4/L5's single token group (`N_TG=1`).
The single-stream ancestor `matmul_144x144_x128.asm` handles this by
hand-duplicating a `tg=0`/`tg=1` block inside its `j_loop` rather than a
runtime tg loop (L3's D layout interleaves tg within each channel's row-pair,
so a real tg loop would need its own runtime-selected data-pointer stride on
top of the already-runtime chunk loop, for only 2 fixed iterations — not
worth it). The four `proj_*_144_p4` kernels mirror this: `stream_loop ->
j_loop -> {tg=0 block, tg=1 block}`, each running the same runtime chunk-loop
body as the L4/L5 family. This is the only nesting-depth difference from
L4/L5's `stream_loop -> j_loop -> chunk_loop`.

**Bug found and fixed while building `proj_outproj_144_p4` (2026-08-06):** the
first draft reset the per-`j` weight-row pointer (`weight_row_off`) to zero at
the top of `j_loop` AND again at the start of the tg=1 block, both of which
stomped the `ADD weight_row_off weight_row_off W_STRIDE` advance at the
bottom of the loop — every output channel `j` silently read `j=0`'s weight
row. 99.3% of output elements mismatched (only `j=0` was ever correct). The
fix: `weight_row_off` is now reset exactly once per stream (alongside
`j_idx`, at `stream_loop`'s top), left untouched across both tg blocks (W has
no tg axis — tg=1 reuses tg=0's weight row for the same `j`), and advanced
exactly once per `j` at the loop's bottom. This is a copy-paste-under-a-new-
nesting-level class of bug, worth flagging for any future kernel that adds a
loop level around an existing per-iteration-advanced pointer: audit every
place the pointer could be touched, not just where it's supposed to be.

## FFN activation is `silu`, not `identity`

Every FFN1 kernel (`matmul_288x144_x128` / `proj_ffn1_144_p4` at L3;
`matmul_384x192_x128` / `proj_ffn1_192_p4` at L4; `matmul_480x240_x128` /
`proj_ffn1_240_p4` at L5) applies `ACTIVATE.QUANTIZE silu` (`x * sigmoid(x)`,
ISA id 11) at its store. This was previously `identity` everywhere — two
stacked linear layers with no nonlinearity between them collapse to a single
linear layer, so the FFN block was doing less than intended until this
changed. FFN2 kernels still use `identity` (nothing follows FFN2 inside the
FFN block itself).

## LayerNorm/unfold → matmul: XMEM content already agrees; only file staging doesn't (2026-08-06)

Prior seam-audit context established that `matmul_*_x128`'s `_load_data`
reads a *tightly packed disk file* and row-expands it into XMEM — a property
of the FILE format the harness imposes, not of XMEM itself. This round tested
that directly: run the real producer kernel (`layernorm_*` or `unfold_*`),
capture its raw XMEM output bytes via `state.xmem.read_address` (no file
round-trip), write those bytes verbatim into the matmul's DATA region at the
correct one-row-per-channel stride (bypassing `_load_data` entirely), and
compare against an independent reference. Poison-then-mutate methodology
throughout; see `test_seam_layernorm_matmul_xmem_direct*.py` and
`test_seam_unfold_matmul_xmem_direct_l{4,5}.py`.

**Result: the raw XMEM content already agrees, at every layer, for both
pairings** — max abs error 1e-5 to 3e-4 (FP32 debug-mode rounding), confirmed
at L3/L4/L5 for LayerNorm→matmul and at L4/L5 for unfold→matmul (L3's unfold
has no analogous crop concern — `unfold_32x32x144` emits full 512 B rows,
N=256 fills two token groups exactly). This means:

- `layernorm_*`'s one-zero-padded-row-per-channel output IS
  `matmul_*_x128`'s expected DATA layout, byte-for-byte, once placed at the
  right row stride. The `layernorm_64x192 → matmul_576x192_x128` "DEFECT"
  recorded in `test_seam_pipeline_boundaries.py` is real but **scoped to the
  FILE-staging code path only**: `_load_data` parses ANY input file as
  tightly-packed (`K*N_TOK*ELEM_BYTES` bytes, no row padding) regardless of
  what actually produced it, so LayerNorm's full-row file is misread there.
  The fix, if ever built, belongs in the FILE contract (either give
  `_load_data` a full-row-aware mode, or write a correctly-strided loader
  that copies XMEM→XMEM without a tightly-packed-file detour) — it does NOT
  need a value-transforming restaging kernel, because there is no value
  transform to do.
- `unfold_*`'s per-channel row layout (`N_TOK` valid lanes + a stale tail —
  see `unfold_16x16x192`'s docstring) is ALSO already what `matmul_*_x128`
  expects, and the stale tail lanes are structurally inert: `matmul_*_x128`'s
  `MULT.RC.VE`/`ACC.ADD` datapath is per-SIMD-lane-independent (see the
  "Cross-lane reduction vs. per-lane independence" section below), and the
  SIMD lane axis is the token axis here, so garbage past lane `N_TOK` can
  only ever land in output lanes past `N_TOK` — which every consumer already
  crops. Proven directly (not just architecturally) by
  `test_unfold_tail_lanes_are_structurally_inert_when_poisoned`: filling
  every tail lane with `999.0` before the handoff left the valid-lane matmul
  output unchanged (max abs error ~1e-5). One correction to the
  `unfold_16x16x192` docstring's premise: the emulator's actual tail-lane
  content in a fresh run reads back as `0.0`, not non-zero garbage — `r_acc`
  starts zeroed and `ACC.STRIDE` only writes its selected output slots (see
  `ipu.py:execute_acc_stride`), so "stale" here means "whatever `r_acc` held
  before this store," which happens to still be its zeroed reset state in an
  isolated single-kernel run. The docstring's "garbage" framing is the
  correct WORST CASE to design against (which the poison test now verifies
  directly) but is not what a fresh run actually observes.

**Conclusion for the two seam defects named at the top of this file's
history:** neither is a missing *data-transformation* bridge. The LayerNorm
seam is a file-contract mismatch fixable in the harness/loader layer with no
new arithmetic; the unfold seam was never actually broken, just untested end
to end. The QKV→scores seam (next section) is the one place in this audit
that DOES need a real, if mechanical, repack.

## QKV projection → attention scores: repack bridge (2026-08-06)

`proj_qkv_192_p4`/`proj_qkv_240_p4` (all 4 streams, one invocation) cannot
feed `attn_scores_km_64x48`/`attn_scores_km_16x60` via pure base+stride
addressing: the projection's store pitch is a full padded row per channel
(512 B, `N_TOK` valid + zero pad) while the score kernels expect a tightly
packed file (`N_TOK * ELEM_BYTES` per channel, no row padding) — the same
class of pitch mismatch as the LayerNorm seam above, but here on the READ
side of a kernel that has no file-bypass option (its own `setup()` always
parses a flat tightly-packed file, there is no XMEM-native alternative).

The bridge is `repack_qkv_to_km()` in
`test_seam_proj_qkv_to_scores_km_repack.py`: strip each row's padding lanes
down to `N_TOK`, keep every index (stream, head, channel order) exactly as
the projection produced it — no permutation, no transpose, no value
transform. Verified two ways: (1) an index-injective test
(`test_repack_is_padding_strip_only_no_permutation`) that encodes each
`(stream, head, channel, token)` position as a unique decodable float value
and checks every output element lands at the SAME index it started at — an
axis swap or reorder would fail this exact-match check even if a
numeric-tolerance test wouldn't; (2) full pipeline agreement at both L4
(`proj_qkv_192_p4 → repack → attn_scores_km_64x48`, one QKV invocation
covering all 4 streams) and L5 (`proj_qkv_240_p4 → repack → attn_scores_km_16x60`,
looped 4x since `attn_scores_km_16x60`, unlike its L4 sibling, takes one
stream's Q/K per call), both against an independent `W @ D` reference, both
poisoned-destination and mutation-first. Head-concatenation before
`out_proj` still needs no repack — confirmed pure addressing in the prior
round, unchanged by this round.

## Layer-independent conventions

- One output channel per XMEM row; sub-row store strides are a bug. Where a
  channel's output is shorter than a row, the row is still exclusively its own.
- Memory-region bases are derived from row counts, never hardcoded byte maps
  (a byte map sized for 1-byte elements overflows 4× at FP32).
- Every store goes through `ACTIVATE.QUANTIZE` + `STR_POST_AAQ_REG`, co-issued in
  one VLIW word where the activation is `identity`. The simulation-only
  `STR_ACC_REG` appears in no kernel.
- Each kernel owns its `gen_debug_data.py` and imports only from its own package
  (plus `ipu_apps.base`), so any single kernel can be cherry-picked alone.
- Unfold emits **per-stream** output; the packed `[k][p·n]` layout remains a
  deferred experiment.

## Crop convention: producers emit full rows, only the final consumer crops

Where a kernel's payload is narrower than a row (`N < LANES`), there are two
*separate* things that can crop, and they must not be conflated:

1. **The kernel's XMEM store pitch** — the row stride `STR_POST_AAQ_REG` writes
   with, inside the emulator. This is **always a whole row** (`ROW_BYTES`),
   never a packed sub-row stride. A store narrower than a row is the "sub-row
   store stride" bug named above.
2. **The harness's file-output extraction** — the byte range `teardown()` (or
   whatever reads XMEM into the output file) pulls out of each row.

The settled rule governs (2), not (1): **the *producer's* `teardown()` emits
the full, uncropped row into its output file; only the *final consumer* in a
chain crops down to the valid prefix**, in its own input-staging code (or, for
a kernel with no downstream consumer, in its test). A kernel is a "final
consumer" if nothing else reads its output — most single-kernel tests crop
there for convenience, which is fine; the rule binds when a kernel's output
*is* another kernel's staged input.

Rationale: a producer's `teardown()` cropping to `N * ELEM_BYTES` silently
picks a pitch that only one particular consumer's staging convention happens
to agree with. Any other consumer addressing "one \<unit\> per row" (which is
the default addressing mode per the row-per-channel convention above) reads
the wrong bytes with no error — this was exactly the `attn_scores_km_64x48` /
`attn_v_bcast_48` bug: `attn_scores_km_64x48` cropped its key rows to
`N * ELEM_BYTES` (256 B) before writing them, while `attn_v_bcast_48` stages
its `p_path` **verbatim** and addresses one key per whole 512 B row — so every
key row after the first was read at the wrong offset. The fix was to make
`attn_scores_km_64x48.teardown()` emit full rows (matching `attn_scores_km_256x36`,
which already did, and `attn_v_bcast_48`'s expectation) and move the crop into
the *test's* own reshape.

As of this audit, kernel status:
- **Full rows (correct):** `qk_scores_256x36`, `attn_v_256x36`,
  `attn_scores_km_256x36`, `attn_v_bcast_36` (L3 — N=256 fills rows exactly, no
  crop needed anyway), `attn_v_16x60`, `attn_v_bcast_60`, `attn_scores_km_16x60`
  (dump full rows via `dump_xmem_to_binary`, test/consumer crops),
  `attn_scores_km_64x48` (fixed by this audit), and — as of the 2026-08-06
  follow-up audit — `matmul_240x240_x128`, `matmul_480x240_x128`,
  `matmul_240x480_x128`, `matmul_720x240_x128` (all 4 L5 `matmul_*_x128`
  kernels: `OUTPUT_ROW_BYTES` was `N_TOK*ELEM_BYTES`, cropped, in all 4; now
  `512`, matching every L3/L4 matmul kernel and `residual_add_16x240`'s
  full-row input assertion — see "L5 matmul output-crop fix" below).
- **Producer crops (still narrower than the rule, but harmless because no
  downstream kernel consumes them verbatim — flagged for consistency, not
  correctness):** `qk_scores_64x48`, `attn_v_64x48`, `attn_v_bcast_48`,
  `qk_scores_16x60`, `layernorm_16x240`, `layernorm_64x192`, `unfold_*`
  (`unfold_8x8x240`'s `teardown` writes both the raw uncropped rows and a
  cropped convenience file). None of these currently feed another kernel's
  verbatim-staged input, so there is no live pitch-mismatch bug today — but a
  future chain that stages one of these verbatim would hit the same class of
  bug 1c fixed. Prefer the full-row form in new kernels.

### L5 matmul output-crop fix (2026-08-06)

All four L5 `matmul_*_x128` kernels (`matmul_240x240_x128` OutProj,
`matmul_480x240_x128` FFN1, `matmul_240x480_x128` FFN2, `matmul_720x240_x128`
QKV) had `OUTPUT_ROW_BYTES = N_TOK * ELEM_BYTES` (64 B) and cropped their
`teardown()` output to that pitch — the same class of bug the
`attn_scores_km_64x48` fix above addressed, just in the matmul family instead
of the attention family. This blocked `matmul_240x240_x128` (L5 OutProj) from
feeding `residual_add_16x240` verbatim: `residual_add_16x240.setup()`
hard-asserts full `N_ROWS*ROW_BYTES` (240*512 B) input and rejected the
cropped 64 B/channel file outright (see
`test_seam_pipeline_boundaries.py`'s now-fixed
`test_seam_outproj_240x240_to_residual_add_16x240_agrees`, formerly
`..._pitch_mismatch`). All four L5 matmul kernels are fixed together (not
just OutProj) since the crop was the same code pattern in all four and the
rule is layer-wide, not seam-specific — FFN2's and QKV's outputs don't
currently feed a downstream consumer test, but would hit the identical bug
the moment one is added. `OUTPUT_ROW_BYTES` is now `512` in all four,
`teardown()` now calls `dump_xmem_to_binary(..., OUTPUT_BASE, 512, N_OUT)`
matching every L3/L4 matmul kernel, and each kernel's own wide test crops the
valid `N_TOK` prefix itself (the "final consumer" role, per the rule above).

## Cross-lane reduction vs. per-lane independence (padding-lane safety)

Kernels whose payload occupies fewer than `LANES` (128) elements of a row
carry **padding lanes**. Whether garbage in those lanes can reach the valid
output depends entirely on whether the kernel's datapath ever *reduces across
lanes*:

- **`AGG.SUM[.FIRST]` / `AGG.MAX[.FIRST]`** reduce across the lane axis into a
  single scalar. Any garbage in a padding lane is summed/maxed straight into
  the valid result. These kernels **must** set `valid_elements` (via
  `state.set_cr_dstructure(valid_elements=N)`) so the reduction is gated
  structurally — never rely on the harness's zero-fill. Affected:
  `qk_scores_*` (gates the `ACTIVATE.QUANTIZE` store window, no AGG to gate),
  `attn_v_64x48`, `attn_v_16x60`.
- **`ACC.ADD[.FIRST]` / `ACC.SUB[.FIRST]`** (the broadcast/matmul template) and
  plain per-lane `MULT.RC.VV`/`MULT.RC.VE`/`MULT.RC.VS` accumulate **each lane
  independently** — lane `i`'s result never depends on lane `j`. Padding lanes
  can only ever waste lanes, not contaminate valid ones, regardless of their
  content. Affected: `attn_v_bcast_*` (broadcast attn@V), `layernorm_*`
  (channel-axis reduction is an outer loop over rows, not a lane reduction),
  `residual_add_*`.
- **`ACC.STRIDE`** is a lane *remapping/decimation*, not a sum: each output
  slot copies one source lane verbatim (or zero, off-grid). A kernel using it
  can still be safe even if the selector reads from padding lanes, as long as
  every selected-from-padding output slot lands somewhere the consumer crops
  away and never in a valid slot. Affected: `unfold_*` — `ACC.STRIDE`'s
  vertical selector does read from the padding half of the source row, but the
  contribution always lands in `r_acc[16:32]`, which `teardown` crops away;
  the valid `r_acc[0:16]` only ever comes from the real-data half.

Every kernel above with padding lanes should have a garbage-padding probe, but
the shape of the probe depends on the mechanism:

- **AGG kernels** (`attn_v_64x48`, `attn_v_16x60`): a `..._padding_is_inert`
  test — rerun with padding lanes filled with garbage (`1e3`) instead of zero
  and assert the *valid-lane output* is bit-identical to the zero-padded run.
  This is the right shape here because AGG pulls padding lanes into the
  reduction, so garbage in the padding is visible on the valid lanes if the
  gate is missing.
- **`qk_scores_*` / `attn_scores_km_16x60`** (no AGG — `valid_elements` gates
  `MULT.RC.VE`'s mask and the `ACTIVATE.QUANTIZE` store window, not a
  reduction): the `..._padding_is_inert` shape does **not** work here. Lanes
  are independent, so garbage in padding lanes stays in padding lanes and
  never reaches the valid output regardless of gating — asserting valid-lane
  equality passes whether the gate is present or not. The correct probe
  instead stages garbage in the padding lanes of the *input* (K for
  `qk_scores_*`, Q for `attn_scores_km_16x60`) and asserts the **stored
  extent**: that the row bytes past `N*ELEM_BYTES` come back zero. All three
  (`qk_scores_64x48`, `qk_scores_16x60`, `attn_scores_km_16x60`) have this
  test as `..._padding_lanes_not_stored`.

Either way, this converts "we checked and the harness happens to zero-fill"
into a structural guarantee that holds regardless of what a real upstream
producer leaves there. Where the kernel is per-lane-independent or a remap
(`attn_v_bcast_*`, `unfold_*`), the existing correctness tests already cover
this — see the `unfold_8x8x240` note below on which test actually proves what.

## `unfold_8x8x240` input contract: `_ROW_PACK_ORDER`

`unfold_8x8x240` requires its caller to pre-permute the 8 spatial rows of each
8×8 channel into a specific packed order before staging them into XMEM — see
`ipu_apps.unfold.unfold_8x8x240._ROW_PACK_ORDER = (0, 2, 1, 3, 4, 6, 5, 7)`. This
follows from a real ISA constraint (`ACC.STRIDE`'s `elements_in_row` only
encodes 16/32/64, and W=8 isn't one of them, so a channel's 8 rows are packed
two-per-view-row and the pairing has to line up with `ACC.STRIDE`'s even/odd
split — full derivation in the kernel's module docstring). It is **not** a
defect, but it *is* an undocumented-until-now input contract, and nothing else
in the repo produces this ordering automatically.

The failure mode if a caller gets it wrong is worse than garbage: it silently
selects the wrong stride-2 phase per output stream — a plausible-looking row
shuffle, not an obviously corrupt result — which is harder to catch than the
pitch mismatch in the crop-convention bug above.

Use `ipu_apps.unfold.unfold_8x8x240.pack_input_rows(x)` (takes a `[C, H, W]` array,
returns the packed `[C, LANES]` XMEM-ready rows) rather than re-deriving the
permutation — it is now the single implementation, and both
`gen_debug_data.pack_stripe` and the kernel's own docstring point to it.

Coverage note: the kernel also has a `..._padding_is_inert`-shaped test, but
that test only proves the *padding lanes* are inert (per the `ACC.STRIDE`
crop argument above) — it does not exercise `_ROW_PACK_ORDER` at all, since
it reuses one fixed packed input and only varies what's in the unused lanes.
What actually proves the pack order is correct across every parity split and
phase is the pre-existing `test_unfold_8x8x240_wide_fp32` /
`test_output_shape_and_stale_lanes`, which check all four (row-phase,
col-phase) output streams against an independent numpy golden computed
directly from the un-packed `[C, H, W]` array. Do not credit the new padding
probe with covering `_ROW_PACK_ORDER` correctness — it doesn't.

## Packed activation layout — first committed documentation of this scheme

**This section is new documentation, not an update to a pre-existing
section.** Two rounds of exploratory work built a packed-activation-layout
kernel family for Layer 5 and Layer 4 respectively, entirely as untracked
files under `src/tools/ipu-apps/test/` with no bazel registration — this is
the first time either round's design gets written up anywhere committed.
The only prior record was `docs/isa_friction_log.md` (also previously
untracked). If you are looking for where these kernels are *used* in
production, they are not — they are a standalone viability/measurement
exercise, runnable only via direct pytest against the `.asm`/`test_*.py`
files listed below.

### What "packed" means here

Every kernel elsewhere in this document uses the repo's default convention:
one channel occupies one whole 128-lane XMEM row, regardless of how many of
those 128 lanes actually hold live data (`N_TOK` tokens, the rest zero
padding). The packed layout instead fits **multiple channels into one row**,
each in a fixed-width lane slot, so the wasted padding lanes of one channel
become live data for its neighbor.

The packing factor is derived from the same `partition_size()` rule
`softmax_rows_partial` already implements for its own (unrelated) row
packing (`src/tools/ipu-apps/src/ipu_apps/softmax/softmax_rows_partial/
__init__.py:117-124`):

```python
def partition_size(n: int) -> int:
    """Next power of two >= n, clamped to [16, 128]."""
    ps = 16
    while ps < n:
        ps *= 2
    return ps
```

`packing_factor = 128 // partition_size(N_TOK)`. Channel `p`'s `N_TOK`
tokens occupy lanes `[p * partition_size(N_TOK), (p+1) * partition_size(N_TOK))`
of its packed row.

| Layer | N_TOK | `partition_size(N_TOK)` | packing factor | packed rows for D_MODEL channels |
|---|---|---|---|---|
| L5 | 16 | 16 | 8 | 240 / 8 = 30 |
| L4 | 64 | 64 | 2 | 192 / 2 = 96 |

L5's 8x packing factor gives large memory and cycle wins on the linear
layers and layernorm. L4's factor of 2 is inherently much smaller — this is
a direct consequence of `partition_size(64)=64` versus `partition_size(16)=16`,
not an implementation gap, and the two rounds' measured results reflect
that: L5's headline numbers should **not** be assumed to generalize, and did
not — see `docs/isa_friction_log.md`'s L4 entry for a formula (replication
slot count) that L4 directly contradicts.

Attention (QK^T / softmax / attn·V) has no channel axis in either layer's
kernels, so it is never packed — it always runs on the existing unpacked
production kernels, with an on-chip pack/unpack conversion kernel at the
seam.

### Where the files live

Both rounds' kernels are untracked, standalone, test-only — `.asm` sources
and their `test_*.py` files sit side by side in `src/tools/ipu-apps/test/`,
assembled and run via the direct `assemble_to_bin_file` → `IpuState(...)` →
`load_program_from_binary` → `run_until_complete` pattern (no
`ipu_apps.<name>.App` wrapper, no `BUILD.bazel` target).

| Kernel role | L5 (240ch × 16tok, packing factor 8) | L4 (192ch × 64tok, packing factor 2) |
|---|---|---|
| Pack (unpacked→packed) | `asm_packed_pack_240x16.asm` | `asm_packed_pack_192x64.asm` |
| Unpack (packed→unpacked) | `asm_packed_unpack_240x16.asm` | `asm_packed_unpack_192x64.asm` |
| LayerNorm | `asm_packed_layernorm_240x16.asm` | `asm_packed_layernorm_192x64.asm` |
| Linear, packed output | `asm_packed_output_linear_generic.asm` / `_silu.asm` / `_1slot.asm` / `_tiny.asm` | `asm_packed_output_linear_generic_p4.asm` / `_silu_p4.asm` |
| Linear, masked (unpacked output) | `asm_packed_linear_240to8_masked.asm` / `_replicated.asm`, `asm_packed_linear_masked_generic.asm` | not built (output-linear construction carried forward instead) |
| Residual add | `asm_packed_residual_add_240x16.asm` | `asm_packed_residual_add_192x64.asm` |
| Cross-partition combine primitive | `asm_primitive_a_combine8x16.asm` | `asm_primitive_a_combine2x64.asm` |
| Full end-to-end layer chain | `test_full_layer_l5_packed.py` | `test_full_layer_l4_packed.py` |
| Instruction-counting fixture | `fixture_packed_l5_measure.py` | `fixture_packed_l4_measure.py` |

### The `rc_idx` formulas, generalized

Every masked gather/scatter in these kernels addresses `R_CYCLIC` (loaded via
`LDR_CYCLIC_MULT_REG`, a 512-element cyclic ring across 4 slots of 128) with
`rc_idx = ps * (...) mod 512`, where `ps = partition_size(N_TOK)` is the
per-partition lane width — **not** a fixed constant carried over between
layers:

| Purpose | Formula | L5 (`ps=16`) | L4 (`ps=64`) |
|---|---|---|---|
| Pack scatter | `rc_idx = (-ps*p_out) mod 512` | `(-16*p_out) mod 512` | `(-64*p_out) mod 512` |
| Unpack gather | `rc_idx = ps*p_in` | `16*p_in` | `64*p_in` |
| Packed-output-linear scatter | `rc_idx = ps*(p_in-p_out) mod 512` | `16*(p_in-p_out) mod 512` | `64*(p_in-p_out) mod 512` |
| LayerNorm broadcast | `rc_idx = (-ps*p) mod 512` | `(-16*p) mod 512` | `(-64*p) mod 512` |
| Primitive-A combine | `rc_idx = ps*p` for `p=0..packing_factor-1` | `16*p`, `p=0..7` | `64*p`, `p=0..1` |

The replication-slot count for the packed-output-linear scatter (how many of
`R_CYCLIC`'s 4 slots must be loaded per packed chunk) is **not** a constant
either — it depends on the actual range `rc_idx` sweeps for the layer's
`(p_in, p_out)` pairs, and must be re-derived by direct enumeration per
layer, not assumed: L5's formula stays within slot 0 (1 slot suffices); L4's
formula reaches slot 3 for one pairing (2 slots needed, 0 and 3). See
`docs/isa_friction_log.md` for both derivations in full.

### No ISA segmented reduce — the shared workaround

Neither layer's packed linear-output construction has access to a
partition-wise (segmented) reduce instruction — `AGG.SUM` collapses all
active lanes to one scalar, `RESHAPE` only moves up to 8 word-lanes
`r_acc`→`r_acc` per call, and `CR15.partition` (or any `CR0`-`CR15` used as
a `dstructure` register) only feeds mask/shift math, never data movement.
Both rounds work around this with the same hand-built "primitive A"
construction: store `r_acc` to XMEM, reload the row into `R_CYCLIC` via
`LDR_CYCLIC_MULT_REG`, then one `MULT.RC.VE ×1.0` + `ACC.ADD[.FIRST]` per
partition at `rc_idx = ps*p`. Cost scales with the packing factor: L5's
8-term combine measured 23 cycles / 22 instructions standalone; L4's 2-term
combine measured 11 cycles / 10 instructions.
