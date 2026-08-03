# L4/L5 Phase 0 — Verification Findings

**Date:** 2026-07-28 · **Branch:** `ZDlinear` · **Basis:** `ipu.py` / `instruction_spec.py` as shipped, plus executed probes.

Every **[VERIFY]** item in the L4+L5 plan is resolved below. Each claim is backed by
either a line-referenced source read or a probe that was actually run.

---

## Summary

| # | Item | Verdict |
|---|---|---|
| Q-1 | Was `silu` re-added after PR #94? | **Yes — X1 unblocked** |
| Q-3 | Load / acc_store co-issue → 2-cycle residual body | **Confirmed, 1161 → 587 cycles** |
| §3.4 | `MULT.RC.VV` + `AGG.SUM` at 1 score/cycle, no `R_ACC` hazard | **Confirmed** |
| §3.1 | L3's actual attention mapping | **Confirmed; §5 pairing resolved** |
| §1 | `matmul_144x288_x128` = 110,592 mult-active | **Reconciled — padding rule is NOT uniform; this kernel is the outlier** |
| P1 | `lane_util` in `print_stats_all.py` | **Landed; AR-1 baseline measured** |
| — | AGG operand signature in the plan | **Stale — now takes an explicit dstructure CR** |

---

## Q-1 — `silu` is present. X1 is unblocked.

`ACTIVATION_FN_NAMES` in `ipu-common/src/ipu_common/activations.py:34-47` lists all
twelve activations, with `silu` at **id 11**:

```
identity, relu, relu6, sigmoid, tanh, gelu,
softplus, elu, exp2, reciprocal, rsqrt, silu
```

The published reference is correct and **project memory is stale** — PR #94 did not
leave silu removed. MobileViT's swish activation is therefore a single
`ACTIVATE.QUANTIZE silu, CRn` in the FFN1 epilogue.

**Consequence:** X1 is a one-instruction change, and the "activation is deferred"
status carried for L3 can be closed as well.

---

## Q-3 — Load and acc_store DO co-issue. The residual body is 2 cycles.

`SLOT_COUNT` (`instruction_spec.py:144-146`) gives `load`, `store` and `acc_store`
one independent slot each, and a probe confirms a `LDR_CYCLIC_MULT_REG` and a
`STR_ACC_REG` issue in the same word with no structural hazard.

Measured on the real 288-row residual shape, INT8, both variants verified against
a numpy golden (all 36,864 elements exact):

| body | cycles | correct |
|---|---|---|
| 4-cycle (shipped `residual_add_256x144`) | 1,161 | yes |
| **2-cycle (this finding)** | **587** | yes |

**49.4% saving**, matching the plan's ~576 prediction.

### The one real constraint — `STR_ACC_REG` reads its offset LIVE

The first attempt produced correct data shifted one row late (row *n* landed at
row *n+1*). Cause: `ADD lr4 lr4 lr8` co-issued with `STR_ACC_REG lr4 cr3`, and the
store saw the **already-incremented** pointer. This is the hazard the shipped
kernel's own header warns about; it is *not* a load/store conflict.

Working 2-cycle body — the output pointer advances in **cycle 1**, so it starts one
row behind (`lr4` init = −512):

```
row_loop:
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; ADD lr4 lr4 lr8;
        MULT.RC.VE lr0 cr10 0 lr0 cr15; ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG lr3 cr9 lr0; ADD lr3 lr3 lr7; ADD lr5 lr5 cr1;
        MULT.RC.VE lr0 cr10 0 lr0 cr15; ACC.ADD; STR_ACC_REG lr4 cr3;
        BLT lr5 lr6 row_loop;;
```

**Build K3/K8 this way, and back-port to `residual_add_256x144`** (1,161 → 587).

---

## §3.4 — AGG dot-product scores: 1 score/cycle confirmed

A full `MULT.RC.VV` + `AGG.SUM.FIRST` loop with the destination stepped by an
`INC` co-issued **in the same bundle** was run in wide FP32 and matched
`K @ q` from numpy exactly:

| N (keys) | body | cycles | correct |
|---|---|---|---|
| 16 | 2-cycle (`INC` isolated) | 40 | yes |
| 16 | **1-cycle** | 25 | yes |
| 64 | **1-cycle** | 73 = 7 setup + 64 + 2 | yes |

All three consequences the plan draws in §3.4 hold:

1. A whole `q_i · k_j` dot product fits one multiply for both head dims (48, 60),
   so **AGG-based scores are valid for L4 and L5**.
2. `valid_elements` handles the ragged head dim directly — the probe used
   `valid_elements=60` with zero masking gymnastics.
3. **The result gather is free.** 64 consecutive AGGs with `INC` on the dest LR
   built a full score vector in `R_ACC` with no stores and **no `R_ACC` write
   hazard** — MULT, AGG, LOAD and the LR sub-instructions all co-issue.

### Correction to the plan: AGG's operand signature is stale

The plan quotes `AGG.SUM dest_slot, cr_idx` but its worked syntax omits the CR.
As shipped, **both operands are mandatory** and the CR must be named explicitly
(`instruction_spec.py:791-814`, `ipu.py:972-992`):

```
AGG.SUM.FIRST lr1, cr14      ;; cr14 supplies valid_elements
```

`dest_slot` is declared `"read": "snapshot"`, which is exactly why an `INC` on
that LR co-issues correctly — AGG uses the pre-increment value, so the 1-cycle
body works without a pipeline bubble. Semantics the plan relies on are unchanged.

---

## §3.1 — L3's attention mapping, read from the assembly

Confirmed exactly as the plan reconstructs it. All three L3 attention kernels use
the same broadcast template — scalar from `R0` indexed by the contraction index,
vector in `R_CYCLIC`, `ACC.ADD` accumulating, `STR_ACC_REG` draining raw:

```
LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD; BLT lr5 lr6 c_loop;;
```

L3 hits 100% lanes only because 256 tokens is exactly 2×128.

### §5 — the chain pairing is now unambiguous

The kernel headers name their own roles, resolving the question the plan flagged:

| chain | scores | attn@V | lanes |
|---|---|---|---|
| **query-major** | `qk_scores_256x36` (Agent C) | `attn_v_256x36` | keys / queries |
| **key-major** | `attn_scores_km_256x36` (Agent D) | `attn_v_bcast_36` (Agent B) | queries |

`qk_scores_256x36` stores query-major raw scores explicitly so "softmax (Agent A)
reads unquantized scores"; `attn_scores_km_256x36` produces key-major rows.
**Ze'evi still needs to confirm which chain the softmax owner is targeting** — that
is a preference, not a missing fact, and it is the only §5 item still open.

---

## §1 — K-chunk padding reconciled: the rule is NOT uniform

**Corrected.** An earlier version of this note generalised a padding rule from
`matmul_144x288_x128` alone. That was wrong, and the d=240 runstats figures are
the evidence — reconciled here.

### The stat itself cannot count padded lanes

`ipu.py:1192-1194` — `mult_active_cycles` increments once per bundle that issues a
non-NOP multiply:

```python
if slot_type == "mult":
    if instruction_name != "NOP":
        stats.mult_active_cycles += 1
```

It has **no notion of lanes or padding at all**; it counts cycles. So the
"inconsistent padded-lane accounting" hypothesis is ruled out. Every difference
between kernels is a difference in their **loop bounds**.

### The real cause: tail chunks use exact bounds, `144x288` does not

Every `_x128` matmul splits K into 128-wide chunks, but the *tail* chunk is
bounded to the real remaining K, not padded up:

| kernel | K | chunk bounds (`lr6` / `lr11`) | MACs/token-group |
|---|---|---|---|
| `matmul_240x240_x128` | 240 | 126 (w=128) + **110 (w=112)** | 240 × 240 |
| `matmul_240x480_x128` | 480 | 126 ×3 + **94 (w=96)** | 240 × 480 |
| `matmul_192x192_x128` | 192 | 126 (w=128) + **62 (w=64)** | 192 × 192 |
| `matmul_576x192_x128` | 192 | 126 (w=128) + **62 (w=64)** | 576 × 192 |
| `matmul_144x288_x128` | 288 | **126 ×3 — no tail bound** | 144 × 384 |

`240x240` → 128+112 = 240 exactly, `240x480` → 3×128+96 = 480 exactly. These match
runstats (57,600 and 115,200) with **no padding**, exactly as predicted.

`matmul_144x288_x128` is the **outlier**: it runs three full 128-wide chunks for
K=288, i.e. 384, and reports 144 × 384 × 2 groups = 110,592. Its own header
documents only a `lr6=126` "per-chunk" bound with no tail-chunk register, so the
third chunk over-runs K by 96 and relies on zero-padded weights for correctness.

### Consequences

1. **The 6.7% "d=240 pads to 256" lane cost claimed for L5 does not exist.** The
   d=240 kernels use exact tail bounds. §4's combined-utilization figure should
   stay at **50%**, not 46.9%, and AR-1's L5 saving needs no adjustment.
2. `matmul_144x288_x128` is doing **~33% redundant multiply work** for its K
   (384 issued vs 288 real). It is an L3 kernel, so it does not affect the L4/L5
   baseline — but it is a real ~27,648-cycle saving available by giving it a
   96-wide tail bound like its siblings. Worth a separate issue.
3. Any new L4/L5 kernel must use the **exact tail bound** pattern
   (`lr11 = width - 2`), not the `144x288` pattern.

---

## §4 layout question — answered. K1 must emit a NEW layout; L3 is no template.

This is the Phase 1 gate: K1 sets the layout, so it had to be settled before K1
is written. Read from the shipped assembly.

### What each side actually does

| | address of (channel, token-group) | note |
|---|---|---|
| `unfold_32x32x144` **output** | `DST_BASE + s*147456 + (c*2 + t)*512` | **four separate stream bases** |
| `matmul_144x144_x128` **input** | `DATA_BASE + k*256 + t*128` | one contiguous 256 B row per channel |

### The correction: L3's `tg` is NOT the pixel-stream axis

The plan's hopeful branch — *"if L3 is already `[k][p·n]`, this is a loop-bound
change and nothing more"* — **does not hold.**

L3 has N=256 tokens per stream, which is exactly 2×128. So `tg` ∈ {0,1} indexes
the **two 128-token chunks within a single stream**, not two pixel streams. L3
processes one stream at a time and never packs across streams, because it never
needs to. Its layout is `[k][n]` for one stream — not `[k][p·n]`.

Consequently **there is no existing kernel that packs tokens across pixel
streams**, and K11 has no precedent to copy. K1/K6 must emit a genuinely new
layout.

Note also the two kernels are never chained today — the matmul apps load a
pre-staged blob (`matmul_144x144_x128/__init__.py:54`), so the unfold→matmul
seam is untested at any level. K11 is the first thing that exercises it.

### Why a 128-lane load cannot span two streams today

For a fixed channel `c`, unfold places consecutive streams **147,456 bytes
apart**. A 128-lane load reads 128 contiguous bytes, so it cannot reach two
streams at any offset. This is a layout property, not an ISA limit — hence
fixable in software, exactly as §4 argues.

### K1/K6 output spec

Emit **`[k][p·n]`** — for each channel `k`, all P·N tokens contiguous across all
four streams — rather than four stream-major blocks. Then re-tile the matmul call
sites to 128-token chunks (K11).

### Ceilings (confirming §4's table)

| | tokens/stream | P·N packed | chunks of 128 | lane fill |
|---|---|---|---|---|
| L4 | 64 (48.2% now) | 256 | 2 | **100%** — 2.0× in software |
| L5 | 16 (12.1% now) | 64 | 1 | **50%** — 4.0× in software |

L4's matmul waste is **entirely** recoverable in software; AR-1 buys it nothing.
L5 recovers 4× in software, and only the final 2× (50→100%) is the architecture
ask. §6's instruction to quote the residual rather than the raw measured gap is
the right call.

---

## Phase 1 — K1 `unfold_16x16x192` (built, 4/4 passing)

**3,074 cycles, 50.0% mult, 12.5% lane.** All three dtypes bit-exact against the
golden, plus a layout-contract test.

### Why it is not a port

L3 loads 4 spatial rows × 32 cols per 128-byte row and has **8 stripes**; L4
loads 8 rows × 16 cols and has **2 stripes**. `ACC.STRIDE` still yields 32
elements per call (verified against `execute_acc_stride`: 4 rows × 8 cols with
the expected even/odd split for all four selectors, `elements_in_row=16`), so
the four-slot structure carries over — but a stream fills only **2** slots.

### The layout decision (Option B, per §4)

64 tokens/stream = 256 B, but `STR_ACC_REG` unconditionally writes all 512 B of
`R_ACC` (`ipu.py:446-455`). A per-stream store therefore cannot fill a row. Two
options existed:

- **A** — pair two streams per store (slots 0,1 = stream *s*; 2,3 = stream *s+1*).
  No waste, half the stores — but that *is* the packed `[k][p·n]` layout for a
  pair, pre-empting the deferred §9 experiment.
- **B** — one stream per store, upper half stale. **Chosen**, per the plan's
  "build per-stream, defer packing".

Each output row is `[64 valid FP32 tokens | 64 stale lanes]`; consumers read only
the first 256 B. `test_output_shape_and_stale_half` pins this contract so a
future change cannot silently violate it. The stale lanes are zero (never
written, `R_ACC` zero at reset), and the golden models this explicitly rather
than assuming it.

### The 12.5% lane figure

49,152 valid tokens / (3,074 × 128). Two multiplicative causes: only 64 of 128
lanes carry a token, and 1 of every 3 bundles is a store. This is the per-stream
cost §9 would recover — record it as the K1 input to that experiment.

### Incidental: L3 unfold has a test-data regeneration hazard

`test_unfold_32x32x144.py` parametrises dtype dirs `fp8_e4m3` / `fp8_e5m2`, and
the **committed data uses those names**, so all three L3 dtypes really are
verified today. But `gen_test_data.py` now writes `fp8_e4` / `fp8_e5`
(`gen_test_data.py:154-155`). Re-running the L3 generator therefore creates two
new directories the test never looks at, silently leaving the stale goldens in
place — the tests would keep passing against old data. A one-line fix in either
file; worth doing before anyone regenerates L3 data.

K1 sidesteps this by using the names its generator produces (`fp8_e4`/`fp8_e5`)
in both the test and the data.

---

## Issues to file

`gh` is not installed in this environment, so these are drafted rather than filed.

**Q-4 — Assembler label table is process-global, not per-program.** See below.

**`matmul_144x288_x128` runs a full 128-wide tail chunk.** Every sibling
(`240x240`, `240x480`, `192x192`, `576x192`) bounds its tail chunk to the real
remaining K via a second bound register (`lr11 = width - 2`); this kernel runs
three full 128-wide chunks for K=288, issuing 384 K-steps. Measured **73.8%
lane utilization vs 95.7% for its d=144 siblings**; a 96-wide tail bound recovers
**~28,000 cycles**. Correctness is unaffected (weights are zero-padded).

**Q-3 contract question — is `STR_ACC_REG`'s offset intended to be read live?**
A pointer update co-issued in the same word takes effect on that store, silently
shifting output by one row. Reasonable as a design, but not stated in the
instruction reference. See §Q-3 above for the reproduction.

---

## Incidental finding — assembler label state leaks across `assemble()` calls

Assembling two programs in one process fails if they share a label name:

```
Label 'done' is defined for the second time at Line 14 ... Previous definition at Line 16
```

Line numbers refer to the *other* program, so labels are global to the process, not
to the program. It bit every multi-variant probe here. Not a kernel blocker
(each app assembles in its own process), but it makes A/B harnesses awkward and is
worth a GitHub issue.

---

## `lane_util` — landed, and the measured AR-1 baseline

`print_stats_all.py` now declares `USEFUL_MACS` per kernel and reports a `lane%`
column beside `mult%`, plus a summary table. TOTAL is **1,400,700** — unchanged,
so nothing regressed.

```
kernel                      cycles   mult%   lane%
matmul_144x144_x128          43345    95.7    95.7     L3, d=144, 256 tok
matmul_144x288_x128         112465    98.3    73.8     <- redundant-K outlier
matmul_192x192_x128          38209    96.5    48.2     L4, d=192, N_TOK=64
matmul_384x192_x128          76417    96.5    48.2
matmul_576x192_x128         114625    96.5    48.2
matmul_192x384_x128          75265    98.0    49.0
matmul_240x240_x128          59281    97.2    12.1     L5, d=240, N_TOK=16
matmul_480x240_x128         118561    97.2    12.1
matmul_720x240_x128         177841    97.2    12.1
matmul_240x480_x128         117361    98.2    12.3
attn_v_bcast_36              77346    95.3    95.3     L3 attention
qk_scores_256x36             20993    87.8    87.8
```

**This is the whole argument for AR-1 in one table.** Every kernel reads ~97%
`mult%`. L5 is doing **12.1%** useful work; L4 **48.2%**. The occupancy column
alone would have shown nothing wrong.

### Measured waste (replaces the plan's estimates)

| | measured cycles/layer | lane% | wasted/layer | × L | **total wasted** |
|---|---|---|---|---|---|
| L4 (L=4) | 304,516 | 48.2% | 157,739 | 4 | **630,957** |
| L5 (L=3) | 473,044 | 12.1% | 415,806 | 3 | **1,247,417** |

L5 measures 473,044 vs the plan's 460,800 estimate (+2.7%).

**The L5 figure is ~1.8× the plan's ~691,200.** The plan derived that from the
50% token loss alone; the measurement shows the array is really only 12.1% full,
because `N_TOK=16` leaves **7/8** of the lanes idle, not half. AR-1 with
`partition = 16` addresses exactly this. Combined L4+L5 waste is **~1.88M
cycles** against a 1.4M project total — the ask is stronger than drafted, and
these are measured numbers, which is what §6 says makes it land.

**Correction to §4:** the "d=240 pads to 256, 6.7%" claim should be dropped (see
§1 above — the d=240 kernels use exact tail bounds). The gap between 12.1% and
the theoretical 12.5% (16/128) is ordinary loop overhead, not padding.

### Side finding: `matmul_144x288_x128` at 73.8%

Its siblings all read 95.7%. Giving it a 96-wide tail bound recovers
**~28,000 cycles**. L3-only, so it does not affect the L4/L5 baseline.

---

## Not done in Phase 0

- **Zero-cost items still unasserted:** the QKV split offsets, head concat, and
  the folded `1/sqrt(head_dim)` scale (L4: 0.144338, L5: 0.129099) still need a
  test, per the plan.
- **Q-2 (unfold/fold asymmetry)** is a question for the architecture owner, not
  something verifiable from `ipu.py`.

---

## Verdict

**Phases 1–3 are unblocked.** Three of the four design decisions that depended on
Phase 0 are settled: the 2-cycle residual body is real, AGG scores sustain
1/cycle, and the L4/L5 attention mapping split in §3.4 stands as written. The
only open item is the §5 chain choice, which is Ze'evi's call.
