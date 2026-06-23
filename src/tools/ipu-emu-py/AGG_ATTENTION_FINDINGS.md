# Aggregation Datapath Verification — Findings (ZDlinear + main, POST-FIX)

**Date:** 2026-06-22
**Branch:** `ZDlinear`, after merging `origin/master` (HEAD `ce33be2`, PR #153 / issue-152). The fix landed in commit **`d67c441` "Update AGG instruction references to use MULT_RES lanes"**. Clean merge: only `ipu.py`, `test_execute.py`, `test_wide_vector_debug.py`, `SKILL.md` changed; all ZDlinear apps preserved.
**Repro:** `cd src/tools/ipu-emu-py && uv run --extra dev pytest test/test_agg_attention_verification.py -v` → **13 passed**.
**Full suite:** `uv run --extra dev pytest test/ -q` → **289 passed, 0 failed.**

---

## VERDICT

**Query-major attn@V IS collision-free on ZDlinear+main, because AGG now reduces `mult_res` (the live multiply result) and writes the scalar straight into R_ACC[dest] — so a full-width MULT followed by AGG.SUM[.FIRST] dest=i co-issues in one bundle, computes O[i] directly, and parks it in R_ACC[i] without ever touching the other 128 lanes; no ACC.FIRST, no per-query reset, no scratch collision.**

---

## §1 — What does AGG reduce? **`mult_res` (fixed).**

All four handlers now read `mult_res` and reduce over it:

| handler | mult_res read | reducer call |
|---|---|---|
| `execute_agg_sum_first` | `ipu.py:899` `mult_res = self.state.regfile.raw("mult_res")` | `ipu.py:901` `_agg_sum_lanes(fmt, mult_res, active)` |
| `execute_agg_sum` | `ipu.py:911` | `ipu.py:915` |
| `execute_agg_max_first` | `ipu.py:930` | `ipu.py:933` |
| `execute_agg_max` | `ipu.py:941` | `ipu.py:945` |

(The reducer helpers `_agg_sum_lanes`/`_agg_max_lanes` at `ipu.py:879/885` still *name* their param `snap_acc`, but the callers pass `mult_res` — cosmetic stale name, behavior is correct.)

**Empirical proof** (`TestFinding1_AggReducesMultRes::test_agg_sum_first_reduces_mult_res_768`, PASS): MULT wrote `mult_res = 6.0`/lane (sum 768); R_ACC pre-seeded to `1.0`/lane (sum 128); `AGG.SUM.FIRST`, no ACC → result **768.0** (= sum of mult_res), not 128. Also confirmed in narrow INT8 (`test_agg_sum_first_int8_reduces_mult_res`: 3×4×128 = 1536).

---

## §2 — Exact semantics (matching shipped code, line-referenced)

| Instruction | Effect | Refs |
|---|---|---|
| `AGG.SUM.FIRST dest` | `R_ACC[dest] = sum(mult_res[0..n-1])` (clean write) | `ipu.py:901,905` |
| `AGG.SUM dest` | `R_ACC[dest] += sum(mult_res[0..n-1])` (seed = snapshot `R_ACC[dest]`) | `ipu.py:914-920` |
| `AGG.MAX.FIRST dest` | `R_ACC[dest] = max(mult_res[0..n-1])` (seed = identity −inf/INT32_MIN) | `ipu.py:932-935` |
| `AGG.MAX dest` | `R_ACC[dest] = max(mult_res[0..n-1], R_ACC[dest])` (seed = snapshot `R_ACC[dest]`) | `ipu.py:944-946` |

- **n** = `128 if full_xmem_row else min(CR15.valid_elements, 128)` — `ipu.py:897` + `_agg_active_lane_count` `ipu.py:868`. Verified: `TestFinding4a` (full_xmem_row=1 → 128 lanes; =0 → valid_elements=10).
- **Masking applies**: deselected `mult_res` lanes are zeroed by the MULT slot *before* AGG reads them, so a masked MULT reduces only the active lanes. Verified narrow INT8: `TestFinding4c` (8-byte mask → 64 lanes → sum 384; full mask → 768). Note: mask gating is a no-op in wide mode (`ipu.py:374`), so this is exercised in narrow INT8 only.
- **`mult_res` read LIVE**: MULT and AGG co-issue in **one bundle** — no intervening ACC. Verified `TestFinding4b::test_mult_and_agg_same_bundle_no_acc` (768 in a single MULT+AGG bundle). ACC and AGG remain mutually exclusive in the single ACC slot (`test_acc_and_agg_cannot_co_issue`, PASS) — but you no longer need ACC at all for the reduce.

Each verified in wide-FP32 first (clarity), then INT8 (`test_agg_sum_first_int8_reduces_mult_res`, `TestFinding4a/4c` are INT8). FP8_E4M3/E5M2 use the same narrow MULT→AGG path (the early-return at `ipu.py:374` is keyed on wide-mode, not dtype); INT8 exercises that path directly.

---

## §3 — Collision check (the decisive one): **survives.**

`TestFinding2_QueryMajorAttnV::test_intended_flow_matches_and_is_collision_free` (PASS). 4 queries × 256 keys (2 chunks) × 1 channel. Per query i: chunk 0 `MULT; AGG.SUM.FIRST dest=i`, chunk 1 `MULT; AGG.SUM dest=i` — dest stepped per query, no ACC.FIRST. Result: all four O[i] match numpy `P@V`, and each query's answer sits in its own R_ACC[i] lane untouched by later queries.

Corroborated at the lane level by `test_racc_lane_behavior.py::test_item4_agg_sum_first_dest70_valid64`: AGG.SUM.FIRST into dest=70 writes only lane 70; the other 127 pre-parked SENTINEL lanes are unchanged.

---

## §4 — Supporting / orientation checks

- **Key-major broadcast** (lanes=queries, ACC over keys, no AGG): still correct + clean 512-byte channel-major `STR_ACC_REG` row — `TestFinding3` (PASS). Either orientation is now viable; query-major no longer carries a collision penalty.
- **full_xmem_row vs valid_elements**: §2. **R_MASK gating**: §2 (narrow INT8).

---

## Scratch-test reconciliation (the 6 pre-fix reds)

The earlier scratch files asserted **pre-fix** behavior and used **pre-merge ISA syntax** (notably the redesigned `MULT.EE`, whose first operand is now `ra_idx`/LR, not `r0`). Reconciled (not deleted — each probes a distinct, still-valid concern), all now green:

- `test_racc_lane_behavior.py` (5): swapped dead `MULT.EE r0 …` → `MULT.RC.VV lr2 r0 0 lr2`; rewrote `item4` to assert AGG-reduces-`mult_res` + collision-free lane preservation.
- `test_aaq_scale_path.py` (2 fixed): AAQ is now a **direct INT8 clamp** (`ipu.py:1010`), not `>>24` — updated `test_aaq_is_fixed_clamp_not_a_shared_scale`; seeded `mult_res` in `test_agg_sum_writes_into_racc_not_post_aaq` (AGG reduces mult_res now).
- `test_str_acc_reg_store.py` (1 fixed): `MULT.EE r0 …` → `MULT.RC.VV` in the co-issue bundle test.

Net: `test/` = **289 passed, 0 failed**.

---

## Recommendation

Query-major attn@V is cleared to build on ZDlinear+main. Use `MULT.<…> ; AGG.SUM.FIRST dest=i` for the first key-chunk and `; AGG.SUM dest=i` for subsequent chunks of the same query, stepping dest per query; drain R_ACC to XMEM once the query block fills the lane budget. No ACC needed in the reduce path.
