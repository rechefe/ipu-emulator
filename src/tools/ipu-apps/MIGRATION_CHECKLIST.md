# Kernel Migration Checklist — ZDlinear → new master architecture

**Merge done:** `8742f03` merged `origin/master` (`b970e97`) into ZDlinear on 2026-06-15.
Pre-merge WIP saved at `9b2b095`. Activation conflict resolved by **adopting master's
`rsqrt` (id 10)** and dropping our `inv_sqrt` (id 12) — same `1/sqrt(x)` math.

## What changed in the architecture (vs our old merge-base `a2c3d54`)

1. **XMEM slot split into 3 slots:** old single `xmem` → `load` / `store` / `acc_store`.
   - `load`: `LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG`, `LDR_MULT_MASK_REG`, `LOAD_NOP`
   - `store`: `STR_POST_AAQ_REG`, `STORE_NOP` (drains POST_AAQ_REG, last pipeline stage)
   - `acc_store`: `STR_ACC_REG`, `ACC_STORE_NOP` — **simulation-only** (not real HW).
     Assembler emits a UserWarning whenever `STR_ACC_REG` is used. Benign for now.
   - Consequence: load + store can now be fused into ONE compound word (future optimization).
2. **9 slot types total**, compound order (MSB→LSB):
   `cond, lr(×3), load, mult, acc, aaq, store, acc_store, break`.
3. **Activation table rewritten.** Now 11 entries (ids 0–10):
   `identity(0) relu(1) relu6(2) sigmoid(3) tanh(4) gelu(5) softplus(6) elu(7) exp2(8) reciprocal(9) rsqrt(10)`.
   - Our old `inv_sqrt` (id 12) is GONE → use **`rsqrt`**.
   - Dropped from old table: leaky_relu, silu, prelu.
4. **`ACTIVATE` signature changed.** Old: `ACTIVATE <lr_valid_elements> <fn>`.
   New: `ACTIVATE <activation_fn>, <full_xmem_row>` where `full_xmem_row` = `1` (all 128
   lanes) or `0` (use CR15.valid_elements). Lives in the **aaq** slot.
   - Example fix: `ACTIVATE lr8 inv_sqrt;;` → `ACTIVATE rsqrt, 0;;` (lr8 held 128 = full row → could also be `rsqrt, 1`).
5. **`AAQ` signature changed.** New: `AAQ <full_xmem_row>` (INT8 quantize of POST_AAQ_REG).
   Lives in **aaq** slot. Verify our layernorm `AAQ` calls match.
6. **AGG.SUM / AGG.MAX moved into the `acc` slot** (`AGG.SUM[.FIRST]`, `AGG.MAX[.FIRST]`),
   signature `<dest_slot_lr>, <full_xmem_row>`. Old `ACC.ADD_AAQ` family is gone.
7. **ADD/SUB now accept an unsigned immediate src_b (0–31):** `ADD LR4, LR1, 7;;`.
   Opportunity to drop some constant CRs/extra SETs.

## Assembly-level status (verified locally 2026-06-15, post-merge)

| Kernel | Assembles? | Notes |
|--------|-----------|-------|
| fully_connected | ✅ PASS | STR_ACC_REG sim-slot warning only |
| matmul_128x128 | ✅ PASS | warning only |
| matmul_128x64x128 | ✅ PASS | warning only |
| matmul_128x64x64 | ✅ PASS | warning only |
| matmul_64x64x64 | ✅ PASS | warning only |
| matmul_144x144_x128 | ✅ PASS | warning only |
| matmul_288x144_x128 | ✅ PASS | warning only |
| matmul_432x144_x128 | ✅ PASS | warning only |
| matmul_144x288_x128 | ✅ PASS | warning only |
| residual_add_256x144 | ✅ PASS | warning only |
| unfold_32x32x144 | ✅ PASS | uses ACC.STRIDE — re-verify semantics on DGX |
| layernorm_128x16 | ✅ PASS | `ACTIVATE rsqrt 1` + GAMMA off CR1→CR11; test passes |
| layernorm_256x144 | ✅ PASS | same fix; WIP rewrite already in HEAD; test passes |

## ⚠️ Assembling ≠ running correctly

`ipu.py` had a 678-line diff in this merge (execution semantics, register file,
POST_AAQ_REG pipeline). **Every kernel must be re-run on DGX** to confirm runtime
output still matches the reference — a clean assemble does NOT prove correctness.

## DGX RESULTS (2026-06-15) + TWO ROOT CAUSES

Core suites PASS (test_execute 119/119, ipu-as 1/1, ipu-common 14/14) — the emulator
engine is correct. Both kernel failures are **code-needs-updating, not emulator bugs.**

**Bug #1 — FIXED (commit 4f946d7):** `IpuState.set_cr_dtype()` was removed; dtype is
now a plain attribute. Applied to all 10 apps:
`state.set_cr_dtype(int(self.dtype))` → `state.dtype = self.dtype`.
Result: fully_connected now 3/3 PASS.

**Bug #2 — ROOT-CAUSED, per-kernel fix pending:** **CR0≡0 and CR1≡1 are now read-only
hardwired constants** (`ipu_config.py::CR_READ_ONLY_INITIAL_VALUES = {0:0, 1:1}`;
CR15 reserved). `regfile.set_scalar` **silently drops** writes to CR0/CR1. So
`set_cr(1, WEIGHTS_BASE)` is a no-op → cr1 stays 1 → loads read garbage. This breaks
all 10 non-FC apps (FC passes because master moved its weights base off CR1 → CR13).

Per-kernel fix for Bug #2:
1. In `__init__.py`: move the CR1 (and any nonzero-CR0) base to a free CR (CR3–CR14).
   Note CR0 base = 0x0 is a harmless no-op (matches hardwired 0) but cleaner to keep
   the asm reading cr-zero where it genuinely wants 0.
2. In the `.asm`: update every `cr0`/`cr1` reference that pointed at a moved base.
3. Re-assemble + re-run all 3 dtypes on DGX.
- unfold_32x32x144 also uses CR0+CR1 as stripe bases AND still has `on_inv` ACC.STRIDE
  encodings (known-bad — decode as disabled; use `on_expand`/`reserved3`). Most rework.

## Per-kernel checklist (do one at a time, in a fresh chat)

For EACH kernel:
- [ ] Assemble (already known pass/fail above)
- [ ] Run `uv run pytest test/test_<app>.py -v` on DGX (all 3 dtypes)
- [ ] If fail: diff behavior vs old; decide emulator-bug vs HW-constraint per CLAUDE.md memory rule
- [ ] Check `state.stats.format_summary()` still reasonable (RunStats baseline in memory)
- [ ] Consider fusing load+store into single words now that slots are split (perf, optional)

### Order of attack (suggested)
1. **layernorm_128x16** — smallest broken kernel; fix `ACTIVATE`/`AAQ` here first, establish the pattern.
2. **layernorm_256x144** — apply same fix; reconcile with the in-flight WIP rewrite (`9b2b095`).
3. **fully_connected** — simplest passing kernel; confirm runtime baseline on new emulator.
4. **matmul_128x128** then the rest of the FC-convention matmuls (64/128 variants).
5. **transformer matmuls** (432/144/288 ×144, 144×288) — MULT.VE.CYCLIC patterns.
6. **residual_add_256x144** — uses AGG? verify.
7. **unfold_32x32x144** — ACC.STRIDE; highest risk of semantic drift; verify bit-encoding still matches.

## layernorm fix recipe (APPLIED 2026-06-21 — both kernels pass locally)
- `ACTIVATE lr8 inv_sqrt;;` → `ACTIVATE rsqrt 1;;`
  - NOTE: the assembler grammar uses **space-separated** operands, NOT commas. The
    docs/spec `ACTIVATE rsqrt, 1` form fails to parse ("No terminal matches ','").
    Use `ACTIVATE rsqrt 1` (1 = full 128 lanes; lr8=128 was the old valid_elements arg).
  - The old `lr8` valid_elements operand is gone → its `SET lr8 cr14` startup is now
    dead and was removed from both kernels.
- GAMMA_BASE was on read-only CR1. Both kernels had `cr11 = const 0` (a free CR) used
  only as the zero source for `SET lrX cr11`. Fix: point those SETs at the **hardwired
  CR0 (=0)** instead, freeing CR11 to hold GAMMA_BASE. Then redirect the γ-load
  `LDR_MULT_REG r0 lr? cr1` → `cr11`. (DATA_BASE=0x0 so CR0 stays correct.)
- No `AAQ`/`ACC.ADD_AAQ`/`AGG` instructions appear in either layernorm — they reduce
  via MULT.EE/MULT.EE.RR + ACC with a ones-vector. No AGG signature changes needed.
- Both are **wide-FP32 debug-mode** kernels: a single `test_*_wide_fp32` test (NOT
  3-dtype parametrized). Both pass locally (atol/rtol 1e-4 vs numpy reference).
  Util: 128x16 = 37.5% mult, 256x144 = 42.7% mult (in the 37–43% baseline).
- These rely on the local TEMP BLT 20-bit sign-extend patch (commit 8983bc0); they will
  only pass on DGX once the upstream BLT fix lands. Re-run on DGX after that.
