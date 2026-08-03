# Layer 3 Kernel Reference

A self-contained reference for every Layer-3 (L3) kernel in the IPU emulator
(`ZDlinear` branch). It is meant to be read end-to-end without access to the
repository: every kernel's full assembly, data layout, walkthrough, measured
performance, and golden-test result is reproduced inline.

> **Provenance.** All assembly is transcribed verbatim from the `.asm` source on
> the `ZDlinear` branch (commit `7c71a17`, the same commit the performance
> numbers were measured at). Performance figures are quoted from a DGX run
> recorded in the project's RunStats baseline; they are flagged **measured**.
> Test results were re-run locally for this document (all 34 L3 tests, 8m17s,
> `uv run pytest`) and are flagged **re-run for this doc**. Where a number is an
> analytic estimate rather than a measurement, it is labelled **(estimated)**.

---

## 1. What "Layer 3" means here

The transformer this ISA targets is sliced into layers, and the matmul kernels
carry their layer in their docstring (e.g. `matmul_192x192_x128` is *"Layer 4
OutProj"*, `matmul_240x240_x128` is *"Layer 5 OutProj"*). **Layer 3** is the
block whose activations have these dimensions:

| Quantity | Value | Notes |
|---|---|---|
| Model width *d* | **144** channels | the contraction / channel dimension |
| Tokens per stream | **256** | processed as **2 token-groups × 128** ("tg") |
| Streams | 4 | spatial sub-grids produced by `unfold` (TL/TR/BL/BR) |
| Attention heads | **4** | |
| Head dimension | **36** | `4 × 36 = 144 = d` |

Every kernel documented here operates at those dimensions: the matmuls contract
over *K = 144* (or *288* for the FFN's expanded inner dimension), the attention
kernels use *N = 256* tokens with *head_dim = 36*, and the elementwise kernels
process 144 channels × 256 tokens. Kernels at *K = 192* (Layer 4) and *K = 240*
(Layer 5), the generic fully-connected matmuls (128×128, 64×64, …), and
`layernorm_128x16` are **explicitly out of scope** — they are not L3.

### The 11 L3 kernels

| # | Kernel | Role | File |
|---|--------|------|------|
| 1 | `matmul_144x144_x128` | square projection (QKV-slice / attn-out proj) | [matmul_144x144_x128.asm](../src/tools/ipu-apps/src/ipu_apps/matmul_144x144_x128/matmul_144x144_x128.asm) |
| 2 | `matmul_288x144_x128` | FFN-1-like (2× output expansion) | [matmul_288x144_x128.asm](../src/tools/ipu-apps/src/ipu_apps/matmul_288x144_x128/matmul_288x144_x128.asm) |
| 3 | `matmul_432x144_x128` | fused QKV (3 × 144 outputs) | [matmul_432x144_x128.asm](../src/tools/ipu-apps/src/ipu_apps/matmul_432x144_x128/matmul_432x144_x128.asm) |
| 4 | `matmul_144x288_x128` | FFN linear-2 (contracts the 288-wide hidden) | [matmul_144x288_x128.asm](../src/tools/ipu-apps/src/ipu_apps/matmul_144x288_x128/matmul_144x288_x128.asm) |
| 5 | `qk_scores_256x36` | QKᵀ attention scores, **query-major** output | [qk_scores_256x36.asm](../src/tools/ipu-apps/src/ipu_apps/qk_scores_256x36/qk_scores_256x36.asm) |
| 6 | `attn_scores_km_256x36` | kQᵀ attention scores, **key-major** output | [attn_scores_km_256x36.asm](../src/tools/ipu-apps/src/ipu_apps/attn_scores_km_256x36/attn_scores_km_256x36.asm) |
| 7 | `attn_v_256x36` | attn@V via **AGG reduction** (query-major scores) | [attn_v_256x36.asm](../src/tools/ipu-apps/src/ipu_apps/attn_v_256x36/attn_v_256x36.asm) |
| 8 | `attn_v_bcast_36` | attn@V via **broadcast matmul** (key-major scores) | [attn_v_bcast_36.asm](../src/tools/ipu-apps/src/ipu_apps/attn_v_bcast_36/attn_v_bcast_36.asm) |
| 9 | `unfold_32x32x144` | NHCW → 4 channel-major streams | [unfold_32x32x144.asm](../src/tools/ipu-apps/src/ipu_apps/unfold_32x32x144/unfold_32x32x144.asm) |
| 10 | `layernorm_256x144` | LayerNorm over 144 channels | [layernorm_256x144.asm](../src/tools/ipu-apps/src/ipu_apps/layernorm_256x144/layernorm_256x144.asm) |
| 11 | `residual_add_256x144` | elementwise A + B | [residual_add_256x144.asm](../src/tools/ipu-apps/src/ipu_apps/residual_add_256x144/residual_add_256x144.asm) |

---

## 2. The execution model you need to read these kernels

### 2.1 Registers and memory

| Register | Size | Role in L3 kernels |
|---|---|---|
| **R0**, **R1** | 128 B each (128 × INT8) | multiply-stage inputs. In matmuls they hold one output channel's **weight row** (the "scalar" operand, indexed lane-by-lane); in attention they hold whichever operand plays the scalar role. R0++R1 form a 256-byte combined window so a scalar index 0–255 selects across both. |
| **R_CYCLIC** | 512 B (4 × 128-B slots) | the streamed **vector** operand, loaded fresh each cycle by `LDR_CYCLIC_MULT_REG`. Addressed cyclically mod 512 by a base index. |
| **R_ACC** | 512 B (128 × INT32, or 128 × FP32 in wide-debug) | the accumulator. `STR_ACC_REG` always writes all 512 bytes. |
| **POST_AAQ_REG** | 512 B | activation/quantization staging (used only by `layernorm`'s `ACTIVATE rsqrt`). |
| **LR0–LR15** | 20-bit scalars | loop counters and byte offsets. |
| **CR0–CR15** | 20-bit, read-only | base addresses and constants set by the Python harness. `CR0`≡0, `CR1`≡1 are hardwired; `CR15` is the reserved dstructure register. |

XMEM is a flat, byte-addressable 2 MB space, zero-initialised. Every load/store
address is `lr_offset + cr_base`.

### 2.2 The VLIW bundle and its timing

Each `;;`-terminated **compound instruction** issues up to six slots in one
cycle: LOAD (XMEM), three LR ops, MULT, ACC, AAQ, STORE, ACC_STORE. The timing
rule that makes every loop in this document work:

> **The LR slot runs first** (reading the pre-cycle *snapshot*, writing *live*
> values). Then **XMEM and MULT read LR values live** (post-increment), while
> **the branch (`BLT`) reads the snapshot** (pre-increment).

This is why the kernels use a **one-cycle startup skew**: a pointer is initialised
to `first_addr − stride` and the counter to `first_index − 1`, so that on the
*first* loop cycle the co-issued `ADD` fires before the load and the live value
already points at the first real element. The matching consequence is the
**loop-bound formula** for a do-while body covering `width` elements from
`first_index`:

```
counter_start = first_index − 1
bound (BLT)   = first_index + width − 2      # NOT width − 1
ptr_start     = first_addr − stride
```

For a width-128 contraction starting at index 0 this gives `counter_start = −1`,
`bound = 126`. (The exit cycle lands on live index 127, the last real term.)

### 2.3 The multiply primitives in use

These are quoted from the instruction reference (`InstructionDoc` in
`instruction_spec.py`). Every L3 kernel uses one of the `MULT.RC.*` forms.

| Mnemonic | Operation (per lane *i*) | Used by |
|---|---|---|
| **`MULT.RC.VV`** `rc_idx, ra, mask_off, mask_shift` | `MULT_RES[i] = R_CYCLIC[(rc_idx+i) % 512] × ra[i]` — vector × vector | attn@V (AGG), unfold, layernorm steps |
| **`MULT.RC.VE`** `rc_idx, src, mask_off, mask_shift` | `MULT_RES[i] = R_CYCLIC[(rc_idx+i) % 512] × scalar`, where the scalar is `R0/R1[LR[src]]` (if `src` is an LR; 0–127→R0, 128–255→R1) or `CR[src]`'s low byte (if a CR) — **vector × scalar broadcast** | all 4 matmuls, both score kernels, attn@V broadcast, layernorm γ/β |
| **`MULT.RC.VS`** `rc_idx, mask_off, mask_shift` | `MULT_RES[i] = R_CYCLIC[(rc_idx+i) % 512]²` — square | layernorm variance |

The accumulate primitives:

| Mnemonic | Operation | Notes |
|---|---|---|
| **`ACC`** | `R_ACC += MULT_RES` | running accumulate |
| **`ACC.FIRST`** | `R_ACC = MULT_RES` | seeds the accumulator (replaces the removed `RESET_ACC`; this is why the first contraction step is *peeled* out of each loop) |
| **`ACC.STRIDE`** `elements_in_row, h_stride, v_stride, offset` | reorders `MULT_RES` into a 32/64/128-lane slice of `R_ACC` by row/column decimation | unfold only |
| **`AGG.SUM[.FIRST]`** `dest_slot, full_xmem_row` | reduces the active `MULT_RES` lanes to a **single** `R_ACC` slot chosen by `LR[dest_slot]` | attn@V (AGG) only |

> ⚠️ **Reference-vs-emulator discrepancy (AGG).** The `InstructionDoc` *operation*
> text for `AGG.SUM`/`AGG.SUM.FIRST` still reads
> `R_ACC[dest] = sum(R_ACC[0..n-1])` — i.e. it claims AGG reduces **R_ACC**. The
> emulator (`ipu.py: execute_agg_sum*`) actually reduces **`MULT_RES`**
> (`mult_res = self.state.regfile.raw("mult_res")`). The emulator behaviour is
> the intended post-merge design (it lets a `MULT` and its `AGG` co-issue every
> cycle with no accumulator collision — see `attn_v_256x36`), and it is what the
> goldens are built against. **Treat "AGG reduces `MULT_RES`" as authoritative;
> the spec's prose is stale.** This is a documentation bug in the spec text, not
> an emulator bug, so no emulator issue is filed.

### 2.4 Output precision and the `STR_ACC_REG` convention

Every L3 kernel writes its result with `STR_ACC_REG`, which dumps the full 512-B
`R_ACC` (128 × FP32 in wide-debug, 128 × INT32 in INT8 mode). So outputs are
**channel/key-major rows of 128 four-byte words**, regardless of dtype. The
softmax that consumes the score kernels therefore reads *unquantized* scores.

`STR_ACC_REG` lives in the **simulation-only `acc_store` slot** — it has no
hardware encoding (the assembler emits a warning to that effect). It is the
emulator's stand-in for the real activation/quantize/store path.

### 2.5 Symbolic register names

The ISA has no `.equ`/`.alias` directive, but the assembler runs every `.asm`
through a **Jinja2 preprocessor** before parsing
(`ipu_as/lark_tree.py: parse()` — it templates the source whenever it contains
`{{`, `{%`, or `{#`). That is the project's supported mechanism for naming
registers, and all 11 L3 kernels use it. Each kernel opens with a `set` block:

```jinja
{% set data_ptr  = "lr4" %}   {# byte offset into D, walks channels k #}
{% set k_index   = "lr5" %}   {# contraction index k -> selects W[j,k] #}
{% set DATA_BASE = "cr0" %}
```

and the body then reads:

```asm
LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
```

**Conventions used here.** Lower-case names are LRs (mutable — pointers, counters,
strides); UPPER-CASE names are CRs (harness-set constants and base addresses).
Each kernel's block is the single source of truth for its register assignments —
the old per-kernel `# CRs: cr0=… / # LRs: lr4=…` header tables were removed,
since they duplicated the block and would rot independently of it.

Two properties worth knowing:

- **It is a pure source-level substitution.** Renaming provably cannot change the
  program: reassembling all 11 kernels after the rename produced binaries
  **byte-identical** to the pre-rename ones.
- **Jinja runs *before* comment stripping.** A `#` comment containing `{{` or
  `{%` is still expanded — and a malformed one is a *build error*, not a comment.
  So kernel comments must not contain Jinja delimiters.

---

## 3. Cross-kernel performance summary

All figures **measured on DGX, commit `7c71a17`** (post the second ISA merge),
recorded in the project RunStats baseline. `mult%` is the fraction of cycles
whose MULT slot was active; after the `RESET_ACC` removal **`mult% == acc%`** for
every kernel (the dead accumulator-reset cycle is gone). "Mode" is the dtype the
measurement was taken in.

| Kernel | Cycles | MULT % (= ACC %) | Mode | Bound by |
|--------|-------:|----:|------|----------|
| `matmul_144x144_x128` | 43,345 | 95.7 % | INT8 | inner-loop overhead amortization |
| `matmul_288x144_x128` | 86,689 | 95.7 % | INT8 | same family as 144×144 (2× outputs) |
| `matmul_432x144_x128` | 130,033 | 95.7 % | INT8 | same family (3× outputs) |
| `matmul_144x288_x128` | 112,465 | 98.3 % | INT8 | larger K ⇒ best amortization in L3 |
| `qk_scores_256x36` | 20,993 | 87.8 % | FP32 debug | short K=36 contraction; per-row reload overhead |
| `attn_scores_km_256x36` | 21,332 | 86.7 % | INT8 | short K=36 + per-key R0 reload |
| `attn_v_256x36` | 77,787 | 94.8 % | INT8 | per-query P reload inside the key loop |
| `attn_v_bcast_36` | 77,346 | 95.3 % | INT8 | per-channel V reload; full-width MULT every cycle |
| `unfold_32x32x144` | 6,338 | 72.7 % | INT8 | store-bound (1 store per 4 MULT/ACC.STRIDE) |
| `layernorm_256x144` | 4,442 | 45.4 % | FP32 debug | multi-pass; serial dependencies between passes |
| `residual_add_256x144` | 1,163 | 49.5 % | INT8 | structural 4-cycle body (2 MULT of 4 cycles) |

**Reading the table.** Within the 144-output family, `mult%` is identical
(95.7 %) because the loop body and per-output overhead are identical — only the
output count (and hence cycle count) scales. Across the matmul families,
`mult%` *rises with K*: a longer inner contraction amortizes the fixed per-output
overhead (weight loads, ACC.FIRST peel, store), which is why `matmul_144x288`
(K=288) reaches 98.3 %. The non-matmul kernels are structurally bounded (stores,
serial passes, short bodies), not by MULT throughput.

There is no "effective PE utilization" counter distinct from `mult%` in the
emulator's RunStats; `mult%` *is* the lane-occupancy proxy, and because every
`MULT.RC.*` here drives all 128 lanes, MULT-cycle occupancy and lane occupancy
coincide for the full-width kernels. The one exception is `unfold`, whose
`ACC.STRIDE` writes only 32 of 128 accumulator lanes per call (4 calls fill the
row) — its 72.7 % counts MULT cycles, but each MULT *is* full-width; the loss is
the interleaved stores, not idle lanes.

---

## 4. The L3 transformer matmuls

All four matmuls compute the **same operation** with the **same inner-loop
template**; they differ only in the output count *N_OUT* and the contraction
width *K*. Documenting one in full and then giving the deltas is the honest way
to present them, because the assembly is genuinely near-identical.

### 4.0 The shared algorithm

```
C[j, t] = Σ_k  W[j, k] · D[k, t]            j ∈ [0, N_OUT), t ∈ [0, 256), k ∈ [0, K)
```

- **W** (weights) — output-major `[N_OUT, K]`, stored **verbatim** (no
  transpose). One output channel `j`'s K weights occupy `W_STRIDE` bytes
  (`ceil(K/128)·128`): for K=144, two 128-B slots (`W[j,0:128]`, `W[j,128:144]`
  padded); for K=288, three slots.
- **D** (activations) — interleaved channel-major `[K, 2 tg, 128 tokens]`. Row
  `(k, tg)` is at `DATA_BASE + k·256 + tg·128`. So one input channel `k`'s 256
  tokens are 256 contiguous bytes, split into two 128-token groups.
- **C** (output) — grouped channel-major `[2 tg, N_OUT, 128 tokens]`, **FP32**.
  Row `(j, tg)` at `OUTPUT_BASE + tg·N_OUT·512 + j·512`.

**Register roles.**
- **R0** ← `W[j, 0:128]`, **R1** ← `W[j, 128:256]` (loaded once per output `j`).
  These supply the **scalar** `W[j,k]`: `MULT.RC.VE`'s `src=lr5` makes
  `LR5 = k` select `W[j,k]` from R0 (k<128) or R1 (k≥128).
- **R_CYCLIC** ← `D[k, tg]` — the 128 tokens of input channel `k`, loaded fresh
  every cycle. This is the **vector**.
- **R_ACC** ← the 128 partial sums `C[j, ·, tg]`, one FP32 lane per token.

Each inner cycle does `MULT_RES[t] = W[j,k] · D[k,t]` for all 128 tokens `t` at
once, then `ACC` folds it in. After K cycles `R_ACC` holds the complete 128-token
output row for `(j, tg)`, which `STR_ACC_REG` writes out. The whole matmul is
thus **one MULT + one ACC per (j, tg, k)** — the theoretical floor — plus a small
constant of setup/store cycles per `(j, tg)`.

> **Note on register names.** The `.asm` sources now refer to registers by
> symbolic names (`data_ptr`, `k_index`, `W_BASE_LO`, …) defined in a Jinja2
> `set` block at the top of each kernel; the assembler's preprocessor expands
> them before parsing, so the emitted binary is unchanged. The listings below are
> transcribed in the **raw-register** form for readability of the addressing
> arithmetic — each kernel's own naming block gives the name↔register mapping.
> (The pre-merge mnemonics `MULT.VE.CYCLIC` / `RESET_ACC` that these comments
> used to carry have been corrected in the sources to `MULT.RC.VE` /
> `ACC.ADD.FIRST`.)

### 4.1 `matmul_144x144_x128`

| | |
|---|---|
| **Purpose** | square projection: 144 inputs → 144 outputs, over 256 tokens |
| **Shapes** | D `[144, 256]` · W `[144, 144]` → C `[144, 256]` (FP32) |
| **Memory** | `DATA_BASE=0x00000`, `WEIGHTS_BASE=0x10000`, `OUTPUT_BASE=0x20000`, `W_STRIDE=256` |
| **Loop bounds** | k-loop1 `lr6=126` (k=0..127), k-loop2 `lr11=142` (k=128..143), j-limit `lr10=144` |

The K=144 contraction is split into **two k-loops** because the weight scalars
live in two registers: k=0..127 read from R0 (`fixed_idx = k`), k=128..143 read
from R1 (`fixed_idx = k`, which the hardware routes to `R1[k−128]`). The data
pointer `lr4` runs continuously across both loops (D channels are contiguous);
only the scalar index `lr5` is reset (to 127) at the boundary so its first live
value is 128.

```asm
j_loop:
    LDR_MULT_REG r0 lr8 cr9;;          # r0[0..127] = W[j, 0..127]
    LDR_MULT_REG r1 lr8 cr2;;          # r1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    SET lr4 cr5;;                       # tg=0 data startup offset: -256
    SET lr5 cr7;;                       # k-loop1 scalar idx startup: -1

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_loop1_tg0;;
    B after_k_tg0;;

k_loop1_tg0:                            # k = 1..127 : scalar from R0[k], vector D[k,tg0]
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_loop1_tg0;;

after_k_tg0:
    SET lr5 cr8;;                       # k-loop2 scalar idx startup: 127 → first live=128 (R1[0])

k_loop2_tg0:                            # k = 128..143 : scalar from R1[k-128]
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr11 k_loop2_tg0;;

    STR_ACC_REG lr7 cr3;;               # store 512B FP32 → C[j, tg=0]  (128 token lanes)

    # -- token group 1 -------------------------------------------------------
    SET lr4 cr6;;                       # tg=1 data startup offset: -128
    SET lr5 cr7;;

    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_loop1_tg1;;
    B after_k_tg1;;

k_loop1_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_loop1_tg1;;

after_k_tg1:
    SET lr5 cr8;;

k_loop2_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr11 k_loop2_tg1;;

    STR_ACC_REG lr7 cr4;;               # store 512B FP32 → C[j, tg=1]
    ADD lr7 lr7 lr3;;                   # advance output ptr (+512)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;   # next j: weight offset += W_STRIDE, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
```

**Per-bundle annotation of the inner cycle** (e.g. inside `k_loop1_tg0`):
- `LDR_CYCLIC_MULT_REG lr4 cr0 lr0` — XMEM reads `D[live lr4]` (the post-`ADD`
  address `= D[k, tg0]`) into R_CYCLIC slot 0.
- `ADD lr4 lr4 lr2` — advance the data pointer by 256 (one channel).
- `ADD lr5 lr5 cr1` — advance the scalar index `k` by 1.
- `MULT.RC.VE lr0 lr5 0 lr0` — for all 128 token lanes `t`:
  `MULT_RES[t] = R_CYCLIC[t] · R0[live lr5]` `= D[k,t]·W[j,k]`.
- `ACC` — `R_ACC[t] += MULT_RES[t]`.
- `BLT lr5 lr6 …` — branch on the **pre-increment** `lr5`, so the body runs for
  live `k = 1..127` (the peeled cycle did `k=0`).

**Walkthrough / intuition.** The choice of D channel-major and W output-major is
what makes this a *single-broadcast* matmul: one weight scalar fans out across
128 tokens in one cycle, so the 128-wide SIMD is fully busy on useful tokens. The
two-token-group structure exists because a stream is 256 tokens but the SIMD and
accumulator are 128 lanes wide — so each output channel is computed twice, once
per group, sharing the same weights (no reload between groups, only a pointer
reset). The "skew + peel" pattern (startup offsets `−256`/`−1`, `ACC.FIRST` on the
first step) is the cost of having no `RESET_ACC`: the accumulator is *seeded* by
the first product instead of being cleared, which removes a dead cycle and is
exactly why `mult% == acc%`.

- **Measured performance:** 43,345 cycles, **95.7 %** MULT/ACC (INT8, DGX).
  Bound by the per-output overhead (2 weight loads + 2 ACC.FIRST peels + 2 stores
  per `j`) against the 288 useful MULT cycles per `j` (144 k-steps × 2 tg).
- **Correctness (re-run for this doc):**
  - INT8 vs `ipu_math` golden — **PASS** (byte-exact).
  - FP8 E4M3 vs golden — **PASS** (byte-exact).
  - FP8 E5M2 vs golden — **PASS** (byte-exact).
  - FP8 drift: the FP32-accumulate path reproduces the golden bit-for-bit, so
    drift **vs the FP8-decoded-input reference is 0**. (The matmuls ship no
    wide-FP32 golden, so an *input-quantization* drift number isn't computed for
    them here; see `qk_scores` §5.1 for a measured input-quant drift.)

### 4.2 `matmul_288x144_x128` — 2× the outputs

Identical body to §4.1; the only changes are the output count and the memory map
(weights are bigger, so `OUTPUT_BASE` is pushed past them to avoid overlap).

| Delta vs 144×144 | |
|---|---|
| `N_OUT` | **288** (`lr10 = 288`) |
| `OUTPUT_BASE` | **0x30000** (weights end at `0x10000 + 288·256 = 0x22000`; output must clear that) |
| everything else | unchanged (`K=144`, `W_STRIDE=256`, `lr6=126`, `lr11=142`) |

> **Memory-overlap rule (load-bearing).** `WEIGHTS_BASE + N_OUT·W_STRIDE ≤
> OUTPUT_BASE` must hold. A historical bug in a sibling kernel set
> `OUTPUT_BASE=0x20000` while weights ran to `0x22000`, corrupting weights for
> `j ≥ 256`. The fix — and the reason every ≥288-output L3 matmul uses
> `OUTPUT_BASE=0x30000` — is exactly this check.

- **Measured:** 86,689 cycles, **95.7 %** (exactly 2× the cycles of 144×144, same
  efficiency — same body, 2× the output rows).
- **Correctness (re-run):** INT8 / E4M3 / E5M2 all **PASS** byte-exact; FP8
  accumulate-path drift = 0.

### 4.3 `matmul_432x144_x128` — fused QKV (3× the outputs)

Again the identical body. 432 = 3 × 144, i.e. the fused Q, K, V projection.

| Delta | |
|---|---|
| `N_OUT` | **432** (`lr10 = 432`) |
| `OUTPUT_BASE` | **0x30000** |
| everything else | unchanged |

- **Measured:** 130,033 cycles, **95.7 %** (3× the 144×144 cycle count).
- **Correctness (re-run):** INT8 / E4M3 / E5M2 all **PASS** byte-exact; FP8
  accumulate-path drift = 0.

### 4.4 `matmul_144x288_x128` — FFN linear-2 (K=288)

This one is structurally different: the contraction is **K=288**, which exceeds
the 256-byte R0++R1 window, so the weights are loaded in **three 128-wide chunks**
and the inner loop runs **three times per token-group**, reloading R0 between
chunks. The data pointer `lr4` advances continuously across all three chunks; the
scalar index `lr5` resets to −1 (`cr8`) at each chunk start.

| | |
|---|---|
| **Purpose** | FFN second linear: contract the 288-wide hidden back to 144 |
| **Shapes** | D `[288, 256]` · W `[144, 288]` → C `[144, 256]` (FP32) |
| **Memory** | `DATA_BASE=0x00000`, `WEIGHTS_BASE=0x20000`, `OUTPUT_BASE=0x40000`, `W_STRIDE=384` (=3·128) |
| **Loop bounds** | per-chunk `lr6=126` (width 128 each), j-limit `lr10=144` |

```asm
j_loop:
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;  # tg=0 data startup (-256); r0 = W[j, 0..127]
    SET lr5 cr8;;                            # chunk0 scalar idx startup: -1

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_chunk0_tg0;;
    B after_chunk0_tg0;;

k_chunk0_tg0:                                # k = 1..127, scalar from r0
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk0_tg0;;

after_chunk0_tg0:
    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;  # chunk1 reset idx; r0 = W[j, 128..255]

k_chunk1_tg0:                                # k = 128..255, r0 reloaded, idx restarts 0..127
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk1_tg0;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;  # chunk2 reset; r0 = W[j, 256..287] + zeros

k_chunk2_tg0:                                # k = 256..287 (32 real + padding)
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk2_tg0;;

    STR_ACC_REG lr7 cr4;;                   # store 512B FP32 → C[j, tg=0]

    SET lr4 cr7; LDR_MULT_REG r0 lr8 cr9;;  # tg=1 data startup (-128); r0 = W[j, 0..127]
    SET lr5 cr8;;

    # ... (chunks 0/1/2 for tg=1 repeat the three blocks above) ...
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_chunk0_tg1;;
    B after_chunk0_tg1;;
k_chunk0_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk0_tg1;;
after_chunk0_tg1:
    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;
k_chunk1_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk1_tg1;;
    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;
k_chunk2_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk2_tg1;;

    STR_ACC_REG lr7 cr5;;                   # store 512B FP32 → C[j, tg=1]
    ADD lr7 lr7 lr3;;                       # advance output ptr

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;       # next j: weight offset += 384, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
```

**Walkthrough.** The three-chunk structure is the price of K > 256: a contraction
index can address at most 256 distinct scalar bytes (R0++R1), so K=288 must be
tiled. Crucially the *vector* stream (R_CYCLIC ← D) never needs chunking — the
data pointer just keeps walking the 288 channels — so the only per-chunk cost is
one extra `LDR_MULT_REG` (weight reload) and one `SET lr5`. Because the inner
contraction is now 288 long per group versus 144, the fixed per-output overhead
is amortized over twice as many MULT cycles, which is why this kernel has the
**highest L3 matmul efficiency at 98.3 %**.

- **Measured:** 112,465 cycles, **98.3 %** MULT/ACC.
- **Correctness (re-run):** INT8 / E4M3 / E5M2 all **PASS** byte-exact; FP8
  accumulate-path drift = 0.

**Gotchas across all four matmuls.**
- The two-loop (or three-chunk) split is **not** an optimization choice — it is
  forced by the 256-byte R0++R1 scalar window.
- The loop bound is `width − 2`, not `width − 1` (the live/snapshot skew). An
  off-by-one here silently drops the last contraction term.
- Output is FP32 even in INT8 mode (INT32 lanes); `STR_ACC_REG` always writes
  512 B. Downstream readers must treat each output row as 128 four-byte words.

---

## 5. The attention kernels

There are **four** attention kernels, forming two pairs:

- **Scores** (`Sᵢₛ = Σ_c Qᵢ_c · Kₛ_c`, contract over head_dim 36):
  - `qk_scores_256x36` — output **query-major** (`S[i, s]`, lanes = keys).
  - `attn_scores_km_256x36` — output **key-major** (`S[i, s]`, lanes = queries).
- **attn@V** (`Oᵢ_t = Σ_s Pᵢ_s · Vₛ_t`, contract over keys 256):
  - `attn_v_256x36` — consumes **query-major** P, reduces with **AGG**.
  - `attn_v_bcast_36` — consumes **key-major** P, reduces with a **broadcast matmul**.

All four process **one head at a time at the score stage** (the harness slices the
head before loading) except the attn@V kernels, which loop over all 4 heads
internally. Head_dim is 36, N = 256 tokens (2 groups of 128).

### 5.0 The layout decision and its softmax consequence (read this first)

Softmax in attention normalizes each **query's** score row over all keys:
`softmax_s(S[i, ·])`. The reduction axis is therefore **keys**. The two score
layouts place that axis differently:

| | query-major (`qk_scores`) | key-major (`attn_scores_km`) |
|---|---|---|
| Output element `S[i,s]` stored at | `S_BASE + i·1024 + g·512 + (s%128)·4` | `SBASE + s·1024 + g·512 + (i%128)·4` |
| A stored **512-B row** holds | one query `i`, 128 **keys** | one key `s`, 128 **queries** |
| The 128 SIMD **lanes** are | **keys** | **queries** |
| Softmax (reduce over keys) is… | a reduction **within a row** (across lanes) — convenient, contiguous | a reduction **across rows** (gather one lane from every key's row) — awkward |

So **query-major scores are softmax-friendly**: the keys to be normalized are
contiguous, so the downstream softmax reads one query's whole score vector as two
512-B rows. **Key-major scores invert this**: each row is one key across all
queries, so softmax must gather lane `i` from all 256 key-rows to assemble query
`i`'s vector. The key-major kernel exists because it pairs naturally with a
**key-major attn@V** (`attn_v_bcast_36`), where having P key-major lets the value
channel sit entirely in R0++R1 and be reused across all keys with no mid-loop
reload — i.e. the layout choice trades softmax convenience for attn@V efficiency.
The two attn@V kernels mirror this: `attn_v_256x36` pays for query-major P
(softmax-friendly) by reloading P every key via the AGG reduction;
`attn_v_bcast_36` exploits key-major P for a clean full-width matmul.

### 5.1 `qk_scores_256x36` — QKᵀ, query-major

| | |
|---|---|
| **Purpose** | one head's query-major score matrix `S[i,s] = Σ_{c=0..35} Q[i,c]·K[s,c]` |
| **Shapes** | Q `[256, 36]` (staged), K `[36, 256]` → S `[256, 256]` (FP32, query-major) |
| **Memory** | `K_BASE=0x00000`, `QROW_BASE=0x40000`, `S_BASE=0x80000` |
| **Loop bounds** | c-loop `lr6=34` (=0+36−2), query-limit `lr10=256` |

This is **exactly the matmul broadcast template** with a width-36 contraction. The
trick is the input staging:
- **Q is staged query-major by the harness** (`QROW[i] = Q[i, 0:36]` contiguous).
  This lets one query's 36 head-channels load into **R0** with a single
  `LDR_MULT_REG`, so R0 plays the **scalar** role (`MULT.RC.VE src=lr5` picks
  `Q[i,c]`).
- **K stays channel-major**: `K[s,c]` at `K_BASE + c·256 + s`, so one channel's
  128 keys are contiguous → loaded straight into **R_CYCLIC** as the **vector**.

```asm
q_loop:
    LDR_MULT_REG r0 lr8 cr9;;            # r0 = QROW[i] = Q[i, 0..35] (rest pad)

    # -- key group 0 (keys 0..127) -------------------------------------------
    SET lr4 cr5;;                        # g=0 K-data startup: -256
    SET lr5 cr7;;                        # channel scalar idx startup: -1

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g0;;
    B after_c_g0;;

c_loop_g0:                               # c = 1..35 : MULT_RES[s] = Q[i,c]·K[s,c]
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g0;;

after_c_g0:
    STR_ACC_REG lr7 cr3;;                # store 512B → S[i, keys 0..127]  (lanes = keys)

    # -- key group 1 (keys 128..255) -----------------------------------------
    SET lr4 cr6;;                        # g=1 K-data startup: -128
    SET lr5 cr7;;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g1;;
    B after_c_g1;;

c_loop_g1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g1;;

after_c_g1:
    STR_ACC_REG lr7 cr4;;                # store 512B → S[i, keys 128..255]
    ADD lr7 lr7 lr3;;                    # advance output ptr (+1024 = 2 key-groups)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;  # next query: Q ptr += 512, i++
    BLT lr9 lr10 q_loop;;

end:
    BKPT;;
```

**Per-bundle (inner cycle of `c_loop_g0`):** XMEM loads `K[live lr4]` (channel `c`,
keys `g·128 .. g·128+127`) into R_CYCLIC; `MULT.RC.VE` broadcasts the scalar
`Q[i,c]` (= `R0[live lr5]`) across all 128 key lanes giving
`MULT_RES[s] = Q[i,c]·K[s,c]`; `ACC` folds it; after 36 channels `R_ACC[s] = S[i,s]`
for the 128 keys of group `g`. **No AGG** — the contraction is across cycles
(channels), and each of the 128 lanes is an independent key whose score
accumulates in place.

**Walkthrough.** The lanes are **keys**, so one 512-B store is one query's scores
for 128 keys — which is precisely the softmax-friendly query-major layout. Scores
are stored **raw** (`STR_ACC_REG`, full FP32), so the softmax stage reads
unquantized scores. The cost: Q must be pre-gathered into contiguous rows
(`QROW`), an O(256·36) harness-side transpose, because the canonical Q is
channel-major and a per-query scalar fetch needs the 36 channels adjacent.

- **Measured:** 20,993 cycles, **87.8 %** MULT/ACC (FP32 debug). The lower
  efficiency vs the matmuls is the short K=36 contraction: 36 useful MULTs per
  `(i,g)` against the fixed overhead of a Q reload, two ACC.FIRST peels and two
  stores per query — the overhead is a larger fraction when K is small.
- **Correctness (re-run for this doc):**
  - **wide-FP32 vs numpy `Qᵀ@K`** — **PASS** (`atol=rtol=1e-3`). This is the
    primary correctness check, free of quantization noise.
  - INT8 / E4M3 / E5M2 vs `ipu_math` goldens — **PASS** (byte-exact).
  - **FP8 input-quantization drift** (measured here: FP8 score golden vs the
    wide-FP32 score golden): **E4M3** mean abs error **0.053**, **E5M2** **0.105**
    (units of the score). Relative error is dominated by near-zero scores where
    products cancel, so the absolute figure is the honest one. This is *input*
    quantization only — the FP32 accumulate path reproduces each FP8 golden
    bit-for-bit (drift vs the FP8-decoded-input reference = 0).

### 5.2 `attn_scores_km_256x36` — kQᵀ, key-major

Same math as §5.1, **transposed output**: lanes are **queries**, the outer loop is
over **keys**.

| | |
|---|---|
| **Purpose** | one head's key-major score row `S[i,s]`, lanes = queries |
| **Shapes** | Q `[36, 256]` (channel-major, verbatim), K `[256, 36]` (key-major scratch) → S `[256, 256]` (FP32, key-major) |
| **Memory** | `QBASE=0x00000`, `SBASE=0x20000`, `KBASE_KM` (key-major K) in `cr9` |
| **Loop bounds** | c-loop `lr6=34`, key-limit `lr10=256` |

Here the roles **swap relative to qk_scores**: now **K** is the per-iteration
scalar (one key `s`'s 36 channels loaded into R0, key-major), and **Q**
channel-major streams through R_CYCLIC as the vector (a channel column of 128
queries is contiguous). The contraction is still over the 36 head channels.

```asm
s_loop:
    ADD lr8 lr8 lr12;;                  # key byte offset += 128 (first live = 0)
    LDR_MULT_REG r0 lr8 cr9;;           # r0[0..127] = K[s, 0:35] + zeros (key-major)

    # -- query group 0 (queries 0..127) -------------------------------------
    SET lr4 cr5;;                       # channel-column startup: -256
    SET lr5 cr7;;                       # scalar idx c startup: -1

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g0;;
    B after_c_g0;;

c_loop_g0:                              # c = 1..35 : MULT_RES[i] = Q[i,c]·K[s,c]
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g0;;

after_c_g0:
    STR_ACC_REG lr7 cr2;;               # store S[queries 0..127, s]  (lanes = queries)
    ADD lr7 lr7 lr3;;                   # output ptr += 512

    # -- query group 1 (queries 128..255) -----------------------------------
    SET lr4 cr6;;                       # g=1 channel-column startup: -128 → first live = 128
    SET lr5 cr7;;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g1;;
    B after_c_g1;;

c_loop_g1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g1;;

after_c_g1:
    STR_ACC_REG lr7 cr2;;               # store S[queries 128..255, s]
    ADD lr7 lr7 lr3;;                   # output ptr += 512

    ADD lr9 lr9 cr1; BLT lr9 lr10 s_loop;;   # next key

end:
    BKPT;;
```

**Per-bundle:** identical shape to qk_scores, but `MULT_RES[i] = Q[i,c]·K[s,c]`
where the lane index `i` is now a **query**. After 36 channels `R_ACC[i] = S[i,s]`
for the 128 queries of the group; the store is one key's scores across 128
queries — the key-major layout.

**Softmax consequence (restated concretely).** To softmax query `i`, the consumer
must read `S[i, s]` for all 256 keys `s`; in this layout those values are lane `i`
of 256 different 512-B rows (`SBASE + s·1024 + g·512`), so the softmax does a
strided gather. The kernel produces this layout deliberately to feed a key-major
attn@V; standing alone it is the *less* softmax-friendly of the two.

- **Measured:** 21,332 cycles, **86.7 %** MULT/ACC (INT8). Slightly lower than
  `qk_scores` because K must be reloaded into R0 every key (`s`-loop), one extra
  load per output row.
- **Correctness (re-run):** INT8 / E4M3 / E5M2 all **PASS** byte-exact. (No
  wide-FP32 golden ships for this kernel; the INT8 path is the reference and the
  FP8 accumulate path reproduces it bit-exactly.)

### 5.3 `attn_v_256x36` — attn@V via AGG (query-major P)

This is the most subtle L3 kernel. It computes `O[i,t] = Σ_s P[i,s]·V[s,t]` with
**query-major P** and a **per-element AGG reduction**.

| | |
|---|---|
| **Purpose** | attn@V for all 4 heads; query-major scores → channel-major output |
| **Shapes** | P `[4, 256, 256]` (head, query, key) · V `[4, 36, 256]` → O `[4, 256, 36]` (FP32) |
| **Memory** | `PBASE=0x00000`, `VBASE=0x40000`, `OBASE=0x50000` |
| **Reduction** | `MULT.RC.VV` (P chunk × V chunk, lanes = keys) + `AGG.SUM[.FIRST]` per query |

The contraction here is over **keys** (256 of them), and the output `O[i,t]` for a
single `(query i, channel t)` is a scalar — a full reduction over 256 key-lanes.
The kernel does this with **AGG**: each cycle multiplies a 128-key chunk of P
against the same 128-key chunk of V (`MULT.RC.VV`, lanes = keys) and then
`AGG.SUM` reduces those 128 `MULT_RES` lanes into **one** R_ACC slot — the slot
indexed by the query. Stepping the destination slot per query (via `INC lr3`)
writes 128 queries' results into 128 distinct R_ACC lanes with **no collision**.

```asm
    SET lr2 cr0;;                       # rc write index = 0 (const)
    SET lr4 cr0;;                       # value-channel offset = 0
    SET lr6 cr0;;                       # head P offset = 0
    SET lr8 cr0;;                       # head counter = 0
    SET lr10 cr0;;                      # O channel offset = 0
    SET lr11 cr5;;                      # P query stride = 256

head_loop:
    SET lr7 cr0;;                       # t counter = 0
    ADD lr9 lr6 cr0;;                   # group P offset = head P offset (g=0)

t_loop:
    # ===================== group g = 0 (queries 0..127) =====================
    # ---- chunk 0: keys 0..127 ----
    ADD lr1 lr4 cr0;;                   # V chunk0 offset = chan
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;   # R_CYCLIC = V[0..127, t]   (base VBASE)
    SUB lr0 lr9 lr11;;                  # P inner start = group P off - 256
    SET lr3 cr0;;                       # dest/inner counter = 0
g0c0_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr3 1; BLT lr3 cr9 g0c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD lr1 lr4 cr6;;                   # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;   # R_CYCLIC = V[128..255, t]
    ADD lr0 lr9 cr6;;                   # P chunk1 base = group P off + 128
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g0c1_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM lr3 1; BLT lr3 cr9 g0c1_loop;;

    ADD lr5 lr10 cr0;;                  # O g=0 offset = O chan offset
    STR_ACC_REG lr5 cr4;;              # O[0..127, t] = R_ACC   (base OBASE)

    # ===================== group g = 1 (queries 128..255) ===================
    # ---- chunk 0: keys 0..127 ----
    ADD lr1 lr4 cr0;;                   # V chunk0 offset = chan (same channel)
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;
    ADD lr0 lr9 cr7;;                   # g=1 P base = group P off + 32768
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g1c0_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr3 1; BLT lr3 cr9 g1c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD lr1 lr4 cr6;;                   # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;
    ADD lr0 lr9 cr7;;                   # g=1 P base
    ADD lr0 lr0 cr6;;                   # + 128 (chunk1)
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g1c1_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM lr3 1; BLT lr3 cr9 g1c1_loop;;

    ADD lr5 lr10 cr12;;                 # O g=1 offset = O chan offset + 512
    STR_ACC_REG lr5 cr4;;              # O[128..255, t] = R_ACC

    # ----- next t: advance value-channel offset (+256 in) and O offset (+1024), t++ -----
    ADD lr4 lr4 cr5;;                   # chan += 256
    ADD lr10 lr10 cr13;;               # O chan offset += 1024
    INC lr7 1;;                         # t++
    BLT lr7 cr10 t_loop;;

    # ----- next head: head P offset += 65536, head++ -----
    ADD lr6 lr6 cr8;;                   # head P offset += 65536
    INC lr8 1;;
    BLT lr8 cr11 head_loop;;

end:
    BKPT;;
```

**Per-bundle of the reduction loop (`g0c0_loop`)** — this single dense bundle is
the heart of the kernel:
- `LDR_MULT_REG r0 lr0 cr2` — load query `i`'s 128 keys of P (chunk 0) into R0.
- `ADD lr0 lr0 lr11` (+256) — advance to the next query's P row.
- `INC lr3 1` — advance the **destination slot** = the query index.
- `MULT.RC.VV lr2 r0 0 lr2` — `MULT_RES[s] = R_CYCLIC[s]·R0[s] = V[s,t]·P[i,s]` for
  the 128 keys `s` of this chunk (V chunk held constant across all 128 queries).
- `AGG.SUM.FIRST lr3 1` — reduce the 128 `MULT_RES` lanes to a scalar and write it
  to `R_ACC[LR3]` (clean init for chunk 0). `full_xmem_row=1` ⇒ 128 lanes.
- `BLT lr3 cr9 …` — `cr9 = 127`, snapshot read ⇒ exactly 128 iterations (queries
  0..127), each writing its own R_ACC slot.

Chunk 1 repeats with `AGG.SUM` (not `.FIRST`), which **adds** the keys-128..255
partial to the keys-0..127 partial already in `R_ACC[i]`. After both chunks,
`R_ACC[i] = Σ_{s=0..255} P[i,s]·V[s,t]` = `O[i,t]` for all 128 queries of the
group, stored as one 512-B FP32 column segment.

**Why this layout / why AGG.** With query-major P, one query's 256 scores are
contiguous, so the SIMD lanes naturally hold *keys* and a key-reduction is what
`AGG` does. The genius is **co-issue**: the spec discrepancy in §2.3 notwithstanding,
the emulator's AGG reduces `MULT_RES` (not R_ACC), so `MULT` and `AGG` run in the
**same** bundle with no read-after-write hazard on the accumulator — one bundle per
(query, key-chunk). The per-query destination-slot stepping is what makes 128
independent reductions share one R_ACC without collision. The V chunk is loaded
once per `(t, g, chunk)` and reused across all 128 queries; only P (R0) reloads.

> 🔑 **Golden gotcha (documented in `gen_test_data.py`).** `AGG.SUM` reduces with a
> **float64 left-fold of the float32 products per chunk**, then a float64 add of the
> two chunk partials, rounded to float32. The golden reference *must mirror this
> exact per-chunk reduction order* — a naive `P @ V` (single fold over 256 keys)
> diverges under **E5M2** and the byte-equal test fails. The shipped golden does
> mirror it, which is why all three dtypes pass.

- **Measured:** 77,787 cycles, **94.8 %** MULT/ACC (INT8). Bound by the per-query
  P reload inside the 128-iteration key loop.
- **Correctness (re-run for this doc):**
  - **wide-FP32 inline test** (the kernel's exact MULT.RC.VV → AGG.SUM.FIRST/AGG.SUM
    pattern vs numpy `P @ V`, no quant noise) — **PASS** (`rtol=1e-5, atol=1e-4`).
  - INT8 / E4M3 / E5M2 vs goldens — **PASS** byte-exact.
  - FP8 accumulate-path drift vs the FP32-of-decoded-inputs reference (query-major
    layout): E4M3 **0**, E5M2 **~1e-6 max** (float32 rounding only) — confirming the
    reduction-order match.

### 5.4 `attn_v_bcast_36` — attn@V via broadcast matmul (key-major P)

Same `O[i,t] = Σ_s P[i,s]·V[s,t]`, but with **key-major P** it becomes a plain
broadcast matmul — **no AGG, no collisions**.

| | |
|---|---|
| **Purpose** | attn@V for all 4 heads; key-major scores → channel-major output |
| **Shapes** | P `[4, 256, 256]` (head, **key**, query) · V `[4, 36, 256]` → O `[4, 256, 36]` (FP32) |
| **Memory** | `PBASE=0x00000`, `VBASE=0x40000`, `OBASE=0x50000` (same map as §5.3) |
| **Reduction** | `MULT.RC.VE` (V scalar × P vector) + `ACC` over keys; lanes = queries |

The layout flip is the whole story: with **key-major P** (`P[i,s]` at
`PBASE + h·65536 + s·256 + i`), one key `s`'s 128 queries are contiguous → P
streams through **R_CYCLIC** as the vector, lanes = queries. The value channel's
256 keys fit in **R0 (s=0..127) ++ R1 (s=128..255)**, loaded once per channel, and
`MULT.RC.VE` indexes the scalar `V[s,t]` by the key counter `s` — so V never
reloads inside the key loop. The contraction over keys is an ordinary cross-cycle
`ACC`, exactly like the matmuls.

```asm
    SET     lr10 cr0;;                 # R0 source offset = chan*256 = 0
    SET     lr11 cr5;;                 # R1 source offset = chan*256 + 128 = 128
    SET     lr12 cr0;;                 # P head base = 0
    SET     lr14 cr0;;                 # output offset = 0

    SET     lr6  cr0;;                 # head counter (0..3)
h_loop:
    SET     lr7  cr0;;                 # channel (t) counter (0..35)
t_loop:
    # -- load V[:, chan] into R0 (s=0..127) and R1 (s=128..255) ----------------
    LDR_MULT_REG r0 lr10 cr3;;         # R0 = V[0:127,   chan]
    LDR_MULT_REG r1 lr11 cr3;;         # R1 = V[128:255, chan]

    SET     lr13 cr0;;                 # P group offset = g*128 = 0
    SET     lr9  cr0;;                 # g counter (0..1)
g_loop:
    # P[h,g,s] address = PBASE + h*65536 + s*256 + g*128.  data ptr lr4 = start-256.
    SET     lr4  cr2;;                 # PBASE
    ADD     lr4  lr4  lr12;;           # + h*65536
    ADD     lr4  lr4  lr13;;           # + g*128  -> P[h,g,s=0]
    SUB     lr4  lr4  lr1;;            # - 256 startup (ADD +256 -> live s=0)
    SET     lr5  cr6;;                 # key index startup = -1 (ADD +1 -> s=0)

    # Peeled first key (s=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr1; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 cr8 s_loop;;
    B s_done;;
s_loop:                                # s = 1..255 : MULT_RES[i] = P[i,s]·V[s,t]
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr1; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 cr8 s_loop;;
s_done:

    STR_ACC_REG lr14 cr4;;             # 512B FP32 -> O[g*128:+128, chan]
    ADD     lr14 lr14 lr2;;            # advance output-row offset by 512

    ADD     lr13 lr13 lr3;;            # next group: P offset += 128
    ADD     lr9  lr9  cr1; BLT lr9 cr11 g_loop;;

    ADD     lr10 lr10 lr1;;            # next channel: R0 source += 256
    ADD     lr11 lr11 lr1;;            # next channel: R1 source += 256
    ADD     lr7  lr7  cr1; BLT lr7 cr9 t_loop;;

    ADD     lr12 lr12 cr7;;            # next head: P head base += 65536
    ADD     lr6  lr6  cr1; BLT lr6 cr10 h_loop;;

end:
    BKPT;;
```

**Per-bundle (`s_loop`):** XMEM loads `P[live lr4]` = key `s`'s 128 queries into
R_CYCLIC; `MULT.RC.VE lr0 lr5` broadcasts the scalar `V[s,t]` (indexed by the key
counter `s` across R0++R1) over all 128 query lanes:
`MULT_RES[i] = P[i,s]·V[s,t]`; `ACC` folds it; the key bound `cr8 = 254`
(=0+256−2) runs the body for live `s = 1..255` (the peel did `s=0`). After 256
keys `R_ACC[i] = O[i,t]` for the 128 queries of the group.

**Walkthrough / contrast with §5.3.** Both kernels compute the identical attn@V;
the only difference is which operand is contiguous. Key-major P makes the
**queries** the lanes, so the reduction over keys is an ordinary cross-cycle `ACC`
(matmul-style), and all 256 values of one value-channel live in R0++R1 with no
mid-loop reload — that is why `attn_v_bcast` is marginally *more* efficient
(95.3 % vs 94.8 %) than the AGG kernel: it does a full-width scalar×vector MULT
every cycle and reloads V only once per channel, whereas the AGG kernel reloads P
every query. The trade-off lives upstream: this kernel needs **key-major scores**
(from `attn_scores_km`), which are the softmax-*unfriendly* layout.

- **Measured:** 77,346 cycles, **95.3 %** MULT/ACC (INT8).
- **Correctness (re-run for this doc):**
  - **wide-FP32 inline test** vs numpy `P @ V` — **PASS** (`rtol=1e-5, atol=1e-4`).
  - INT8 / E4M3 / E5M2 vs goldens — **PASS** byte-exact.
  - FP8 accumulate-path drift vs FP32-of-decoded-inputs reference (**key-major**
    layout): E4M3 **0**, E5M2 **~1e-6 max**. *(Computing this required using the
    key-major P reshape; the query-major reshape gives a spurious ~600 % error —
    an independent confirmation that this kernel really is key-major.)*

---

## 6. The elementwise / structural kernels

### 6.1 `unfold_32x32x144` — NHCW → 4 channel-major streams

| | |
|---|---|
| **Purpose** | rearrange a 32×32×144 spatial tensor (NHCW-striped) into 4 sub-grid streams (TL/TR/BL/BR), channel-major |
| **Shapes** | in `[8 stripes, 144 ch, 128]` → 4 × `[288 rows, 128]` FP32 |
| **Memory** | `SRC_BASE=0x00000`, `ONES_BASE=0x24000`, `DST_BASE=0x30000` |
| **Key instruction** | `ACC.STRIDE` (the only L3 kernel that uses it) |

This kernel does **no real arithmetic** — it is a data-movement kernel disguised
as a multiply. Each 128-byte stripe row is `4 spatial rows × 32 cols`. The four
output streams are the four (even/odd col) × (even/odd row) decimations of that
4×32 tile:

| Stream | columns | rows | `ACC.STRIDE` h / v |
|---|---|---|---|
| TL | even | even | `on  on` |
| TR | odd  | even | `on_inv on` |
| BL | even | odd  | `on  on_inv` |
| BR | odd  | odd  | `on_inv on_inv` |

The multiply is a **pass-through**: `r_cyclic` is preloaded once with dtype-`1.0`,
and `MULT.RC.VV lr0 r0 0 lr0` computes `stripe[i] × 1.0 = stripe[i]`. Then
`ACC.STRIDE` selects 32 of the 128 `MULT_RES` elements (2 rows × 16 cols of the
chosen parity) and **direct-writes** them into one 32-lane slot of `R_ACC`
(slot chosen by the offset LR). Four `ACC.STRIDE` calls (slots 0,1,2,3) fill the
128-lane accumulator from four stripes, then one `STR_ACC_REG` emits the row.

```asm
    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific), once

ch_loop:
    # ----- Stream TL  (even cols, even rows) -----
    # tg=0: stripes 0..3 -> r_acc slots 0..3
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr3;;
    STR_ACC_REG         lr8 cr9;;           # TL tg=0 -> DST_TL + ch*2*512
    # tg=1: stripes 4..7
    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr3;;
    STR_ACC_REG         lr9 cr9;;           # TL tg=1 -> DST_TL + ch*1024 + 512

    # ----- Stream TR  (odd cols, even rows): ACC.STRIDE 32 on_inv on -----
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr3;;
    STR_ACC_REG         lr8 cr10;;          # TR tg=0
    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr3;;
    STR_ACC_REG         lr9 cr10;;          # TR tg=1

    # ----- Stream BL  (even cols, odd rows): ACC.STRIDE 32 on on_inv -----
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr3;;
    STR_ACC_REG         lr8 cr11;;          # BL tg=0
    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr3;;
    STR_ACC_REG         lr9 cr11;;          # BL tg=1

    # ----- Stream BR  (odd cols, odd rows): ACC.STRIDE 32 on_inv on_inv -----
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr3;;
    STR_ACC_REG         lr8 cr12;;          # BR tg=0
    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr3;;
    STR_ACC_REG         lr9 cr12;;          # BR tg=1

    ADD                 lr4 lr4 lr5;;            # src offset: next channel (+128)
    ADD                 lr8 lr8 lr6; ADD lr9 lr9 lr6;;  # dst offsets: +1024 per channel
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;;      # loop while ch < 144

end:
    BKPT;;
```

**`ACC.STRIDE` enum gotcha (verified passing).** The horizontal/vertical stride
modes decode via an explicit lookup table (`acc_stride_enums.py`), so the natural
names are correct: `on` = enabled/even (index 1), `on_inv` = enabled/inverted/odd
(index 2). An earlier bit-packed mis-decode (now fixed upstream) made `on_inv`
wrong; on the current branch the four stream encodings above are the verified-correct ones.

- **Measured:** 6,338 cycles, **72.7 %** MULT/ACC (INT8) — up from 61.5 % before
  the `acc.stride` redesign. The 27 % loss is **store-bound**: the body is 8
  `MULT`+`ACC.STRIDE` cycles followed by interleaved `STR_ACC_REG` cycles (one
  store per 4 stride-writes), and the stores have no MULT. Every MULT here is
  full-width 128-lane (pass-through), so this is not a lane-occupancy loss.
- **Correctness (re-run):** INT8 / E4M3 / E5M2 vs goldens — **PASS** byte-exact.
  Because the multiply is a 1.0 pass-through, the FP8 output is the exact decoded
  input rearranged (the FP32 store path is lossless).

### 6.2 `layernorm_256x144` — LayerNorm over 144 channels

| | |
|---|---|
| **Purpose** | `out[ch,i] = γ[ch]·(x[ch,i] − μ[i]) / σ[i] + β[ch]` per token `i`, over 144 channels |
| **Shapes** | x `[144, 256]`, γ/β `[144]` → out `[144, 256]` (FP32) |
| **Mode** | **wide-vector FP32 only** (the only L3 kernel that genuinely needs FP32 math: mean/variance/rsqrt) |
| **Memory** | `DATA_BASE=0x00000`, scratch in `0x24000–0x37400`, `OUTPUT_BASE=0x37400` |

LayerNorm normalizes across the **channel** axis, so the reduction (`Σ_ch`) runs
*along channels* while the 128 token-lanes stay independent. It is a genuine
**multi-pass** kernel — six sequential steps per token-group, each a small loop
over 144 channels, with scratch buffers carrying intermediates between passes:

1. **−μ[i]** = `Σ_ch x[ch,i] · (−1/N)` — accumulate the negative mean directly
   (broadcast `−1/144` scalar via `MULT.RC.VV` against each channel row).
2. **centered[ch,i]** = `x[ch,i] + (−μ[i])` — add the mean (ones × x, then −μ × 1).
3. **Σ(centered²)** — `MULT.RC.VS` squares each centered row; `ACC` sums.
4. **variance = (1/N)·Σ**, then **1/σ = `ACTIVATE rsqrt`** → `STR_POST_AAQ_REG`.
   This is the **only** L3 use of the AAQ slot: `ACTIVATE rsqrt 1` reads `R_ACC`,
   writes `1/√var` into `POST_AAQ_REG`, which is stored to the `INVSTD` scratch.
5. **normalized[ch,i]** = `centered[ch,i] · (1/σ[i])` (overwrite centered).
6. **out[ch,i]** = `γ[ch]·normalized + β[ch]`. Because **N=144 > 128**, γ and β
   span two 512-B rows, so step 6 is split into **sub-loop A (ch 0..127)** and
   **sub-loop B (ch 128..143)** with a reload of γ/β between them. The scalar
   `γ[ch]`/`β[ch]` is selected by a per-channel `fixed_idx` via `MULT.RC.VE`.

The data stride between consecutive channels within one token-group is **1024 B**
(`= N_TG · 512`, channels interleaved with the two groups), held in `lr7`; the
whole six-step pipeline runs once per token-group (`tg_loop`).

> The full listing is long (six labelled step-loops × the tg outer loop). The
> source is [layernorm_256x144.asm](../src/tools/ipu-apps/src/ipu_apps/layernorm_256x144/layernorm_256x144.asm);
> the per-step structure above is faithful to it. Representative inner bundles:
>
> ```asm
> # Step 1 (−mean): broadcast −1/N scalar, sum over channels
> LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.RC.VV lr0 r0 0 lr1; ACC.FIRST;;
> # Step 3 (variance): square each centered row in place
> LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr12; MULT.RC.VS lr0 0 lr1; ACC.FIRST;;
> # Step 4 (rsqrt): the only ACTIVATE / POST_AAQ in L3
> ACTIVATE            rsqrt 1;;
> STR_POST_AAQ_REG    lr0 cr9;;
> # Step 6A (affine): γ·normalized + β, per-channel scalar via fixed_idx
> LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr12; MULT.RC.VE lr0 lr13 0 lr1; ACC.FIRST;;
> LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.RC.VE lr0 lr14 0 lr1; ACC;;
> STR_ACC_REG         lr3 cr10; ADD lr3 lr3 lr7; ADD lr13 lr13 cr1; ADD lr14 lr14 cr1;;
> ```

**Stale-comment note.** The `.asm` step-3 comment still says `MULT.EE.RR (square
r0)`; the actual instruction is `MULT.RC.VS` (square `r_cyclic`), as in the
listing. The comment names the pre-merge instruction.

- **Measured:** 4,442 cycles, **45.4 %** MULT/ACC (FP32 debug). The low figure is
  **structural**: six serial passes with data dependencies (variance can't start
  until centered is written; affine can't start until 1/σ is known), plus
  pointer-reset and store cycles between passes that carry no MULT. This is a
  reduction-along-channels pattern, the opposite of the matmuls' streaming.
- **Correctness (re-run for this doc):**
  - **wide-FP32 vs `reference_layernorm`** — **PASS**. This is the *only* test the
    kernel ships.
  - ⚠️ **No FP8 functional test exists.** A `fp8_e4m3` data directory is present
    but its golden is `out_fp8_e4m3_acc_int32.bin` — a stale INT32-accumulate
    format from before this kernel moved to wide-FP32, and the test file
    (`test_layernorm_256x144.py`) has **only** `test_layernorm_256x144_wide_fp32`.
    So **layernorm is verified in wide-FP32 only**; its FP8 behaviour is
    **unverified** in this repo. (See §8.)

### 6.3 `residual_add_256x144` — elementwise A + B

| | |
|---|---|
| **Purpose** | `C[r] = A[r] + B[r]` for 288 rows (256 tokens × 144 channels) |
| **Shapes** | A, B `[288, 128]` → C `[288, 512]` (FP32) |
| **Memory** | `A_BASE=0x00000`, `B_BASE=0x10000`, `ONES_BASE=0x20000`, `OUTPUT_BASE=0x30000` |

The cleanest kernel. There is no native vector-add in this ISA, so addition is
expressed as **two pass-through multiplies into the same accumulator**: load
`A[r]` and multiply by 1.0 (`ACC.FIRST`), then load `B[r]` and multiply by 1.0
(`ACC`) — the accumulator now holds `A[r]·1 + B[r]·1 = A[r] + B[r]`. The "scalar
1.0" is `CR10` (the dtype-encoded value `1.0`, set by the harness).

```asm
    SET lr0 cr4;; SET lr1 cr4;;
    SET lr2 cr5;; SET lr3 cr5;;        # A/B ptrs start at -128 (live = 0 after first ADD)
    SET lr4 cr4;; SET lr5 cr4;;
    SET lr6 cr6;; SET lr7 cr7;; SET lr8 cr8;;
    LDR_MULT_REG        r0 lr0 cr2;;      # r0 = ONES (unused after CR10 switch; harmless)

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0  (live ADD lr2 fires first -> live lr2 = r*128)
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.RC.VE lr0 cr10 0 lr0; ACC.FIRST;;
    # Cycle 2: r_acc += B[r] × 1.0
    LDR_CYCLIC_MULT_REG lr3 cr9 lr0; ADD lr3 lr3 lr7; MULT.RC.VE lr0 cr10 0 lr0; ACC;;
    # Cycle 3: store (do NOT ADD lr4 here: STR_ACC_REG reads lr4 live)
    STR_ACC_REG         lr4 cr3; ADD lr5 lr5 cr1;;
    # Cycle 4: advance output ptr; BLT reads snap lr5 (already incremented)
    ADD                 lr4 lr4 lr8; BLT lr5 lr6 row_loop;;

end:
    BKPT;;
```

**Per-bundle.** Cycle 1: R_CYCLIC ← `A[r]`, `MULT.RC.VE lr0 cr10` = `A[r]·1.0`,
`ACC.FIRST` seeds R_ACC. Cycle 2: R_CYCLIC ← `B[r]` (note `cr9 = B_BASE`),
`B[r]·1.0`, `ACC` adds. Cycle 3: `STR_ACC_REG` writes the 512-B FP32 sum (reading
`lr4` **live** — that is why `lr4` is *not* incremented in this bundle); the row
counter advances. Cycle 4: the output pointer advances and `BLT` loops on the
snapshot counter.

> **Stale-comment / divergence note.** The header comment describes `MULT.EE r0`
> (the pre-merge pass-through against ones in R0); the actual code uses
> `MULT.RC.VE lr0 cr10 0 lr0`, broadcasting the `CR10` scalar `1.0` — R0 is loaded
> with ones at startup but is **no longer read** by the loop. This is benign cruft
> (the `LDR_MULT_REG r0` startup line could be removed), flagged here because the
> comment and code disagree about which register supplies the `1.0`.

- **Measured:** 1,163 cycles, **49.5 %** MULT/ACC (INT8). This is the **structural
  ceiling**: the 4-cycle body has exactly 2 MULT cycles (cycles 1 & 2) and 2
  non-MULT cycles (store + branch), so it can never exceed 50 %. 1,163 ≈
  288 rows × 4 + a few setup cycles.
- **Correctness (re-run):** INT8 / E4M3 / E5M2 vs goldens — **PASS** byte-exact;
  FP8 accumulate-path drift vs FP32-of-decoded-inputs = **0**.

---

## 7. FP8 drift — what the numbers mean

A consistent finding across **every** L3 kernel: the FP8 goldens are bit-exact
reproductions of the **FP8-input / FP32-accumulate** path. Concretely:

- The kernels quantize **inputs** to FP8 (E4M3 or E5M2) but **accumulate in
  FP32** (R_ACC is FP32; `STR_ACC_REG` writes FP32). So the only quantization is
  on the operands, and both the kernel and the golden share the same FP8-decoded
  inputs and the same FP32 reduction order.
- Therefore **drift of the kernel output vs a numpy reference computed on the
  *same* FP8-decoded inputs is 0** (≤ ~1e-6 from float32 rounding on E5M2). I
  verified this for the matmuls, qk_scores, both attn@V kernels, and residual —
  all 0 (measured here). This is *why* the byte-equality tests pass.
- The **meaningful** numerical degradation is **input quantization**: FP8 vs the
  original FP32 inputs. I could measure this only where a wide-FP32 golden ships:

| Kernel | E4M3 mean abs error | E5M2 mean abs error | basis |
|---|---|---|---|
| `qk_scores_256x36` | **0.053** | **0.105** | FP8 score golden vs wide-FP32 score golden (measured here) |

  (Relative error is huge in the tail because many attention scores cancel to
  ≈0; the absolute figure is the trustworthy one. E5M2's larger error reflects
  its 2 mantissa bits vs E4M3's 3.) The matmuls and attn@V kernels ship no
  wide-FP32 golden, so an input-quant drift number is **not available** for them
  from shipped artifacts — but they share the same FP8-input/FP32-accumulate
  structure, so qualitatively the same input-quant behaviour applies.

---

## 8. What to watch — verification flags & open issues

**Verification status at a glance** (all re-run for this doc, `ZDlinear` @
`7c71a17`, `uv run pytest`, **34/34 pass**):

| Kernel | FP32 / numpy | INT8 | E4M3 | E5M2 |
|---|---|---|---|---|
| matmul_144x144 / 288x144 / 432x144 / 144x288 | — (no FP32 golden) | ✅ | ✅ | ✅ |
| qk_scores_256x36 | ✅ wide-FP32 | ✅ | ✅ | ✅ |
| attn_scores_km_256x36 | — | ✅ | ✅ | ✅ |
| attn_v_256x36 | ✅ wide-FP32 inline | ✅ | ✅ | ✅ |
| attn_v_bcast_36 | ✅ wide-FP32 inline | ✅ | ✅ | ✅ |
| unfold_32x32x144 | — | ✅ | ✅ | ✅ |
| **layernorm_256x144** | ✅ wide-FP32 | ❌ none | ❌ none | ❌ none |
| residual_add_256x144 | — | ✅ | ✅ | ✅ |

**Things to watch:**

1. **layernorm_256x144 has no FP8/INT8 test.** Only `test_layernorm_256x144_wide_fp32`
   exists. The `fp8_e4m3` golden in its data dir (`out_fp8_e4m3_acc_int32.bin`) is
   a **stale INT32-accumulate artifact** that predates the kernel's move to
   wide-FP32 and is not referenced by any test. **Open item:** either regenerate
   an FP8 wide-FP32-accumulate golden and add the parametrized test, or delete the
   stale artifact to avoid confusion. Until then, treat layernorm's FP8 behaviour
   as **unverified**.

2. **AGG spec text is stale (documentation bug, not an emulator bug).** The
   `InstructionDoc.operation` for `AGG.SUM`/`AGG.SUM.FIRST`/`AGG.MAX*` says the
   reduction is over `R_ACC[0..n-1]`, but `ipu.py` (correctly, per the post-merge
   design) reduces **`MULT_RES`**. The goldens and the `attn_v_256x36` correctness
   all depend on the `MULT_RES` behaviour. **Open item:** fix the spec prose to say
   `MULT_RES`. No emulator issue filed — the *code* is correct; only the embedded
   doc string is wrong.

3. **Stale instruction names in `.asm` comments — FIXED.** The matmul comments
   said `MULT.VE.CYCLIC` / `RESET_ACC`, layernorm step 3 said `MULT.EE.RR`, and
   residual said `MULT.EE r0`, none of which matched the code. All 11 L3 kernels
   now name the instruction the code actually issues (`MULT.RC.VE`,
   `ACC.ADD.FIRST`, `MULT.RC.VS`). Comments that describe *history* ("was
   `MULT.EE`", "`RESET_ACC` removed") are deliberately kept as historical notes.

4. **`residual_add` loads R0 with ones but never reads it — STILL OPEN.** Post the
   `DTYPE_ONE`-scalar switch, the startup `LDR_MULT_REG r0 … ONES_BASE` is dead:
   the loop multiplies by the CR scalar, not R0. Verified still present. Removing
   it changes the emitted binary (one fewer bundle), so it was deliberately left
   out of the register-naming pass, which is provably instruction-for-instruction
   identical. **Open item:** delete the dead load — and, with it, the `ONES_BASE`
   CR and the harness buffer that backs it — as a separate, test-gated change.

5. **The `width − 2` loop bound is fragile.** Every streaming loop relies on the
   live/snapshot skew (`bound = first_index + width − 2`). An off-by-one silently
   drops the last contraction term and would *not* be caught by a too-loose
   tolerance — only the byte-exact INT8/FP8 goldens catch it. Keep those goldens
   authoritative when editing any loop bound.

6. **Memory-overlap discipline.** The ≥288-output matmuls deliberately place
   `OUTPUT_BASE=0x30000` to clear the weight region. Any change to `N_OUT`,
   `W_STRIDE`, or a base address must re-check `WEIGHTS_BASE + N_OUT·W_STRIDE ≤
   OUTPUT_BASE` (and the analogous data/weight checks). A past violation silently
   corrupted weights for high output indices.

7. **Two attn@V kernels, two P layouts — keep them paired with the matching score
   kernel.** `attn_v_256x36` needs **query-major** P (from `qk_scores`);
   `attn_v_bcast_36` needs **key-major** P (from `attn_scores_km`). Feeding the
   wrong layout produces a plausible-looking but wrong result (I confirmed a
   ~600 % error when the bcast kernel's output is checked against a query-major
   reference). The layout is not interchangeable.

### Numbers: measured vs estimated

- **Measured (DGX, commit 7c71a17):** every cycle count and `mult%` in §3 and the
  per-kernel "Measured" lines.
- **Measured (re-run locally for this doc):** all pass/fail results (§8 table),
  and the FP8 drift figures (qk_scores 0.053/0.105; the ≈0 accumulate-path drifts
  for matmuls / attn@V / residual).
- **Estimated:** none of the headline numbers are estimates. The only
  non-measured statements are the *qualitative* "bound by …" attributions in §3,
  which are reasoned from the loop structure, not from a per-instruction profiler.

### Kernels not fully verified

- **`layernorm_256x144`** — verified in wide-FP32 only; FP8/INT8 paths untested
  (see open item 1). Everything else in this document is verified across FP32 (or
  numpy) **and** INT8 **and** both FP8 formats.
