# ISA alias catalogue — where `mult%` is not MAC utilization

Audit of `pr-linear-kernels`, `pr-projections`, `pr-attention`, and
`pr-kernel-registry-docs`.

The IPU can retire 128 MACs/cycle, and `mult%` from `RunStats` is the number we
steer by. It is not a MAC-utilization figure. Three separate mechanisms push it
away from the truth, and they push in both directions:

| # | Mechanism | Effect on `mult%` | Effect on real MACs |
|---|---|---|---|
| 1 | **Fake multiplies** — `×1.0` used to move data | inflates | zero |
| 2 | **Dead lanes** — a 16-token row still bills 128 lanes | inflates | up to 8× overstated |
| 3 | **Addressing bubbles** — mult slot idle in LR-only bundles | deflates | real work, hidden cost |

Mechanism 1 exists because **every `ACC.*` and `AGG.*` op consumes `MULT_RES`**.
There is no path into the accumulator that does not cross the multiplier, and the
mult slot has no move, add, subtract, or pass-through form. So a multiply against
a constant `1.0` is the universal stand-in — and `mult_active_cycles` bills it as
a full active cycle.

The constant's interpretation depends on the arithmetic mode:

- `CR0 ≡ 0`, `CR1 ≡ 1` as raw register values, both read-only; attempts to
  write them raise `EmulatorError`. CR1 supplies numeric one in INT8 and wide
  vector modes. In narrow FP8, the multiplier decodes that byte using the active
  dtype, so CR1 is **not** floating-point one. Put `dtype_one_byte(dtype)` in a
  writable CR and use that register for FP8 identity multiplies.
- Packed and `residual_add` kernels instead load `1.0` into **`CR10`** —
  `residual_add_256x144/__init__.py:94` is a literal `set_cr(10, 1)`, aliased in
  the assembly as `DTYPE_ONE`.

---

## The catalogue

Each entry: the operation that does not exist, the real instruction sequence
standing in for it, and what that does to the counters.

### A1 — `MOV.RC` — move a vector into the accumulator

The root gap; every other fake-multiply entry is a special case of it.

```asm
MULT.RC.VE  rc, CR1, 0, lr, CR15 ;   # x 1.0  <-- pure routing, 0 MACs
ACC.ADD.FIRST ;;
```

Variants: `MULT.RC.VV` against a resident all-`1.0` row (layernorm, unfold);
`MULT.EE ra, CR1` for the single-element form.

**Sites** — softmax pass 3 in all five kernels; every `unfold_*`; `residual_add_*`.
**Distortion** — `mult_active_cycles` +1 per move, 100% fake. `acc_active_cycles`
rises 1:1.

### A2 — `ADD.VV` / `ADD.VS` — vector add

Two pass-through multiplies into one accumulator.

```asm
MULT.RC.VE rc(A), CR10, ... ; ACC.ADD.FIRST ;;   # A x 1.0
MULT.RC.VE rc(B), CR10, ... ; ACC.ADD ;;         # B x 1.0
```

**Admitted verbatim**, `residual_add_256x144.asm:17-19`:

> There is no add instruction in the vector path, so each addend is passed
> through the multiplier against a constant 1.0 and summed in the accumulator.

and `L3_kernel_reference.md:1130-1134`:

> There is no native vector-add in this ISA, so addition is expressed as two
> pass-through multiplies into the same accumulator.

**Sites** — all `residual_add_*` (unpacked and packed), layernorm step 2 and the
`+beta` step, `asm_packed_layernorm_*:310,312,342,344,549,581`.
**Distortion** — `residual_add_*` kernels are **100% fake mult**: 576 mult cycles
at a reported 49.5% utilization, doing zero MACs.

### A3 — `SUB.VV` / `NEG` — vector subtract

`ACC.SUB` exists, but its operand must still be staged through the multiplier.

```asm
MULT.EE max_idx, CR1, 0, lr, CR15 ;   # broadcast x 1.0
ACC.SUB ;;                            # softmax's x - max
```

**Sites** — `softmax_rows.asm:103`, `_long:123`, `_partial:122`,
`softmax_columns.asm:105`, `columns_packed:130`.
**Evidence of an earlier alias** — `softmax_rows.asm:35` still records
`CR11 = free (was -1.0; ACC.SUB + CR1 replaced it)`: multiply-by-−1 was the
previous workaround for the same gap.

> **`softmax.md:53-55` is actively misleading.** It says the subtract is
> "free here — `ACC.SUB` against CR1 does it in the accumulate slot." It is not
> free: the subtrahend must first be pushed through the multiplier by a `×1.0`.
> This is the clearest case of the workaround being invisible to whoever reads
> the utilization number.

### A4 — `AGG.SUM.RC` / `AGG.MAX.RC` — reduce a register directly

A reduction cannot read a vector register; it only reduces `MULT_RES`.

```asm
MULT.RC.VE lr_cyc, CR1, 0, lr_cyc, CR15 ;   # num x 1.0  <-- FAKE
AGG.SUM.FIRST lr_row, CR15 ;;
```

**Sites** — `softmax_rows.asm:123`; `columns.asm:132,138`;
`rows_long.asm:149,157,165`; `rows_partial.asm:145`; `columns_packed.asm:148,154`.
Source comments admit it inline as `{#- num*1.0 -#}`.

### A5 — `BCAST` / `SPLAT` — broadcast one element to 128 lanes

`MULT.EE` *is* the only broadcast path, so a pure splat costs a multiply.
The windowed form is worse:
`MULT.RC.VE rc_idx=(-ps*p) % 512, CR1, mask_offset=p`, repeated 8×
(`asm_packed_layernorm_240x16.asm:287,445`).

### A6 — `GATHER` / `SCATTER` / `PERMUTE` / `ROTATE`

A masked `×1.0` where `rc_idx` picks the read window and `mask_offset` picks the
write window. **Admitted**, `asm_packed_layernorm_240x16.asm:16-20`:

> masked `MULT.RC.VE` gather/scatter construction (rc_idx picks the READ window,
> mask_offset picks the WRITE window).

**Sites** — all `asm_packed_pack_*` / `unpack_*`, `softmax_rows_partial.asm:85,180,185`,
`columns_packed.asm:102,108,168,174`.
**Distortion** — the pack/unpack kernels are **100% fake mult**, 0 MACs. Cross-lane
rotate has no primitive at all: `columns_packed` stores to XMEM and reloads into two
ring slots because the ring will not wrap at 128.

### A7 — `REDUCE.SEG` — segmented (partition-wise) reduce

Every reduce collapses to a *single* scalar. Summing 8 partitions of 16 lanes
requires the hand-built "primitive A": store `r_acc` → XMEM → reload into
`R_CYCLIC` → 8 × (`MULT.RC.VE ×1.0` + `ACC.ADD`).

Documented at length in `docs/isa_friction_log.md` (see *Cross-references*),
measured at 23 cycles / 22 instructions standalone, ~18–19 embedded. At
out-proj's 240 output channels that is **~4,560 of 27,606 cycles (~16.5%)**.
`CR15.partition` already encodes P8 = 8 groups of 16, but it only feeds
mask-shift math — it does not move or reduce data.

### A8 — `MOV.ACC → R0/R1/R_CYCLIC`

`R_ACC` has no path back to the multiply-stage inputs. Multi-pass kernels drain it
via `ACTIVATE.QUANTIZE identity` + `STR_POST_AAQ_REG` + `LDR_*` — a register move
executed through external memory.

**Distortion** — `xmem_reads` / `xmem_writes`, entirely non-algorithmic.
`layernorm_16x240` alone: **~+723 reads, ~+483 writes**. Roughly 30 sites across the
packed kernels. `ACTIVATE.QUANTIZE identity` is itself a move instruction in disguise.

### A9 — `EXP`

The activation unit provides `exp2`, not `exp` (`softmax.md:40-52`). Softmax carries
a resident `log2(e)` vector and spends one real multiply per element on the base
change. Full activation set (`activations.py:34-47`): identity, relu, relu6, sigmoid,
tanh, gelu, softplus, elu, **exp2**, **reciprocal**, **rsqrt**, silu. No `exp`, `log`,
`sqrt`, `abs`, `sign`, `clamp`, `pow`.

Because `reciprocal` and `rsqrt` *do* exist, there are no Newton-Raphson or LUT
divide chains anywhere — a useful negative result.

### A10 — fractional scalar in a CR

In INT8 and wide vector modes, `MULT.RC.VE` interprets a CR scalar's low byte
as a signed integer (`asm_packed_layernorm_240x16.asm:26-30`), so fractional
constants cannot use that scalar path. Wide FP32 kernels instead materialise
them as vector data. Narrow FP8 can use fractional CR scalars directly by
placing the active dtype's FP8 encoding in the low byte.

### A11 — `ACC.MUL` — scale the accumulator by `MULT_RES`

`ACC.ADD` always *adds* `MULT_RES` into `R_ACC`; there is no elementwise-multiply
accumulate, so `R_ACC[i] *= MULT_RES[i]` cannot be expressed. Admitted at
`asm_packed_layernorm_240x16.asm:505-513` — "which is A+B, not A*B" — where the gap
**caused a real bug**. There is no in-register alias at all: the only way through is
to drain `R_ACC` to XMEM (A8) and re-multiply on the way back in.

### A12 — post-increment addressing / same-cycle load→mult forwarding

The largest *absolute* waste found, and it is invisible as a fake multiply because
it shows up as mult-slot **idle**. The snapshot contract (issue #157) forces an
extra LR-only bundle per inner-loop iteration:

```asm
s_loop_pre:
    ADD key_index, key_index, CR1 ;;   # entire bundle, mult slot NOP
```

**Cost** — `attn_v_bcast_36`: **73,440** bubble cycles; `attn_v_bcast_48`: 48,384;
`qk_scores_256x36` and `attn_scores_km_256x36`: 17,920 each. It caps
`matmul_128x128` at 50% (32,768 cycles for 16,384 MULTs). Present in 9 of 12
attention kernels.

`attn_v_256x36` is the one genuinely clean kernel at **~96%** — the `MULT.RC.VV`
form avoids the bubble because `AGG`'s `dest_slot` reads the snapshot.

### A13 — narrow / `valid_elements`-costed MULT

Not a fake multiply — a fully real one that is only fractionally useful. The
projection harnesses zero-pad every row to 128 lanes: `N_TOK` is 128 for the
144-family, **64** for the 192-family, **16** for the 240-family. `CR15` is never
written, so `valid_elements` stays at its default 128 and all 128 lanes really are
multiplied.

This is the single largest distortion in the repo: **2.2 M of 4.35 M mult cycles**
across the projections family are spent on padding.

### A14 — multiple accumulators / `MULT.OUTER`

Only one `R_ACC` exists, so only one output channel can be in flight. Each
projection kernel re-streams the entire activation block once per output channel.
**`proj_qkv_240_p4` performs 691,200 XMEM reads where 960 would suffice — 720×
amplification.**

### A15 — `SELECT` / `MASK.VV`, `CSEL`, and scalar `MUL` in the LR slot

Three smaller gaps. Predication is emulated multiplicatively — `columns_packed:46-49`
folds a "KEEP-mask (1.0 real / 0.0 pad)" in via `MULT.RC.VV`. Branch diamonds stand
in for a conditional move (`proj_ffn2_240_p4.asm:169-181`, ~3.7% of that kernel).
And `softmax_rows_partial/STATUS.md:98` notes "**The LR slot has no multiply**, so
instead of computing `cbound*P`…".

### A16 — `STR_ACC_REG` is simulation-only

Flagged `"hardware": False` in `instruction_spec.py` — it has no hardware encoding
(`L3_kernel_reference.md:133-140`). **One live use remains:
`fully_connected/fully_connected.asm:48`.** Cycle counts from that kernel are not
achievable on silicon.

---

## Generic VLIW forms

Every alias above, spelled out as bundles. These are shape-independent templates —
substitute your own registers.

**Bundle syntax.** `;` separates slot instructions, `;;` ends the bundle and the
cycle. Slot is inferred from the mnemonic, so textual order within a bundle is free.
The LR slot has three sub-slots per cycle.

!!! warning "Commas are documentation style, not assembler syntax"

    Operands are written comma-separated here per the project's documentation
    convention, but **the assembler's grammar rejects commas** — it expects
    whitespace-separated operands. Strip the commas before assembling.

    This affects more than this page: every `example=` string in
    `instruction_spec.py` is comma-separated, so the examples in the generated
    instruction reference do not assemble as written either. Worth reconciling —
    either the grammar should accept an optional comma, or the convention should
    change to match the grammar.

Every bundle below was validated by assembling it (commas stripped) with
`//src/tools/ipu-as-py:ipu-as`.

**Register convention used below:**

| Register | Holds |
|---|---|
| `LR0` | constant `0` — neutral `rc_idx` base and neutral `mask_shift` |
| `LR1` | element index (scalar-select for `MULT.RC.VE` / `MULT.EE`) |
| `LR2` | XMEM pointer |
| `LR3` | `AGG` destination slot |
| `CR0` | hardwired `0` — also the XMEM base for ring loads |
| `CR1` | hardwired raw `1` — numeric one in INT8 and wide modes (`CR10` in packed kernels) |
| `CR2` | data base address |
| `CR15` | dstructure: `valid_elements`, `partition`, `pad_mode` |

The identity examples using CR1 assume INT8 or wide vector arithmetic. For
narrow FP8, substitute a writable CR containing `dtype_one_byte(dtype)`.
Fractional vector examples assume a floating-point data mode.

**Snapshot contract (issue #157).** `MULT.RC.*` reads `R_CYCLIC` from the
start-of-cycle snapshot, so it cannot consume a row loaded in its own bundle. Every
load below therefore sits one bundle ahead of the multiply that uses it. Co-issuing
a load and an unrelated multiply is fine.

Cost lines below count *fake* mult cycles — cycles billed to `mult_active_cycles`
that retire zero useful MACs.

---

### A1 — `MOV.RC`: move a vector into `R_ACC`

```asm
# Form 1 - ring vector x CR one (the dominant idiom)
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # ring <- src, one bundle ahead
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- 0 MACs
    ACC.ADD.FIRST ;;                            # ACC.ADD to accumulate instead

# Form 2 - Ra vector x CR one (source already in R0/R1)
    MULT.VE LR0, CR1, 0, LR0, CR15 ;
    ACC.ADD.FIRST ;;

# Form 3 - ring x a resident all-ones row in R0 (used by layernorm, unfold)
    LDR_MULT_REG R0, LR2, CR2 ;;                # R0 <- 128 x 1.0, hoisted once
    MULT.RC.VV LR0, R0, 0, LR0, CR15 ;
    ACC.ADD.FIRST ;;

# Form 4 - single element, broadcast to all 128 lanes
    MULT.EE LR1, CR1, 0, LR0, CR15 ;
    ACC.ADD.FIRST ;;
```

**Cost** 1 fake mult + 1 acc per move. Form 3 additionally burns a `R0`/`R1` slot
and one XMEM row on the ones vector.
**Wanted** `ACC.ADD.RC rc_idx, cr_idx` — let the acc slot read a vector register.

### A2 — `ADD.VV` / `ADD.VS`: vector add

```asm
# Form 1 - vector + vector (residual add). Two passes, one accumulator.
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # ring <- A
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # A x 1.0   <-- FAKE
    ACC.ADD.FIRST ;
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # ring <- B, co-issued
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # B x 1.0   <-- FAKE
    ACC.ADD ;;

# Form 2 - vector + scalar (layernorm's + beta)
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # vector x 1.0  <-- FAKE
    ACC.ADD.FIRST ;;
    MULT.EE LR1, CR1, 0, LR0, CR15 ;            # scalar broadcast x 1.0  <-- FAKE
    ACC.ADD ;;

# Form 3 - Ra-side, when both addends are already resident
    MULT.VE LR0, CR1, 0, LR0, CR15 ; ACC.ADD.FIRST ;;
    MULT.VE LR1, CR1, 0, LR0, CR15 ; ACC.ADD ;;
```

**Cost** 2 fake mult + 2 acc per add. 100% of the mult cycles in every
`residual_add_*` kernel.
**Wanted** `ACC.ADD.RC` (A1) covers this — the second addend needs no multiplier.

### A3 — `SUB.VV` / `SUB.VS` / `NEG`: vector subtract

```asm
# Form 1 - current idiom: stage the subtrahend, subtract in the acc slot
    MULT.EE LR1, CR1, 0, LR0, CR15 ;            # broadcast max x 1.0  <-- FAKE
    ACC.SUB ;;                                  # softmax's x - max

# Form 2 - vector subtrahend
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.SUB ;;

# Form 3 - negate only (no prior accumulator content)
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.SUB.FIRST ;;                            # R_ACC = -MULT_RES

# Form 4 - HISTORICAL, superseded: multiply by a CR holding -1
    MULT.RC.VE LR0, CR11, 0, LR0, CR15 ;        # CR11 = -1.0
    ACC.ADD ;;
```

Form 4 is the prior generation of this alias — `softmax_rows.asm:35` still carries
`CR11 = free (was -1.0; ACC.SUB + CR1 replaced it)`. Both forms cost the same fake
multiply; only the sign moved from the multiplier to the acc slot.

**Cost** 1 fake mult + 1 acc per subtract.

### A4 — `AGG.SUM.RC` / `AGG.MAX.RC`: reduce a register directly

```asm
# Form 1 - sum-reduce a ring vector to one R_ACC slot
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    AGG.SUM.FIRST LR3, CR15 ;;                  # AGG.SUM to accumulate across cycles

# Form 2 - max-reduce
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    AGG.MAX.FIRST LR3, CR15 ;;

# Form 3 - elementwise (non-collapsing) running max
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.MAX ;;
```

`AGG.*` collapses lanes `0 .. valid_elements-1` to the single slot `LR[LR3] % 128`.
**Cost** 1 fake mult per reduced vector.
**Wanted** `AGG.SUM.RC rc_idx, dest_slot, cr_idx`.

### A5 — `BCAST` / `SPLAT`: replicate one value across lanes

```asm
# Form 1 - one Ra element -> all 128 lanes. MULT.EE is the ONLY broadcast path.
    MULT.EE LR1, CR1, 0, LR0, CR15 ;            # x 1.0  <-- FAKE
    ACC.ADD.FIRST ;;

# Form 2 - windowed splat: one partition-sized window -> every partition.
# Repeat per partition p, with rc_idx = (-partition_size * p) mod 512
# selecting the READ window and mask_offset = p selecting the WRITE window.
    SET LR4, CR3 ;;                             # CR3 = (-partition_size * p) mod 512
    MULT.RC.VE LR4, CR1, 1, LR0, CR15 ;         # mask_offset = p = 1  <-- FAKE
    ACC.ADD ;;
    # ... unrolled for p = 0 .. partitions-1

# Form 3 - ACC.RESHAPE, limited to 8 lanes per call and still mult-gated
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.RESHAPE LRD0, LRD2, 0 ;;                # MULT_RES[src[i]] -> R_ACC[dst[i]]
```

Form 3 reads *8 byte-elements* of an `LRDn` pair as source and destination lane
indices, so replicating one scalar across a 16-lane partition would take two calls
and still needs the fake multiply to populate `MULT_RES`.

**Cost** Form 1: 1 fake mult. Form 2: one fake mult *per partition* (8 for P8).

### A6 — `GATHER` / `SCATTER` / `PERMUTE` / `ROTATE`

```asm
# Form 1 - gather/scatter: rc_idx picks the READ window, mask_offset the WRITE window
    MULT.RC.VE LR4, CR1, 3, LR0, CR15 ;         # x 1.0, write window 3  <-- FAKE
    ACC.ADD ;;

# Form 2 - cross-lane rotate. The ring does NOT wrap at 128, so the row must be
# materialised twice, back to back, and read at a shifted offset.
    ACTIVATE.QUANTIZE identity, CR15 ;
    STR_POST_AAQ_REG LR2, CR2 ;;                # drain R_ACC to scratch
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # reload copy 1 at ring slot 0
    LDR_CYCLIC_MULT_REG LR2, CR0, LR5 ;;        # reload copy 2 at ring slot 128
    MULT.RC.VE LR6, CR1, 0, LR0, CR15 ;         # LR6 = rotate amount  <-- FAKE
    ACC.MAX ;;

# Form 3 - ACC.RESHAPE: an arbitrary 8-lane permutation, MULT_RES -> R_ACC
    MULT.RC.VE LR0, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.RESHAPE LRD0, LRD2, 0 ;;
```

**Cost** Form 1: 1 fake mult per window — the pack/unpack kernels are 100% this.
Form 2: 1 fake mult per rotate step, plus 1 fake XMEM write and 2 fake reads.
**Wanted** `PERMUTE.RC` operating on a full 128 lanes without a multiplier.

### A7 — `REDUCE.SEG`: segmented reduce ("primitive A")

Every `AGG` collapses to *one* scalar, so N partial sums require N passes:

```asm
# Drain the 128 lanes holding N partitions of partition_size
    ACTIVATE.QUANTIZE identity, CR15 ;
    STR_POST_AAQ_REG LR2, CR2 ;;
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # reload into the ring

# Then one pass per partition p, at rc_idx = partition_size * p
    SET LR4, CR3 ;;                             # CR3 = partition_size * 0
    MULT.RC.VE LR4, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.ADD.FIRST ;;
    SET LR4, CR4 ;;                             # CR4 = partition_size * 1
    MULT.RC.VE LR4, CR1, 0, LR0, CR15 ;         # x 1.0  <-- FAKE
    ACC.ADD ;;
    # ... repeated to p = N-1
```

**Cost** measured at 23 cycles / 22 instructions for the 8-partition case, ~18–19
embedded — of which N are fake multiplies, plus the store/reload pair.
**Wanted** an `AGG` variant driven by `CR15.partition` writing `128/N` partial sums
instead of collapsing to one. That field already encodes P8; it just does not reduce.

### A8 — `MOV.ACC`: get `R_ACC` back to the multiply stage

There is no direct path. The only route is through external memory:

```asm
# Drain
    ACTIVATE.QUANTIZE identity, CR15 ;          # 'identity' = this is a move
    STR_POST_AAQ_REG LR2, CR2 ;;

# Re-enter, either side
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # -> R_CYCLIC
    LDR_MULT_REG R0, LR2, CR2 ;;                # -> R0 / R1
```

**Cost** 0 fake mult, but +1 `xmem_write` and +1 `xmem_read` per move, plus the
round-trip latency. `layernorm_16x240` spends ~723 reads and ~483 writes here.
Note this also **quantizes to INT8 on the way out**, so it is lossy as a register move.

### A9 — `EXP`: only `exp2` exists

```asm
# Rebase by log2(e), then use the native exp2
    LDR_MULT_REG R0, LR2, CR2 ;;                # R0 <- 128 x log2(e), hoisted once
    MULT.RC.VV LR0, R0, 0, LR0, CR15 ;          # REAL multiply, genuinely needed
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE exp2, CR15 ;
    STR_POST_AAQ_REG LR2, CR2 ;;
```

**Cost** 1 *real* mult per element vector, plus one resident vector register and one
XMEM row for the constant. Not a fake multiply — but it is work a native `exp` would
not need. Full activation set: `identity`, `relu`, `relu6`, `sigmoid`, `tanh`,
`gelu`, `softplus`, `elu`, `exp2`, `reciprocal`, `rsqrt`, `silu`.

### A10 — fractional scalar in a CR

```asm
# Wide FP32: a CR scalar is still a signed integer byte; it cannot encode 0.5
# (CRs store integers, so assigning a Python float is not a supported setup.)

# Alias - materialise the constant as a full 128-lane row
    LDR_MULT_REG R0, LR2, CR2 ;;                # R0 <- 128 x 0.5
    MULT.RC.VV LR0, R0, 0, LR0, CR15 ;
    ACC.ADD.FIRST ;;
```

**Cost** one XMEM row and one `R0`/`R1` slot per distinct fractional constant.
This workaround is for wide FP32. In narrow FP8, store the encoded byte for
0.5 in CR3 and use `MULT.RC.VE LR0, CR3, 0, LR0, CR15` directly. In INT8,
neither a scalar byte nor a vector byte directly represents a fraction.

### A11 — `ACC.MUL`: scale the accumulator

`R_ACC[i] *= MULT_RES[i]` has no encoding, and no in-register alias. The only route
is a full A8 round-trip:

```asm
    ACTIVATE.QUANTIZE identity, CR15 ;
    STR_POST_AAQ_REG LR2, CR2 ;;                # drain R_ACC
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;;        # reload as ring data
    MULT.RC.VV LR0, R0, 0, LR0, CR15 ;          # now multiply against R0
    ACC.ADD.FIRST ;;
```

**Cost** +1 XMEM write, +1 read, +2 cycles, and an INT8 quantization round-trip.
Writing `ACC.ADD` here instead computes `A+B` — the bug at
`asm_packed_layernorm_240x16.asm:505-513`.

### A12 — post-increment addressing

```asm
# Current: the pointer bump needs its own bundle, mult slot idle
loop_pre:
    ADD LR1, LR1, CR1 ;;                        # <-- ENTIRE BUNDLE, 0 MACs
loop:
    MULT.RC.VE LR0, LR1, 0, LR0, CR15 ;         # real MAC
    ACC.ADD ;
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;
    ADD LR2, LR2, CR4 ;
    BLT LR1, LR5, loop_pre ;;

# Wanted: post-increment folded into the operand, halving the loop
loop:
    MULT.RC.VE LR0, LR1++, 0, LR0, CR15 ;
    ACC.ADD ;
    LDR_CYCLIC_MULT_REG LR2++, CR0, LR0 ;
    BLT LR1, LR5, loop ;;
```

**Cost** 1 idle mult cycle per iteration — 73,440 in `attn_v_bcast_36`. This one
*lowers* `mult%` while wasting time, so it hides from the current counter entirely.
`attn_v_256x36` escapes it: `AGG`'s `dest_slot` reads the snapshot, so no
pre-increment bundle is needed.

### A13 — narrow / lane-costed MULT

```asm
# Same bundle bills 128 lanes whether 16 are live or 128
    MULT.RC.VE LR0, LR1, 0, LR0, CR15 ;         # CR15.valid_elements = 128 (default)
    ACC.ADD ;;

# Wanted: cost follows the data
    MULT.RC.VE.N LR0, LR1, 0, LR0, CR15, 16 ;   # or: bill min(valid_elements, 128)
    ACC.ADD ;;
```

`valid_elements` does not gate multiply results: `_mult_mask_and_shift` gates
them with R_MASK and the partition-dependent shift. The new lane counter also
intersects that mask with lanes `0 .. valid_elements-1`, using the multiply's
dstructure CR (conventionally CR15), for statistics only. A zero value falls
back to the full mask as an undeclared width. AGG operations separately use
`valid_elements` to bound their reduction. Declaring a narrower width therefore
changes the multiply statistic, not execution or issue cost.

**Cost** 8× overstatement at `N_TOK=16`, 2× at `N_TOK=64`.

### A14 — multiple accumulators / `MULT.OUTER`

```asm
# Current: one R_ACC, so the whole activation block re-streams per output channel
j_loop:
    SET LR2, CR2 ;;                             # rewind data pointer EVERY j
k_loop:
    MULT.RC.VE LR0, LR1, 0, LR0, CR15 ;
    ACC.ADD ;
    LDR_CYCLIC_MULT_REG LR2, CR0, LR0 ;         # re-read, N_OUT times over
    ...

# Wanted: several accumulators in flight, data read once
    MULT.RC.VE LR0, LR1, 0, LR0, CR15 ;
    ACC.ADD 0 ;;                                # accumulator index
```

**Cost** `xmem_reads` inflated by `N_OUT`× — 691,200 where 960 would do in
`proj_qkv_240_p4`.

### A15 — `SELECT`, `CSEL`, and scalar multiply in the LR slot

```asm
# Missing SELECT/MASK.VV - predication folded in multiplicatively
    LDR_MULT_REG R0, LR2, CR2 ;;                # R0 <- KEEP mask, 1.0 real / 0.0 pad
    MULT.RC.VV LR0, R0, 0, LR0, CR15 ;          # gate by multiplying
    ACC.ADD ;;

# Missing CSEL - a branch diamond stands in for a conditional move
    BLT LR1, LR5, use_full ;;
    SET LR6, CR8 ;;                             # tail bound
    B joined ;;
use_full:
    SET LR6, CR7 ;;                             # full bound
joined:

# Missing scalar MUL in the LR slot - repeated ADD instead of cbound * P
    ADD LR6, LR6, LR7 ;;
    ADD LR6, LR6, LR7 ;;
```

**Cost** the branch diamond costs 6–7 mult-idle bundles per taken path
(~3.7% of `proj_ffn2_240_p4`); it *deflates* `mult%`.

### A16 — `STR_ACC_REG` is simulation-only

```asm
# Simulation only - "hardware": False, no encoding on silicon
    STR_ACC_REG LR2, CR2 ;;

# Real-hardware equivalent
    ACTIVATE.QUANTIZE identity, CR15 ;
    STR_POST_AAQ_REG LR2, CR2 ;;
```

The real form quantizes to INT8; `STR_ACC_REG` writes raw accumulator words. Any
kernel relying on it is not measuring achievable cycles —
`fully_connected/fully_connected.asm:48` is the one live use.

---

## Scoreboard

Reported `mult%` versus theoretical useful-work estimates from the audit.
These estimates are not necessarily `RunStats.effective_mac_utilization`: the
runtime metric subtracts detected CR-sourced and explicitly declared vector
identity multiplies and uses declared lane widths. It does not infer constant
intent from ordinary data values.
✅ = executed end-to-end; other rows are closed-form audit models.

| Kernel | Total cycles | `mult%` reported | **Audit useful-work estimate** | Cause |
|---|---:|---:|---:|---|
| `proj_qkv_240_p4` | 748,832 | 92.3% | **11.5%** | A13 |
| `proj_ffn1_240_p4` | 499,232 | 92.3% | **11.5%** | A13 |
| `proj_outproj_240_p4` ✅ | 249,632 | 92.3% | **11.5%** | A13 |
| `proj_qkv_192_p4` | 488,480 | 90.6% | **45.3%** | A13 |
| `proj_*_144_p4` (×4) | 1,507,544 | 87–91% | 87–91% | clean |
| **projections total** | **4,788,696** | **90.8%** | **44.8%** | 2.2 M wasted |
| `unfold_32x32x144` | 6,338 | 72.7% | **0%** | A1, 100% fake |
| `residual_add_256x144` | 1,163 | 49.5% | **0%** | A2, 100% fake |
| `asm_packed_pack/unpack_240x16` | ~930 ea | ~26% | **0%** | A6, 100% fake |
| `softmax_rows_partial` n=16 | 9,435 | 27.1% | **~2.0%** | A1–A6 + lanes, **13×** |
| `layernorm_16x240` | ~3,365 | ~50% | **3.6%** | A2 + 16/128 lanes |
| `attn_v_bcast_60` | — | ~41% | **~5%** | A12 + lanes |
| `attn_v_bcast_36` | — | ~50% | ~49% | A12 bubbles |
| `attn_v_256x36` | — | ~96% | **~96%** | clean |
| `matmul_128x128` | 32,768 | 50% | 50% | A12 |

Softmax per-row MULT counts reproduce each kernel's committed
`benchmark/results.md` `mult%` to within 0.1%.

Four default softmax cases have 40% CR-sourced identity multiplies. The packed
column case (64 rows, width 16) has 32 identities out of 57 active multiplies,
or 56.1%: its cross-group max and sum folds add identity work. A current
`softmax_rows` default run (128 rows) confirms
640 active multiplies, 256 identities (40%), 81,920 lane ops, and 32,768 identity
lane ops over 2,323 cycles. Its **effective MAC utilization is 16.5%**:
`(81920 - 32768) / (2323 * 128)`. Its existing mult utilization remains 27.6%.
The remaining multiplications include base conversion and normalization work;
the runtime counter includes them even though they are not a matrix-product
MAC workload. Neither 0% nor the earlier 5.5% estimate describes this counter.

---

## Why `acc%` always equals `mult%`

Every `results.md` row shows `mult% == acc%` to the decimal. That is not a
coincidence and not a healthy sign: since `STR_ACC_REG` was dropped, every store
needs `R_ACC` freshly seeded, so kernels carry a bare `acc.add.first` purely to move
`MULT_RES` into `R_ACC` for `ACTIVATE.QUANTIZE identity`
(`softmax_rows.asm:143-144`). `acc_active_cycles` is inflated 1:1 with the fake
multiplies and carries no independent information today.

---

## Measuring it — implemented

The emulator now counts these idioms directly. `RunStats` carries:

1. **`mult_lane_ops`** — lanes that both passed the multiply mask and lay within
   the MULT instruction's dstructure descriptor `valid_elements`, summed over
   every multiply. The descriptor is conventionally in CR15, but kernels can
   select another CR. Addresses A13.
2. **`mult_identity_cycles`** — multiplies whose multiplicand was a constant
   `1.0` from a CR or a declared ONES vector, with **`mult_identity_lane_ops`** as the lane-weighted
   counterpart. Flags A1–A6.
3. **`mult_idle_cycles`** — incremented when the mult slot issues NOP or an empty
   instruction executes. It is current at debugger stops, before the runner
   assigns `total_cycles`, and equals `total_cycles − mult_active_cycles` after
   a completed run. Surfaces A12, otherwise invisible because it *lowers* `mult%`.

Headline number: **`effective_mac_utilization`** =
`(mult_lane_ops − mult_identity_lane_ops) / (total_cycles × 128)`.

Two boundaries are deliberate. A CR supplying 1.0 counts as an identity scalar;
an LR selecting an ordinary data element equal to one does not. For constant
vectors, a harness writes its intentional ONES row with
`state.write_constant_ones(xmem_addr)`. This writes the active dtype's encoding,
and register loads carry its provenance into cycle-start snapshots. Overwrites
revoke provenance even when the replacement bytes also contain ones. Ordinary
data or weights written through the usual memory API are never promoted to
constants by value equality alone. `MULT.RC.VV` and `MULT.RC.VE` recognize declared
ONES vectors; this covers unfold routing and layernorm's centering/beta addends.
Mutable register handles (`raw` or writable word views) revoke and block
provenance for that buffer because retained handles can bypass setters. Use
copying getters or `raw_readonly` for inspection that should preserve provenance.
For MULT,
`valid_elements` is read for statistics only; it never changes the multiplication.
A kernel that pads without
declaring a narrower row still reports 128 lanes, so the run summary prints a
warning whenever every multiply used all 128 lanes.

### Declaring aliases

The idioms above are declared in `ipu_common/isa_alias_spec.py` as a table
validated at import, and each run reports per-alias hit counts. Entries are
ordered most-specific-first with first-match-wins, so A1–A6 hits partition
`mult_identity_cycles` rather than overlapping. A16 independently counts the
simulation-only accumulator store. Because a bundle is dispatched as
a unit, a constraint can name a **co-issued slot** — which is the only thing
separating A1–A4 and A6, since all of them use an identical `MULT.RC.VE` with a
1.0 scalar or a declared ONES vector. A1–A4 differ in the acc op beside it.
A6 takes precedence for nonzero mask slots, mask-zero windows narrower than
the declared destination span, and declared-vector routing through ACC.STRIDE.
Thus packed pack/unpack count every window as A6, including slot zero. A mask
matching a narrower `valid_elements` declaration is ordinary padding and does
not by itself trigger A6. A completely unmasked move remains indistinguishable
from routing with identical operands, so algorithmic intent is not inferred.

Scalar constraints must name a supported scalar multiplicand. Each match may
contain at most one immediate constraint and one co-slot constraint;
`hardware_only` must stand alone. Duplicate match signatures are rejected.

## Ranked ISA additions, by cycles recovered

1. **Narrow/`valid_elements`-costed MULT** (A13) — up to 8× on the 240 family.
2. **`ACC.ADD.RC` / `AGG.SUM.RC` / `MOV.RC`** (A1, A4) — retires every `×1.0` in
   the repo and pre-empts the bias trap below.
3. **Post-increment addressing** (A12) — largest absolute cycle count.
4. **Multiple accumulators / `MULT.OUTER`** (A14) — removes 720× read amplification.
5. **`ADD.VV`** (A2) → **`BCAST`** (A5) → **`GATHER`/`SCATTER`** (A6) →
   **`REDUCE.SEG`** (A7) → **`MOV.ACC.RC`** (A8) → **`EXP`** (A9) → FP CR scalar (A10).

---

## Latent trap: bias

The projection kernels compute `C = W·D` with **no `+b`**. The moment a bias term is
added, A2 applies and every output channel pays a fake multiply:

```asm
LDR_CYCLIC_MULT_REG bias_ptr, CR0, LR1 ;;
MULT.RC.VE LR1, CR1, 0, LR1, CR15 ;   # bias x 1.0  <-- FAKE
ACC.ADD ;;
```

`+20,160` fake mult cycles across the projections family, for arithmetic that is a
single vector add. `ACC.ADD.RC` would cost zero.

---

## Negative results

Checked for and **not** found — worth recording so nobody re-audits:

- No multiply-by-zero anywhere; `ACC.*.FIRST` is used correctly throughout.
- No Newton-Raphson or LUT reciprocal (native `reciprocal` / `rsqrt` exist).
- No polynomial `exp` chains (native `exp2`); FFN1's `silu` is a real activation.
- No AGG-collision NOP padding — every "collision" mention in the sources reads
  *"collision-free"*.
- The 12 projection kernels contain **zero** multiply-by-one: every `MULT.RC.VE`
  scalar operand is `LR2`, an index selecting a real weight. Verified across all 12.
- No K-dimension padding waste in projections: `BLT` reads the snapshot while MULT
  reads LR live, so the bounds yield exactly `width` MULTs.
- Clean kernels with no fake multiply: all 16 `matmul_*`, all 12 `attention/*`,
  the 4 `asm_packed_output_linear_*`. They remain lane-limited (A13).

## Known hazards found in passing

- **Cross-kernel `R_MASK` state bleed** — `LDR_MULT_MASK_REG` replaces all 1024 bits
  with no revert-to-default, forcing mask reloads at every step boundary
  (`asm_packed_layernorm_240x16.asm:160-174`, `residual_add_240x16.asm:29-36`).
- **Docstring rot** — `proj_qkv_240_p4/__init__.py:4` claims `N_OUT=576, N_TOK=64`;
  the actual constants at `:55-57` are `720` and `16`. `proj_ffn2_240_p4` disagrees
  with itself the same way. Anyone estimating lane occupancy from docstrings gets
  the wrong answer.
- **`1/√head_dim` attention scaling has no implementation site** anywhere
  (`kernel_layer_map.md:132-139`).
- **`L3_kernel_reference.md:209-215` is wrong** — it states `mult%` *is* the
  lane-occupancy proxy. Every packed kernel masks to 16 or 64 of 128 lanes; the
  overstatement is 4–13×. This file should be corrected.

## Cross-references

`docs/isa_friction_log.md` is the project's existing ISA-gap log and covers A7 in
depth, plus `rc_idx` byte-vs-element addressing, packing factors, and scheduling.
**It lives only on `upstream/ZDlinear` and `upstream/zdlinear_before_pr`.** It is
cited nine times from sources on `pr-kernel-registry-docs`
(`asm_packed_layernorm_240x16.asm:40,234,339,398,563`; `residual_add_240x16.asm:36`;
`output_linear_generic.asm:77`; `pack_240x16.asm:37`;
`kernel_layer_map.md:503,546,596`) — every one of those citations is currently
dangling. It should be carried onto the mainline.

## Optional full ISA measurements

`RunStats.alias_hits` remains the lightweight identity/hardware-only counter.
Full profiling is a separate, opt-in measurement system covering A1–A16. It
records observed instruction patterns and traffic, **not estimated hardware
speedups**. Overlapping aliases never alter effective-MAC accounting.

Run any registered case with `--profile-aliases`, or use
`--alias-report /tmp/aliases.json` to enable profiling and export versioned JSON.
For a harness or direct emulator invocation:

```python
from ipu_emu.alias_profile import AliasProfile
from ipu_emu.ipu_state import IpuState

profile = AliasProfile(metadata={"experiment": "hardware-baseline"})
state = IpuState(alias_profile=profile)
# Alternatively: app.run(alias_profile=profile)
# Execute normally, then inspect profile.to_dict() / profile.to_json().
```

Each report contains all registered aliases, subtype/status counts, instruction
sites, bounded examples, participating instruction/cycle counts, elapsed spans,
active lanes, and actual XMEM bytes. Metrics on sequence occurrences are
inclusive: a hoisted constant load can participate in many occurrences. Do not
sum their costs as exclusive totals. `overlap_instruction_groups` and the unique
profiled instruction/cycle totals identify shared evidence. `unique_read_bytes`
is the union of read addresses, irrespective of memory version.

Detection contracts:

- A1–A6/A16 reuse the existing identity and hardware-only classification.
- A7 measures disjoint partition-window accumulation and complete rotating
  folds over duplicated accumulator data, including MAX folds. It does not
  interpret `CR.partition` as a reduction instruction.
- A8 follows an identity-AAQ store into an actual data-register reload. A11
  additionally requires a full-vector multiplication returning to the same
  accumulator version. Quantizing transfers are marked lossy.
- A9 verifies FP32 `log2(e)` multiplication feeding `exp2`; A10 measures fractional
  broadcast vector operands in wide FP32. Other arithmetic modes are explicitly
  unsupported by these two detectors.
  Centered exponentials are verified when every activated lane has proven
  `log2(e)` scale, including the offset. Lane provenance follows supported
  ADD/SUB/MAX operations, packed SUM/MAX reductions, identity AAQ stores and
  reloads, cyclic windows, and scalar extraction. Quantization, unsupported
  operations, and overwritten sources erase the corresponding proof.
- A12 separates isolated address updates and first load-to-MULT dependencies.
  Distance is observed latency, not proof of removable stalls. A13 reports
  lane-count distributions and undeclared widths without inferring width from
  zero-valued data.
- A14 counts repeated reads of unchanged regions and records distance and
  accumulator-chain changes. This is reuse evidence, not a claim that a given
  accumulator count could eliminate those reads.
- A15 separately measures binary-vector masks, canonical two-arm SET diamonds,
  and consecutive dependent scalar additions. Other control-flow shapes do not
  silently count as verified conditional selections.

Uniform fractional/log2(e) vectors and binary masks without declared provenance
are reported as **candidates**. Declare a known constant after initializing it:
`state.xmem.mark_profile_constant(address, encoded_row_bytes)`. Contents are
validated; overlapping writes revoke the declaration, even if bytes are equal.
This declaration does not turn arbitrary constants into identity multiplies.
Register overwrites invalidate load provenance. Mutable register exports make
subsequent source verification conservative because retained references can
bypass setters.

### Adding an alias

The profiling engine contains no numbered alias cases. Implement a detector
whose `observe(event)` yields `Match` objects, then register its factory and
`AliasDefinition` in `default_registry()` in `ipu_emu.alias_detectors`. A detector
can subscribe to named execution slots through its `slots` tuple; omitting it
subscribes to all events. A `cycle` event contains the completed bundle and its
actual next PC. `Event.facts` provides decoded operands and the shared execution
provenance collected by `alias_observer`.

For an experiment, copy `default_registry()`, register the detector on that
copy, and pass it to `AliasProfile(registry)`. Factories create independent state
for each profile. Duplicate IDs and matches emitted under another detector's
ID are rejected. Add positive/negative detector tests and a real-kernel
expectation alongside the new entry. No emulator dispatch or report formatting
changes are needed unless the new detector requires a genuinely new execution
fact.

The default example limit is three per alias/subtype/status. Full instruction
traces are not retained, but exact overlap membership grows with execution
length. Profiling therefore adds both CPU and memory overhead and stays off
unless explicitly requested.
