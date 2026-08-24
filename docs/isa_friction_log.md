# ISA / spec friction log

Running list of places where the ISA or its spec made a kernel harder to
write, or a bug harder to find, than it needed to be — surfaced while doing
the MobileViT kernel work. Not a bug tracker (those get filed as GitHub
issues); this is for friction that was real but didn't necessarily rise to
a filed issue, or whose issue is worth cross-referencing here for context.
Entries are short by design — the point is that the log exists and
accumulates, not that any one entry is exhaustive.

---

## `rc_idx`: byte vs. element addressing (PR #196 / issue #182)

**What happened:** `MULT.RC.VV` / `MULT.RC.VE` / `MULT.RC.VS`'s `rc_idx`
operand addresses `R_CYCLIC`. The spec's own pseudocode
(`R_CYCLIC[(rc_idx + i) % 512]`) never states a unit, and for a long time
the implementation treated it as a raw **byte** offset while
`LDR_CYCLIC_MULT_REG`'s `index` operand — which writes the same register —
used **elements**. Nothing caught the mismatch until wide-mode's 4-byte
elements made a byte-vs-element bug produce a 4x address drift instead of a
no-op.

**Cost:** two kernels (`softmax_rows_partial`, `softmax_columns_packed`)
were written correctly against the element convention, then produced
order-1 wrong answers (0.7–2.4 absolute error) against every multi-row/
multi-group shape until the branches carrying the fix (`Ipu.
_rc_element_to_byte_offset`) merged. Single-row shapes never exercised the
cross-partition repack, so the bug was invisible until a specific — and not
obviously "the first one to test" — shape was tried.

**Where it stands:** fixed in `ipu.py` (all three `MULT.RC.*` handlers call
`_rc_element_to_byte_offset`). The generated instruction reference's operand
prose was updated to say "base ELEMENT index ... same unit as
`LDR_CYCLIC_MULT_REG`'s `index`" for all three mnemonics
(`instruction_spec.py`), but the `operation=` pseudocode line for all three
still reads `R_CYCLIC[(rc_idx + i) % 512]` with no unit annotation — the
same ambiguity that caused the original drift is still present in the one
place (the formula, not the prose) a future reader is most likely to copy
from.

**Takeaway:** an operand whose unit isn't stated in the formula that uses
it will eventually be implemented in two different units by two different
people. This one only surfaced because wide mode's 4-byte elements turned a
no-op into a visible 4x error — a narrow-mode-only version of the same
mismatch would have been silent.

---

## No segmented (partition-wise) reduce — cross-partition combine costs a
## full store/reload round trip

**What happened:** investigating whether a packed L5 layout (8 channels
packed side-by-side in 16-lane groups within one 128-lane row, instead of
one channel per row) is viable. Packing is free for elementwise ops
(residual add: exactly 8x fewer cycles/instructions/XMEM bytes, measured,
`test_packed_residual_add_240x16.py`) but a packed **linear layer**
(contraction across channels) needs a step with no direct instruction:
after accumulating 30 packed chunks into `r_acc`'s 128 lanes (8 partitions
of 16), the 8 partitions must be summed together to produce the true
16-lane result. Every reduce-shaped instruction in the ISA reduces to a
*single* scalar, unconditionally:
- `AGG.SUM`/`AGG.SUM.FIRST` collapse **all** active lanes (`0..valid_elements-1`,
  contiguous from lane 0) to one `r_acc` slot — no way to keep 8 partial
  sums separate.
- `RESHAPE` permutes `r_acc` word lanes, but only 8 lanes per call and only
  `r_acc`→`r_acc` (not `R_CYCLIC`/`R0`/`R1`, the multiply-stage inputs) —
  replicating one scalar across a 16-lane partition would take 16 calls,
  worse than the workaround below.
- `CR15.partition` (`P0/P2/P4/P8/P16`, `ipu_config.py:39-46`) looks like it
  should help — `P8` gives exactly 8 groups of 16 lanes — but it only feeds
  `_mult_mask_and_shift`'s mask-shift math (`ipu.py:434-486`), which
  **masks/shifts**, it doesn't move data across partitions or reduce.

**Workaround used** (`asm_packed_linear_240to8_replicated.asm`,
`test_primitive_a_combine8x16.py`): store `r_acc`'s 128 lanes to XMEM via
`ACTIVATE.QUANTIZE`+`STR_POST_AAQ_REG`, reload into `R_CYCLIC` via
`LDR_CYCLIC_MULT_REG`, then do 8 `MULT.RC.VE ×1.0` calls at
`rc_idx = 16p` for `p=0..7`, each landing partition `p`'s 16 values at
`mult_res` lanes 0..15, accumulated via `ACC.ADD`/`.FIRST`. Measured cost:
23 cycles / 22 dynamic instructions standalone (`test_primitive_a_combine8x16.py`),
~18-19 cycles embedded once per output channel inside the packed linear
kernel (`asm_packed_linear_240to8_replicated.asm`, lines `after_chunks`
through the second `ACTIVATE.QUANTIZE`).

**Cost at real sizes:** for a packed 240-channel linear layer with 240
output channels (out-proj), this combine runs once per output channel —
240 times — contributing roughly `240 × 19 ≈ 4560` of the kernel's total
27606 measured cycles (~16.5%). It doesn't dominate (the 30-chunk
contraction itself is larger), but it's a fixed tax with no way to shrink
it further given the current ISA, and it's the only reason a packed linear
kernel needs a second pass (store+reload) instead of finishing entirely in
registers.

**What would fix it:** a segmented/strided reduce — e.g. an `AGG` variant
that reduces `mult_res` (or `r_acc`) in `N`-wide segments and writes `128/N`
partial sums instead of collapsing to one, driven by the same
`CR15.partition` field that already exists for masking (P8 → 8 segments of
16). That would collapse the 4-instruction-per-partition combine (SET +
MULT.RC.VE + ACC.ADD, x8, plus the store/reload pair) into a single `AGG`
call reading directly from `mult_res` — no XMEM round trip, no `R_CYCLIC`
reload. Rough value: turns 18-19 cycles/output into approximately 1-2,
saving ~15-17 cycles per output channel wherever packed cross-partition
reduction is used (linear layers, layernorm's channel-axis mean/variance).
At out-proj's 240 output channels alone that's ~3600-4000 cycles saved per
invocation, before counting QKV/FFN or the two layernorms per layer.

**Where it stands:** not filed as a GitHub issue (this is a design-space
finding from an experiment, not a bug in shipped code) — worth filing if
packed layouts move from "measured feasibility" to "adopted."

---

## Packed-linear masked-pass IPC gap — mostly a scheduling bug in my own
## kernel, not the ISA; ~8% residual tax is real and load-bound

**What happened:** path (b) of the packed-linear experiment above (masked
`MULT.RC.VE` passes over packed activations, unpacked/unreplicated weights —
the memory-viable path once path (a)'s 16x weight blowup was ruled out) was
first measured at 240→8 with every per-partition index-advance (`rc_idx_reg
+= 16`, `k_idx += 1`) issued as a **separate** VLIW bundle from the
`MULT.RC.VE; ACC.ADD` that used the pre-advance value: 4412 cycles / 4121
instructions (0.93 IPC), against the unpacked baseline's 2076 cycles / 7745
instructions (3.73 IPC) at the same shape — fewer instructions, twice the
cycles. That gap was **not** inherent to the masked-pass structure: `rc_idx`
and `src`(`k_idx`, resolved via `_mult_resolve_lcr_scalar_wide`'s live
`regfile.get_lr` read) are both `"read": "live"` operands
(`instruction_spec.py:591,594`, `ipu.py:777`), and LR-slot writes dispatch
before load/mult in the same bundle — the same pre-increment rule already
used throughout this codebase (`residual_add_16x240.asm`,
`proj_qkv_240_p4.asm`) to co-issue a pointer's *next* advance with the *load*
that should see it. The masked kernel simply hadn't applied that trick to
its own index registers.

**Fix applied** (`asm_packed_linear_240to8_masked.asm`,
`asm_packed_linear_masked_generic.asm`): seed `rc_idx_reg`/`k_idx` one step
behind (`-16`/`-1`) and co-issue their advance `ADD`s in the same bundle as
the `MULT.RC.VE`/`ACC.ADD` that consumes the advanced value — plus merging
the per-packed-chunk `SET rc_idx_reg` into the preceding load's bundle
(`SET` is an `lr`-slot, `LDR_CYCLIC_MULT_REG` is a `load`-slot, different
slots, freely co-issuable). Result at 240→8: **2252 cycles**, same 4121
instructions (a static program has a fixed instruction count; only the
*scheduling* changed) — a 48.9% cycle reduction, IPC 0.93 → 1.83.
Correctness unaffected (`max_abs_err` unchanged at 7.57e-06,
`test_packed_linear_240to8.py::test_packed_linear_path_b_masked_passes`).

**What's left (real, not schedulable away):** unpacked's 2092 cycles for
1920 `MULT.RC.VE`+`ACC.ADD` ops is 1.09 cycles/op; packed(b)'s 2252 cycles
for the same 1920 ops is 1.17 cycles/op — an **8% residual gap**, not the
original 111%. The remaining tax is the load cycle every 8th step: a packed
chunk's `LDR_CYCLIC_MULT_REG` must land in the cycle *before* the 8
`MULT.RC.VE`s that read it (the MULT snapshot contract — same-bundle loads
aren't visible until the next cycle, issue #157's rule, unrelated to this
kernel), so every 8-partition group costs 9 cycles instead of 8. Unpacked
pays an equivalent one-load-per-step tax but amortizes it differently (1
load per *individual* k, prefetched a cycle ahead inside the same pipelined
bundle as the previous MULT). This is a fixed ~12.5% tax on the *load*
portion specifically, diluted to ~8% overall once mixed with the 8
uniformly-fast partition steps it buys.

**Where it stands:** the 48.9%-of-original gap was a scheduling bug in the
throwaway kernel, fixed, not an ISA gap — no friction-log or issue action
needed for that part. The remaining ~8% is a real, inherent consequence of
the snapshot contract interacting with 8-wide unrolling and is not further
schedulable within the current ISA; it is small enough that it does not
change the packed(b) viability verdict (memory-optimal, cycle-neutral-ish)
and is not worth a separate ISA-change proposal on its own — noted here for
completeness rather than as an actionable gap.

---

## CORRECTION to the entry above: packed OUTPUT is possible without a
## scatter-into-partition instruction — a masked-write construction does it

**What happened:** the "No segmented (partition-wise) reduce" entry (two
entries up) concluded that a packed linear layer's output could not be
produced in packed form — only packed input, via path (b) — because nothing
in the ISA moves 8 independently-computed output channels' scalars into 8
different partitions of one shared row. That conclusion was **wrong**,
found via a follow-up construction that was verified both algebraically and
by direct emulator execution before being trusted:

    rc_idx = 16*(p_in - p_out) mod 512

Reading `R_CYCLIC` at this offset lands input partition `p_in`'s 16 lanes at
`mult_res` lanes `16*p_out..16*p_out+15` — for **any** `p_out`, not just
`p_out == p_in`. Masked to that 16-lane window (`mask_offset = p_out`,
selecting one of `R_MASK`'s 8 pre-built one-hot slots) and accumulated
without resetting `r_acc` between output channels, each output partition's
`ACC.ADD` only ever touches its own disjoint 16 lanes — the other 7
partitions' running sums ride through as `+0` (the mask's pad value) every
other partition's turn. After all 8 output channels' passes, `r_acc` holds a
genuine packed row; one store writes it. No scatter instruction needed —
the "scatter" is achieved by choosing which 16-lane window a *read* (not a
write) lands in, then gating the write side with the existing per-lane mask.

**Verified:** lane arithmetic checked against `_debug_rb_lane_vals`/
`_rc_element_to_byte_offset` for all 64 `(p_in, p_out)` pairs
(`Ipu._rc_element_to_byte_offset`, `ipu.py:327-338`), then built as a real
kernel (`asm_packed_output_linear_generic.asm`) and run against numpy
float64 at K=8, 16, 240, 480 — all pass, max abs error 3e-7 to 1.6e-5
(FP32-consistent). `test_packed_output_linear_tiny.py`,
`test_packed_output_linear_generic.py`.

**Two real costs this construction adds, neither of which the original
entry anticipated:**
1. **R_CYCLIC must be replicated into all 4 slots per packed chunk**
   (`rc_idx` wraps mod 512, but `LDR_CYCLIC_MULT_REG` only ever writes one
   128-element slot per call) — 4 loads instead of path (b)'s 1 per chunk.
2. **The weight row must be reloaded once per (packed chunk, output
   partition)** pair, not once per weight-chunk — each output partition has
   its own weight row, and R0 only holds one row at a time. At K=240,
   N_OUT=8: 361 load-slot instructions (`test_packed_output_linear_generic.py`'s
   `by_slot` output) vs path (b)'s 256 at the same shape.

**Cost at 240→8** (`test_packed_output_linear_generic.py::
test_packed_output_linear_240to8`): 3739 cycles / 4264 instructions / 1
store (vs 8 unpacked) / 512 output activation bytes (vs 4096 unpacked) — see
the updated viability report for the real-size numbers and the corrected
seam verdict this enables.

**Where it stands:** this does not change the earlier entry's finding about
the cross-partition **reduce** (accumulating 8 partitions down to 1 scalar,
e.g. for a non-packed consumer) — that combine primitive still doesn't
exist and the segmented-`AGG` proposal there still stands as the fix for
*that* problem. This correction is specifically about **scatter-on-write**:
producing a packed row from 8 independently-accumulated partitions never
needed a dedicated instruction in the first place, because the existing
mask-gated `MULT.RC.VE` + non-resetting `ACC.ADD` already expresses it once
the read-side addressing is chosen correctly. Not filed as a GitHub
issue — this corrects a design-space finding from an earlier session in
the same log, not a bug in shipped code.

---

## CORRECTION to the packed-layernorm task brief: broadcast `rc_idx` formula
## is `mod 512`, not `mod 128` — and per-token-broadcast vs per-channel-scale
## are two different operations that must not share a construction

**What happened:** building `asm_packed_layernorm_240x16.asm` (packed
LayerNorm, 240 channels × 16 tokens, 8 channels/row), the task brief's
proposed broadcast step read: "push mean and sigma back out to every
partition with masked passes at `rc_idx = -16*p mod 128`." Verified by
direct computation before writing any code: `(-16*p) mod 128` and
`(-16*p) mod 512` diverge for every `p ≥ 1` (e.g. `p=1`: 112 vs 496), and
only the `mod 512` form actually lands `R_CYCLIC[(rc_idx+i) % 512]` back
at element `i` for `i=0..15` — the brief's `mod 128` was simply wrong
arithmetic, not a valid alternate encoding. Caught before implementation
by direct modular-arithmetic verification, not by a failing test.

**A second, costlier confusion surfaced only after the kernel ran and gave
wrong numbers three separate times:** LayerNorm's mean/1-over-sigma are
per-TOKEN values (one scalar per token, broadcast identically into every
channel's 16-lane window), while gamma/beta are per-CHANNEL values (a
different scalar in every 16-lane window, replicated only across that
window's own 16 tokens). These look superficially similar ("a 128-lane
tile built from a smaller source") but require structurally different
constructions: the per-token broadcast needs the masked
`rc_idx=(-16p)%512` scatter (one shared 16-value source folded into every
window); the per-channel tile is simply pre-replicated by the harness at
load time (16 copies of each channel's own scalar, no on-chip broadcast
at all). Conflating them — e.g. debugging normalized-token output against
an `invstd` value indexed by channel instead of by token — produced a
plausible-looking but wrong reference during debugging (session-internal
only; never shipped) before the indexing mismatch was found.

**Also found in the same debugging pass, both real kernel bugs (not ISA
issues):**
- `MULT.RC.VE(x1.0)+ACC.ADD`, run twice (once per operand), computes
  `A+B`, not `A*B` — `ACC.ADD` only ever adds `mult_res` into `r_acc`;
  there is no elementwise-multiply-accumulate. The centering step
  (`x + neg_mean_tile`) is genuinely an addition, so this construction is
  correct there — but copying it for the normalize step
  (`centered * invstd_tile`, a genuine tensor-tensor product) silently
  computed `centered + invstd` instead. Fixed by loading the constant
  operand into a mult-stage register (R0/R1, via `LDR_MULT_REG`, not
  `LDR_CYCLIC_MULT_REG`) and using `MULT.RC.VV(ra=R0/R1)`, which computes
  `R_CYCLIC[i] * Ra[i]` in one instruction — the same mechanism the
  gamma-multiply step already used correctly. Caught by comparing `r_acc`
  directly against the hand-computed expected product after the first
  chunk (kernel gave `centered+invstd`; expected `centered*invstd`).
- `STR_POST_AAQ_REG`'s `offset` operand is a **live** read, and LR-slot
  writes dispatch before `store` in `execute_vliw_cycle` — so a
  same-bundle `ADD` to the offset register IS visible to that same
  store, requiring the well-established "-1 startup offset" convention
  (seed the pointer one step behind, let the first bundle's ADD bring it
  to the correct value) — this project's other kernels already do this
  correctly, but a rewritten pointer-init sequence here briefly dropped
  the `-1` seed, shifting every store in the affected step by one row.
- A **loop-count off-by-one** distinct from the pointer bug above:
  `BLT`'s operands are `"read": "snapshot"` (the pre-this-bundle value),
  so a loop counter's own `ADD` and the `BLT` that checks it must be
  fused in the SAME bundle as the loop body's trailing prefetch load —
  exactly the existing pattern in `layernorm_16x240.asm` (`LDR_CYCLIC...;
  ADD ch_index...; BLT ch_index...` all in one bundle). Splitting the
  counter's `ADD`+`BLT` into a bundle one cycle after the load (which
  looked equivalent, and does NOT break anything if the counter is never
  touched by the intervening bundle) still shifts the loop's effective
  trip count by one, because the reference kernel's PEEL block *also*
  performs the counter's first `ADD`+`BLT` fused with the load in its own
  trailing bundle — a step this rewrite's peel omitted, expecting a
  separately-initialized counter to compensate. It doesn't: the correct
  fix was `row_limit = ROW_COUNT - 1`, derived by explicitly tracing
  which loop-body pass numbers satisfy the snapshot-read inequality
  rather than by pattern-matching the reference kernel's surface syntax.
  Caught by counting dynamic `STR_POST_AAQ_REG` calls to a known-size
  region (31 stores observed for 30 valid rows) — the numeric error
  alone was misleadingly small at first (zero-valued unwritten XMEM
  masqueraded as a harmless extra chunk) until a later step's
  out-of-bounds write corrupted an unrelated tensor's row.

**Takeaway:** four independent, differently-shaped bugs — one arithmetic
(brief's own formula), one indexing (token vs. channel), one algebraic
(addition standing in for multiplication), one timing (loop-counter
snapshot semantics) — all had to be found by direct verification against
hand-computed expected values at each pipeline stage, not by pattern-
matching "this looks like the working kernel's shape." All four are now
fixed in `asm_packed_layernorm_240x16.asm`
(`test_packed_layernorm_240x16.py` passes, max abs error 6.02e-07,
identical to the unpacked kernel's own precision). Not filed as GitHub
issues — all four are kernel-authoring bugs in new code, not defects in
shipped `ipu.py`/`instruction_spec.py`.

---

## Pack/unpack seam kernels: cross-chunk `ACC.ADD` reset bug (own new code)

**What happened:** building `asm_packed_pack_240x16.asm` (the reverse of
`asm_packed_unpack_240x16.asm` — one-channel-per-row → packed, for the
attention seam), the per-chunk scatter-write loop's `p_out=0` case used
unconditional `ACC.ADD` instead of `ACC.ADD.FIRST`. Each packed output
row is an independent accumulation (8 source rows scattered into 8
disjoint 16-lane windows of one shared `r_acc`, then stored) — copying
the "loop body always uses `ACC.ADD`, only the outer peel uses `.FIRST`"
shape from `asm_packed_layernorm_240x16.asm`'s step 1 (where the loop
genuinely sums *across* chunks) was the wrong pattern here, since this
kernel's chunks do not accumulate into each other at all. The bug let
chunk *c*'s first window silently add onto chunk *(c-1)*'s stale `r_acc`
content, compounding across chunks — row 0 correct, error growing
monotonically row-by-row afterward (diff ≈3.0 at row 1, ≈13.3 at row 9),
the signature of unbounded cross-iteration accumulation rather than a
one-off indexing slip. Fixed by using `ACC.ADD.FIRST` at `p_out==0` in
both the peeled first chunk and the runtime loop body.

**Where it stands:** fixed in `asm_packed_pack_240x16.asm`
(`test_packed_pack_unpack_240x16.py` passes, exact 0.0 error — pack/
unpack are pure data movement with no arithmetic beyond pass-through
multiplies by 1.0, so bit-exact equality is the correct bar and both
directions hit it, including a pack(unpack(X)) round trip). Not filed as
a GitHub issue — kernel-authoring bug in new code.

---

## Cross-kernel `R_MASK` state bleed: a real chaining hazard, not a bug in
## either kernel individually

**What happened:** chaining `asm_packed_output_linear_generic.asm`
(out-proj) directly into `asm_packed_residual_add_240x16.asm` (residual
add) within one shared `IpuState`, for the full-L5-layer task, residual
add's output was correct only in partition 0 of every packed row —
partitions 1–7 read back as exact zero. `asm_packed_residual_add_240x16.asm`
never calls `LDR_MULT_MASK_REG` at all; it relies entirely on `R_MASK`'s
regfile-init default (all bits set), which is correct — and has *always*
been correct — every time this kernel has been tested standalone, since
a fresh `IpuState` always starts with that default. The packed-output
linear kernel, however, loads a one-hot 8-slot `R_MASK` (one 16-lane
window per output partition) for its own scatter-write construction and
never restores it before halting. `R_MASK` is process-wide register
state, not owned or reset by any kernel — so residual-add's *own*,
independently-correct assumption ("`R_MASK` is all-ones because nothing
sets it otherwise") silently breaks the moment it runs after a *different*
kernel that has touched `R_MASK`, in the *same* emulator state.

**Why this matters beyond this one instance:** neither kernel is buggy in
isolation, and no single-kernel test — however thorough — would ever
catch this, because the failure only exists in the *composition* of two
kernels' hardware-state side effects across a shared `IpuState`. Any
kernel in this codebase that omits `LDR_MULT_MASK_REG` and relies on the
all-ones default is implicitly assuming it is either the first kernel to
run in its `IpuState`, or that no prior kernel in the same state has
touched `R_MASK` — an assumption that held by accident for every
kernel-chaining test written before full-layer chaining was attempted,
because no earlier test ran two `R_MASK`-touching kernels back-to-back in
one state.

**Where it stands:** fixed at the call site (the full-layer-chain test
harness explicitly reloads an all-ones `R_MASK` via
`state.regfile.set_r_mask(bytes([0xFF]*128))` before every residual-add
call), not in either kernel — both are correct and validated
independently; the fix belongs in whatever eventually owns
kernel-to-kernel scheduling for real chained execution, not in a
retroactive edit to a validated production kernel. Worth flagging if
packed-kernel chaining becomes a standing pattern: **any kernel that
omits an explicit `R_MASK`/`R_CYCLIC`-slot setup step is only safe to run
first in a fresh state**, and that constraint is invisible from reading
the kernel alone.

---

## Replication-count optimization: 1 slot suffices, not the 2 the brief
## suggested checking

**What happened:** task brief asked whether `asm_packed_output_linear_
generic.asm`'s `replicate_chunk()` macro (4 `R_CYCLIC` slot loads per
packed chunk) could shrink to 2, given `rc_idx = 16·((p_in - p_out) mod 8)`
always lands in `[0, 112]` and the brief's own bound is "every read
window `r+i ≤ 239` stays inside slots 0 and 1." Verified by direct
enumeration over all 64 `(p_in, p_out)` pairs before writing any code:
the 16-lane read window `[rc_idx, rc_idx+15]` has a maximum index of
`112+15=127` — it never reaches slot 1 (elements 128–255) at all, let
alone slots 2–3. The brief's own bound is real but looser than what this
specific `rc_idx` formula actually achieves.

**Verified:** a 1-slot replication variant (`asm_packed_output_linear_
1slot.asm`, `replicate_chunk()` reduced to its first `LDR_CYCLIC_MULT_REG`
call only) matches the 4-slot baseline's correctness exactly (both
6.28e-06 max abs error at K=240) and passes at the K ∈ {240, 480, 720}
boundary set. Replication loads drop from 120→30 per output-channel call
(4 loads × 30 chunks → 1 load × 30 chunks); measured load-slot
instruction count falls 361→271 (a 25% reduction), and total cycles fall
3739→3409 (8.8%) at K=240
(`test_packed_output_linear_1slot_replication.py`).

**Where it stands:** not filed as a GitHub issue — this is a
design-space optimization confirmed for the packed-output linear
kernel's specific `rc_idx` range, not a defect. Worth keeping in mind if
this construction is generalized to a different `p_in`/`p_out` mapping
in the future: the "1 slot suffices" result is a consequence of this
exact `rc_idx` formula's range, not a general property of the ISA, and
should be re-derived (not assumed) if the formula changes.

---
this same log, not a bug in shipped code.

---

## L4 packed-layout port: `partition_size(64)=64` gives a packing factor
## of 2, not 8 — every L5 constant had to be re-derived, not ported

**Context:** the entries above document the packed-activation-layout
kernel family built for Layer 5 (d=240, N_TOK=16, packing factor 8, per
`softmax_rows_partial.partition_size(16)=16` → `128/16=8`). This entry
covers the L4 port (d=192, N_TOK=64, P_STREAM=4, N_HEAD=4, HEAD_DIM=48).

**The core fact driving every difference below:**
`partition_size(64) = 64` (next power of two ≥ 64, floored at 16 — the
same rule already implemented at
`softmax_rows_partial/__init__.py:117-124`, applied here for the first
time outside softmax itself). So `parts_per_chunk = 128/64 = 2`, not 8 —
L4 packs only 2 channels per 128-lane row, giving 96 packed rows for
D_MODEL=192 (vs L5's 30 packed rows for 240 channels). Every `16` and
`range(8)` in the L5 `.asm` files had to become `64` and `range(2)`
respectively, re-derived from this fact — not found by search-and-replace,
since L5's files hardcode "8 partitions of 16 lanes" throughout (fixed
8-entry `SEED_CR` tables, `range(8)` unrolls, mask rows with 8 one-hot
16-lane slots).

## Replication-slot finding CONTRADICTS the L5 result — 2 slots (0 and 3),
## not 1

**What L5 found** (see the "Replication-count optimization" entry above):
for L5's `rc_idx = 16*(p_in-p_out) mod 512` formula, all 64 `(p_in,p_out)`
pairs' 16-lane read windows stay within `[0,127]`, so replicating the
packed chunk into R_CYCLIC slot 0 only is sufficient — a genuine 25%
reduction in load-slot instructions.

**L4's formula is `rc_idx = 64*(p_in-p_out) mod 512`.** Direct
enumeration over all 4 `(p_in,p_out)` pairs (`p_in,p_out ∈ {0,1}`,
script-verified before writing any `.asm`):

```
p_in=0 p_out=0  rc_idx=0    max_read=63   slot=[0,0]
p_in=0 p_out=1  rc_idx=448  max_read=511  slot=[3,3]
p_in=1 p_out=0  rc_idx=64   max_read=127  slot=[0,0]
p_in=1 p_out=1  rc_idx=0    max_read=63   slot=[0,0]
```

3 of 4 pairs stay in slot 0, but `(p_in=0, p_out=1)` lands **entirely in
slot 3** (elements 448–511) — the negative wrap (`-64 mod 512 = 448`)
reaches all the way to the far end of the 512-element ring, something
L5's `16`-wide step never did. Slots 1 and 2 are never touched by either
formula. **L4's replication therefore needs exactly 2 slots (0 and 3),
not L5's validated 1, and not the naive 4.** This is a genuine
contradiction of the L5 finding, not a refinement of it — L5's own report
already flagged that "1 slot suffices" was "a consequence of this exact
`rc_idx` formula's range, not a general property of the ISA," and this is
the confirming counterexample. Verified both by the standalone enumeration
test (`test_packed_output_linear_generic_p4.py::test_replication_slot_
enumeration`) and by the 2-slot kernel's own correctness
(`asm_packed_output_linear_generic_p4.asm`, err 7.6e-07 at K=192, err
5.1e-07 at K=384 with silu).

## `primitive_a` scales down trivially — 2 terms instead of 8

Not a surprise, but confirms the L5 report's own framing: at
`parts_per_chunk=2`, the cross-partition combine primitive
(`asm_primitive_a_combine2x64.asm`) reduces to exactly 2
`MULT.RC.VE ×1.0` + `ACC.ADD[.FIRST]` terms (`rc_idx ∈ {0, 64}`), measured
at 11 cycles / 10 dynamic instructions standalone (vs L5's 23 cycles / 22
instructions for 8 terms) — roughly proportional to term count, as
expected. L4's packed-output-linear construction (which embeds the same
gather-and-combine idea directly, not via the standalone primitive) never
needed to invoke this in isolation for the full chain, since the 2-slot
replication + 2-way unroll already does the combine inline.

## Packing win is real but modest — nowhere near L5's 8x headline

Measured (not scaled) results, `1e-3`/`1e-4`/`1e-5` error bounds per the
L5 convention:

| Kernel | packed/unpacked cycles | packed/unpacked instructions | packed/unpacked XMEM bytes |
|---|---|---|---|
| LayerNorm (192ch×64tok) | 0.806 | 0.581 | 0.836 |
| Residual add (192ch×64tok) | 0.507 | 0.501 | 0.500 |

Residual add lands almost exactly at the theoretical 0.5x (packing factor
2, pure elementwise, no combine needed) — the cleanest possible outcome.
LayerNorm's win is smaller than its own packing factor would suggest
(0.81x cycles, not ~0.5x) because the six-step algorithm's broadcast/
combine overhead (steps 1/2/4/5) doesn't shrink proportionally with
`parts_per_chunk` — the same fixed per-chunk bookkeeping (mask
loads, R_CYCLIC reloads, loop-counter bundles) now amortizes over only 2
partitions/chunk instead of 8, so relatively more of the kernel's cost is
per-chunk overhead rather than per-partition work. This was flagged as
the expected outcome in the task brief ("2x packing density is a much
smaller win than L5's 8x") and the measurement confirms it directly — no
extrapolation was needed or used.

## No `ipu.py` edits, no softmax changes

Same standing constraints as the L5 session, re-confirmed: nothing in this
port required an emulator change (every re-derived formula fits inside
the existing `MULT.RC.*`/`ACC.*`/`AGG.*`/`LDR_CYCLIC_MULT_REG`/
`STR_POST_AAQ_REG` instruction set), and `softmax_rows_partial` was used
strictly read-only (imported directly, never modified) for the
attention sub-chain's real-softmax stage.
