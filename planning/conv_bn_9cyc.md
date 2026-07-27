# conv_universal_bn_activation → 9 cyc/ch  ✅ DONE (2026-06-27)

## RESULT: implemented, all 7 conv_bn tests pass, 814 vs 898 cyc (28ch/3chunks) = exactly
## -1 cyc/ch x 84 channel-iters.  No regressions (depthwise_bn 7/7; the 5 conv_universal
## ERRORS are the pre-existing unmigrated base app).  scratchpad/dbg2.py = single-channel
## 3-chunk harness that isolated the main section (chunk1 correct while g0/gN still old).

## VALIDATED DESIGN v3 (2026-06-27) — role-rotating slots, kr+1 loaded same loop

Attempts 1 & 2 FAILED (reverted).  Root cause of BOTH: I prefetched the NEXT channel's
kr+1 onto a slot, and my lr_write chain landed a load on the ALWAYS-LIVE kr0 slot.

CORRECT lifetimes (user, slot-level): kr0 slot live ALL 9 taps; kr-1 slot live taps 1-3;
kr+1 slot live taps 7-9.  The kr0 slot is NEVER a valid load target.

CORRECT load phasing (user): each loop loads THIS channel's kr+1 during taps 1-6 (read
taps 7-9 same loop), and prefetches the NEXT channel's kr-1 and kr0.  3 loads/loop; the
NEXT kr+1 is loaded in ITS OWN loop, not this one.

Slot roles rotate; lr_read (kr0 slot) advances -128 (= +384)/channel.  Per-channel slots
(scratchpad/roles.py, matches user's "slot0 = ch0 kr-1, ch1 kr0, ..."):
  ch c: kr0@slot s; kr-1@(s-1); kr+1@(s+1); spare@(s+2).  s sequence 1,0,3,2 (off 128,0,384,256).

Per-loop load schedule (computing ch c), verified no load hits the kr0 slot:
  L1: THIS ch c kr+1  -> kr+1 slot (slot(s+1)); load taps 1-6 (free then; read 7-9).
  L2: NEXT ch c+1 kr-1 -> the SPARE slot (slot(s+2)); free always; prefetch any tap.
  L3: NEXT ch c+1 kr0  -> THIS kr-1 slot (slot(s-1)); free from tap 4; prefetch tap 4+.
  => place at e.g. tap 6 (this kr+1), tap 1 (next kr-1 -> spare), tap 4 (next kr0 -> kr-1 slot).

DIFFERENCE from the working 10-cyc code: the 10-cyc loads THIS kr-1 in a standalone word
and prefetches NEXT kr0(tap4)+kr+1(tap7).  The 9-cyc loads THIS kr+1 (taps1-6) +
NEXT kr-1(spare) + NEXT kr0(this kr-1 slot), rotation -128 vs +256.  ext addrs:
THIS kr+1 = lr_off_zero + cr6; NEXT kr-1 = E' - cr6; NEXT kr0 = E'.

TODO: derive lr_write chain (4 distinct slot targets/loop), walk seed/cr14 for -128
rotation (walk itself UNCHANGED cols-stride), preamble (seed first ch's slots + roles).
Implement, test against single-channel FIRST (in_ch=1 catches same-loop corruption),
then full suite.  scratchpad/roles.py has the slot math.

## (superseded) VALIDATED DESIGN (2026-06-26, attempt 2) — +384 rotation, masking frees slots early

KEY INSIGHT (from user): mask_shift on kc=-1/+1 ZEROES the straddle lane, so a tap
reads ONLY its own kr chunk slot — it does NOT keep the neighbouring chunk slot alive.
Real slot lifetimes (masked), lr_read=0: kr-1 slot last read tap 3; kr0 slot tap 6;
kr+1 slot tap 9.  (My attempt-1 sim wrongly counted masked straddle bytes as reads.)

Byte-accurate + masking feasibility (scratchpad/decide2.py), 4 slots:
  rot +256 (CURRENT): NOT feasible — N+1 kr-1 lands on a slot free only at tap10
                      -> standalone kr-1 word -> 10 cyc.  THIS is why current = 10.
  rot +384 (= -128):  FEASIBLE — prefetch free-taps: kr-1@1, kr0@4, kr+1@7. -> 9 cyc.
  rot +128:           NOT feasible.

THE FIX: change channel rotation +256 -> +384, fold the standalone kr-1 load into
tap 1 (its target = the free 4th slot, available from tap1 under +384), keep EVERYTHING
ELSE identical to the working code — the cols-stride walk, ±128 slot placement, masking.
cr14 (walk wrap) does change +256-2cols-2 -> +384-2cols-2 (the rotation grew by 128).

ATTEMPT-1 POST-MORTEM: I wrongly believed the walk strided by 128 and rewrote the walk
seed/steps.  In fact the walk is UNCHANGED (cols-stride); attempt-1's real bug was in
the prefetch lr_write targets / E' timing, NOT the walk.  Attempt 2: minimal diff from
the working code — touch only rotation amount + kr-1 load tap + cr14.  WIP of the broken
attempt saved to wip_9cyc_attempt1.patch (do NOT reapply; reference only).

## OLD (attempt-1, INVALID) NOTES BELOW
---

## ATTEMPT 1 FAILED — wrong layout model (2026-06-26). WIP saved to wip_9cyc_attempt1.patch; reverted.

My +384 "tape" slot model assumed the 3 kr-rows of a channel sit in slots 128 bytes
apart and the walk strides by 128.  **That is false.**  The walking pointer reads the
3 vertical neighbours at offsets `lr_read - cols`, `lr_read`, `lr_read + cols` — i.e.
**`cols` apart (16/32/64), NOT 128**.  `get_r_cyclic_at(offset,128)` reads 128 contiguous
bytes wrapping mod 512, so a read at `lr_read ± cols` STRADDLES two adjacent 128-byte
chunks.  The "+256 rotation / 3 chunks at ±128" is the PHYSICAL chunk placement; the
walk distance between kr rows is `cols`.  My slot-lifetime analysis (taps 1-3 read slot
R-128, etc.) was therefore wrong, and so was the conclusion about which slots free when.

Symptom that exposed it: with the +384 walk seed, tap 7-9 (kr=+1) landed in slot 0
(offset 15/16/17) instead of the kr=+1 chunk at slot 128 — because the walk strides by
`cols`, the kr=+1 read at `lr_read+cols` = offset +16, which is inside slot 0, relying
on the straddle to reach the kr=+1 chunk's data.  The whole 9-cyc redesign must be
re-derived against the STRADDLING layout, not a 128-spaced slot model.

TODO next session: re-read how the straddle maps (lr_read±cols reads spanning chunk
boundaries); redo the slot/byte lifetime analysis with cols-spaced reads; only then
decide whether 9 cyc is reachable.  scratchpad/ has the (now-invalid) 128-spaced sims.

## ORIGINAL (INVALID) DESIGN BELOW — kept for reference only
---

# conv_universal_bn_activation → 9 cyc/ch (virtual-tape +384 ring, natural row order)

## Chosen design (user's "virtual tape" model — verified in scratchpad/tape.py)

Treat the 512B cyclic register as a tape: each input channel's 3 rows are written
into 3 **consecutive** 128B slots, right after the previous channel's. Advance the
write base by **+384 (3 slots) per channel** (mod 512). Read in the **natural order**
kr=-1, kr=0, kr=+1 — the walking pointer sweeps forward in fixed intervals; NO tap
reordering.

This is simpler than the +128-ring-with-reorder idea (now discarded). Why +384 works
where +128 did not: under natural read order slots free in the order
kr=-1 (after tap 3) → kr=0 (after tap 6) → kr=+1 (after tap 9). With +384 the next
channel must FILL its slots in exactly that same order, so free-order == fill-order.

### Prefetch schedule (channel N reads; prefetch is for channel N+1)

Channel N base = B (mod 512); slots: kr=-1 @ B, kr=0 @ B+128, kr=+1 @ B+256.
last-read tap: B→3, B+128→6, B+256→9. The 1 free slot this iter = B+384 ≡ B-128.
Channel N+1 base = B+384.

| N+1 row | target slot          | free from tap | issue prefetch at |
|---------|----------------------|---------------|-------------------|
| kr=-1   | B+384 (free 4th slot)| tap 1         | tap 1 (or 1..3)   |
| kr=0    | B   (N's kr=-1 slot) | tap 4         | tap 4 (or 4..6)   |
| kr=+1   | B+128 (N's kr=0 slot)| tap 7         | tap 7 (or 7..9)   |

All 3 prefetch loads fit inside the 9-tap window → standalone load word removed → 9 cyc/ch.

## Walking pointer (natural order, unchanged shape)

lr_walk visits, relative to read base B (kr=-1 row starts at B - 1 ... wait: kr=-1 is the
TOP row = smallest address. With natural taps 1..9 = (kr=-1 kc=-1) ... (kr=+1 kc=+1):
   tap1 kr=-1 kc=-1 = B - 1            (B is kr=-1 row's center; kc=-1 = -1)
   ... +1, +1 within row
   tap4 kr=0  kc=-1 = B + cols - 1     (step +(cols-2) from tap3)
   tap7 kr=+1 kc=-1 = B + 2cols - 1    (step +(cols-2) from tap6)
   tap9 kr=+1 kc=+1 = B + 2cols + 1
Wrap (tap9 → next iter tap1): next base = B+384; next tap1 = (B+384) - 1.
Step tap9→tap1 = (B+384-1) - (B+2cols+1) = 384 - 2cols - 2 = 382 - 2cols.
  (current cr14 = 256 - 2cols - 2; new wrap CR = 384 - 2cols - 2 = 382 - 2cols.)
Within-row +1; row-transition +(cols-2); wrap +(382-2cols). All fixed → simple sweep.
NOTE the read offsets stay within a forward-moving 384B window that itself advances
+384/iter; cyclic wrap (get_r_cyclic_at mod 512) handles the modulo.

## Mask slots (vertical borders) — UNCHANGED from current

Natural read order is preserved, so g0 (top border) still masks kr=-1 = taps 1-3
(slot 3); gN (bottom border) still masks kr=+1 = taps 7-9 (slot 6). No mask remap.
This is a big simplification vs the reorder approach.

## Edits

- mn/g0/gN tap bodies: keep natural tap order; place 3 prefetch loads at taps 1/4/7
  (writing N+1's kr=-1/kr=0/kr=+1 into slots B+384 / B / B+128); delete standalone load word.
- Base advance: +384/channel instead of +256. lr_write computations and lr_read rotate
  by +384 (≡ -128) mod 512. Re-derive lr_write target per prefetch tap.
- Walk wrap CR: 256-2cols-2 → 384-2cols-2. Check harness sets this (cr14).
- Preamble (ch_loop cy1-4): seed first channel's 3 slots at base 0; seed lr_walk to
  tap1 = B-1; seed lr_write chain.
- Verify lr_read/lr_write arithmetic for the +384 (=-128 mod512) rotation.

## Verify

- Direct pytest test_conv_universal_bn_activation.py (Bazel blocked).
- Don't touch base conv_universal.asm (unmigrated) or the depthwise twin.
- scratchpad/tape.py holds the slot-timing proof.
