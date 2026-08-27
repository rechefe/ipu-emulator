# Reading `depthwise_conv_universal`: a guided walkthrough

This is a line-by-line guide to `depthwise_conv_universal.asm` — how a 3×3
depthwise convolution gets built out of VLIW words on this ISA, and why the
inner loop looks the way it does. It's written for someone who can read the
ISA reference but hasn't yet built the mental model of how a real kernel
threads loads, multiplies, accumulation, and stores through the pipeline.

If you haven't read the project's [`SKILL.md`](../../../../../../../../SKILL.md)
(architecture, register file, ISA reference) yet, do that first — this guide
assumes you know what `R_CYCLIC`, `R_ACC`, `R_MASK`, and the seven VLIW slots
are, and just walks through how they're *used*.

## 1. What the app computes

Depthwise 3×3 convolution, stride 1, zero-padding ("same"), INT8, **no bias,
no activation** (its BN+ReLU twin, `depthwise_conv_universal_bn_activation`,
adds those two things on top of this same pipeline — see its own docstring
for the delta). Depthwise means each output channel depends on exactly one
input channel — there's no cross-channel reduction, which is what makes this
app a simpler read than `conv_universal` (which sums over `in_channels` per
output).

```python
# input: [channels, rows, cols] int8, kernel: [channels, 9] int8 (dr*3+dc taps)
for ch in range(channels):
    for r in range(rows):
        for c in range(cols):
            acc = 0
            for dr in range(3):
                for dc in range(3):
                    ir, ic = r + dr - 1, c + dc - 1
                    if 0 <= ir < rows and 0 <= ic < cols:
                        acc += kernel[ch, dr*3+dc] * input[ch, ir, ic]
            output[ch, r, c] = clamp(acc, -128, 127)
```

The IPU version does this **128 columns at a time** (one XMEM chunk = one
lane vector = up to `128/cols` packed spatial rows), one channel at a time,
in a 9-cycle inner loop — one cycle per tap.

## 2. The core idea: taps as scalar-times-vector multiplies

A 3×3 conv has 9 taps. Each tap `(dr, dc)` needs, for every output lane, the
product of one kernel weight (a scalar, same for the whole channel) and one
*shifted* copy of the input row (a 128-lane vector, shifted by `dc` columns
and pulled from a different row for `dr ≠ 0`).

The instruction that does exactly this is `MULT.RC.VE`:

```
MULT.RC.VE  rc_idx, src, mask_offset, mask_shift, cr_idx
MULT_RES[i] = R_CYCLIC[(rc_idx + i) % 512] * src_value      # src_value: scalar from R0/R1[idx] or CR[idx]
```

`rc_idx` is a *base index into a 512-element ring buffer* (`R_CYCLIC`), so
reading `[rc_idx .. rc_idx+128)` for `rc_idx` not aligned to a slot boundary
gives you a **shifted window** across two adjacent slots — that shift is
exactly the `dc` column offset. `src` here is always `lr6` (the kernel byte
index), interpreted by `MULT.RC.VE` as a byte offset into the combined
`R0++R1` kernel block: `0..127 → R0[idx]`, `128..255 → R1[idx-128]`.

So each tap is one `MULT.RC.VE` (get the shifted+scaled product) plus one
`ACC.ADD` (accumulate it into `r_acc`) — and the whole 9-tap body is 9 nearly
identical VLIW words, differing only in which `rc_idx`/mask/shift they use.

## 3. R_CYCLIC layout: three rows resident at once

`R_CYCLIC` is a 512-element ring. This app keeps **three 128-element slots**
resident — the row above (`kr=-1`), the current row (`kr=0`), and the row
below (`kr=+1`) — using slot boundaries `{0, 128, 256, 384}` (4 slots
available, 3 used at a time, the 4th absorbs the rotation below).

The **walk index** `lr3` starts at a base for `kr=-1, kc=-1` and is advanced
by `+1` per tap for `kc` steps and by `lr1` (`= cols - 2`) to jump between
rows — e.g. after tap 3 (`kr=-1, kc=+1`), tap 4 needs `kr=0, kc=-1`, which is
`+lr1` elements ahead in a slot 128 elements wide but only `cols` of it
"real" (the rest is other packed rows/garbage never read at these `rc_idx`
values). This walk requires **no reloads mid-body** — the loads all happen
in the 4-cycle preamble (`*_ch_pre`) and get *prefetched one channel ahead*
inside the tap body (see §5).

## 4. Which row is "row 0" of R_CYCLIC — the slot-rotation trick

Naively, you'd reload all three `kr` rows for every channel. Instead, this
app keeps a **single running write pointer** `lr5`, advanced by
`incr_mod_pow2` (never overflows the 512-element ring), and a **separate read
pointer** `lr4` used only to track which slot is "stale" and safe to
overwrite next. Within one channel's 9-cycle body:

- tap 4 (`kr=0, kc=-1`): `lr5 += 384 (mod 512)` then load *next channel's*
  `kr=0` row there.
- tap 5 (`kr=0, kc=0`): `lr5 += 128 (mod 512)`.
- tap 7 (`kr=+1, kc=-1`): load *next channel's* `kr=+1` row (computed one tap
  earlier, at tap 6/8 depending on section — see below).
- tap 9 (`kr=+1, kc=+1`): `lr5 += 256 (mod 512)`, load *next channel's*
  `kr=-1` row there, fused into the **same cycle** as the final `acc.add` +
  `ACTIVATE.QUANTIZE`.

Net effect: **each channel change costs zero extra cycles for reloading**
input data — by the time tap 1 of the next channel needs `kr=-1`, it was
already loaded during the *previous* channel's tap 9. This is a general
one-iteration-ahead prefetch idiom used throughout this app family: loads
are issued far enough in advance that the consuming `MULT.RC.VE` always sees
fresh data in the same cycle it's needed, never a cycle late.

## 5. Borders: masks, not branches

Three border cases, all handled **without extra cycles**:

- **Top/bottom rows** (`kr=-1` at row 0, or `kr=+1` at the last row): a
  single 128-byte mask blob is loaded *once* at program init
  (`ldr_mult_mask_reg`), carrying three slots — `0` = keep-all, `3` =
  zero-top-row, `6` = zero-bottom-row. The asm is split into three sections
  (`g0_*`, `main`/`mn_*`, `gN_*`) that are just the same 9-tap body with a
  different **mask_offset immediate** (`3`, `0`, `6`) baked into the relevant
  taps' `MULT.RC.VE`. No mid-program mask reload — the blob has everything.
- **Left/right columns** (`kc=-1` at column 0, or `kc=+1` at the last
  column): handled by `mask_shift`, **not** a separate mask slot. `CR15`'s
  `partition` field is set to `cols` (via `Partition.P2/P4/P8` depending on
  `cols=64/32/16`) so each partition group is exactly one packed spatial row;
  `mask_shift = lr9 (+1)` zeros the start of each row, `lr13 (-1)` zeros the
  end. Every `kc=-1` tap uses `lr9`, every `kc=+1` tap uses `lr13`, every
  `kc=0` tap uses `lr0` (no shift) — this is baked directly into each tap's
  `MULT.RC.VE` operand, visible as the 4th column across the 9 taps in the
  asm.
- **Combining both**: a corner tap (e.g. `kr=-1, kc=-1` in the top section)
  just uses mask slot `3` *and* shift `lr9` in the same instruction — masking
  composes for free, no separate corner case.

## 6. r_acc and the store pipeline

- **Tap 1** always issues `acc.add.first`, which *resets* `r_acc` to this
  tap's product instead of adding to the previous channel's leftover value —
  this is the "bias seed" in the BN twin, replaced here by just letting tap
  1's own product be the reset value (no bias to add).
- **Taps 2–8** issue `acc.add` (accumulate).
- **Tap 9** issues `acc.add` *and* `ACTIVATE.QUANTIZE identity, cr15` in the
  **same VLIW word**. This is only possible because of a specific pipeline
  fact: `ACTIVATE.QUANTIZE` reads `r_acc` **live** (post-ACC-phase within the
  cycle — see `execute_activate_quantize` in `ipu.py`), and the VLIW
  intra-cycle dispatch order runs ACC before AAQ (§"VLIW Execution Order" in
  the project's architecture notes). So tap 9's `acc.add` and that same
  cycle's `ACTIVATE.QUANTIZE` see the *same*, fully-accumulated value — no
  extra "finalize" cycle needed. (This fusion is a relatively recent win: an
  earlier version of this app needed a standalone ACTIVATE cycle, i.e.
  10 cyc/ch instead of 9 — see `activate_quantize_live_read_fix.md` in
  project memory for the upstream fix that made this possible.)

The actual memory store is **deferred by one full channel**: `STR_POST_AAQ_REG`
fires at **tap 2**, storing the *previous* channel's `post_aaq_reg` (already
written by that channel's tap 9). This spreads the store across a cycle that
has a free XMEM slot, rather than trying to also cram a store into tap 9's
already-busy word. The very first channel's tap-2 store writes to a harmless
scratch address (never read back); the very last channel's store is flushed
explicitly in the `end:` epilogue.

## 7. The kernel byte walk (`lr6`)

Weights are packed FPB=28 channels per 256-byte "super-block", 9 bytes per
channel (no bias byte — see the BN twin for how a bias byte gets added to
this layout). `lr6` just counts `0, 1, 2, ..., 8` across one channel's 9
taps (`INC lr6 1` every tap) and is fed as `MULT.RC.VE`'s `src` operand,
which `MULT.RC.VE` interprets as a byte offset into `R0++R1`. Because taps
land at consecutive bytes and channels are 9 bytes apart, this one counter
naturally rolls over into the next channel's first weight byte without any
extra reset logic — it just keeps incrementing across the whole
channel-group loop.

## 8. Reading the whole body top to bottom

Put together, here's the 9-cycle body (main-section version,
`mn_tap_body:`), tap by tap — this is the piece worth reading side-by-side
with the `.asm` file itself:

| tap | kr,kc | mask | shift | extra work this cycle |
|---|---|---|---|---|
| 1 | -1,-1 | 0 | lr9 (+1) | `acc.add.first` (reset r_acc); loop counter += |
| 2 | -1,0  | 0 | lr0 (0)  | deferred store of **previous** channel |
| 3 | -1,+1 | 0 | lr13(-1) | advance lr2 → next channel's `kr=0` address |
| 4 | 0,-1  | 0 | lr9      | rotate write slot (+384); load next ch `kr=0` |
| 5 | 0,0   | 0 | lr0      | rotate write slot (+128) |
| 6 | 0,+1  | 0 | lr13     | rotate **read** slot (lr4, tracks staleness) |
| 7 | +1,-1 | 0 | lr9      | load next ch `kr=+1` |
| 8 | +1,0  | 0 | lr0      | precompute next ch's `kr=-1` address |
| 9 | +1,+1 | 0 | lr13     | rotate write slot (+256); load next ch `kr=-1`; **fused** `ACTIVATE.QUANTIZE`; loop branch |

The `g0_tap_body`/`gN_tap_body` variants are identical except taps 1–3 (in
`g0`) or 7–9 (in `gN`) use mask slot `3`/`6` instead of `0`, since those taps
are the out-of-bounds row for that section.

## 9. Why this is worth 9 cyc/ch, and what would make it fewer

Every one of the 9 cycles does a real tap (`MULT.RC.VE` + `ACC.ADD`) — there
is no idle "setup" or "teardown" cycle inside the steady-state loop; all
prefetching, pointer rotation, and the store are folded into taps that
otherwise have a spare LR/XMEM/ACC slot that cycle. That's the floor for a
9-tap kernel on this ISA (one MULT + one ACC issue per cycle, 9 taps) unless
multiple channels could somehow be computed in the same cycle — which the
ISA doesn't support (`R_ACC` is one accumulator, not banked per-channel).
Benchmark data (`benchmark/results.md`) shows 89–95% MULT-slot utilization
across channel counts, consistent with "essentially every cycle is doing
useful multiply work."

## 10. Where to look next

- `depthwise_conv_universal_bn_activation/` — same pipeline plus a
  bias-seed cycle and ReLU; a good "what changes when you add a feature"
  diff to read against this app.
- `depthwise_conv_stride2_128/` and `depthwise_conv_stride2_narrow/` — reuse
  this app *unmodified* as "stage 1" of a two-stage stride-2 pipeline; see
  the top-level [`../../README.md`](../../README.md) for that pattern.
