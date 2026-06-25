# Universal Depthwise 3x3 Convolution + folded bias + ReLU.
#
# Derived from depthwise_conv_universal.asm. Same walking-pointer / rotating
# cyclic-slot pipeline and deferred-store scheme, with three additions:
#   * per-channel folded bias seeded via MULT.EE ("x1" broadcast from Ra[lr6] * CR1),
#   * ReLU via ACTIVATE relu before quantization,
#   * mask-based top/bottom borders (no cr9 zero region).
#
# Per-channel budget: 11 cyc/ch = 1 bias-seed cycle + 9 weight taps + 1 standalone
# ACTIVATE cycle.  (The ACTIVATE cycle is a placeholder: a future revision folds
# it back into tap 9's acc once ACTIVATE reads r_acc live, restoring 10 cyc/ch.)
#
# KERNEL PACKING (FPB=25, 10-byte stride):  each channel s in a 256-byte
# super-block occupies bytes [s*10 .. s*10+10): byte 0 = bias, 1..9 = taps.
# The kernel byte index lr6 walks +1 every cycle, so the 10-cycle body advances
# lr6 by exactly one channel stride; the bias cycle reads fixed_idx = s*10, the
# 9 taps read s*10+1 .. s*10+9.
#
# CR registers (set by harness; master-ISA CR remap):
#   cr0  = read-only 0 (zero constant)
#   cr1  = read-only 1 (used as bias multiplier for MULT.EE and for sub/set immediates)
#   cr2  = output base - 128 (deferred-store pre-bias)
#   cr3  = mask blob base (single blob, slots 0/3/6; loaded once at init)
#   cr4  = cols
#   cr5  = kernel base
#   cr6  = group_stride (= channels * 128)
#   cr7  = FPB*128 (= 25*128 = 3200; channel-group inner-loop size in bytes)
#   cr8  = total_kernel_bytes (= num_super_blocks * 256)
#   cr9  = 384 (slot-pointer step: +384 mod 512 for the running write ptr lr5)
#   cr10 = input base                (and cyclic-load base)
#   cr11 = chunk-loop limit (= (num_chunks - 1) * group_stride)
#   cr12 = 128
#   cr13 = 256
#   cr14 = 256 - 2*cols - 2  (end-of-9 walking step from tap 9 to next bias)
#
# Mask scheme (mask shift for L/R, single 3-slot blob for vertical borders):
#   Left/right edge columns are zeroed by mask_shift (NOT by slots), with
#   CR15.partition = cols so each partition group is one packed spatial row:
#     kc=-1 -> mask_shift = lr9  (+1): zeros START col of each row.
#     kc= 0 -> mask_shift = lr0  ( 0): no shift.
#     kc=+1 -> mask_shift = lr13 (-1): zeros END col of each row.
#   Vertical off-image rows use mask slots from a SINGLE blob (loaded once at
#   init from cr3); no mid-program R_MASK reload, no zero region:
#     slot 0 = all-pass (KEEP) -> interior / kr=0 row taps.
#     slot 3 = top-row zero     -> g0 taps 1-3 (kr=-1, row 0 out of bounds).
#     slot 6 = bottom-row zero  -> gN taps 7-9 (kr=+1, last row out of bounds).
#   The kc shift adds the edge column on top of the row zero.
#
# LR allocation:
# Slot pointers: lr4 = read pointer (computation; rotated +256 mod 512 at tap 8).
#   lr5 = the SOLE running write/load-slot pointer, advanced only by
#   incr_mod_pow2 (mod 512) so it never overflows the 512-byte cyclic register;
#   per channel it steps +384, +128, +256 (taps 4, 5, 9) — net +256 = lr4's
#   rotation.  lr5 is self-contained: it is NOT recomputed from lr4.
#
#   lr0=0  lr1=cols-2  lr2=this-ch kr=0 ext  lr3=walk  lr4=read  lr5=write slot
#   lr6=kernel byte idx  lr7=output ptr  lr8=chunk base  lr9=+1 (kc=-1 shift)
#   lr10=ch byte counter  lr11=clamp limit  lr12=kernel super-block offset
#   lr13=-1 (kc=+1 shift)  lr14=scratch

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr0 cr0;
    ldr_mult_mask_reg   lr0 cr3;;           # load mask blob (slots 0/3/6)

    add                 lr1 lr0 cr4;        # lr1 = cols (temp)
    SET                 lr8 cr0;
    SET                 lr7 cr0;;

    # lr7 = -128 so ch 0's tap-2 advance lands at 0 (scratch store target).
    sub                 lr7 lr7 cr12;;      # lr7 = -128

    DEC                 lr1 2;              # lr1 = cols - 2 (permanent)
    add                 lr9 lr0 cr1;        # lr9 = +1 (mask_shift for kc=-1, permanent)
    sub                 lr13 lr0 cr1;;      # lr13 = -1 (mask_shift for kc=+1, permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — kr=-1 row masked (slots 3/4/5).
# ===========================================================================

    SET                 lr12 cr0;;

g0_kg_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    SET                 lr10 cr0;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 g0_clamp;;
    b                   g0_ch_pre;;
g0_clamp:
    add                 lr11 lr0 cr6;;

g0_ch_pre:
    # Preamble (4 cyc): set up rotating slots for first ch of this kernel-group.

    # Cy 1: lr4 = 0; lr5 = 0; lr2 = lr8+lr10 (THIS-ch kr=0 ext).
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr5 = 128; ext = lr2+cr6 (kr=+1); lr3 = 1 (seed for -255).
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    SET                 lr3 cr1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols → cols-255.  lr6 = -1 (bias cycle's +1 -> 0 = ch0
    # bias byte index; the 10-cycle body then advances lr6 by exactly 10/channel).
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 cr1;;

    b                   g0_tap_body;;

g0_tap_body:
    # 11 cyc/ch.  cyc0 = bias seed; taps 1..9 = weights; cyc10 = ACTIVATE.  G0: kr=-1 row masked.
    #
    # Quantize pipeline (2-deep, no extra cycle):
    #   ch K tap 9:    acc (final) + ACTIVATE relu   -> post_aaq_reg = relu(r_acc)
    #   ch K+1 cyc 0:  aaq                            -> latch = clamp(post_aaq_reg)
    #                  (aaq does NOT read r_acc, so co-issues with the bias acc.first)
    #   ch K+1 tap 2:  deferred store of ch K's latch
    # ch 0's cyc-0 aaq quantizes undefined data into the latch; ch 0's tap-2 store
    # writes it to the -128 scratch slot, so it is harmless.

    # --- cyc 0: BIAS seed + quantize PREVIOUS channel.
    # lr6 += 1 -> s*10 (this ch's bias byte index; the +1 here plus +1 on each of
    # taps 1..9 = +10 total per channel = one 10-byte channel slot).  Bias's
    # MULT.EE reads Ra[lr6]*CR1=1, broadcasting the bias byte to all lanes.
    # lr3 is NOT walked here; tap 1 keeps the base's +cr14 walk.  aaq quantizes prev ch.
    # Also pre-load this ch's kr=-1 (own valid base lr2) into slot lr5 here, one
    # cycle before tap 1 reads it (snapshot; cyc-0's LOAD slot is free).
    INC                 lr6 1;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    MULT.EE             lr6 cr1 0 lr0;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  Top row out of bounds: slot 3; kc=-1 shift (lr9)
    #     zeros the left edge column.  kr=-1 (own valid base lr2) pre-loaded in
    #     cyc 0; row-0 lanes masked.  Walk +cr14 (as base).
    add                 lr3 lr3 cr14;
    INC                 lr6 1;
    MULT.RC.VE          lr3 lr6 3 lr9;
    acc;;

    # --- tap 2: kr=-1 kc=0.  slot 3 = top row only, no shift.  Deferred store + lr7 advance.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr7 lr7 cr12;
    STR_POST_AAQ_REG    lr7 cr2;
    MULT.RC.VE          lr3 lr6 3 lr0;
    acc;;

    # --- tap 3: kr=-1 kc=+1.  slot 3 + kc=+1 shift (lr13).  Advance lr2 → NEXT-ch kr=0 ext.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr2 lr2 cr12;
    MULT.RC.VE          lr3 lr6 3 lr13;
    acc;;

    # --- tap 4: kr=0 kc=-1.  slot 0 + kc=-1 shift.  lr5 += 384 (mod 512) → R+256;
    #     load NEXT-ch kr=0 → slot lr5.
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr9 9;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 5: kr=0 kc=0.  slot 0, no shift.  lr5 += 128 (mod 512) → R+384.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr12 9;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 6: kr=0 kc=+1.  slot 0 + kc=+1 shift.  Loop counter += 128.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr10 lr10 cr12;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  slot 0 + kc=-1 shift.  Load NEXT-ch kr=+1 → slot lr5 (= R+384).
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 8: kr=+1 kc=0.  slot 0, no shift.  Rotate lr_read.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 9: kr=+1 kc=+1.  slot 0 + kc=+1 shift.  Final acc.  lr5 += 256
    #     (mod 512) → R+128 = next ch's R'-128 (its tap-1 kr=-1 slot).  lr6 now =
    #     s*10 + 9; +1 at next cyc0 -> (s+1)*10.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr13 9;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- cyc 10: ACTIVATE only.  ACTIVATE reads the cycle-start SNAPSHOT r_acc,
    #     which now holds tap 9's just-finalized accumulator (acc ran in the
    #     previous VLIW word).  This standalone cycle costs +1 cyc/ch (11 total);
    #     a future revision can fold ACTIVATE back into tap 9's acc once the
    #     emulator reads r_acc live there.  The per-channel loop branch lives here.
    ACTIVATE            relu 1;
    blt                 lr10 lr11 g0_tap_body;;

    # All channels in this kernel-group done.  The last channel's ACTIVATE has
    # run (cyc 10) but its aaq + store are still pending; they fire on the NEXT
    # body's cyc-0 aaq / tap-2 store (next group, same or next chunk), or at the
    # program epilogue for the very last channel.
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 g0_reload;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 g0_reload_clamp;;
    b                   g0_ch_pre;;
g0_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   g0_ch_pre;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from cr10, no masks.
# ===========================================================================

main_setup:
    add                 lr8 lr0 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

row_loop:
    SET                 lr12 cr0;;

kg_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    SET                 lr10 cr0;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 mn_clamp;;
    b                   ch_pre;;
mn_clamp:
    add                 lr11 lr0 cr6;;

ch_pre:
    # Cy 1: lr2 = lr8+lr10; load ch kr=0 → slot 0.
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr5 = 128; lr3 = 1 (seed for -255); load ch kr=+1 → slot 128.
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    SET                 lr3 cr1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols; lr6 = -1 (bias cycle +1 -> 0).
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 cr1;;

    b                   mn_tap_body;;

mn_tap_body:
    # 11 cyc/ch (cyc10 = ACTIVATE).  All loads use cr10; no border masks (slot 0 + shift).

    # --- cyc 0: BIAS seed + quantize PREVIOUS channel (lr3 not walked here).
    #     Also pre-load THIS-ch kr=-1 (ext = lr2-cr6) into its slot lr5 here, one
    #     cycle before tap 1 reads it — under snapshot a same-cycle LDR is not
    #     visible to the mult, and cyc-0's LOAD slot is free.  lr5 is already the
    #     kr=-1 slot; lr2 already holds THIS-ch kr=0 ext.  Keeps 11 cyc/ch.
    INC                 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    MULT.EE             lr6 cr1 0 lr0;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (lr9).  kr=-1 pre-loaded in
    #     cyc 0.  Walk +cr14.
    add                 lr3 lr3 cr14;
    INC                 lr6 1;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 2: kc=0, no shift.  Deferred store + lr7 advance.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr7 lr7 cr12;
    STR_POST_AAQ_REG    lr7 cr2;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 3: kc=+1 shift (lr13).  Advance lr2 → NEXT-ch kr=0 ext.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr2 lr2 cr12;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- tap 4: kr=0 kc=-1 shift.  NEXT-ch kr=0 ext = lr2 LIVE.
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr9 9;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 5: kc=0, no shift.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr12 9;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 6: kc=+1 shift.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr10 lr10 cr12;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- tap 7: kr=+1 kc=-1 shift.  NEXT-ch kr=+1 ext = lr2+cr6.
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 8: kc=0, no shift.  rotate lr_read.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 9: kc=+1 shift.  Final acc; prep lr5 for next iter tap 1.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr13 9;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- cyc 10: ACTIVATE only (reads snapshot = tap 9's finalized r_acc).
    #     Standalone cycle (+1 cyc/ch); foldable into tap 9 later.  Loop branch here.
    ACTIVATE            relu 1;
    blt                 lr10 lr11 mn_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 mn_reload;;

    add                 lr8 lr8 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

mn_reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 mn_reload_clamp;;
    b                   ch_pre;;
mn_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   ch_pre;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 row masked (slots 3/4/5).
# ===========================================================================

gN_section:
    # No R_MASK reload: the single blob already carries slot 6 (bottom-row zero),
    # selected by the kr=+1 taps below.
    SET                 lr12 cr0;;

gN_kg_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    SET                 lr10 cr0;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 gN_clamp;;
    b                   gN_ch_pre;;
gN_clamp:
    add                 lr11 lr0 cr6;;

gN_ch_pre:
    # Cy 1: lr2 = lr8+lr10; load ch kr=0 → slot 0.
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr5 = 128; lr3 = 1 (seed for -255); load THIS-ch kr=+1 with its own
    # valid base (lr2); bottom row's lanes are masked on the kr=+1 taps.
    add                 lr5 lr5 cr12;
    SET                 lr3 cr1;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols; lr6 = -1.
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 cr1;;

    b                   gN_tap_body;;

gN_tap_body:
    # 11 cyc/ch (cyc10 = ACTIVATE).  NEXT-ch kr=-1/kr=0 from cr10; kr=+1 taps (7/8/9) masked (bottom).

    # --- cyc 0: BIAS seed + quantize PREVIOUS channel (lr3 not walked here).
    #     Also pre-load THIS-ch kr=-1 (ext = lr2-cr6) into its slot lr5 here, one
    #     cycle before tap 1 reads it (snapshot; cyc-0's LOAD slot is free).
    #     Keeps 11 cyc/ch.
    INC                 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    MULT.EE             lr6 cr1 0 lr0;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (lr9).  kr=-1 pre-loaded in
    #     cyc 0.  Walk +cr14.
    add                 lr3 lr3 cr14;
    INC                 lr6 1;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 2: kc=0, no shift.  Deferred store + lr7 advance.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr7 lr7 cr12;
    STR_POST_AAQ_REG    lr7 cr2;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 3: kc=+1 shift (lr13).  Advance lr2 → NEXT-ch kr=0 ext.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr2 lr2 cr12;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- tap 4: kr=0 kc=-1 shift.  NEXT-ch kr=0 from cr10.
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr9 9;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr9;
    acc;;

    # --- tap 5: kc=0, no shift.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr12 9;
    MULT.RC.VE          lr3 lr6 0 lr0;
    acc;;

    # --- tap 6: kc=+1 shift.  Also pre-load the kr=+1 slot (consumed by tap 7)
    #     here, one cycle ahead (snapshot; tap 6's LOAD slot is free).  lr5 is
    #     unchanged between here and tap 7, so it targets the same slot.  Bottom
    #     border: load this ch's own valid base (lr2); bottom row's lanes masked.
    INC                 lr3 1;
    INC                 lr6 1;
    add                 lr10 lr10 cr12;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    MULT.RC.VE          lr3 lr6 0 lr13;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Bottom row out of bounds: slot 6; kc=-1 shift (lr9)
    #     zeros the left edge column.  kr=+1 chunk pre-loaded in tap 6's word.
    add                 lr3 lr3 lr1;
    INC                 lr6 1;
    MULT.RC.VE          lr3 lr6 6 lr9;
    acc;;

    # --- tap 8: kr=+1 kc=0.  slot 6 = bottom row only, no shift.  Rotate lr_read.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    MULT.RC.VE          lr3 lr6 6 lr0;
    acc;;

    # --- tap 9: kr=+1 kc=+1.  slot 6 + kc=+1 shift (lr13).  Final acc; prep lr5
    #     for next iter tap 1.
    INC                 lr3 1;
    INC                 lr6 1;
    incr_mod_pow2       lr5 cr13 9;
    MULT.RC.VE          lr3 lr6 6 lr13;
    acc;;

    # --- cyc 10: ACTIVATE only (reads snapshot = tap 9's finalized r_acc).
    #     Standalone cycle (+1 cyc/ch); foldable into tap 9 later.  Loop branch here.
    ACTIVATE            relu 1;
    blt                 lr10 lr11 gN_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 gN_reload;;

end:
    # Epilogue: the very last channel's ACTIVATE has run (cyc 10), but its aaq +
    # store are still pending.  Quantize it, advance lr7, and store.
    aaq                 1;;
    add                 lr7 lr7 cr12;
    STR_POST_AAQ_REG    lr7 cr2;;

    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr5;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 gN_reload_clamp;;
    b                   gN_ch_pre;;
gN_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   gN_ch_pre;;
