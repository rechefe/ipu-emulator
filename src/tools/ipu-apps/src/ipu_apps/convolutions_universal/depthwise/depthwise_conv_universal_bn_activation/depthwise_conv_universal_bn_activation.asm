# Universal Depthwise 3x3 Convolution + folded bias + ReLU.
#
# Derived from depthwise_conv_universal.asm. Same walking-pointer / rotating
# cyclic-slot pipeline and deferred-store scheme, with three additions:
#   * per-channel folded bias seeded via mult.ve.padded ("x1" broadcast),
#   * ReLU via ACTIVATE relu before quantization,
#   * mask-based top/bottom borders (no cr9 zero region).
#
# Per-channel budget: 10 cyc/ch = 1 bias-seed cycle + 9 weight taps.
#
# KERNEL PACKING (FPB=25, 10-byte stride):  each channel s in a 256-byte
# super-block occupies bytes [s*10 .. s*10+10): byte 0 = bias, 1..9 = taps.
# The kernel byte index lr6 walks +1 every cycle, so the 10-cycle body advances
# lr6 by exactly one channel stride; the bias cycle reads fixed_idx = s*10, the
# 9 taps read s*10+1 .. s*10+9.
#
# CR registers (set by harness; master-ISA CR remap):
#   cr0  = read-only 0 (zero constant)
#   cr2  = output base - 128 (deferred-store pre-bias)
#   cr3  = TOP mask blob base       (g0 section)
#   cr4  = cols
#   cr5  = kernel base
#   cr6  = group_stride (= channels * 128)
#   cr7  = FPB*128 (= 25*128 = 3200; channel-group inner-loop size in bytes)
#   cr8  = total_kernel_bytes (= num_super_blocks * 256)
#   cr9  = BOTTOM mask blob base     (gN section reload)
#   cr10 = input base                (and cyclic-load base)
#   cr11 = chunk-loop limit (= (num_chunks - 1) * group_stride)
#   cr12 = 128
#   cr13 = 256
#   cr14 = 256 - 2*cols - 2  (end-of-9 walking step from tap 9 to next bias)
#
# Mask slots (TOP and BOTTOM blobs share numbering; R_MASK reloaded per section):
#   0=none  1=left  2=right  3=vert  4=left+vert  5=right+vert
#
# LR allocation (as base, plus lr15 = 512 pad-offset constant for bias):
#   lr0=0  lr1=cols-2  lr2=this-ch kr=0 ext  lr3=walk  lr4=read  lr5=slot
#   lr6=kernel byte idx  lr7=output ptr  lr8=chunk base  lr10=ch byte counter
#   lr11=clamp limit  lr12=kernel super-block offset  lr14=scratch  lr15=512

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr0 cr0;
    ldr_mult_mask_reg   lr0 cr3;;           # load TOP mask blob

    add                 lr1 lr0 cr4;        # lr1 = cols (temp)
    SET                 lr8 cr0;
    SET                 lr7 cr0;;

    # lr7 = -128 so ch 0's tap-2 advance lands at 0 (scratch store target).
    sub                 lr7 lr7 cr12;;      # lr7 = -128

    sub                 lr1 lr1 2;;         # lr1 = cols - 2 (permanent)

    # lr15 = 512: a padded cyclic_offset >= 512 makes every lane of
    # mult.ve.padded read the dtype-1 pad, so "bias_byte * 1" broadcasts the
    # channel bias across all 128 lanes for the per-channel bias seed.
    SET                 lr15 cr13;;         # lr15 = 256
    add                 lr15 lr15 cr13;;    # lr15 = 512

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
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols → cols-255.  lr6 = -1 (bias cycle's +1 -> 0 = ch0
    # bias byte index; the 10-cycle body then advances lr6 by exactly 10/channel).
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   g0_tap_body;;

g0_tap_body:
    # 10 cyc/ch.  cyc0 = bias seed; taps 1..9 = weights.  G0: kr=-1 row masked.
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
    # mult.ve.padded ignores lr3 (cyclic_offset=lr15>=512), so lr3 is NOT walked
    # here; tap 1 keeps the base's +cr14 walk.  aaq quantizes prev ch.
    add                 lr6 lr6 1;
    mult.ve.padded      lr15 0 lr0 lr6;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  Top border slot 4 (left+top).  Load this ch kr=-1
    #     with its own valid base (lr2); row-0 lanes masked.  Walk +cr14 (as base).
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 4 lr0 lr6;
    acc;;

    # --- tap 2: kr=-1 kc=0.  slot 3 = top row only.  Deferred store + lr7 advance.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr7 lr7 cr12;
    xmem.store_aaq_result lr7 cr2;
    mult.ve.cyclic      lr3 3 lr0 lr6;
    acc;;

    # --- tap 3: kr=-1 kc=+1.  slot 5 = right+top.  Advance lr2 → NEXT-ch kr=0 ext.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 5 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1.  Load NEXT-ch kr=0 → slot lr5 (= lr_read+256).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  lr5 += cr12 → lr_read+384.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1.  Loop counter += 128.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Load NEXT-ch kr=+1 → slot lr5 (= lr_read+384).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0.  Rotate lr_read.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1.  Final acc; ACTIVATE relu reads the just-finalized
    #     r_acc (ACC runs before the AAQ slot in one VLIW word).  Compute lr5 for
    #     next iter tap-1 slot.  lr6 now = s*10 + 9; +1 at next cyc0 -> (s+1)*10.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;
    ACTIVATE            relu 1;
    blt                 lr10 lr11 g0_tap_body;;

    # All channels in this kernel-group done.  The last channel's ACTIVATE has
    # run (tap 9) but its aaq + store are still pending; they fire on the NEXT
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
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols; lr6 = -1 (bias cycle +1 -> 0).
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   mn_tap_body;;

mn_tap_body:
    # 10 cyc/ch.  All loads use cr10; no border masks (slots 1/0/2).

    # --- cyc 0: BIAS seed + quantize PREVIOUS channel (lr3 not walked here).
    add                 lr6 lr6 1;
    mult.ve.padded      lr15 0 lr0 lr6;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  Load THIS-ch kr=-1 ext = lr2-cr6.  Walk +cr14.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 2.  Deferred store + lr7 advance.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr7 lr7 cr12;
    xmem.store_aaq_result lr7 cr2;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3.  Advance lr2 → NEXT-ch kr=0 ext.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 4: NEXT-ch kr=0 ext = lr2 LIVE.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 5.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 7: NEXT-ch kr=+1 ext = lr2+cr6.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 8: rotate lr_read.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9.  Final acc + ACTIVATE relu; prep lr5 for next iter tap 1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;
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
    # Reload R_MASK with the BOTTOM-border composites (cr9 = bottom blob base).
    SET                 lr12 cr0;
    ldr_mult_mask_reg   lr0 cr9;;

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
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: walk seed += cols; lr6 = -1.
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   gN_tap_body;;

gN_tap_body:
    # 10 cyc/ch.  NEXT-ch kr=-1/kr=0 from cr10; kr=+1 taps (7/8/9) masked (bottom).

    # --- cyc 0: BIAS seed + quantize PREVIOUS channel (lr3 not walked here).
    add                 lr6 lr6 1;
    mult.ve.padded      lr15 0 lr0 lr6;
    acc.first;
    aaq                 1;;

    # --- tap 1: kr=-1 kc=-1.  Load THIS-ch kr=-1 ext = lr2-cr6.  Walk +cr14.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 2.  Deferred store + lr7 advance.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr7 lr7 cr12;
    xmem.store_aaq_result lr7 cr2;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3.  Advance lr2 → NEXT-ch kr=0 ext.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 4: NEXT-ch kr=0 from cr10.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 5.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Bottom border slot 4 (left+bottom).  Load kr=+1
    #     slot with this ch's own valid base (lr2); bottom row's lanes masked.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 4 lr0 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0.  slot 3 = bottom row only.  Rotate lr_read.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 3 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1.  slot 5 = right+bottom.  Final acc + ACTIVATE relu;
    #     prep lr5 for next iter tap 1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 5 lr0 lr6;
    acc;
    ACTIVATE            relu 1;
    blt                 lr10 lr11 gN_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 gN_reload;;

end:
    # Epilogue: the very last channel's ACTIVATE has run (tap 9), but its aaq +
    # store are still pending.  Quantize it, advance lr7, and store.
    aaq                 1;;
    add                 lr7 lr7 cr12;
    xmem.store_aaq_result lr7 cr2;;

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
