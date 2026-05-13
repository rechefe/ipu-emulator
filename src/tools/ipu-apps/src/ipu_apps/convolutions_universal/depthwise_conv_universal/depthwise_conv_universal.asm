# Universal Depthwise 3x3 Convolution — walking-pointer / rotating-slot variant.
#
# Mirrors conv_universal.asm's pipeline for the depthwise case: same walking
# pointer, same rotating cyclic-slot scheme, same FPB=28 super-block kernel
# packing.  The only structural difference vs. standard conv: each channel
# produces an independent output (no out_filter loop accumulating across
# in-channels), so each channel's r_acc must be stored to memory.
#
# Per-channel cycle budget: 10 cyc/ch.
#   Taps 1-9 = standard conv_universal body (kr=-1 load at tap 1; NEXT-ch
#   kr=0 / kr=+1 loads at taps 4 / 7; lr_read rotation at tap 8).  Tap 10 is
#   a dedicated store cycle: XMEM = str_acc_reg, LR advances lr7, MULT/ACC nop.
#
#   Why 10, not 9?  The store reads r_acc LIVE at XMEM stage; ACC's `acc.first`
#   for the next channel overwrites r_acc same-cycle.  But tap 1's XMEM is
#   already used by the kr=-1 load (which must complete before tap 1's mult
#   reads RC, and overwrites a slot that only becomes free after the previous
#   channel's tap 9).  Both XMEM uses are needed and conflict, so we add a
#   10th cycle.
#
# Supported configurations:
#   - cols ∈ {16, 32, 64}; rows*cols >= 256 (>= 2 chunks).
#     (cols=128 lives in a separate binary.)
#   - channels >= 1.
#
# CYCLIC REGISTER LAYOUT (rotating slots, lr_read = lr4):
#   The 512-byte cyclic register holds three vertically-neighboring 128-byte
#   chunks of one input channel.  Slot bases (mod 512):
#     kr=-1 chunk → lr_read - 128
#     kr= 0 chunk → lr_read
#     kr=+1 chunk → lr_read + 128
#   At each new input channel lr_read advances by +256 mod 512.
#
# WALKING POINTER lr3 (= lr_walk):
#     tap 1 (kr=-1 kc=-1): lr_read - cols - 1
#     tap 2 (kr=-1 kc= 0): lr_read - cols
#     tap 3 (kr=-1 kc=+1): lr_read - cols + 1
#     tap 4 (kr= 0 kc=-1): lr_read - 1
#     tap 5 (kr= 0 kc= 0): lr_read
#     tap 6 (kr= 0 kc=+1): lr_read + 1
#     tap 7 (kr=+1 kc=-1): lr_read + cols - 1
#     tap 8 (kr=+1 kc= 0): lr_read + cols
#     tap 9 (kr=+1 kc=+1): lr_read + cols + 1
#   Steps:  tap 1: +cr14 (= 256-2cols-2);  taps 4, 7: +lr1 (= cols-2);
#           all other taps: +1.  Tap 10: no walk (lr3 holds tap-9 value).
#   Preamble seed: lr3 = (tap 1 offset) - cr14 = cols - 255.
#
# LOAD SCHEDULE (per channel iteration):
#   tap 1: load THIS-ch kr=-1 → slot lr_read - 128.
#          ext = lr2 - cr6 (lr2 already holds THIS-ch's kr=0 ext from prev
#          iter's tap-3 advance, or from preamble for first ch of group).
#   tap 4: load NEXT-ch kr=0 → slot lr_read + 256.
#          ext = lr2 LIVE (just advanced at tap 3 to NEXT-ch's kr=0 ext).
#   tap 7: load NEXT-ch kr=+1 → slot lr_read + 384.
#          ext = lr14 = lr2 + cr6.
#
# STORE (tap 10):  LR `add lr7 lr7 cr10` (commits to LIVE before XMEM stage),
#   then XMEM `str_acc_reg lr7 cr2` writes r_acc to Memory[lr7+cr2] using the
#   updated lr7.  Hence lr7 must start at -cr10 (= -512) so that ch 0's
#   tap-10 advance lands at output offset 0.
#
# CR registers (set by harness):
#   cr0  = input base
#   cr1  = kernel base
#   cr2  = output base
#   cr3  = mask base
#   cr4  = cols
#   cr5  = num_chunks
#   cr6  = group_stride (= channels * 128)
#   cr7  = FPB*128 (= 28*128 = 3584; channel-group inner-loop size in bytes)
#   cr8  = total_kernel_bytes (= channel_groups * 256)
#   cr9  = zero region address (128 bytes of zeros, for top/bottom border)
#   cr10 = 512  (output-pointer step per channel)
#   cr11 = chunk-loop limit (= (num_chunks - 1) * group_stride)
#   cr12 = 128
#   cr13 = 256
#   cr14 = 256 - 2*cols - 2  (end-of-9 walking step)
#
# LR allocation:
#   lr0  = 0            (zero constant)
#   lr1  = cols - 2     (init at startup, permanent)
#   lr2  = THIS-ch kr=0 ext addr at tap 1; advanced at tap 3 to NEXT-ch's
#   lr3  = lr_walk
#   lr4  = lr_read
#   lr5  = cyclic slot index for current load
#   lr6  = kernel byte index (fixed_idx, 0..251 per super-block)
#   lr7  = output pointer
#   lr8  = chunk base address
#   lr10 = channel byte counter (for blt loop term)
#   lr11 = clamp limit (channel-group inner limit)
#   lr12 = kernel super-block offset
#   lr14 = scratch (kr=-1 ext at tap 1; kr=+1 ext at tap 7)
#
# Mask slots: 0=no mask, 1=zero left col (kc=-1), 2=zero right col (kc=+1).

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;
    ldr_mult_mask_reg   lr0 cr3;;

    add                 lr1 lr0 cr4;        # lr1 = cols (temp)
    set                 lr8 0;
    sub                 lr7 lr0 cr10;;      # lr7 = -512 (tap-10 LR adds cr10 → 0 for ch 0)

    sub                 lr1 lr1 2;;         # lr1 = cols - 2 (permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — kr=-1 row is zeros (loaded from cr9).
# ===========================================================================

    set                 lr12 0;;

g0_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    set                 lr10 0;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 g0_clamp;;
    b                   g0_ch_pre;;
g0_clamp:
    add                 lr11 lr0 cr6;;

g0_ch_pre:
    # Preamble (4 cyc): set up rotating slots for first ch of this kernel-group.
    # G0: kr=-1 always loads zeros from cr9.

    # Cy 1: lr4 = 0; lr5 = 0; lr2 = lr8+lr10 (THIS-ch kr=0 ext).
    # Load THIS-ch kr=0 → slot 0.
    set                 lr4 0;
    set                 lr5 0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr0 lr5;;

    # Cy 2: lr5 = 128; ext = lr2+cr6 (kr=+1).  Load → slot 128.
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr5;;

    # Cy 3: lr3 = -255; lr5 = 384 (= lr_read - 128 mod 512, slot for tap-1
    # kr=-1 load of first ch).
    set                 lr3 -255;
    set                 lr5 384;;

    # Cy 4: walk seed += cols → cols-255 (one cr14 short of tap-1 offset).
    # lr6 = -1 (so tap 1's `add lr6 lr6 1` brings it to 0).
    add                 lr3 lr3 cr4;
    set                 lr6 -1;;

    b                   g0_tap_body;;

g0_tap_body:
    # 10 cyc/ch body.  G0 specialization: kr=-1 loads always come from cr9.

    # --- tap 1: kr=-1 kc=-1.  Load THIS-ch kr=-1 from cr9 → slot lr5.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr0 cr9 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc.first;;

    # --- tap 2.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3.  Advance lr2 → NEXT-ch kr=0 ext.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1.  Load NEXT-ch kr=0 → slot lr5 (= lr_read+256).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr0 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 5.  lr5 += cr12 → lr_read+384.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6.  Loop counter.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Load NEXT-ch kr=+1 → slot lr5 (= lr_read+384).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 8.  Rotate lr_read.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9.  Compute lr5 = lr_read(N+1) - 128 for next iter's tap-1
    # kr=-1 slot.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 10 (STORE).  Store this ch's r_acc; advance lr7.
    add                 lr7 lr7 cr10;
    str_acc_reg         lr7 cr2;
    blt                 lr10 lr11 g0_tap_body;;

    # All channels in this kernel-group done.  Advance kernel super-block.
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 g0_reload;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 g0_reload_clamp;;
    b                   g0_ch_pre;;
g0_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   g0_ch_pre;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from cr0.
# ===========================================================================

main_setup:
    add                 lr8 lr0 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

row_loop:
    set                 lr12 0;;

kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    set                 lr10 0;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 mn_clamp;;
    b                   ch_pre;;
mn_clamp:
    add                 lr11 lr0 cr6;;

ch_pre:
    # Preamble: set up first ch.  kr=-1 loaded at body's tap 1 from cr0.

    # Cy 1: lr2 = lr8+lr10; load ch kr=0 → slot 0.
    set                 lr4 0;
    set                 lr5 0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr0 lr5;;

    # Cy 2: lr5 = 128; load ch kr=+1 → slot 128.
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr5;;

    # Cy 3: lr3 = -255; lr5 = 384 (slot for tap-1 kr=-1 load).
    set                 lr3 -255;
    set                 lr5 384;;

    # Cy 4: walk seed += cols; lr6 = -1.
    add                 lr3 lr3 cr4;
    set                 lr6 -1;;

    b                   mn_tap_body;;

mn_tap_body:
    # All loads use cr0.

    # --- tap 1: kr=-1 kc=-1.  Load THIS-ch kr=-1 ext = lr2-cr6.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc.first;;

    # --- tap 2.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 4: NEXT-ch kr=0 ext = lr2 LIVE.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr0 lr5;
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
    ldr_cyclic_mult_reg lr14 cr0 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 8.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 10 (STORE).
    add                 lr7 lr7 cr10;
    str_acc_reg         lr7 cr2;
    blt                 lr10 lr11 mn_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 mn_reload;;

    add                 lr8 lr8 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

mn_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 mn_reload_clamp;;
    b                   ch_pre;;
mn_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   ch_pre;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 row is zeros (loaded from cr9).
# ===========================================================================

gN_section:
    set                 lr12 0;;

gN_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    set                 lr10 0;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 gN_clamp;;
    b                   gN_ch_pre;;
gN_clamp:
    add                 lr11 lr0 cr6;;

gN_ch_pre:
    # Preamble: load THIS-ch kr=0 from cr0; THIS-ch kr=+1 from cr9 (zeros);
    # tap-1 will load THIS-ch kr=-1 from cr0.

    # Cy 1: lr2 = lr8+lr10; load ch kr=0 → slot 0.
    set                 lr4 0;
    set                 lr5 0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr0 lr5;;

    # Cy 2: lr5 = 128; load THIS-ch kr=+1 from cr9 (zeros).
    add                 lr5 lr5 cr12;
    ldr_cyclic_mult_reg lr0 cr9 lr5;;

    # Cy 3: lr3 = -255; lr5 = 384.
    set                 lr3 -255;
    set                 lr5 384;;

    # Cy 4.
    add                 lr3 lr3 cr4;
    set                 lr6 -1;;

    b                   gN_tap_body;;

gN_tap_body:
    # NEXT-ch kr=-1 / kr=0 from cr0; NEXT-ch kr=+1 from cr9 (zeros).

    # --- tap 1: load THIS-ch kr=-1 from cr0.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc.first;;

    # --- tap 2.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 4: NEXT-ch kr=0 from cr0.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr0 lr5;
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

    # --- tap 7: NEXT-ch kr=+1 from cr9 (zeros).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr0 cr9 lr5;
    mult.ve.cyclic      lr3 1 lr0 lr6;
    acc;;

    # --- tap 8.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 2 lr0 lr6;
    acc;;

    # --- tap 10 (STORE).
    add                 lr7 lr7 cr10;
    str_acc_reg         lr7 cr2;
    blt                 lr10 lr11 gN_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 gN_reload;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr14 lr12 cr12;
    ldr_mult_reg        r1 lr14 cr1;;

    add                 lr11 lr10 cr7;;
    blt                 cr6 lr11 gN_reload_clamp;;
    b                   gN_ch_pre;;
gN_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   gN_ch_pre;;
