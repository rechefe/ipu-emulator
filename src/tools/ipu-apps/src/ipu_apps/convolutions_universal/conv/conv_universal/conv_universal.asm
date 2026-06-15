# Universal Standard 3x3 Convolution — walking-pointer / rotating-slot variant.
#
# Supported configurations:
#   - cols ∈ {16, 32, 64}; rows*cols >= 256 (>= 2 chunks).
#     (cols=128 lives in a separate binary.)
#   - in_channels >= 1, out_channels >= 1.
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
#   Single offset that walks through 9 mult.ve.cyclic taps:
#     tap 1 (kr=-1 kc=-1): lr_read - cols - 1
#     tap 2 (kr=-1 kc= 0): lr_read - cols
#     tap 3 (kr=-1 kc=+1): lr_read - cols + 1
#     tap 4 (kr= 0 kc=-1): lr_read - 1
#     tap 5 (kr= 0 kc= 0): lr_read
#     tap 6 (kr= 0 kc=+1): lr_read + 1
#     tap 7 (kr=+1 kc=-1): lr_read + cols - 1
#     tap 8 (kr=+1 kc= 0): lr_read + cols
#     tap 9 (kr=+1 kc=+1): lr_read + cols + 1
#
#   Convention: at tap K, the LR sub-slot writes lr3 = (tap K-1's offset)
#   + step.  mult.ve.cyclic at tap K reads lr3 LIVE = tap K's offset.
#   Steps: tap 1: +cr14 (= 256-2cols-2, end-of-9 from prev ch's tap 9);
#          taps 4, 7: +lr1 (= cols - 2);
#          all other taps: +1.
#
#   Preamble seed: lr3 = (tap 1 offset) - cr14 = cols - 255.
#
# LOAD SCHEDULE (each load same-cycle with the mult that needs it):
#   tap 1: load THIS ch's kr=-1 → slot lr_read(N)-128 mod 512
#          (tap 1 mult reads kr=-1 in same cycle, sees fresh data)
#   tap 4: load NEXT ch's kr=0 → slot lr_read(N)+256
#          (= NEXT ch's kr=0 slot under +256 rotation)
#   tap 7: load NEXT ch's kr=+1 → slot lr_read(N)+384 (= lr_read(N)-128 mod 512)
#          (= NEXT ch's kr=+1 slot; physically same slot as THIS ch's kr=-1
#          slot, but kr=-1 is dead since tap 4)
#
#   NEXT ch's kr=0 chunk is loaded one iter ahead (at iter N tap 4) so iter
#   N+1's tap 1 mult — which spans kr=-1 (just loaded same-cycle) and kr=0
#   (loaded one iter ago) — sees correct data.  Same for kr=+1 (loaded at
#   iter N's tap 7, used by iter N+1's taps 7-9).
#
# lr5 (lr_write) values at each load tap, computed fresh from lr_read:
#   tap 1: lr5 = lr_read - cr12  (= lr_read - 128 = lr_read + 384 mod 512)
#   tap 4: lr5 = lr_read + cr13  (= lr_read + 256)
#   tap 7: lr5 = (tap 4 lr5) + cr12 = lr_read + 384
#   The tap-1 compute is done at the prev iter's tap 9 (so lr5 LIVE at tap 1
#   = lr_read(N)-128, where lr_read was just rotated at tap 8).
#
# lr2 (= lr_off_zero) holds THIS ch's kr=0 ext addr (= lr8 + lr10(N)).
# Used at: tap 1 (sub cr6 → kr=-1 ext), tap 4 (after +cr12 → NEXT kr=0 ext —
# but the +cr12 is folded by computing tap-4 ext as lr2+cr12 directly).
# Implementation: update lr2 += cr12 at tap 3 so SNAPSHOT lr2 at tap 4
# already = lr8+lr10(N+1).  Tap 1 reads SNAPSHOT lr2 (pre-iter) = lr8+lr10(N).
#
# CR registers (set by harness):
#   cr10..cr11: as before (cr11 = chunk-loop limit)
#   cr12 = 128
#   cr13 = 256
#   cr14 = 256 - 2*cols - 2  (end-of-9 walking step)
#
# LR allocation:
#   lr0  = 0
#   lr1  = cols - 2 (init at startup, permanent)
#   lr2  = lr_off_zero
#   lr3  = lr_walk
#   lr4  = lr_read
#   lr5  = lr_write
#   lr6  = kernel byte index
#   lr7  = output pointer
#   lr8  = chunk base address
#   lr9  = -1 (mask_shift for kc=+1, init at startup, permanent)
#   lr10 = channel byte counter (for blt loop term)
#   lr11 = clamp limit
#   lr12 = kernel super-block offset
#   lr13 = +1 (mask_shift for kc=-1, init at startup, permanent)
#   lr14 = scratch
#
# Mask scheme ("3 masks overall"):
#   Slot 0 = all-ones (KEEP all) -> used by every interior tap.
#   Slot 1 = all-zeros           -> TOP off-image row (g0 taps 1,2,3).
#   Slot 2 = all-zeros           -> BOTTOM off-image row (gN taps 7,8,9).
#   Left/right columns are zeroed by mask_shift (NOT by slots), with
#   CR15.partition = cols so each partition group is one packed spatial row:
#     kc=-1 -> mask_shift = lr13 (+1): partition_vector zeros START of each row.
#     kc= 0 -> mask_shift = lr0  ( 0): no shift.
#     kc=+1 -> mask_shift = lr9  (-1): inverse_partition_vector zeros END of row.
#   Off-image rows in g0/gN load a REAL in-bounds row (no zero region); the
#   whole-row zeroing comes from selecting slot 1 / slot 2 (all-zeros).

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr0 cr0;
    ldr_mult_mask_reg   lr0 cr3;;

    add                 lr1 lr0 cr4;        # lr1 = cols (temp)
    SET                 lr8 cr0;
    SET                 lr7 cr0;;

    sub                 lr1 lr1 2;          # lr1 = cols - 2 (permanent)
    add                 lr13 lr0 1;         # lr13 = +1 (mask_shift for kc=-1, permanent)
    sub                 lr9 lr0 1;;         # lr9  = -1 (mask_shift for kc=+1, permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — kr=-1 off-image row zeroed via mask slot 1.
# ===========================================================================

    SET                 lr12 cr0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    SET                 lr10 cr0;
    reset_acc;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 g0_clamp;;
    b                   g0_ch_loop;;
g0_clamp:
    add                 lr11 lr0 cr6;;

g0_ch_loop:
    # Preamble: load FIRST-ch-of-block's kr=0 (slot 0) and kr=+1 (slot 128).
    # Its kr=-1 will be loaded same-cycle at the body's iter-0 tap 1 (g0:
    # the kr=-1 slot is zeroed by mask slot 1).  Seed walking-pointer state.
    #
    # Note: lr10 is NOT reset here.  filter_loop / g0_filter_loop initialise
    # lr10 = 0 for a fresh filter; reload paths keep lr10 at its end-of-body
    # value so the preamble loads the correct channel within the next FPB
    # super-block.

    # Cy 1: lr_read=0, lr_write=0; lr2 = lr8+lr10 (ch base ext addr).
    # Load ch's kr=0 → slot 0.
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr_write = 128.  Ext = lr2+cr6 (kr=+1) — load → slot 128.
    # lr3 = 1 (seed for -255: next cycle sub lr3 lr3 cr13 → 1-256 = -255).
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4: finalize seeds.  lr3 += cols → cols-255.
    # lr6 = -1 (so tap 1's `add lr6 lr6 1` brings it to 0).
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   g0_tap_body;;

g0_tap_body:
    # 9 cyc/ch body.  G0 specialization: kr=-1 taps (1,2,3) use mask slot 1.
    #
    # Per-cycle plan (LR1 walks lr3, LR2 advances lr6, LR3 = misc):
    #   tap 1: walk +cr14;  +lr6;  (free).  XMEM: load real row into kr=-1 slot lr5.
    #   tap 2: walk +1;     +lr6;  (free).
    #   tap 3: walk +1;     +lr6;  add lr2 lr2 cr12  (NEXT ch kr=0 ext addr).
    #   tap 4: walk +lr1;   +lr6;  add lr5 lr4 cr13  (lr_write := lr_read+256).
    #          XMEM: load NEXT ch kr=0 from cr10, ext = lr2 (post-tap-3 = lr8+lr10(N+1)).
    #   tap 5: walk +1;     +lr6;  add lr5 lr5 cr12  (lr_write := lr_read+384).
    #   tap 6: walk +1;     +lr6;  add lr10 lr10 cr12  (loop counter).
    #   tap 7: walk +lr1;   +lr6;  add lr14 lr2 cr6  (NEXT ch kr=+1 ext addr).
    #          XMEM: load NEXT ch kr=+1 from cr10, ext = lr14.
    #   tap 8: walk +1;     +lr6;  incr_mod_pow2 lr4 cr13 9  (rotate lr_read).
    #   tap 9: walk +1;     +lr6;  sub lr5 lr4 cr12  (lr_write := lr_read(N+1)-128).

    # --- tap 1: kr=-1 kc=-1.  Load this ch's kr=-1 chunk from cr9 (the
    #     off-image chunk above chunk 0 — all zeros).  Vertical top border is
    #     handled by DATA (zero region), since only local_row 0 reads the
    #     off-image neighbour; rows 1..N-1 read their real kr=-1 from the kr=0
    #     chunk via the walking pointer.  Mask slot 0 + kc=-1 shift (lr13)
    #     zeros the left edge column of each packed row.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr0 cr9 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 2: kr=-1 kc=0 → slot 0, no shift.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3: kr=-1 kc=+1 → slot 0 + kc=+1 shift (lr9).  lr2 += cr12 (NEXT ch kr=0).
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1.  Load NEXT ch kr=0 → slot lr5 (= lr_read+256).
    #     XMEM offset = lr2 LIVE (post-tap-3 = lr8+lr10(N+1)).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  lr5 += cr12 → lr_read+384.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1.  lr10 += cr12 (loop counter).
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Load NEXT ch kr=+1 → slot lr5 (= lr_read+384).
    #     XMEM offset = lr14 (= lr2 + cr6 = lr8+lr10(N+1)+cr6).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0.  Rotate lr_read for next iter.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1.  Compute lr_write for NEXT iter's tap 1.
    #     aaq quantizes r_acc -> aaq_result (fires every iter; only last matters).
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;
    blt                 lr10 lr11 g0_tap_body;;

    # Advance kernel offset; check for more channel groups.
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 g0_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE identity 1;;
    aaq 1;;
    str_post_aaq_reg lr7 cr2;;

    add                 lr7 lr7 cr12;
    blt                 lr12 cr8 g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 g0_reload_clamp;;
    b                   g0_ch_loop;;
g0_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from cr10.
# ===========================================================================

main_setup:
    add                 lr8 lr0 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

row_loop:
    SET                 lr12 cr0;;

filter_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    SET                 lr10 cr0;
    reset_acc;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 mn_clamp;;
    b                   ch_loop;;
mn_clamp:
    add                 lr11 lr0 cr6;;

ch_loop:
    # Preamble: see g0_ch_loop.  All loads from cr10.

    # Cy 1: lr2 = lr8+lr10; load ch kr=0 → slot 0.
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr_write = 128; ext = lr2+cr6 (kr=+1).  lr3 = 1 (seed for -255).
    add                 lr5 lr5 cr12;
    add                 lr14 lr2 cr6;
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr14 cr10 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4.
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   mn_tap_body;;

mn_tap_body:
    # All loads use cr10.

    # --- tap 1: kr=-1 kc=-1.  Load this ch kr=-1 ext = lr8+lr10(N)-cr6 = lr2-cr6.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 2: kc=0.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3: kc=+1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 4: NEXT ch kr=0 ext = lr2 LIVE.  kc=-1.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 5: kc=0.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6: kc=+1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 7: NEXT ch kr=+1 ext = lr2+cr6.  kc=-1.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 8: rotate lr_read.  kc=0.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: prep lr5 for next iter tap 1.  kc=+1.
    #     aaq fires every iter; only the final aaq_result (after all in_ch) matters.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;
    blt                 lr10 lr11 mn_tap_body;;

mn_after_block:
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE identity 1;;
    aaq 1;;
    str_post_aaq_reg lr7 cr2;;

    add                 lr7 lr7 cr12;
    blt                 lr12 cr8 filter_loop;;

    add                 lr8 lr8 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 mn_reload_clamp;;
    b                   ch_loop;;
mn_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   ch_loop;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 off-image row zeroed via mask slot 2.
# ===========================================================================

gN_section:
    SET                 lr12 cr0;;

gN_filter_loop:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    SET                 lr10 cr0;
    reset_acc;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 gN_clamp;;
    b                   gN_ch_loop;;
gN_clamp:
    add                 lr11 lr0 cr6;;

gN_ch_loop:
    # Preamble: load first-ch-of-block kr=0 (real) and kr=+1 (real; masked by slot 2).
    # Mirrors g0_ch_loop/ch_loop: lr2 = lr8+lr10 so reload paths load the
    # correct channel offset within each super-block.

    # Cy 1: lr2 = lr8+lr10 (ch base ext addr); load ch kr=0 → slot 0.
    SET                 lr4 cr0;
    SET                 lr5 cr0;
    add                 lr2 lr8 lr10;
    ldr_cyclic_mult_reg lr2 cr10 lr5;;

    # Cy 2: lr5 = 128; lr3 = 1 (seed for -255); load ch kr=+1 (zeros from cr9).
    # kr=+1 is the off-image chunk below the last chunk; only the last packed
    # row reads it.  Vertical bottom border handled by DATA (zero region).
    add                 lr5 lr5 cr12;
    add                 lr3 lr0 1;
    ldr_cyclic_mult_reg lr0 cr9 lr5;;

    # Cy 3: lr3 = 1-256 = -255; lr5 = 128+256 = 384.
    # (lr10 is NOT reset here — filter_loop sets it to 0 for a fresh filter;
    #  reload re-enters here with lr10 at its end-of-block value.)
    sub                 lr3 lr3 cr13;
    add                 lr5 lr5 cr13;;

    # Cy 4.
    add                 lr3 lr3 cr4;
    sub                 lr6 lr0 1;;

    b                   gN_tap_body;;

gN_tap_body:
    # NEXT ch kr=-1 / kr=0 from cr10; kr=+1 taps (7,8,9) use mask slot 2.

    # --- tap 1: this ch kr=-1 from cr10.  kc=-1.
    add                 lr3 lr3 cr14;
    add                 lr6 lr6 1;
    sub                 lr14 lr2 cr6;
    ldr_cyclic_mult_reg lr14 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 2: kc=0.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 3: kc=+1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr2 lr2 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 4: NEXT ch kr=0 from cr10.  kc=-1.
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    add                 lr5 lr4 cr13;
    ldr_cyclic_mult_reg lr2 cr10 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 5: kc=0.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr5 lr5 cr12;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 6: kc=+1.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Load this ch's kr=+1 chunk from cr9 (the
    #     off-image chunk below the last chunk — all zeros).  Vertical bottom
    #     border handled by DATA (zero region): only the last packed row reads
    #     the off-image neighbour.  Mask slot 0 + kc=-1 shift (lr13).
    add                 lr3 lr3 lr1;
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr0 cr9 lr5;
    mult.ve.cyclic      lr3 0 lr13 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0 → slot 0, no shift.  Rotate lr_read.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    incr_mod_pow2       lr4 cr13 9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1 → slot 0 + kc=+1 shift (lr9).  Prep lr5 for next iter.
    #     aaq fires every iter; only the final aaq_result matters.
    add                 lr3 lr3 1;
    add                 lr6 lr6 1;
    sub                 lr5 lr4 cr12;
    mult.ve.cyclic      lr3 0 lr9 lr6;
    acc;
    blt                 lr10 lr11 gN_tap_body;;

    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 gN_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE identity 1;;
    aaq 1;;
    str_post_aaq_reg lr7 cr2;;

    add                 lr7 lr7 cr12;
    blt                 lr12 cr8 gN_filter_loop;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr5;
    SET                 lr6 cr0;;

    add                 lr14 lr12 cr12;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr5;;
    blt                 cr6 lr11 gN_reload_clamp;;
    b                   gN_ch_loop;;
gN_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   gN_ch_loop;;
