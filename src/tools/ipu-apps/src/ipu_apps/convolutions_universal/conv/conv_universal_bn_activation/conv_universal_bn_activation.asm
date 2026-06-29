{#- Register aliases (named via Jinja preprocessing; expand to lrN/crN). -#}
{%- set lr_zero = "lr0" -%}
{%- set lr_cols_m2 = "lr1" -%}
{%- set lr_off_zero = "lr2" -%}
{%- set lr_walk = "lr3" -%}
{%- set lr_read = "lr4" -%}
{%- set lr_write = "lr5" -%}
{%- set lr_kern_idx = "lr6" -%}
{%- set lr_out_ptr = "lr7" -%}
{%- set lr_chunk_base = "lr8" -%}
{%- set lr_shift_m1 = "lr9" -%}
{%- set lr_ch_ctr = "lr10" -%}
{%- set lr_clamp = "lr11" -%}
{%- set lr_sb_off = "lr12" -%}
{%- set lr_shift_p1 = "lr13" -%}
{%- set lr_scratch = "lr14" -%}
{%- set cr_zero = "cr0" -%}
{%- set cr_out_base = "cr2" -%}
{%- set cr_mask_base = "cr3" -%}
{%- set cr_cols = "cr4" -%}
{%- set cr_kernel_base = "cr5" -%}
{%- set cr_group_stride = "cr6" -%}
{%- set cr_sb_ch_bytes = "cr7" -%}
{%- set cr_total_kern = "cr8" -%}
{%- set cr_ring_adv = "cr9" -%}
{%- set cr_input_base = "cr10" -%}
{%- set cr_chunk_limit = "cr11" -%}
{%- set cr_chunk_bytes = "cr12" -%}
{%- set cr_sb_bytes = "cr13" -%}
{%- set cr_walk_step = "cr14" -%}
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
#   Single offset that walks through 9 MULT.RC.VE taps:
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
#   + step.  MULT.RC.VE at tap K reads lr3 LIVE = tap K's offset.
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
#   lr10 = channel byte counter (for blt loop term)
#   lr11 = clamp limit
#   lr12 = kernel super-block offset
#   lr13 = scratch
#   lr14 = scratch
#   (lr15 free)
#
# Mask scheme (mask shift for L/R, composite slots for vertical borders):
#   Left/right edge columns are zeroed by mask_shift (NOT by slots), with
#   CR15.partition = cols so each partition group is one packed spatial row:
#     kc=-1 -> mask_shift = lr13 (+1): zeros START col of each row.
#     kc= 0 -> mask_shift = lr0  ( 0): no shift.
#     kc=+1 -> mask_shift = lr9  (-1): zeros END col of each row.
#   Vertical off-image rows use mask slots (data-free, no zero region).  A
#   single R_MASK blob (loaded once at init from cr3) carries all 3 slots:
#     slot 0 = all ones (KEEP) -> interior / kr=0 row taps.
#     slot 3 = top-row zero    -> g0 taps 1-3 (kr=-1, row 0 out of bounds).
#     slot 6 = bottom-row zero -> gN taps 7-9 (kr=+1, last row out of bounds).
#   The kc shift adds the edge column on top of the row zero.  No mid-program
#   reload: g0 selects slot 3, gN selects slot 6.

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 {{ lr_zero }} {{ cr_zero }};
    ldr_mult_mask_reg   {{ lr_zero }} {{ cr_mask_base }};;

    add                 {{ lr_cols_m2 }} {{ lr_zero }} {{ cr_cols }};        # {{ lr_cols_m2 }} = cols (temp)
    SET                 {{ lr_chunk_base }} {{ cr_zero }};
    SET                 {{ lr_out_ptr }} {{ cr_zero }};;

    DEC                 {{ lr_cols_m2 }} 2;                           # {{ lr_cols_m2 }} = cols - 2 (permanent)
    ADD                 {{ lr_shift_p1 }} {{ lr_zero }} cr1;         # {{ lr_shift_p1 }} = +1 (mask_shift for kc=-1, permanent)
    SUB                 {{ lr_shift_m1 }} {{ lr_zero }} cr1;;        # {{ lr_shift_m1 }} = -1 (mask_shift for kc=+1, permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — kr=-1 row is zeros (loaded from cr9).
# ===========================================================================

    SET                 {{ lr_sb_off }} {{ cr_zero }};;

g0_filter_loop:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    SET                 {{ lr_ch_ctr }} {{ cr_zero }};
    MULT.EE             {{ lr_kern_idx }} cr1 0 {{ lr_zero }};           # bias = R0[0] * CR1(=1), broadcast
    acc.first;;                            # r_acc = bias (seed; conv taps add on top)

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} g0_clamp;;
    b                   g0_ch_loop;;
g0_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};;

g0_ch_loop:
    # Preamble: load FIRST-ch-of-block's kr=0 (slot 0) and kr=+1 (slot 128).
    # Its kr=-1 will be loaded same-cycle at the body's iter-0 tap 1 (g0:
    # masked via slots 4/3/5).  Seed walking-pointer state.
    #
    # Note: lr10 is NOT reset here.  filter_loop / g0_filter_loop initialise
    # lr10 = 0 for a fresh filter; reload paths keep lr10 at its end-of-body
    # value so the preamble loads the correct channel within the next FPB
    # super-block.

    # Cy 1: lr2 = E0 = ch0 kr=0 ext = lr8+lr10; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} {{ cr_zero }};
    SET                 {{ lr_write }} {{ cr_zero }};
    add                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 2: lr_write = 384; load ch0 kr=-1 (ext = E0 - cr6) → slot 384 (masked by slot 3
    #     in body, value don't-care).  lr_walk = -384.
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_ring_adv }};
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr_ring_adv }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} cr1;;

    # Cy 4: lr_walk = cols - 383.  lr_write := lr_read+128 = 128.  lr_kern_idx := 0.
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_cols }};
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_chunk_bytes }};
    add                 {{ lr_kern_idx }} {{ lr_zero }} {{ lr_zero }};;  # +1 for bias byte at index 0

    b                   g0_tap_body;;

g0_tap_body:
    # 9 cyc/ch body — role-rotating ring (see mn_tap_body).  G0 specialization: kr=-1
    # taps (1-3) mask the top row via mask slot 3.  Loads/lr_write identical to mn.
    # All channels in chunk 0 have kr=-1 out of bounds (masked), so the kr=-1 data is
    # don't-care.  THIS ch kr=+1 (real) is loaded at tap 1, read at taps 7-9.

    # --- tap 1: kr=-1 kc=-1.  slot 3 zeros row 0; kc=-1 shift.  Load THIS ch kr=+1
    #     (ext = lr_off_zero + cr6) into the kr=+1 slot (lr_write = lr_read+128).
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_p1 }};
    acc;;

    # --- tap 2: kr=-1 kc=0.  slot 3, no shift.  Bump lr_off_zero to E'.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_zero }};
    acc;;

    # --- tap 3: kr=-1 kc=+1.  slot 3 + kc=+1 shift.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_m1 }};
    acc;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - cr6) into the SPARE slot.
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 6.  lr10 += cr12 (loop counter).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Prefetch NEXT kr=0 (ext = E') into THIS dead kr=-1 slot.
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 8.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 9: lr_write += 128 -> next iter tap-1 slot.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} g0_tap_body;;

    # Advance kernel offset; check for more channel groups.
    add                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr_sb_bytes }};
    blt                 {{ lr_ch_ctr }} {{ cr_group_stride }} g0_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE relu 1;;        # conv+bias -> ReLU -> quantize
    aaq 1;;
    str_post_aaq_reg {{ lr_out_ptr }} {{ cr_out_base }};;

    add                 {{ lr_out_ptr }} {{ lr_out_ptr }} {{ cr_chunk_bytes }};
    blt                 {{ lr_sb_off }} {{ cr_total_kern }} g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} g0_reload_clamp;;
    b                   g0_ch_loop;;
g0_reload_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};
    b                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from cr10.
# ===========================================================================

main_setup:
    add                 {{ lr_chunk_base }} {{ lr_zero }} {{ cr_group_stride }};;
    blt                 {{ lr_chunk_base }} {{ cr_chunk_limit }} row_loop;;

    b                   gN_section;;

row_loop:
    SET                 {{ lr_sb_off }} {{ cr_zero }};;

filter_loop:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    SET                 {{ lr_ch_ctr }} {{ cr_zero }};
    MULT.EE             {{ lr_kern_idx }} cr1 0 {{ lr_zero }};           # bias = R0[0] * CR1(=1), broadcast
    acc.first;;                            # r_acc = bias (seed; conv taps add on top)

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} mn_clamp;;
    b                   ch_loop;;
mn_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};;

ch_loop:
    # Preamble (9-cyc role scheme): load ch0's kr=0 (slot0) and kr=-1 (slot3); ch0's
    # kr=+1 is loaded by the body at tap 1.  lr_read=0; lr_write enters body = 128
    # (= lr_read+128, ch0 kr=+1 slot1).  lr_off_zero = E0 (ch0 kr=0 ext, NOT bumped:
    # the body bumps it at tap 2).  walk seed = cols-383 (for -128 rotation, cr14=384-2cols-2).

    # Cy 1: lr2 = E0 = ch0 kr=0 ext = lr8+lr10; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} {{ cr_zero }};
    SET                 {{ lr_write }} {{ cr_zero }};
    add                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 2: lr_write = 384 (= cr_ring_adv); ext = E0 - cr6 (ch0 kr=-1) → slot 384.
    #     lr_walk = -384 (= -cr_ring_adv; toward cols-383 via +1 then +cols below).
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_ring_adv }};
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr_ring_adv }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} cr1;;

    # Cy 4: lr_walk = cols - 383 (tap-1 seed).  lr_write := lr_read+128 = 128.
    #     lr_kern_idx := 0 (bias byte index).  lr_off_zero stays = E0.
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_cols }};
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_chunk_bytes }};
    add                 {{ lr_kern_idx }} {{ lr_zero }} {{ lr_zero }};;  # +1 for bias byte at index 0

    b                   mn_tap_body;;

mn_tap_body:
    # 9 cyc/ch — all loads use cr10.  Role-rotating 4-slot ring; lr_read (kr=0 slot)
    # advances -128 (= +384 mod 512) per channel.  The walk is UNCHANGED from the
    # 10-cyc version (cols-stride, masked straddle).  Slot lifetimes (user): kr=0 slot
    # live ALL 9 taps (NEVER a load target); kr=-1 slot live taps 1-3; kr=+1 slot live
    # taps 7-9.  Per loop the 3 loads NEVER touch the kr=0 slot:
    #   tap 1: load THIS ch kr=+1 (read taps 7-9 this loop) -> kr=+1 slot (lr_read+128).
    #   tap 4: prefetch NEXT ch kr=-1 -> the SPARE slot (lr_read+256).
    #   tap 7: prefetch NEXT ch kr=0  -> THIS dead kr=-1 slot (lr_read+384).
    # lr_write threads a uniform +128 mod 512 chain (enter = lr_read+128).  lr_read is
    # dead (chain self-sustains).  lr_off_zero holds THIS kr=0 ext at tap 1 (for THIS
    # kr=+1 = +cr6), then is bumped to E' (NEXT kr=0 ext) at tap 2 for taps 4/7.

    # --- tap 1: kr=-1 kc=-1.  Load THIS ch kr=+1 (ext = lr_off_zero + cr6) into the
    #     kr=+1 slot (lr_write = lr_read+128).  Read by taps 7-9 this loop (snapshot OK).
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 2.  Bump lr_off_zero to E' = NEXT ch kr=0 ext (for taps 4/7 prefetch).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 3.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - cr6) into the SPARE slot
    #     (lr_write = lr_read+256 after the tap-3 +128).
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 6.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Prefetch NEXT kr=0 (ext = E' = lr_off_zero LIVE) into
    #     THIS dead kr=-1 slot (lr_write = lr_read+384 after the tap-5 +128).
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 8.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 9: lr_write += 128 -> next iter tap-1 slot (= next lr_read + 128).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} mn_tap_body;;

mn_after_block:
    add                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr_sb_bytes }};
    blt                 {{ lr_ch_ctr }} {{ cr_group_stride }} reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE relu 1;;        # conv+bias -> ReLU -> quantize
    aaq 1;;
    str_post_aaq_reg {{ lr_out_ptr }} {{ cr_out_base }};;

    add                 {{ lr_out_ptr }} {{ lr_out_ptr }} {{ cr_chunk_bytes }};
    blt                 {{ lr_sb_off }} {{ cr_total_kern }} filter_loop;;

    add                 {{ lr_chunk_base }} {{ lr_chunk_base }} {{ cr_group_stride }};;
    blt                 {{ lr_chunk_base }} {{ cr_chunk_limit }} row_loop;;

    b                   gN_section;;

reload:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} mn_reload_clamp;;
    b                   ch_loop;;
mn_reload_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};
    b                   ch_loop;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 out-of-bounds row zeroed by
#            mask slot 6 (no R_MASK reload; one blob carries all 3 slots).
# ===========================================================================

gN_section:
    # No R_MASK reload: the single blob loaded at init carries slot 0 (none),
    # slot 3 (top-row zero, used by g0), and slot 6 (bottom-row zero, used here).
    # gN's kr=+1 taps select slot 6 directly.
    SET                 {{ lr_sb_off }} {{ cr_zero }};;

gN_filter_loop:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    SET                 {{ lr_ch_ctr }} {{ cr_zero }};
    MULT.EE             {{ lr_kern_idx }} cr1 0 {{ lr_zero }};           # bias = R0[0] * CR1(=1), broadcast
    acc.first;;                            # r_acc = bias (seed; conv taps add on top)

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} gN_clamp;;
    b                   gN_ch_loop;;
gN_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};;

gN_ch_loop:
    # Preamble: load first-ch-of-block kr=0 (real) and kr=+1 (real lr2; bottom row masked).
    # Mirrors g0_ch_loop/ch_loop: lr2 = lr8+lr10 so reload paths load the
    # correct channel offset within each super-block.

    # Cy 1: lr2 = E0 = ch0 kr=0 ext = lr8+lr10; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} {{ cr_zero }};
    SET                 {{ lr_write }} {{ cr_zero }};
    add                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 2: lr_write = 384; load ch0 kr=-1 (real, ext = E0 - cr6) → slot 384.
    #     lr_walk = -384.
    #     (lr10 is NOT reset here — filter_loop sets it to 0 for a fresh filter;
    #      reload re-enters here with lr10 at its end-of-block value.)
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_ring_adv }};
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr_ring_adv }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} cr1;;

    # Cy 4: lr_walk = cols - 383.  lr_write := 128.  lr_kern_idx := 0.
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_cols }};
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_chunk_bytes }};
    add                 {{ lr_kern_idx }} {{ lr_zero }} {{ lr_zero }};;  # +1 for bias byte at index 0

    b                   gN_tap_body;;

gN_tap_body:
    # 9 cyc/ch body — role-rotating ring (see mn_tap_body).  gN specialization: kr=+1
    # taps (7-9) mask the BOTTOM packed row via mask slot 6.  Only that one row is out
    # of bounds; the other kr=+1 neighbours are real, so THIS ch kr=+1 is loaded with
    # real data (lr_off_zero + cr6) at tap 1, like mn.  Loads/lr_write identical to mn.

    # --- tap 1: kr=-1 kc=-1.  Load THIS ch kr=+1 (ext = lr_off_zero + cr6) into the
    #     kr=+1 slot (lr_write = lr_read+128).
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 2.  Bump lr_off_zero to E'.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 3.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - cr6) into the SPARE slot.
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }};
    acc;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }};
    acc;;

    # --- tap 6.  lr10 += cr12 (loop counter).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }};
    acc;;

    # --- tap 7: kr=+1 kc=-1.  Bottom border: slot 6 zeros the last packed row; kc=-1
    #     shift.  Prefetch NEXT kr=0 (ext = E') into THIS dead kr=-1 slot.
    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_shift_p1 }};
    acc;;

    # --- tap 8: kr=+1 kc=0.  slot 6 = bottom row only.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_zero }};
    acc;;

    # --- tap 9: kr=+1 kc=+1.  slot 6 + kc=+1 shift.  lr_write += 128 -> next tap-1 slot.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_shift_m1 }};
    acc;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} gN_tap_body;;

    add                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr_sb_bytes }};
    blt                 {{ lr_ch_ctr }} {{ cr_group_stride }} gN_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    ACTIVATE relu 1;;        # conv+bias -> ReLU -> quantize
    aaq 1;;
    str_post_aaq_reg {{ lr_out_ptr }} {{ cr_out_base }};;

    add                 {{ lr_out_ptr }} {{ lr_out_ptr }} {{ cr_chunk_bytes }};
    blt                 {{ lr_sb_off }} {{ cr_total_kern }} gN_filter_loop;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} gN_reload_clamp;;
    b                   gN_ch_loop;;
gN_reload_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};
    b                   gN_ch_loop;;
