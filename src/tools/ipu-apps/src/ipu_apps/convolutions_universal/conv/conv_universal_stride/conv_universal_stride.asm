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
{%- set lr_ring_adv = "lr15" -%}
{%- set cr_zero = "cr0" -%}
{%- set cr_out_base = "cr2" -%}
{%- set cr_mask_base = "cr3" -%}
{%- set cr_cols = "cr4" -%}
{%- set cr_kernel_base = "cr5" -%}
{%- set cr_group_stride = "cr6" -%}
{%- set cr_sb_ch_bytes = "cr7" -%}
{%- set cr_total_kern = "cr8" -%}
{%- set cr_temp_base = "cr9" -%}
{%- set cr_input_base = "cr10" -%}
{%- set cr_pair_limit = "cr11" -%}
{%- set cr_chunk_bytes = "cr12" -%}
{%- set cr_sb_bytes = "cr13" -%}
{%- set cr_walk_step = "cr14" -%}
{%- set cr_ds128 = "cr15" -%}
# Universal Standard 3x3 Convolution, stride 2 — cols=128 only (milestone 1).
#
# Same walking-pointer / role-rotating 9-cyc-per-channel body as
# conv_universal.asm, with cols fixed at 128 (one input spatial row == one
# 128-byte chunk; CR15.partition = P0, no sub-row packing) and two changes:
#
#  1. The outer loop only visits EVEN input rows (0, 2, 4, ..., rows-2) as
#     convolution centers (stride 2, pad 1). Because rows is even, the last
#     center row's kr=+1 neighbour (rows-1) is always in-bounds, so there is
#     no bottom-border (gN) case -- only g0 (row 0's kr=-1 off-image, via
#     mask slot 3) is needed.
#
#  2. Output rows are only 64 lanes wide (out_cols = cols/2 = 64), so two
#     output rows pack into one 128-byte output chunk. ACC.STRIDE decimates
#     one MULT_RES row into 64 lanes, but ACC.ADD/ACC.ADD.FIRST always
#     overwrite ALL 128 r_acc lanes -- two independently-accumulated rows'
#     decimated results cannot both live in r_acc at once. So each row's
#     64-lane decimated result is drained through a TEMP xmem scratch region
#     (ACTIVATE.QUANTIZE identity -> STR_POST_AAQ_REG), row 2k -> TEMP[0:64],
#     row 2k+1 -> TEMP[64:128]; once both halves are written, TEMP[0:128] is
#     reloaded (identity MULT + ACC.ADD.FIRST) and stored to the REAL output
#     chunk in one combined ACTIVATE.QUANTIZE(identity) -> STR_POST_AAQ_REG.
#     3 XMEM round-trips per output-row PAIR (not per channel): one per row's
#     drain-and-decimate, one to combine+store.
#
# ACC.STRIDE mechanics (execute_acc_stride): elements_in_row=64 treats the
# 128-lane MULT_RES as 2 fake 64-wide "rows"; horizontal_stride=on halves
# each to 32, giving 64 contiguous output lanes written at r_acc[0:64]
# (offset=0). Source must be MULT_RES, not r_acc directly, hence the
# reload-through-temp step (mirrors conv_first_layer's identical trick).
#
# CR registers (set by harness). CR1 is the ISA-wide read-only constant 1
# (never assignable), so unlike conv_universal.asm, the ring-advance constant
# (384) is computed into an LR once at init instead of taking a CR slot --
# CR space is otherwise identical in spirit but fully packed (cr2..cr15, 14
# registers) since this app needs one more constant (TEMP_BASE_ADDR) than
# conv_universal.asm does:
#   cr0 = 0 (read-only)           cr2 = OUTPUT_BASE_ADDR
#   cr3 = MASK_BASE_ADDR          cr4 = 128 (cols)
#   cr5 = KERNEL_BASE_ADDR        cr6 = in_group_stride (= in_channels * 128)
#   cr7 = FPB * 128 (channel group size = 28 * 128 = 3584)
#   cr8 = total_kernel_bytes      cr9 = TEMP_BASE_ADDR
#   cr10 = INPUT_BASE_ADDR
#   cr11 = pair-loop limit = (num_pairs - 1) * (2 * in_group_stride)
#   cr12 = 128 (CHUNK_BYTES)      cr13 = 256 (SUPER_BLOCK_BYTES)
#   cr14 = end-of-9 walking-pointer wrap step = 384 - 2*cols - 2 = 126
#   cr15 = dstructure: valid_elements=128, partition=P0 (single group)
#
# LR allocation: same roles as conv_universal.asm's role-rotating scheme
# (lr1 = cols-2, computed at init), plus:
#   lr15 = ring advance (384), computed once at init from cr_chunk_bytes
#          (frees a CR slot for TEMP_BASE_ADDR; conv_universal.asm has this
#          as a dedicated CR since it doesn't need a temp scratch region).

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 {{ lr_zero }} {{ cr_zero }};
    ldr_mult_mask_reg   {{ lr_zero }} {{ cr_mask_base }};;

    SET                 {{ lr_chunk_base }} {{ cr_zero }};
    SET                 {{ lr_out_ptr }} {{ cr_zero }};;

    ADD                 {{ lr_shift_p1 }} {{ lr_zero }} cr1;         # +1 (mask_shift kc=-1)
    SUB                 {{ lr_shift_m1 }} {{ lr_zero }} cr1;;        # -1 (mask_shift kc=+1)

    add                 {{ lr_ring_adv }} {{ lr_zero }} {{ cr_chunk_bytes }};;   # lr_ring_adv = 128

    add                 {{ lr_ring_adv }} {{ lr_ring_adv }} {{ cr_chunk_bytes }};;  # = 256

    add                 {{ lr_ring_adv }} {{ lr_ring_adv }} {{ cr_chunk_bytes }};;  # = 384 (permanent)

    add                 {{ lr_cols_m2 }} {{ lr_zero }} {{ cr_cols }};;             # cols (temp)

    DEC                 {{ lr_cols_m2 }} 2;;                                       # cols - 2 (permanent)

# ===========================================================================
# Output-row-pair loop.  lr8 (lr_chunk_base) tracks the CURRENT input row's
# base byte offset (row 2k while processing row 2k, then advanced by
# cr_group_stride to row 2k+1's offset after row 2k's drain).
# ===========================================================================

pair_loop:
    SET                 {{ lr_sb_off }} {{ cr_zero }};;

filter_loop:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    SET                 {{ lr_ch_ctr }} {{ cr_zero }};
    MULT.EE             {{ lr_kern_idx }} {{ cr_zero }} 0 {{ lr_zero }} cr15;   # 0 * R0[0], broadcast
    acc.add.first;;                                     # r_acc = 0 (seed; conv taps add on top)

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} row0_clamp;;
    b                   row0_ch_loop;;
row0_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};;

row0_ch_loop:
    # Cy 1: lr2 = E0 = ch0 kr=0 ext = lr8+lr10; load ch0 kr=0 -> slot 0.
    SET                 {{ lr_read }} {{ cr_zero }};
    SET                 {{ lr_write }} {{ cr_zero }};
    add                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 2: lr_write = 384; load ch0 kr=-1 (row 2k-1, real if row0_main_body,
    #     don't-care/masked if row0_g0_body) -> slot 384.  lr_walk = -384.
    add                 {{ lr_write }} {{ lr_zero }} {{ lr_ring_adv }};
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ lr_ring_adv }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} cr1;;

    # Cy 4: lr_walk = cols - 383.  lr_write := lr_read+128 = 128.  lr_kern_idx := -1.
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_cols }};
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_chunk_bytes }};
    DEC                 {{ lr_kern_idx }} 1;;

    BZ                  {{ lr_chunk_base }} row0_g0_body;;
    b                   row0_main_body;;

row0_g0_body:
    # Row 0 of pair 0 ONLY (lr_chunk_base == 0): kr=-1 off-image, mask slot 3.
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} row0_g0_body;;

    b                   row0_after_block;;

row0_main_body:
    # This pair's first row is an ordinary in-bounds row: kr=-1 = row 2k-1,
    # real, loaded from cr10 at ext = lr2 - cr6 (already done at Cy2 above).
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} row0_main_body;;

row0_after_block:
    add                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr_sb_bytes }};
    blt                 {{ lr_ch_ctr }} {{ cr_group_stride }} row0_reload;;

    b                   row0_drain;;

row0_reload:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} row0_reload_clamp;;
    BZ                  {{ lr_chunk_base }} row0_g0_body;;
    b                   row0_main_body;;
row0_reload_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};
    BZ                  {{ lr_chunk_base }} row0_g0_body;;
    b                   row0_main_body;;

row0_drain:
    # All input channels accumulated for row 2k.  Drain the 128-lane
    # accumulation through the reload-through-temp trick, decimate to 64
    # lanes, quantize and store to TEMP[0:64].
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    {{ lr_zero }} {{ cr_temp_base }};;

    ldr_cyclic_mult_reg {{ lr_zero }} {{ cr_temp_base }} {{ lr_zero }};;
    MULT.RC.VE          {{ lr_zero }} {{ lr_zero }} 0 {{ lr_zero }} cr15;
    acc.add.first;;
    acc.stride          64 on off {{ lr_zero }};
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    {{ lr_zero }} {{ cr_temp_base }};;

    # ---- Row 2k+1 (this pair's SECOND row).  Always a real, in-bounds row
    #      (kr=-1 = row 2k, kr=+1 = row 2k+2 or the last real row) -- never
    #      needs the g0 top-border body.
    add                 {{ lr_chunk_base }} {{ lr_chunk_base }} {{ cr_group_stride }};;

    SET                 {{ lr_ch_ctr }} {{ cr_zero }};
    MULT.EE             {{ lr_kern_idx }} {{ cr_zero }} 0 {{ lr_zero }} cr15;
    acc.add.first;;

    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    blt                 {{ cr_group_stride }} {{ lr_clamp }} row1_clamp0;;
    b                   row1_ch_loop;;
row1_clamp0:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};;

row1_ch_loop:
    SET                 {{ lr_read }} {{ cr_zero }};
    SET                 {{ lr_write }} {{ cr_zero }};
    add                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};;

    add                 {{ lr_write }} {{ lr_zero }} {{ lr_ring_adv }};
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ lr_ring_adv }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};;

    ADD                 {{ lr_walk }} {{ lr_walk }} cr1;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_cols }};
    add                 {{ lr_write }} {{ lr_zero }} {{ cr_chunk_bytes }};
    DEC                 {{ lr_kern_idx }} 1;;

    b                   row1_body;;

row1_body:
    add                 {{ lr_walk }} {{ lr_walk }} {{ cr_walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_off_zero }} {{ lr_off_zero }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    sub                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr_group_stride }};
    ldr_cyclic_mult_reg {{ lr_scratch }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    add                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} {{ cr_chunk_bytes }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;;

    add                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    ldr_cyclic_mult_reg {{ lr_off_zero }} {{ cr_input_base }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} cr15;
    acc.add;;

    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    incr_mod_pow2       {{ lr_write }} {{ cr_chunk_bytes }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} cr15;
    acc.add;
    blt                 {{ lr_ch_ctr }} {{ lr_clamp }} row1_body;;

    add                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr_sb_bytes }};
    blt                 {{ lr_ch_ctr }} {{ cr_group_stride }} row1_reload;;

    b                   row1_drain;;

row1_reload:
    ldr_mult_reg        r0 {{ lr_sb_off }} {{ cr_kernel_base }};
    SET                 {{ lr_kern_idx }} {{ cr_zero }};;

    add                 {{ lr_scratch }} {{ lr_sb_off }} {{ cr_chunk_bytes }};
    add                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr_sb_ch_bytes }};
    ldr_mult_reg        r1 {{ lr_scratch }} {{ cr_kernel_base }};;
    blt                 {{ cr_group_stride }} {{ lr_clamp }} row1_reload_clamp;;
    b                   row1_ch_loop;;
row1_reload_clamp:
    add                 {{ lr_clamp }} {{ lr_zero }} {{ cr_group_stride }};
    b                   row1_ch_loop;;

row1_drain:
    # Row 2k+1's 64-lane decimated result -> TEMP[64:128].
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    {{ lr_zero }} {{ cr_temp_base }};;

    ldr_cyclic_mult_reg {{ lr_zero }} {{ cr_temp_base }} {{ lr_zero }};;
    MULT.RC.VE          {{ lr_zero }} {{ lr_zero }} 0 {{ lr_zero }} cr15;
    acc.add.first;;
    acc.stride          64 on off {{ lr_zero }};
    SET                 {{ lr_write }} {{ cr_chunk_bytes }};
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    {{ lr_write }} {{ cr_temp_base }};;

    # ---- Combine: reload TEMP[0:128] (row 2k's half + row 2k+1's half) and
    #      store the finished 128-byte output chunk in one shot.
    ldr_cyclic_mult_reg {{ lr_zero }} {{ cr_temp_base }} {{ lr_zero }};;
    MULT.RC.VE          {{ lr_zero }} {{ lr_zero }} 0 {{ lr_zero }} cr15;
    acc.add.first;
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    {{ lr_out_ptr }} {{ cr_out_base }};;

    add                 {{ lr_out_ptr }} {{ lr_out_ptr }} {{ cr_chunk_bytes }};
    blt                 {{ lr_sb_off }} {{ cr_total_kern }} filter_loop;;

    add                 {{ lr_chunk_base }} {{ lr_chunk_base }} {{ cr_group_stride }};;
    blt                 {{ lr_chunk_base }} {{ cr_pair_limit }} pair_loop;;

end:
    bkpt;;
