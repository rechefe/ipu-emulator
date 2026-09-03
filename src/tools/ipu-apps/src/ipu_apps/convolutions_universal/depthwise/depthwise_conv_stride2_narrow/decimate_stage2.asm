# Stage 2 of depthwise_conv_stride2_narrow: joint row+col decimation 2x for
# packed (multi-row-per-chunk) layouts, cols in {16, 32, 64}.
#
# Stage 1 (depthwise_conv_universal, run UNMODIFIED against the full-
# resolution input, its own base-app tests already prove it correct) wrote a
# full-width conv result, chunk-interleaved, `rows_per_chunk = 128/cols`
# packed spatial rows per 128-byte chunk.
#
# Key structural fact (see module docstring in __init__.py for the full
# derivation): loading ONE stage-1 chunk into R0 and running
# ACC.STRIDE({{ elements_in_row }}, on, on, offset) column- AND row-decimates
# it in a single instruction, because:
#   - elements_in_row == cols means ACC.STRIDE treats the 128-lane MULT_RES
#     as `rows_per_chunk` rows of `cols` elements each -- EXACTLY the packed
#     local-row layout stage 1 used.
#   - horizontal stride (on) keeps every other COLUMN within each local row.
#   - vertical stride (on) keeps every other LOCAL row. Because rows_per_chunk
#     is a power of 2 and every chunk starts at local_row=0, local-row parity
#     == absolute-row parity, so "every other local row" IS "every other
#     absolute row" -- no separate vertical-stride bookkeeping needed.
#   - output size per call = (rows_per_chunk/2) * (cols/2) = 32 elements,
#     ALWAYS (128/4), regardless of cols -- this lets 4 stage-1 chunks (at
#     absolute row-groups g, g+1, g+2, g+3 for the same channel) fill exactly
#     one 128-byte output chunk via 4 ACC.STRIDE calls at offsets 0,1,2,3
#     (base = (offset%4)*32).
#
# ACC.STRIDE reads MULT_RES, not R_ACC, hence the identity-MULT passthrough
# (MULT.VE ra_idx=0, CR1 scalar ("x1"), no mask) before each ACC.STRIDE call.
# The k=0..3 inner step is unrolled (4 explicit load+mult+acc.stride bodies)
# rather than a real loop, since each k needs a distinct load-row address
# (lr3 + lr4 + k*channels) that is cheapest to just compute directly with a
# fresh ADD chain per k -- a real loop would need a multiply-by-k that this
# ISA doesn't have in the LR slot, or repeated same-word ADD accumulation
# (blocked: ADD's src_a reads the cycle-START snapshot, not same-word writes).
#
# Layout (row-addressed ISA; every CR below is a ROW number unless noted):
#   cr0 = read-only 0          cr1 = read-only 1
#   cr3 = stage1 output base row (chunk-interleaved, `channels` chunks/row-group)
#   cr4 = channels             cr5 = final output base row
#   cr6 = num_out_groups (outer loop bound)
#
# LR: lr0=0  lr1=out-group counter  lr2=out chunk row ptr (channels/iter)
#     lr3=in row-group base for this out-group (4*channels rows/iter)
#     lr4=channel counter  lr5=scratch (this-k's in-row-group pointer)
#     lr6=128 (ra_idx constant; MULT.VE index into combined Ra, byte-space)
#     lr9=ACC.STRIDE offset constant 0  lr10=offset constant 1
#     lr11=offset constant 2            lr12=offset constant 3

    SET                 lr0 cr0;;

    ADD                 lr6 lr0 cr1;
    SET                 lr1 cr0;
    SET                 lr2 cr0;;
    SET                 lr3 cr0;
    INC                 lr6 127;;                       # lr6 = 128

    ADD                 lr9 lr0 cr0;
    ADD                 lr10 lr0 cr1;;
    ADD                 lr11 lr10 cr1;;
    ADD                 lr12 lr11 cr1;;

outgroup_loop:
    SET                 lr4 cr0;;

ch_loop:
    # --- k=0: load stage1 row-group (lr3 + lr4), ACC.STRIDE offset 0 (writes r_acc[0:32]).
    ADD                 lr5 lr3 lr4;;
    ldr_mult_reg        r0 lr5 cr3;;
    MULT.VE             lr0 cr1 0 lr0 cr15;;
    acc.stride          {{ elements_in_row }} on on lr9;;

    # --- k=1: row-group + channels, offset 1 (writes r_acc[32:64]).
    ADD                 lr5 lr5 cr4;;
    ldr_mult_reg        r0 lr5 cr3;;
    MULT.VE             lr0 cr1 0 lr0 cr15;;
    acc.stride          {{ elements_in_row }} on on lr10;;

    # --- k=2: +channels again, offset 2 (writes r_acc[64:96]).
    ADD                 lr5 lr5 cr4;;
    ldr_mult_reg        r0 lr5 cr3;;
    MULT.VE             lr0 cr1 0 lr0 cr15;;
    acc.stride          {{ elements_in_row }} on on lr11;;

    # --- k=3: +channels again, offset 3 (writes r_acc[96:128]).  Quantize the
    #     full 128-element r_acc (identity, no clamp change) then store.
    ADD                 lr5 lr5 cr4;;
    ldr_mult_reg        r0 lr5 cr3;;
    MULT.VE             lr0 cr1 0 lr0 cr15;;
    acc.stride          {{ elements_in_row }} on on lr12;
    ACTIVATE.QUANTIZE   identity cr15;;

    ADD                 lr5 lr2 lr4;;
    str_post_aaq_reg    lr5 cr5;;

    INC                 lr4 1;
    blt                 lr4 cr4 ch_loop;;

    ADD                 lr2 lr2 cr4;;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;

    INC                 lr1 1;
    blt                 lr1 cr6 outgroup_loop;;

end:
    bkpt;;
