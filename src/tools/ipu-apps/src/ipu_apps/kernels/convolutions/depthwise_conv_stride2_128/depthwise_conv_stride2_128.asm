# Stage 2 of depthwise_conv_stride2_128: horizontal decimation 128->64,
# reading row-pairs from stage 1's full-resolution output and packing each
# pair into one 128-byte output chunk.
#
# Stage 1 (depthwise_conv_universal, run UNMODIFIED against the full-
# resolution input, its own base-app tests already prove it correct) wrote a
# full-width conv result for every input row to its normal chunk-interleaved
# output region. A genuine row+col stride-2 result keeps only EVEN stage-1
# rows (r_center = 2*orow); this stage packs KEPT-row pair (4*rp, 4*rp+2)
# into one output chunk (local_row0=orow 2*rp, local_row1=orow 2*rp+1),
# skipping the odd stage-1 rows in between entirely -- that skip IS the
# vertical stride-2 decimation (every row was already computed by stage 1,
# so skipping the odd ones is free -- no separate "vertical" logic needed).
#
# Layout (row-addressed ISA; every CR below is a ROW number):
#   cr0 = read-only 0          cr1 = read-only 1
#   cr3 = stage1 output base row (chunk-interleaved, `channels` chunks/row)
#   cr4 = channels             cr5 = final output base row
#   cr6 = num_row_pairs (outer loop bound)
#
# Per channel: load stage1 row 4*rp's chunk -> R0, stage1 row (4*rp+2)'s
# chunk -> R1 (plain ldr_mult_reg, XMEM-space -- no r_cyclic/tap/border
# logic, this is a pure reshuffle+decimate). ACC.STRIDE needs its source in
# MULT_RES, so R0/R1 are passed through an identity MULT.VE (x1 via CR1)
# first.
#
# LR: lr0=0  lr1=out-row-pair counter  lr2=out chunk ptr (channels/iter)
#     lr3=in row-group base (4*channels/iter, i.e. 4 kept-apart stage1 rows)
#     lr4=channel counter
#     lr5=scratch (addr calc)  lr6=2 (acc.stride offset value whose %4==2,
#     selecting r_acc base=(2%4)*32=64, the 2nd-half write start)
#     lr7=128 (ra_idx for R1: MULT.VE's ra_idx is a BYTE offset into
#     combined Ra = R0++R1, 256 bytes; 0 selects R0, 128 selects R1)

    SET                 lr0 cr0;;

    ADD                 lr6 lr0 cr1;;
    INC                 lr6 1;;                         # lr6 = 2 (1+1)

    ADD                 lr7 lr0 cr1;;
    INC                 lr7 127;;                      # lr7 = 128

    SET                 lr1 cr0;
    SET                 lr2 cr0;
    SET                 lr3 cr0;;

rowpair_loop:
    SET                 lr4 cr0;;

ch_loop:
    # lr5 = this channel's stage1-row-(4*rp) chunk row = lr3 + lr4.
    ADD                 lr5 lr3 lr4;
    ldr_mult_reg        r0 lr5 cr3;;

    # R1 = stage1 row (4*rp + 2)'s chunk: +2*channels rows from R0's row.
    # Two separate ADDs (ADD's src_a reads snapshot, so a same-word double
    # ADD into lr5 would not see the first result) -- split into two words.
    ADD                 lr5 lr5 cr4;;
    ADD                 lr5 lr5 cr4;;
    ldr_mult_reg        r1 lr5 cr3;;

    MULT.VE             lr0 cr1 0 lr0 cr15;;
    acc.stride          64 on off lr0;;
    MULT.VE             lr7 cr1 0 lr0 cr15;
    acc.stride          64 on off lr6;
    ACTIVATE.QUANTIZE   identity cr15;
    ADD                 lr5 lr2 lr4;
    str_post_aaq_reg    lr5 cr5;;

    INC                 lr4 1;
    blt                 lr4 cr4 ch_loop;;

    ADD                 lr2 lr2 cr4;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;
    ADD                 lr3 lr3 cr4;;

    INC                 lr1 1;;
    blt                 lr1 cr6 rowpair_loop;;

end:
    bkpt;;
