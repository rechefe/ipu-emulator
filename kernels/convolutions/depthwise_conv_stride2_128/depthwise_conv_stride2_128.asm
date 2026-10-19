{#- depthwise_conv_stride2_128.asm -- stage 2 of a 3x3 stride-2 depthwise conv at width 128: keep every other row and column of stage 1's output, two output rows per chunk.

    Stage 2 of depthwise_conv_stride2_128: horizontal decimation 128->64,
    reading row-pairs from stage 1's full-resolution output and packing each
    pair into one 128-element output chunk.

    Stage 1 (depthwise_conv_tpl's program, as depthwise_conv_universal runs
    it: no bias, no ReLU; run UNMODIFIED against the full-resolution input,
    into the scratch region y, stages.py) wrote a
    full-width conv result for every input row to its normal chunk-interleaved
    output region. A genuine row+col stride-2 result keeps only EVEN stage-1
    rows (r_center = 2*orow); this stage packs KEPT-row pair (4*rp, 4*rp+2)
    into one output chunk (local_row0=orow 2*rp, local_row1=orow 2*rp+1),
    skipping the odd stage-1 rows in between entirely -- that skip IS the
    vertical stride-2 decimation (every row was already computed by stage 1,
    so skipping the odd ones is free -- no separate "vertical" logic needed).

    Layout (row-addressed ISA; every CR below is a ROW number):
      cr0 = read-only 0          cr1 = read-only 1
      cr3 = stage1 output base row (chunk-interleaved, `channels` chunks/row)
      cr4 = channels             cr5 = final output base row
      cr6 = num_row_pairs (outer loop bound)

    Per channel: load stage1 row 4*rp's chunk -> R0, stage1 row (4*rp+2)'s
    chunk -> R1 (plain ld.mult, XMEM-space -- no r_cyclic/tap/border
    logic, this is a pure reshuffle+decimate). ACC.STRIDE needs its source in
    MULT_RES, so R0/R1 are passed through an identity MULT.VE (x1 via CR1)
    first.

    LR: lr0=0  lr1=out-row-pair counter  lr2=out chunk ptr (channels/iter)
        lr3=in row-group base (4*channels/iter, i.e. 4 kept-apart stage1 rows)
        lr4=channel counter
        lr5=scratch (addr calc)  lr6=2 (acc.stride offset value whose %4==2,
        selecting r_acc base=(2%4)*32=64, the 2nd-half write start)
        lr7=128 (ra_idx for R1: MULT.VE's ra_idx is an element offset into
        combined Ra = R0++R1, 256 elements; 0 selects R0, 128 selects R1)
-#}
    SET                 LR0 CR0;;

    ADD                 LR6 LR0 CR1;;
    INC                 LR6 1;;                         # lr6 = 2 (1+1)

    ADD                 LR7 LR0 CR1;;
    INC                 LR7 127;;                      # lr7 = 128

    SET                 LR1 CR0;
    SET                 LR2 CR0;
    SET                 LR3 CR0;;

rowpair_loop:
    SET                 LR4 CR0;;

ch_loop:
    # lr5 = this channel's stage1-row-(4*rp) chunk row = lr3 + lr4.
    ADD                 LR5 LR3 LR4;
    LD.MULT        R0 LR5 {{ cr.y }};;

    # R1 = stage1 row (4*rp + 2)'s chunk: +2*channels rows from R0's row.
    # Two separate ADDs (ADD's src_a reads snapshot, so a same-word double
    # ADD into lr5 would not see the first result) -- split into two words.
    ADD                 LR5 LR5 {{ cr.in_channels }};;
    ADD                 LR5 LR5 {{ cr.in_channels }};;
    LD.MULT        R1 LR5 {{ cr.y }};;

    MULT.VE             LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          64 on off LR0;;
    MULT.VE             LR7 CR1 0 LR0 CR15;
    ACC.STRIDE          64 on off LR6;
    ACTIVATE.QUANTIZE   identity CR15;
    ADD                 LR5 LR2 LR4;
    ST.AAQ    LR5 {{ cr.out }};;

    INC                 LR4 1;
    BLT.S                 LR4 {{ cr.in_channels }} ch_loop;;

    ADD                 LR2 LR2 {{ cr.in_channels }};
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;

    INC                 LR1 1;;
    BLT.S                 LR1 {{ cr.num_row_pairs }} rowpair_loop;;

end:
    HALT;;
