{#- depthwise_conv_stride2_narrow.asm -- stage 2 of a 3x3 stride-2 depthwise conv at width 16/32/64: decimate stage 1's full-resolution output in both dimensions with ACC.STRIDE.

    Stage 2 of depthwise_conv_stride2_narrow: joint row+col decimation 2x for
    packed (multi-row-per-chunk) layouts, cols in {16, 32, 64}.

    Stage 1 (depthwise_conv_tpl's program, as depthwise_conv_universal runs
    it: no bias, no ReLU; run UNMODIFIED against the full-resolution input,
    into the scratch region y, stages.py) wrote a
    full-width conv result, chunk-interleaved, `rows_per_chunk = 128/cols`
    packed spatial rows per 128-element chunk.

    Key structural fact: loading ONE stage-1 chunk into R0 and running
    ACC.STRIDE({{ elements_in_row }}, on, on, offset) column- AND row-decimates
    it in a single instruction, because:
      - elements_in_row == cols means ACC.STRIDE treats the 128-lane MULT_RES
        as `rows_per_chunk` rows of `cols` elements each -- EXACTLY the packed
        local-row layout stage 1 used.
      - horizontal stride (on) keeps every other COLUMN within each local row.
      - vertical stride (on) keeps every other LOCAL row. Because rows_per_chunk
        is a power of 2 and every chunk starts at local_row=0, local-row parity
        == absolute-row parity, so "every other local row" IS "every other
        absolute row" -- no separate vertical-stride bookkeeping needed.
      - output size per call = (rows_per_chunk/2) * (cols/2) = 32 elements,
        ALWAYS (128/4), regardless of cols -- this lets 4 stage-1 chunks (at
        absolute row-groups g, g+1, g+2, g+3 for the same channel) fill exactly
        one 128-element output chunk via 4 ACC.STRIDE calls at offsets 0,1,2,3
        (base = (offset%4)*32).

    ACC.STRIDE reads MULT_RES, not R_ACC, hence the identity-MULT passthrough
    (MULT.VE ra_idx=0, CR1 scalar ("x1"), no mask) before each ACC.STRIDE call.
    The k=0..3 inner step is unrolled (4 explicit load+mult+acc.stride bodies)
    rather than a real loop, since each k needs a distinct load-row address
    (lr3 + lr4 + k*channels) that is cheapest to just compute directly with a
    fresh ADD chain per k -- a real loop would need a multiply-by-k that this
    ISA doesn't have in the LR slot, or repeated same-word ADD accumulation
    (blocked: ADD's src_a reads the cycle-START snapshot, not same-word writes).

    Layout (row-addressed ISA; every CR below is a ROW number unless noted):
      cr0 = read-only 0          cr1 = read-only 1
      cr3 = stage1 output base row (chunk-interleaved, `channels` chunks/row-group)
      cr4 = channels             cr5 = final output base row
      cr6 = num_out_groups (outer loop bound)

    LR: lr0=0  lr1=out-group counter  lr2=out chunk row ptr (channels/iter)
        lr3=in row-group base for this out-group (4*channels rows/iter)
        lr4=channel counter  lr5=scratch (this-k's in-row-group pointer)
        lr6=128 (ra_idx constant; MULT.VE index into combined Ra, byte-space)
        lr9=ACC.STRIDE offset constant 0  lr10=offset constant 1
        lr11=offset constant 2            lr12=offset constant 3
-#}
    SET                 LR0 CR0;;

    ADD                 LR6 LR0 CR1;
    SET                 LR1 CR0;
    SET                 LR2 CR0;;
    SET                 LR3 CR0;
    INC                 LR6 127;;                       # lr6 = 128

    ADD                 LR9 LR0 CR0;
    ADD                 LR10 LR0 CR1;;
    ADD                 LR11 LR10 CR1;;
    ADD                 LR12 LR11 CR1;;

outgroup_loop:
    SET                 LR4 CR0;;

ch_loop:
    # --- k=0: load stage1 row-group (lr3 + lr4), ACC.STRIDE offset 0 (writes r_acc[0:32]).
    ADD                 LR5 LR3 LR4;;
    LD.MULT        R0 LR5 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          {{ elements_in_row }} on on LR9;;

    # --- k=1: row-group + channels, offset 1 (writes r_acc[32:64]).
    ADD                 LR5 LR5 {{ cr.in_channels }};;
    LD.MULT        R0 LR5 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          {{ elements_in_row }} on on LR10;;

    # --- k=2: +channels again, offset 2 (writes r_acc[64:96]).
    ADD                 LR5 LR5 {{ cr.in_channels }};;
    LD.MULT        R0 LR5 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          {{ elements_in_row }} on on LR11;;

    # --- k=3: +channels again, offset 3 (writes r_acc[96:128]).  Quantize the
    #     full 128-element r_acc (identity, no clamp change) then store.
    ADD                 LR5 LR5 {{ cr.in_channels }};;
    LD.MULT        R0 LR5 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          {{ elements_in_row }} on on LR12;
    ACTIVATE.QUANTIZE   identity CR15;;

    ADD                 LR5 LR2 LR4;;
    ST.AAQ    LR5 {{ cr.out }};;

    INC                 LR4 1;
    BLT.S                 LR4 {{ cr.in_channels }} ch_loop;;

    ADD                 LR2 LR2 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;
    ADD                 LR3 LR3 {{ cr.in_channels }};;

    INC                 LR1 1;
    BLT.S                 LR1 {{ cr.num_out_groups }} outgroup_loop;;

end:
    HALT;;
