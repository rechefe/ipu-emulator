{#- depthwise_conv_stride2_16.asm -- stage 2 of a 3x3 stride-2 depthwise conv on 16x16: decimate stage 1's output and pack two channels per output chunk.

    Stage 2 of depthwise_conv_stride2_16: joint row+col decimation 2x for the
    degenerate width=16, height=16 shape, where the TRUE output (8 rows x 8
    cols = 64 elements per channel) is only HALF of one 128-element chunk.

    Stage 1 (depthwise_conv_tpl's program, as depthwise_conv_universal runs
    it: no bias, no ReLU; run UNMODIFIED against the full-resolution input,
    into the scratch region y, stages.py) wrote a
    full-width conv result, chunk-interleaved, `rows_per_chunk = 128/16 = 8`
    packed spatial rows per chunk. At height=16 that is exactly TWO stage-1
    chunks per channel (row-groups 0 and 1) -- not four, unlike the general
    depthwise_conv_stride2_narrow sibling, which always reads FOUR row-groups
    per output chunk and therefore can't represent a shape this small (the gap this app closes).

    Key structural fact (same ACC.STRIDE mechanics as the narrow sibling, see
    its .asm header for the general derivation): loading ONE stage-1
    chunk (8 packed rows of 16 cols) into R0 and running
    ACC.STRIDE(16, on, on, offset) column- AND row-decimates it in a single
    instruction, producing (8/2)*(16/2) = 32 elements written to
    r_acc[(offset%4)*32 : +32].

    One channel's TRUE output is exactly 2 stage-1 chunks * 32 elements/call =
    64 elements -- HALF a chunk. Rather than pad the other half with garbage
    (which the general narrow kernel would need 4 row-groups per channel to
    avoid), this kernel packs TWO channels per output chunk:
      channel A (row-groups 0,1) -> ACC.STRIDE offsets 0,1 -> r_acc[0:64]
      channel B (row-groups 0,1) -> ACC.STRIDE offsets 2,3 -> r_acc[64:128]
    filling the chunk EXACTLY, with zero padding and zero wasted computation.
    `channels` must therefore be even (kernel.toml: in_channels is a multiple of 2).

    ACC.STRIDE reads MULT_RES, not R_ACC, hence the identity-MULT passthrough
    (MULT.VE ra_idx=0, CR1 scalar ("x1"), no mask) before each ACC.STRIDE call.
    The k=0..3 inner step is unrolled (4 explicit load+mult+acc.stride bodies)
    rather than a real loop, matching the narrow sibling's rationale: each k
    needs a distinct load-row address that is cheapest to compute directly.

    Layout (row-addressed ISA; every CR below is a ROW number unless noted):
      cr0 = read-only 0          cr1 = read-only 1
      cr3 = stage1 output base row (chunk-interleaved, `channels` chunks/row-group)
      cr4 = channels             cr5 = final output base row
      cr6 = num_channel_pairs (outer loop bound, = channels/2)

    Row addressing: address(row_group, channel) = base_row + row_group*channels
    + channel. For pair p (channels A=2p, B=2p+1): A.rg0 = base+2p,
    A.rg1 = A.rg0+channels, B.rg0 = A.rg0+1 (NOT A.rg1+channels -- that would
    land on channel 2p's row-group 2, which doesn't exist), B.rg1 = B.rg0+channels.

    LR: lr0=0  lr1=channel-pair counter  lr2=out chunk row ptr (1/iter)
        lr3=in row-group-0 base for this pair's channel A (+2 every iteration)
        lr4=scratch (this-k's in-row-group pointer)
        lr9=ACC.STRIDE offset constant 0  lr10=offset constant 1
        lr11=offset constant 2            lr12=offset constant 3
-#}
    SET                 LR0 CR0;;

    SET                 LR1 CR0;
    SET                 LR2 CR0;;
    SET                 LR3 CR0;;

    ADD                 LR9 LR0 CR0;
    ADD                 LR10 LR0 CR1;;
    ADD                 LR11 LR10 CR1;;
    ADD                 LR12 LR11 CR1;;

pair_loop:
    # --- channel A (=2p), row-group 0: offset 0 (writes r_acc[0:32]).
    ADD                 LR4 LR3 CR0;;
    LD.MULT        R0 LR4 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15.L CR15.H;;
    ACC.STRIDE          16 on on LR9;;

    # --- channel A, row-group 1: +channels rows, offset 1 (writes r_acc[32:64]).
    ADD                 LR4 LR4 {{ cr.in_channels }};;
    LD.MULT        R0 LR4 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15.L CR15.H;;
    ACC.STRIDE          16 on on LR10;;

    # --- channel B (=2p+1), row-group 0: lr3+1 (NOT lr4+channels -- see
    #     header note), offset 2 (writes r_acc[64:96]).
    ADD                 LR4 LR3 CR1;;
    LD.MULT        R0 LR4 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15.L CR15.H;;
    ACC.STRIDE          16 on on LR11;;

    # --- channel B, row-group 1: +channels rows, offset 3 (writes
    #     r_acc[96:128]). Store the full 128-element r_acc (identity) in the
    #     same word.
    ADD                 LR4 LR4 {{ cr.in_channels }};;
    LD.MULT        R0 LR4 {{ cr.y }};;
    MULT.VE             LR0 CR1 0 LR0 CR15.L CR15.H;;
    ACC.STRIDE          16 on on LR12;
    ACTIVATE.EXACT      identity CR15.L {{ cr.out }} LR2;;

    INC                LR2 1;;
    # Two separate ADDs (ADD's src_a reads the cycle-start snapshot, so a
    # same-word double ADD into lr3 would not see the first result).
    ADD                 LR3 LR3 CR1;;
    ADD                 LR3 LR3 CR1;;

    INC                 LR1 1;
    BLT.S                 LR1 {{ cr.num_channel_pairs }} pair_loop;;

end:
    HALT;;
