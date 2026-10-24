{#- conv3x3_relu_cin1.asm -- out = relu(bias[o] + 3x3 zero-padded FP32 conv) for ONE input channel, all output channels.

    out[o, y, x] = relu( bias[o] + SUM_kr SUM_kc W[o,0,kr,kc] * in[0, y+kr, x+kc] )

    SuperPoint's conv1a (1 -> 64) and nothing else. The bespoke twin of
    conv3x3_relu, specialised on Cin = 1.

    WHY THIS EXISTS. conv3x3_relu is already ~95% MULT-occupied at Cin = 64:
    its nine taps are unrolled and its channel loop is software-pipelined to
    9 words / 9 MACs, so there is almost nothing left to reclaim. At Cin = 1
    that inverts -- there is a single channel to amortise the per-tile and
    per-group bookkeeping against, and only 9 of 23 words do any multiplying:

        conv3x3_relu at Cin=1        conv3x3_relu_cin1
        2  tile head + bias          3  row loads (bias rides in the third)
        2  prime two rows            9  taps (tap 9 carries ACTIVATE.EXACT)
        5  group sizing (partial)    1  advance + branch
        9  taps
        2  group advance + branch   13 words per output tile
        3  drain
       23 words per output tile      -> 1.77x

    Cin = 1 is what makes the specialisation both worthwhile and possible:
    a fully unrolled channel loop costs 9*Cin + 4 words, so it only fits the
    128-word IMEM bank while Cin <= 13. Every other SuperPoint layer has
    Cin >= 64 and would need 580 to 1156 words -- more than a bank, and at
    Cin=128 more than the whole instruction memory. Those layers keep the
    general kernel, where they lose only ~5% anyway.

    WHAT THE UNROLL REMOVES:
      * the channel loop, and with it the exact-group-size computation
        (ADD gend / BLT.S / SET / DEC) that a partial final group forces;
      * the guard input plane -- the general kernel's pipelined loop prefetches
        one channel past the end, this one has no next channel to prefetch;
      * the separate drain word -- tap 9 co-issues ACTIVATE.EXACT, which
        stores, because ACC runs before AAQ_EXACT within a word.

    HALO TILING and the vertical border are conv3x3_relu's, unchanged: tiles of
    126 output columns, element e of tile t is input column t*126 + e - 1, and
    each plane carries one all-zero row above and below. Output lane j reads
    element j + kc, so every tap is satisfied from inside the one row.

    THE WALKING READ INDEX. Three rows sit in R_CYCLIC slots 0, 1, 2 and the
    nine taps step through them in weight order (kr outer, kc inner):

        tap  1   2  3    4    5    6    7    8    9
        rc   0   1  2  128  129  130  256  257  258
        step  -  +1 +1 +126  +1   +1  +126  +1   +1

    lr_widx walks +1 in lockstep, so the weights are W[o, 0].ravel() -- nine
    consecutive elements of one LD.MULT row.


    MEMORY (regions placed and packed by the harness, kernel.toml; x, w, b, y
    back to back from row 0):
      x: height + 2 padded rows of tiles = ceil(width / 126) halo tiles, no
         guard plane (pack halo_tiled).
      w: one row per output channel, the nine weights W[o, 0].ravel() in
         elements 0..8 (pack conv_weight_groups).
      b: one row per output channel, bias[o] in element 0 (pack lane0_rows).
      y: one row per (o, y, tile), lanes 0..125 valid (pack halo_tiled_output).

    Registers ([abi]): tiles = ceil(width / 126) is TPR, rows per spatial row
    and the row step; rc_step = 126 is the rc slot-to-slot step; lanes = 128
    is the R_CYCLIC slot-1 index and the Ra index selecting R1[0];
    tiles_last = tiles - 1 because the tile branch reads its counter
    pre-increment. CR15.L is the default DS (ds(128)); CR14 holds the QF.
    Routing prefers this kernel (cost 0) over conv3x3_relu (cost 1) when
    in_channels = 1.

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> AAQ_EXACT ->
    COND.
-#}

{%- set lr_zero = "LR0"  -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_o    = "LR1"  -%}  {#- output channel; also the weight/bias row -#}
{%- set lr_out  = "LR2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_y    = "LR3"  -%}  {#- output spatial row -#}
{%- set lr_rowb = "LR4"  -%}  {#- y * TPR -- input row base for this spatial row -#}
{%- set lr_t    = "LR5"  -%}  {#- tile counter AND within-row row offset -#}
{%- set lr_addr = "LR6"  -%}  {#- walking input row address -#}
{%- set lr_rc   = "LR7"  -%}  {#- walking R_CYCLIC read index (the nine taps) -#}
{%- set lr_s1   = "LR8"  -%}  {#- 128: slot-1 index AND the R1[0] bias selector -#}
{%- set lr_s2   = "LR9"  -%}  {#- 256: slot-2 index -#}
{%- set lr_widx = "LR10" -%}  {#- weight element index in R0, seeded -1 per tile -#}

    SET {{lr_zero}} CR0 ;
    SET {{lr_o}}    CR0 ;
    SET {{lr_out}}  CR0 ;;
    SET {{lr_s1}}   {{ cr.lanes }} ;;                              {#- 128 -#}
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;                 {#- 256 -#}

ochan_loop:
    LD.MULT R0 {{lr_o}} {{ cr.w }} ;;                      {#- R0 = W[o, 0] (nine weights) -#}
    LD.MULT R1 {{lr_o}} {{ cr.b }} ;;                      {#- R1 = bias row o -#}
    SET {{lr_y}}    CR0 ;
    SET {{lr_rowb}} CR0 ;;

row_loop:
    SET {{lr_t}} CR0 ;;

tile_loop:
{#- ---- three row loads; the bias reset rides in the third ---------------- -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;
    LD.MULT.CYCLIC {{lr_addr}} {{ cr.x }} {{lr_zero}} ;;   {#- slot0 <- padded row y -#}
    ADD {{lr_addr}} {{lr_addr}} {{ cr.tiles }} ;
    LD.MULT.CYCLIC {{lr_addr}} {{ cr.x }} {{lr_s1}} ;;     {#- slot1 <- padded row y+1 -#}
    ADD {{lr_addr}} {{lr_addr}} {{ cr.tiles }} ;
    SUB {{lr_widx}} {{lr_zero}} CR1 ;
    LD.MULT.CYCLIC {{lr_addr}} {{ cr.x }} {{lr_s2}} ;
    MULT.EE {{lr_s1}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD.FIRST ;;
                                        {#- slot2 <- y+2 | R_ACC = bias[o] | widx = -1 -#}

{#- ---- nine taps; tap 9 drains the tile in the same word ----------------- -#}
    ADD {{lr_rc}} {{lr_zero}} CR0 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr-1 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr-1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr-1 kc+1 -#}
    ADD {{lr_rc}} {{lr_rc}} {{ cr.rc_step }} ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr 0 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr 0 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr 0 kc+1 -#}
    ADD {{lr_rc}} {{lr_rc}} {{ cr.rc_step }} ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr+1 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;;   {#- kr+1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} CR1 ; ADD {{lr_widx}} {{lr_widx}} CR1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.ADD ;
    ACTIVATE.EXACT relu CR15.L {{ cr.y }} {{lr_out}} ;; {#- kr+1 kc+1 | OUT[o, y, t] -#}

    ADD {{lr_out}} {{lr_out}} CR1 ;
    ADD {{lr_t}} {{lr_t}} CR1 ;
    BLT.S {{lr_t}} {{ cr.tiles_last }} tile_loop ;;                       {#- pre-increment: bound TPR-1 -#}

    ADD {{lr_rowb}} {{lr_rowb}} {{ cr.tiles }} ;
    ADD {{lr_y}} {{lr_y}} CR1 ;;
    BLT.S {{lr_y}} {{ cr.height }} row_loop ;;

    ADD {{lr_o}} {{lr_o}} CR1 ;;
    BLT.S {{lr_o}} {{ cr.out_channels }} ochan_loop ;;

end:
    HALT ;;
