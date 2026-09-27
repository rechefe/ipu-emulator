{#- ==========================================================================
    conv3x3_relu_cin1.asm -- 3x3 conv + bias + ReLU for a SINGLE input channel

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
        2  prime two rows            9  taps (tap 9 carries ACTIVATE + STR)
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
        (ADD gend / BLT / SET / DEC) that a partial final group forces;
      * the guard input plane -- the general kernel's pipelined loop prefetches
        one channel past the end, this one has no next channel to prefetch;
      * the separate drain word -- tap 9 co-issues ACTIVATE.QUANTIZE and the
        store, because ACC runs before AAQ runs before STORE within a word.

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
    consecutive elements of one LDR_MULT_REG row.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = WEIGHT_BASE  (rows)      CR5  = BIAS_BASE   (rows)
      CR6  = TPR = ceil(W/126) -- rows per spatial row, and the row step
      CR7  = H                        CR8  = COUT
      CR9  = 126 (the rc slot-to-slot step)
      CR10 = 128 (R_CYCLIC slot-1 index; also the Ra index selecting R1[0])
      CR11 = TPR - 1 (the tile branch reads its counter pre-increment)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND.
========================================================================== -#}

{%- set lr_zero = "lr0"  -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_o    = "lr1"  -%}  {#- output channel; also the weight/bias row -#}
{%- set lr_out  = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_y    = "lr3"  -%}  {#- output spatial row -#}
{%- set lr_rowb = "lr4"  -%}  {#- y * TPR -- input row base for this spatial row -#}
{%- set lr_t    = "lr5"  -%}  {#- tile counter AND within-row row offset -#}
{%- set lr_addr = "lr6"  -%}  {#- walking input row address -#}
{%- set lr_rc   = "lr7"  -%}  {#- walking R_CYCLIC read index (the nine taps) -#}
{%- set lr_s1   = "lr8"  -%}  {#- 128: slot-1 index AND the R1[0] bias selector -#}
{%- set lr_s2   = "lr9"  -%}  {#- 256: slot-2 index -#}
{%- set lr_widx = "lr10" -%}  {#- weight element index in R0, seeded -1 per tile -#}

    SET {{lr_zero}} cr0 ;
    SET {{lr_o}}    cr0 ;
    SET {{lr_out}}  cr0 ;;
    SET {{lr_s1}}   cr10 ;;                              {#- 128 -#}
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;                 {#- 256 -#}

ochan_loop:
    LDR_MULT_REG r0 {{lr_o}} cr4 ;;                      {#- R0 = W[o, 0] (nine weights) -#}
    LDR_MULT_REG r1 {{lr_o}} cr5 ;;                      {#- R1 = bias row o -#}
    SET {{lr_y}}    cr0 ;
    SET {{lr_rowb}} cr0 ;;

row_loop:
    SET {{lr_t}} cr0 ;;

tile_loop:
{#- ---- three row loads; the bias reset rides in the third ---------------- -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- padded row y -#}
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s1}} ;;     {#- slot1 <- padded row y+1 -#}
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    SUB {{lr_widx}} {{lr_zero}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s2}} ;
    MULT.EE {{lr_s1}} cr1 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;;
                                        {#- slot2 <- y+2 | R_ACC = bias[o] | widx = -1 -#}

{#- ---- nine taps; tap 9 drains the tile in the same word ----------------- -#}
    ADD {{lr_rc}} {{lr_zero}} cr0 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr-1 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr-1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr-1 kc+1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr9 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr 0 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr 0 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr 0 kc+1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr9 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr+1 kc-1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr+1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE relu cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- kr+1 kc+1 | OUT[o, y, t] -#}

    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_t}} {{lr_t}} cr1 ;
    BLT {{lr_t}} cr11 tile_loop ;;                       {#- pre-increment: bound TPR-1 -#}

    ADD {{lr_rowb}} {{lr_rowb}} cr6 ;
    ADD {{lr_y}} {{lr_y}} cr1 ;;
    BLT {{lr_y}} cr7 row_loop ;;

    ADD {{lr_o}} {{lr_o}} cr1 ;;
    BLT {{lr_o}} cr8 ochan_loop ;;

end:
    BKPT ;;
