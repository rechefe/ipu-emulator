{#- ==========================================================================
    conv3x3_relu.asm -- 3x3 FP32 convolution + bias + ReLU, zero-padded

    out[o, y, x] = relu( bias[o]
                         + SUM_ci SUM_kr SUM_kc W[o,ci,kr,kc] * in[ci, y+kr, x+kc] )

    with kr, kc in {-1, 0, +1} and zero padding outside the image.

    WHY THE TAPS LIVE IN R_CYCLIC, NOT IN THE LOAD:
      XMEM operands are ROW numbers -- a load addresses whole 128-element rows
      and cannot shift by one element. The older byte-addressed kernel got its
      nine taps from nine shifted *loads*; that is simply not expressible now.
      MULT.RC.* instead reads R_CYCLIC at an arbitrary ELEMENT index and may
      cross slot boundaries, so the horizontal shift happens inside the
      register: three vertically-neighbouring rows are loaded into three of
      R_CYCLIC's four slots, and kc is a +/-1 step on the read index.

    HALO TILING (this is what makes the horizontal shift work at a tile edge):
      A spatial row is cut into tiles of {{ '126' }} output columns, each stored as one
      128-element XMEM row:

        element  0        = input column (t*126 - 1)   <- left halo
        elements 1..126   = input columns t*126 .. t*126+125
        element  127      = input column (t*126 + 126) <- right halo

      Output lane j (0..125) is column t*126+j, and tap kc reads element
      j+kc+1 -- so every tap of every valid lane is satisfied from within the
      one row, with no neighbouring-tile dependency. Columns outside the image
      are stored as zero, which is exactly the convolution's zero padding.
      Lanes 126 and 127 read past the slot; their results are discarded.

    VERTICAL BORDER COSTS NOTHING:
      Each input plane carries one all-zero row band above and below (the plane
      is H+2 spatial rows). Output row y then reads padded rows y, y+1, y+2
      unconditionally -- there is no top/bottom special case anywhere in the
      kernel, because the zero rows *are* the padding.

    SOFTWARE-PIPELINED CHANNEL LOOP -- 9 WORDS, 9 MACS:
      The three row loads and the loop branch all hide inside the nine tap
      words, so the MULT slot issues every cycle. Slots are read
      0,0,0 / 1,1,1 / 2,2,2, which frees each one three taps before it is
      needed again:

        tap 1  read slot 0 | load slot 2 <- THIS channel's row y+2
        tap 4  read slot 1 | load slot 0 <- NEXT channel's row y
        tap 7  read slot 2 | load slot 1 <- NEXT channel's row y+1
        tap 9  read slot 2 | branch

      Every load lands six words before its first read, which is required:
      MULT.RC.* reads R_CYCLIC from the start-of-word SNAPSHOT, so a load
      co-issued with the multiply that consumes it would be one word too late.
      (The ZDconv kernel this is modelled on co-issues them -- that ISA made a
      load visible in its own cycle. This one does not, so the schedule is
      re-derived rather than copied.) The last channel prefetches into the
      GUARD PLANE, which is what the +1 on the input region is for.

      lr_addr walks one delta per load, cycling +TPR, +H*TPR, +TPR -- the
      middle step crossing from row y+2 of one channel to row y of the next.

    THE WALKING READ INDEX:
      lr_rc steps through the nine taps in weight order (kr = -1, 0, +1 outer;
      kc = -1, 0, +1 inner):

        tap  1     2  3    4    5   6    7    8  9
        rc   0     1  2  128  129 130  256  257 258
        step -258 +1 +1  +126  +1  +1  +126  +1  +1

      Three step constants: CR1 (=1), CR14 (=126) and lr_wrap (=-258). rc_idx
      is read LIVE, so the same-word add lands before the read; tap 1's step
      wraps the index from 258 straight back to 0, so the per-channel reset
      costs no word of its own and lr_rc is seeded ONCE, at startup.
      lr_widx walks +1 in lockstep, so the weights are simply
      W[o, ci].ravel() -- (kr, kc) row-major, nine per channel.

    CHANNEL GROUPS:
      Nine weights per channel and 128 per LDR_MULT_REG row gives 14 channels
      per group (126 of 128 elements used). Group size is exact: CR13 = 13, so
      done+CR13 is the group's LAST channel index, which is what tap 9's BLT
      wants (it reads lr_done pre-increment). Only a partial final group needs
      clamping. R_ACC is never reset between groups, which is what makes an
      arbitrary Cin work.

    BIAS AND ACCUMULATOR RESET IN ONE WORD:
      MULT.EE broadcasts R1[0] (= bias[o]) x CR1 (= 1.0) into ACC.ADD.FIRST,
      which both clears R_ACC and seeds it. Every MAC is then a uniform
      ACC.ADD, and the word touches no input data so it cannot make a 0*inf
      NaN.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = WEIGHT_BASE  (rows)      CR5  = BIAS_BASE   (rows)
      CR6  = H * TPR   -- advance from row (y+2, t) of channel c to row (y, t)
                          of channel c+1: (H+2)*TPR - 2*TPR
      CR7  = TPR = ceil(W/126)  (rows per spatial row; tile-loop bound)
      CR8  = H                        CR9  = CIN
      CR10 = COUT                     CR11 = NGROUPS (weight-row advance per
                                             output channel)
      CR12 = 128 (Ra index selecting R1[0] = bias; R_CYCLIC slot-1 index)
      CR13 = 13 (group cap - 1)       CR14 = 126 (rc walking step)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND. That is why the first row load takes lr_addr as it stands while the
    second pre-advances it, why a load can share a word with the LR add that
    positions it, and why the store's offset register is incremented in the
    word after the store.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- constant 0: mask_shift and R_CYCLIC slot-0 index -#}
{%- set lr_o     = "lr1"  -%}  {#- output channel; doubles as the bias row offset -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_wbase = "lr3"  -%}  {#- first weight row of this output channel -#}
{%- set lr_y     = "lr4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- y * TPR -- input row base for this spatial row -#}
{%- set lr_t     = "lr6"  -%}  {#- tile: counter AND within-row row offset -#}
{%- set lr_addr  = "lr7"  -%}  {#- walking input row address -#}
{%- set lr_wrow  = "lr8"  -%}  {#- weight row for the current group -#}
{%- set lr_done  = "lr9"  -%}  {#- absolute input-channel index -#}
{%- set lr_gend  = "lr10" -%}  {#- LAST channel index of the current group -#}
{%- set lr_wrap  = "lr11" -%}  {#- -258: tap 1's rc step, wrapping 258 -> 0 in place -#}
{%- set lr_widx  = "lr12" -%}  {#- weight element index in R0, seeded -1 per group -#}
{%- set lr_rc    = "lr13" -%}  {#- walking R_CYCLIC read index (the nine taps) -#}
{%- set lr_s1    = "lr14" -%}  {#- 128: R_CYCLIC slot-1 index AND R1[0] selector -#}
{%- set lr_s2    = "lr15" -%}  {#- 256: R_CYCLIC slot-2 index -#}

    SET {{lr_zero}}  cr0 ;
    SET {{lr_o}}     cr0 ;
    SET {{lr_out}}   cr0 ;;
    SET {{lr_wbase}} cr0 ;
    SET {{lr_s1}}    cr12 ;;                             {#- 128 -#}
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;                 {#- 256 (snapshot: lr_s1 set last word) -#}
    ADD {{lr_wrap}} {{lr_s2}} cr0 ;;                     {#- 256 -#}
    INC {{lr_wrap}} 2 ;;                                 {#- 258 -#}
    SUB {{lr_wrap}} {{lr_zero}} {{lr_wrap}} ;;           {#- -258 = tap 1's wrapping rc step -#}
{#- rc is seeded ONCE. Every channel ends at 258 and tap 1 subtracts 258 to  -#}
{#- re-enter at 0, so the index carries itself across channels, tiles and    -#}
{#- output channels -- no per-tile reseed.                                   -#}
    ADD {{lr_rc}} {{lr_s2}} cr0 ;;                       {#- 256 -#}
    INC {{lr_rc}} 2 ;;                                   {#- 258 -#}

ochan_loop:
    LDR_MULT_REG r1 {{lr_o}} cr5 ;;                      {#- R1 = bias row o (visible NEXT word) -#}
    SET {{lr_y}}    cr0 ;
    SET {{lr_rowb}} cr0 ;;

row_loop:
    SET {{lr_t}} cr0 ;;

tile_loop:
{#- One output tile: R_ACC accumulates all nine taps of every input channel. -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;               {#- channel 0, padded row y -#}
    SET {{lr_done}} cr0 ;
    ADD {{lr_wrow}} {{lr_wbase}} cr0 ;;                  {#- wrow = wbase (SET takes a CR only) -#}
    MULT.EE {{lr_s1}} cr1 0 {{lr_zero}} cr15 ;
    ACC.ADD.FIRST ;;                                     {#- R_ACC = bias[o] -- reset and seed -#}
{#- ---- prime the pipeline: slots 0 and 1 hold channel 0's rows y and y+1, -#}
{#- which in steady state the previous channel's taps 4 and 7 would load.   -#}
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot 0 <- ch0 row y -#}
    ADD {{lr_addr}} {{lr_addr}} cr7 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s1}} ;;     {#- slot 1 <- ch0 row y+1 -#}

group_loop:
{#- ---- exact group size. CR13 = cap-1, so done+CR13 IS the group's last -#}
{#- channel index -- which is what tap 9's BLT wants, since it reads       -#}
{#- lr_done pre-increment. Only the final group needs clamping.            -#}
    ADD {{lr_gend}} {{lr_done}} cr13 ;;
    BLT {{lr_gend}} cr9 group_sized ;;                   {#- full group: last index is in range -#}
    SET {{lr_gend}} cr9 ;;
    DEC {{lr_gend}} 1 ;;                                 {#- final group: last index = CIN-1 -#}
group_sized:
    LDR_MULT_REG r0 {{lr_wrow}} cr4 ;                    {#- R0 = W[o][group] (visible NEXT word) -#}
    SUB {{lr_widx}} {{lr_zero}} cr1 ;;                   {#- widx = -1 (live +1 lands before the read) -#}

chan_loop:
{#- Nine taps, nine MACs, no overhead words: the three row loads and the loop -#}
{#- branch all ride inside them, so the MULT slot is busy every cycle. Slots  -#}
{#- are read 0,0,0 / 1,1,1 / 2,2,2, which frees each one three taps before it -#}
{#- is next needed; every load lands six words before its first read.         -#}
{#- lr_addr walks +TPR, +H*TPR, +TPR -- the middle step crossing from row y+2 -#}
{#- of this channel to row y of the next.                                     -#}
    ADD {{lr_rc}} {{lr_rc}} {{lr_wrap}} ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    ADD {{lr_addr}} {{lr_addr}} cr7 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s2}} ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;
                                          {#- kr-1 kc-1 | slot2 <- THIS ch row y+2 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr-1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr-1 kc+1 -#}

    ADD {{lr_rc}} {{lr_rc}} cr14 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;
                                          {#- kr 0 kc-1 | slot0 <- NEXT ch row y -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr 0 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr 0 kc+1 -#}

    ADD {{lr_rc}} {{lr_rc}} cr14 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    ADD {{lr_addr}} {{lr_addr}} cr7 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_s1}} ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;
                                          {#- kr+1 kc-1 | slot1 <- NEXT ch row y+1 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;;   {#- kr+1 kc 0 -#}
    ADD {{lr_rc}} {{lr_rc}} cr1 ; ADD {{lr_widx}} {{lr_widx}} cr1 ;
    ADD {{lr_done}} {{lr_done}} cr1 ;
    MULT.RC.VE {{lr_rc}} {{lr_widx}} 0 {{lr_zero}} cr15 ; ACC.ADD ;
    BLT {{lr_done}} {{lr_gend}} chan_loop ;;
                                          {#- kr+1 kc+1 | BLT reads lr_done pre-increment -#}

    ADD {{lr_wrow}} {{lr_wrow}} cr1 ;;                   {#- next group's weight row -#}
    BLT {{lr_done}} cr9 group_loop ;;

{#- ---- drain the tile. The store's offset is advanced in the NEXT word: LR -#}
{#- runs before STORE, so incrementing it here would store one row too far.  -#}
    ACTIVATE.QUANTIZE relu cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[o, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_t}} {{lr_t}} cr1 ;;
    BLT {{lr_t}} cr7 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr7 ;                    {#- next spatial row: += TPR -#}
    ADD {{lr_y}} {{lr_y}} cr1 ;;
    BLT {{lr_y}} cr8 row_loop ;;

    ADD {{lr_wbase}} {{lr_wbase}} cr11 ;                 {#- next output channel: += NGROUPS -#}
    ADD {{lr_o}} {{lr_o}} cr1 ;;
    BLT {{lr_o}} cr10 ochan_loop ;;

end:
    BKPT ;;
