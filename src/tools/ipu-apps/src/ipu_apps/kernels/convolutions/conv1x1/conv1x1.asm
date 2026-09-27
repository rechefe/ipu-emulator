{#- ==========================================================================
    conv1x1.asm -- pointwise (1x1) FP32 convolution, all output channels

    out[o, y, x] = bias[o] + SUM_ci W[o, ci] * in[ci, y, x]

    A 1x1 convolution has no spatial window, so each output element is a pure
    channel-space dot product and every lane of a 128-element tile reduces
    independently. That makes the whole kernel one accumulation loop over input
    channels, with no taps, no border and no mask.

    MEMORY (all XMEM operands are ROW numbers; one row = 128 elements = 512 B
    in wide-vector debug mode -- see issue #179):
      Input   CR2: channel-major planes. Channel ci occupies PLANE_STRIDE rows
                   at CR2 + ci*PLANE_STRIDE; spatial row y occupies NCT rows at
                   +y*NCT, one row per 128-column tile. A spatial row shorter
                   than NCT*128 pads with idle lanes -- addressing is
                   row-granular, so there is no tighter packing to exploit.
                   ONE GUARD PLANE follows the last real channel (see below).
      Weight  CR4: NGROUPS rows per output channel at CR4 + o*NGROUPS. Row g
                   holds W[o, g*128 : (g+1)*128], zero-padded in the last group.
      Bias    CR5: one row per output channel; bias[o] in element 0.
      Output  CR3: same plane layout as the input, COUT planes, written in
                   (o, y, tile) order by a single running row counter.

    BIAS AND ACCUMULATOR RESET IN ONE WORD:
      There is no RESET_ACC in this ISA, so the reset folds into an
      ACC.ADD.FIRST. Rather than peel the first channel's MAC out of both
      loops to carry it, the bias supplies it:

        MULT.EE lr_bias cr1 0 lr0 cr15 ; ACC.ADD.FIRST ;;

      broadcasts R1[0] (= bias[o]) x CR1 (= 1.0) and overwrites R_ACC with it.
      Every MAC in every group is then a uniform ACC.ADD -- no peel, no
      duplicated group body -- and because the word touches no input data it
      cannot manufacture a 0*inf NaN the way a multiply-by-zero reset would.
      This also removes the bias-as-an-all-ones-input-channel trick the older
      byte-addressed kernel used, so Cin' == Cin (no synthetic plane).

    SOFTWARE-PIPELINED CHANNEL LOOP:
      MULT.RC.* reads R_CYCLIC from the start-of-word SNAPSHOT while
      LDR_CYCLIC_MULT_REG's offset is read LIVE, so a load co-issued with the
      multiply that consumes it would be one channel too late. The body instead
      advances lr_addr and loads channel c+1 in the same word whose multiply
      consumes channel c from the snapshot:

        ADD lr_addr +PLANE ; ADD lr_widx +1 ; ADD lr_c +1 ;
        LDR_CYCLIC(lr_addr) ; MULT.RC.VE(snapshot) ; ACC.ADD ;;

      The final iteration therefore prefetches channel index Cin, whose data is
      never consumed but whose row must still be in bounds -- hence the ONE
      GUARD PLANE the harness reserves past the last real channel.

      lr_widx is seeded to -1 (not 0) because MULT.RC.VE's `src` operand is
      resolved LIVE inside the handler, so the same-word increment lands before
      the read. The -1 is never itself read; the LR add wraps 0xFFFFFFFF -> 0.

    EXACT CHANNEL GROUPS (no padding):
      One LDR_MULT_REG row holds 128 FP32 weights and a 1x1 kernel needs one
      weight per channel, so a group covers up to 128 channels. The group size
      is computed exactly as min(128, Cin - done), so a 129-channel input runs
      128 then 1. R_ACC is never reset between groups -- the accumulation
      simply continues, which is what makes an arbitrary Cin work.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0  (zero source)          CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)       CR3  = OUTPUT_BASE (rows)
      CR4  = WEIGHT_BASE  (rows)       CR5  = BIAS_BASE   (rows)
      CR6  = PLANE_STRIDE = H*NCT (rows, input and output alike)
      CR7  = NCT = ceil(W/128)  (rows per spatial row; column-tile bound)
      CR8  = H (spatial rows)          CR9  = CIN
      CR10 = COUT                      CR11 = NGROUPS = ceil(CIN/128)
                                             (also the weight-row advance per
                                              output channel)
      CR12 = 128 (group cap; also the Ra element index selecting R1[0] = bias)
      CR13, CR14 = free
      CR15 = dstructure (valid_elements = 128), named by MULT/ACTIVATE

    The group weight stride is one row, which is CR1 -- it needs no CR of its
    own. CR6 = CR8*CR7 is precomputed by the harness because the LR slot has no
    multiply.

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND, which is why the store's offset register is never incremented in the
    store's own word.
    r_cyclic loads use index 0 in wide mode (full 512 B load).
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- constant 0: r_cyclic slot index and mask_shift -#}
{%- set lr_o     = "lr1"  -%}  {#- output channel; doubles as the bias row offset -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_wbase = "lr3"  -%}  {#- first weight row of this output channel -#}
{%- set lr_y     = "lr4"  -%}  {#- spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- input row base for this spatial row = y*NCT -#}
{%- set lr_ct    = "lr6"  -%}  {#- column tile: counter AND within-row row offset -#}
{%- set lr_addr  = "lr8"  -%}  {#- walking input row offset (channel being prefetched) -#}
{%- set lr_wrow  = "lr9"  -%}  {#- weight row for the current group -#}
{%- set lr_done  = "lr10" -%}  {#- channels completed across all groups so far -#}
{%- set lr_bound = "lr11" -%}  {#- this group's exact channel count -#}
{%- set lr_cin   = "lr12" -%}  {#- CIN (LR copy; SUB's src_a must be an LR) -#}
{%- set lr_widx  = "lr13" -%}  {#- weight element index inside R0, seeded -1 per group -#}
{%- set lr_c     = "lr14" -%}  {#- channel counter inside the current group -#}
{%- set lr_bias  = "lr15" -%}  {#- 128 -> Ra element 128 = R1[0] = bias[o] -#}

    SET {{lr_zero}}  cr0 ;
    SET {{lr_o}}     cr0 ;
    SET {{lr_out}}   cr0 ;;
    SET {{lr_wbase}} cr0 ;
    SET {{lr_cin}}   cr9 ;                               {#- LR copy of CIN for SUB/BLT -#}
    SET {{lr_bias}}  cr12 ;;                             {#- 128 = R1[0] selector -#}

ochan_loop:
    LDR_MULT_REG r1 {{lr_o}} cr5 ;;                      {#- R1 = bias row o (snapshot: visible NEXT word) -#}
    SET {{lr_y}}    cr0 ;
    SET {{lr_rowb}} cr0 ;;

row_loop:
    SET {{lr_ct}} cr0 ;;

coltile_loop:
{#- One output tile: R_ACC accumulates every input channel at this (y, tile). -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_ct}} ;              {#- channel 0's row for this tile -#}
    SET {{lr_done}} cr0 ;
    ADD {{lr_wrow}} {{lr_wbase}} cr0 ;;                  {#- wrow = wbase (ADD: SET takes a CR only) -#}
    MULT.EE {{lr_bias}} cr1 0 {{lr_zero}} cr15 ;
    ACC.ADD.FIRST ;;                                     {#- R_ACC = bias[o] -- reset and seed in one word -#}

group_loop:
{#- ---- exact group size: bound = min(128, CIN - done) -------------------- -#}
    SUB {{lr_bound}} {{lr_cin}} {{lr_done}} ;;           {#- remaining = CIN - done (>= 1) -#}
    BLT {{lr_bound}} cr12 group_size_set ;;              {#- if remaining < 128, keep it ... -#}
    SET {{lr_bound}} cr12 ;;                             {#- ... else cap bound = 128 -#}
group_size_set:
    LDR_MULT_REG r0 {{lr_wrow}} cr4 ;;                   {#- R0 = W[o][group] (visible NEXT word) -#}
    SUB {{lr_widx}} {{lr_zero}} cr1 ;                    {#- widx = -1 (live +1 lands before the read) -#}
    SET {{lr_c}} cr0 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- prime the pipeline: load channel `done` -#}

cin_loop:
{#- Load channel c+1 (offset read LIVE, so the same-word ADD applies) while the -#}
{#- multiply consumes channel c from the start-of-word r_cyclic SNAPSHOT.       -#}
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    ADD {{lr_widx}} {{lr_widx}} cr1 ;
    ADD {{lr_c}} {{lr_c}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;
    MULT.RC.VE {{lr_zero}} {{lr_widx}} 0 {{lr_zero}} cr15 ;
    ACC.ADD ;;
    BLT {{lr_c}} {{lr_bound}} cin_loop ;;                {#- BLT reads the snapshot -- lr_c is already advanced -#}

    ADD {{lr_done}} {{lr_done}} {{lr_bound}} ;           {#- done += this group -#}
    ADD {{lr_wrow}} {{lr_wrow}} cr1 ;;                   {#- next group's weight row -#}
    BLT {{lr_done}} {{lr_cin}} group_loop ;;

{#- ---- drain the tile. The store's offset is advanced in the NEXT word: LR -#}
{#- runs before STORE, so incrementing it here would store one row too far.  -#}
    ACTIVATE.QUANTIZE identity cr15 ;                    {#- FP32 pass-through (reads r_acc LIVE) -#}
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[o, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_ct}} {{lr_ct}} cr1 ;;
    BLT {{lr_ct}} cr7 coltile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr7 ;                    {#- next spatial row: += NCT -#}
    ADD {{lr_y}} {{lr_y}} cr1 ;;
    BLT {{lr_y}} cr8 row_loop ;;

    ADD {{lr_wbase}} {{lr_wbase}} cr11 ;                 {#- next output channel's weights: += NGROUPS -#}
    ADD {{lr_o}} {{lr_o}} cr1 ;;
    BLT {{lr_o}} cr10 ochan_loop ;;

end:
    BKPT ;;
