{#- conv1x1.asm -- out[o, y, x] = bias[o] + SUM_ci W[o, ci] * in[ci, y, x]: pointwise (1x1) FP32 convolution, all output channels.

    A 1x1 convolution has no spatial window, so each output element is a pure
    channel-space dot product and every lane of a 128-element tile reduces
    independently. That makes the whole kernel one accumulation loop over input
    channels, with no taps, no border and no mask. The query vocabulary is the
    tiled one (in_channels, height, width, out_channels, kernel_size = 1,
    stride = 1, padding = 0, activation = none); the universal conv2d kernels
    take a query that also names dilation, groups, has_bias and apply_relu, so
    each vocabulary reaches only its own kernels.

    Design notes

    - MEMORY. All XMEM operands are ROW numbers; one row is 128 elements (512 B
      in FP32, issue #179). The harness places and packs the regions
      (kernel.toml), back to back from row 0 in the order x, w, b, y:
        x: channel-major planes (pack tiled_guard). Channel ci occupies
           plane_stride = height * tiles rows at x + ci*plane_stride; spatial
           row y occupies tiles = ceil(width / 128) rows at +y*tiles, one row
           per 128-column tile. A spatial row shorter than tiles*128 pads
           with idle lanes: addressing is row-granular, so there is no
           tighter packing to exploit. ONE GUARD PLANE (all zero) follows
           the last real channel, inside x's region (see below).
        w: channel_groups = ceil(in_channels / 128) rows per output channel
           at w + o*channel_groups. Row g holds W[o, g*128 : (g+1)*128],
           zero-padded in the last group (pack conv_weight_groups).
        b: one row per output channel; bias[o] in element 0 (pack lane0_rows).
        y: the input's plane layout, out_channels planes, written in
           (o, y, tile) order by a single running row counter.

    - BIAS AND ACCUMULATOR RESET IN ONE WORD. There is no RESET_ACC in this
      ISA, so the reset folds into an ACC.ADD.FIRST. Rather than peel the
      first channel's MAC out of both loops to carry it, the bias supplies it:

        MULT.EE lr_bias CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;

      broadcasts R1[0] (= bias[o]) x CR1 (= 1.0) and overwrites R_ACC with it.
      Every MAC in every group is then a uniform ACC.ADD -- no peel, no
      duplicated group body -- and because the word touches no input data it
      cannot manufacture a 0*inf NaN the way a multiply-by-zero reset would.
      This also removes the bias-as-an-all-ones-input-channel trick the older
      byte-addressed kernel used, so Cin' == Cin (no synthetic plane).

    - SOFTWARE-PIPELINED CHANNEL LOOP. MULT.RC.* reads R_CYCLIC from the
      start-of-word SNAPSHOT while LD.MULT.CYCLIC's offset is read LIVE, so a
      load co-issued with the multiply that consumes it would be one channel
      too late. The body instead advances lr_addr and loads channel c+1 in the
      same word whose multiply consumes channel c from the snapshot:

        ADD lr_addr +plane_stride ; ADD lr_widx +1 ; ADD lr_c +1 ;
        LD.MULT.CYCLIC(lr_addr) ; MULT.RC.VE(snapshot) ; ACC.ADD ;;

      The final iteration therefore prefetches channel index in_channels,
      whose data is never consumed but whose row must still be in bounds --
      hence the ONE GUARD PLANE past the last real channel.

      lr_widx is seeded to -1 (not 0) because MULT.RC.VE's `src` operand is
      resolved LIVE inside the handler, so the same-word increment lands before
      the read. The -1 is never itself read; the LR add wraps 0xFFFFFFFF -> 0.

    - EXACT CHANNEL GROUPS (no padding). One LD.MULT row holds 128 FP32
      weights and a 1x1 kernel needs one weight per channel, so a group covers
      up to 128 channels. The group size is computed exactly as
      min(128, in_channels - done), so a 129-channel input runs 128 then 1.
      R_ACC is never reset between groups -- the accumulation simply
      continues, which is what makes an arbitrary in_channels work.

    - Registers ([abi]): lanes = 128 is the group cap and also the Ra element
      index selecting R1[0] = bias; channel_groups is also the weight-row
      advance per output channel. The group weight stride is one row, which is
      CR1 -- it needs no CR of its own. plane_stride = height * tiles is a
      [derived] value because the LR slot has no multiply. CR15 is the default
      dstructure (valid_elements = 128), named by MULT/ACTIVATE.

    - LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates
      sub-instructions. Slot order within a word is LR -> LOAD -> MULT -> ACC
      -> AAQ -> STORE -> COND, which is why the store's offset register is
      never incremented in the store's own word. R_CYCLIC loads use index 0 in
      wide mode (full 512 B load).
-#}

{%- set lr_zero  = "LR0"  -%}  {#- constant 0: r_cyclic slot index and mask_shift -#}
{%- set lr_o     = "LR1"  -%}  {#- output channel; doubles as the bias row offset -#}
{%- set lr_out   = "LR2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_wbase = "LR3"  -%}  {#- first weight row of this output channel -#}
{%- set lr_y     = "LR4"  -%}  {#- spatial row -#}
{%- set lr_rowb  = "LR5"  -%}  {#- input row base for this spatial row = y*NCT -#}
{%- set lr_ct    = "LR6"  -%}  {#- column tile: counter AND within-row row offset -#}
{%- set lr_addr  = "LR8"  -%}  {#- walking input row offset (channel being prefetched) -#}
{%- set lr_wrow  = "LR9"  -%}  {#- weight row for the current group -#}
{%- set lr_done  = "LR10" -%}  {#- channels completed across all groups so far -#}
{%- set lr_bound = "LR11" -%}  {#- this group's exact channel count -#}
{%- set lr_cin   = "LR12" -%}  {#- CIN (LR copy; SUB's src_a must be an LR) -#}
{%- set lr_widx  = "LR13" -%}  {#- weight element index inside R0, seeded -1 per group -#}
{%- set lr_c     = "LR14" -%}  {#- channel counter inside the current group -#}
{%- set lr_bias  = "LR15" -%}  {#- 128 -> Ra element 128 = R1[0] = bias[o] -#}

    SET {{lr_zero}}  CR0 ;
    SET {{lr_o}}     CR0 ;
    SET {{lr_out}}   CR0 ;;
    SET {{lr_wbase}} CR0 ;
    SET {{lr_cin}}   {{ cr.in_channels }} ;                               {#- LR copy of CIN for SUB/BLT.S -#}
    SET {{lr_bias}}  {{ cr.lanes }} ;;                             {#- 128 = R1[0] selector -#}

ochan_loop:
    LD.MULT R1 {{lr_o}} {{ cr.b }} ;;                      {#- R1 = bias row o (snapshot: visible NEXT word) -#}
    SET {{lr_y}}    CR0 ;
    SET {{lr_rowb}} CR0 ;;

row_loop:
    SET {{lr_ct}} CR0 ;;

coltile_loop:
{#- One output tile: R_ACC accumulates every input channel at this (y, tile). -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_ct}} ;              {#- channel 0's row for this tile -#}
    SET {{lr_done}} CR0 ;
    ADD {{lr_wrow}} {{lr_wbase}} CR0 ;;                  {#- wrow = wbase (ADD: SET takes a CR only) -#}
    MULT.EE {{lr_bias}} CR1 0 {{lr_zero}} CR15 ;
    ACC.ADD.FIRST ;;                                     {#- R_ACC = bias[o] -- reset and seed in one word -#}

group_loop:
{#- ---- exact group size: bound = min(128, CIN - done) -------------------- -#}
    SUB {{lr_bound}} {{lr_cin}} {{lr_done}} ;;           {#- remaining = CIN - done (>= 1) -#}
    BLT.S {{lr_bound}} {{ cr.lanes }} group_size_set ;;              {#- if remaining < 128, keep it ... -#}
    SET {{lr_bound}} {{ cr.lanes }} ;;                             {#- ... else cap bound = 128 -#}
group_size_set:
    LD.MULT R0 {{lr_wrow}} {{ cr.w }} ;;                   {#- R0 = W[o][group] (visible NEXT word) -#}
    SUB {{lr_widx}} {{lr_zero}} CR1 ;                    {#- widx = -1 (live +1 lands before the read) -#}
    SET {{lr_c}} CR0 ;
    LD.MULT.CYCLIC {{lr_addr}} {{ cr.x }} {{lr_zero}} ;;   {#- prime the pipeline: load channel `done` -#}

cin_loop:
{#- Load channel c+1 (offset read LIVE, so the same-word ADD applies) while the -#}
{#- multiply consumes channel c from the start-of-word r_cyclic SNAPSHOT.       -#}
    ADD {{lr_addr}} {{lr_addr}} {{ cr.plane_stride }} ;
    ADD {{lr_widx}} {{lr_widx}} CR1 ;
    ADD {{lr_c}} {{lr_c}} CR1 ;
    LD.MULT.CYCLIC {{lr_addr}} {{ cr.x }} {{lr_zero}} ;
    MULT.RC.VE {{lr_zero}} {{lr_widx}} 0 {{lr_zero}} CR15 ;
    ACC.ADD ;;
    BLT.S {{lr_c}} {{lr_bound}} cin_loop ;;                {#- BLT.S reads the snapshot -- lr_c is already advanced -#}

    ADD {{lr_done}} {{lr_done}} {{lr_bound}} ;           {#- done += this group -#}
    ADD {{lr_wrow}} {{lr_wrow}} CR1 ;;                   {#- next group's weight row -#}
    BLT.S {{lr_done}} {{lr_cin}} group_loop ;;

{#- ---- drain the tile. The store's offset is advanced in the NEXT word: LR -#}
{#- runs before STORE, so incrementing it here would store one row too far.  -#}
    ACTIVATE.QUANTIZE identity CR15 ;                    {#- FP32 pass-through (reads r_acc LIVE) -#}
    ST.AAQ {{lr_out}} {{ cr.y }} ;;                   {#- OUT[o, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} CR1 ;
    ADD {{lr_ct}} {{lr_ct}} CR1 ;;
    BLT.S {{lr_ct}} {{ cr.tiles }} coltile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} {{ cr.tiles }} ;                    {#- next spatial row: += NCT -#}
    ADD {{lr_y}} {{lr_y}} CR1 ;;
    BLT.S {{lr_y}} {{ cr.height }} row_loop ;;

    ADD {{lr_wbase}} {{lr_wbase}} {{ cr.channel_groups }} ;                 {#- next output channel's weights: += NGROUPS -#}
    ADD {{lr_o}} {{lr_o}} CR1 ;;
    BLT.S {{lr_o}} {{ cr.out_channels }} ochan_loop ;;

end:
    HALT ;;
