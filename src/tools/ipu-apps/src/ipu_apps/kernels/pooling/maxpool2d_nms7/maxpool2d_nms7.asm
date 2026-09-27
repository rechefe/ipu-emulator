{#- ==========================================================================
    maxpool2d_nms7.asm -- 7x7 stride-1 centred max, FULLY UNROLLED

    out[c, y, x] = max over dy, dx in [0, 7) of pad(in)[c, y+dy-3, x+dx-3]

    SuperPoint's simple_nms window at nms_radius = 3, a tighter suppression than the default. This is the
    bespoke twin of maxpool2d_window: same result, K fixed at 7 and the taps
    unrolled, which buys three things a run-time K cannot have.

    1. NO SEPARATION WORD. The general kernel spends one dead word per row
       waiting for R_CYCLIC (its CONTENTS are read from the start-of-word
       snapshot, so a load needs a word before its first consumer). Unrolled,
       each row's load is issued inside the PREVIOUS row's tap stream and lands
       nine words early, so the wait costs nothing.

    2. NO SEED ROW, NO SEED WORD. The general kernel's tap loop is uniform, so
       it cannot peel the first tap to carry ACC.MAX.FIRST -- it keeps a
       resident -FLT_MAX row and spends a word maxing against it once per tile.
       Here tap (0,0) simply IS ACC.MAX.FIRST.

    3. NO GUARD ROW. A rolling prefetch would read one row past the plane on
       the last iteration. Unrolled, the last two rows just do not issue a
       load.

        per output tile:   K^2 + 3K + 5 = 75 words    ->   K^2 + 5 = 54
        480x640, 1 plane:        217,447 cycles       ->   ~157,000  (-28%)

    THREE ROTATING R_CYCLIC SLOTS, NOT TWO:
      Row dy is read from slot dy%3 while row dy+2 is loaded into slot (dy+2)%3
      -- the slot holding row dy-1, whose taps are finished. Two slots cannot
      do this: the one not being read holds the row needed next.

        dy=0  read slot0   load row2 -> slot2
        dy=1  read slot1   load row3 -> slot0   (row 0 is done)
        dy=2  read slot2   load row4 -> slot1   (row 1 is done)

      Rows 0 and 1 are preloaded before the tap stream. The rc base for row dy
      is therefore 128*(dy%3), copied from the LR holding it rather than SET
      from a CR, which keeps the CR map down to seven entries.

    HALO TILING is maxpool2d_window's, unchanged: TC = 128-6 = 122 output
    columns per row, element e of tile t is input column t*122 + e - 3, and the
    largest element a valid lane reads is 121+6 = 127. The P=3 border rows and
    the halo elements hold -FLT_MAX.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = 128 (R_CYCLIC slot stride)
      CR5  = TILES_PER_ROW -- tile-loop bound AND the row step between dy taps
      CR6  = IN_PLANE_STRIDE = (H + 6) * TILES_PER_ROW
      CR7  = HEIGHT                   CR8  = CHANNELS
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
========================================================================== -#}

{%- set K = 7 -%}
{%- set lr_zero  = "lr0"  -%}  {#- 0: mask_shift, slot-0 index, rc base for slot 0 -#}
{%- set lr_c     = "lr1"  -%}  {#- channel counter -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter -#}
{%- set lr_cbase = "lr3"  -%}  {#- first input row of this channel's plane -#}
{%- set lr_y     = "lr4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- input row of padded row y, tile 0 -#}
{%- set lr_t     = "lr6"  -%}  {#- tile counter AND within-row row offset -#}
{%- set lr_addr  = "lr7"  -%}  {#- walking input row address -#}
{%- set lr_rc    = "lr8"  -%}  {#- R_CYCLIC read index -#}
{%- set slots    = ["lr0", "lr9", "lr10"] -%}  {#- slot indices 0, 128, 256 -#}

    SET {{lr_zero}}  cr0 ;
    SET {{lr_c}}     cr0 ;
    SET {{lr_out}}   cr0 ;;
    SET {{lr_cbase}} cr0 ;
    SET lr9 cr4 ;;                                       {#- 128 -#}
    ADD lr10 lr9 lr9 ;;                                  {#- 256 -#}

chan_loop:
    SET {{lr_y}} cr0 ;
    ADD {{lr_rowb}} {{lr_cbase}} cr0 ;;

row_loop:
    SET {{lr_t}} cr0 ;;

tile_loop:
{#- ---- preload rows 0 and 1 into slots 0 and 1 --------------------------- -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{slots[0]}} ;;  {#- slot0 <- padded row y+0 -#}
    ADD {{lr_addr}} {{lr_addr}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{slots[1]}} ;;  {#- slot1 <- padded row y+1 -#}
{%- for dy in range(K) %}
{#- row {{dy}}: read slot {{dy % 3}}{% if dy + 2 < K %}, load row {{dy + 2}} into slot {{(dy + 2) % 3}}{% endif %} -#}
    ADD {{lr_rc}} {{slots[dy % 3]}} cr0 ;
{%- if dy + 2 < K %}
    ADD {{lr_addr}} {{lr_addr}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{slots[(dy + 2) % 3]}} ;
{%- endif %}
    MULT.RC.VE {{lr_rc}} cr1 0 {{lr_zero}} cr15 ; {% if dy == 0 %}ACC.MAX.FIRST{% else %}ACC.MAX{% endif %} ;;
{%- for dx in range(1, K) %}
    ADD {{lr_rc}} {{lr_rc}} cr1 ;
    MULT.RC.VE {{lr_rc}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
{%- endfor %}
{%- endfor %}

    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[c, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_t}} {{lr_t}} cr1 ;;
    BLT {{lr_t}} cr5 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr5 ;
    ADD {{lr_y}} {{lr_y}} cr1 ;;
    BLT {{lr_y}} cr7 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} cr6 ;
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    BLT {{lr_c}} cr8 chan_loop ;;

end:
    BKPT ;;
