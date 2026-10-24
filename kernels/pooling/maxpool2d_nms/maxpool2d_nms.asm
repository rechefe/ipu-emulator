{#- ==========================================================================
    maxpool2d_nms.asm -- KxK stride-1 centred max, FULLY UNROLLED

    out[c, y, x] = max over dy, dx in [0, K) of pad(in)[c, y+dy-P, x+dx-P]

    K = kernel_size and P = padding = K//2. The template of the registry.toml
    entries maxpool2d_nms7 (K = 7) and maxpool2d_nms9 (K = 9): SuperPoint's
    simple_nms window at nms_radius = 3 (a tighter suppression than the
    default) and at its default nms_radius = 4. This is the bespoke twin of
    maxpool2d_window: same result, K fixed per entry and the taps unrolled,
    which buys three things a run-time K cannot have.

    1. NO SEPARATION WORD. The general kernel spends one dead word per row
       waiting for R_CYCLIC (its CONTENTS are read from the start-of-word
       snapshot, so a load needs a word before its first consumer). Unrolled,
       each row's load is issued inside the PREVIOUS row's tap stream and lands
       K words early, so the wait costs nothing.

    2. NO SEED ROW, NO SEED WORD. The general kernel's tap loop is uniform, so
       it cannot peel the first tap to carry ACC.MAX.FIRST -- it keeps a
       resident -FLT_MAX row and spends a word maxing against it once per tile.
       Here tap (0,0) simply IS ACC.MAX.FIRST.

    3. NO GUARD ROW. A rolling prefetch would read one row past the plane on
       the last iteration. Unrolled, the last two rows just do not issue a
       load.

        per output tile:   K^2 + 3K + 5 words (window)  ->  K^2 + 5 (unrolled)
                     K = 7:        75 words  ->  54
                     K = 9:       113 words  ->  86
        480x640, 1 plane, K = 7:  217,447 cycles  ->  ~157,000  (-28%)
        480x640, 1 plane, K = 9:  326,887 cycles  ->  ~249,000  (-24%)

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

    HALO TILING is maxpool2d_window's, unchanged: TC = 128-(K-1) output
    columns per row (122 for K = 7, 120 for K = 9), element e of tile t is
    input column t*TC + e - P, and the largest element a valid lane reads is
    (TC-1)+(K-1) = 127. The P border rows and the halo elements hold -FLT_MAX
    (the family pack transform halo_tiled, ../pack.py).

    ABI (kernel.toml [abi]; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = 128 (R_CYCLIC slot stride)
      CR5  = TILES_PER_ROW -- tile-loop bound AND the row step between dy taps
      CR6  = IN_PLANE_STRIDE = (H + K - 1) * TILES_PER_ROW
      CR7  = HEIGHT                   CR8  = CHANNELS
      CR15.L = the default DS (ds(128)); CR14 = the QF

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
========================================================================== -#}

{%- set K = kernel_size -%}
{%- set lr_zero  = "LR0"  -%}  {#- 0: mask_shift, slot-0 index, rc base for slot 0 -#}
{%- set lr_c     = "LR1"  -%}  {#- channel counter -#}
{%- set lr_out   = "LR2"  -%}  {#- running OUTPUT row counter -#}
{%- set lr_cbase = "LR3"  -%}  {#- first input row of this channel's plane -#}
{%- set lr_y     = "LR4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "LR5"  -%}  {#- input row of padded row y, tile 0 -#}
{%- set lr_t     = "LR6"  -%}  {#- tile counter AND within-row row offset -#}
{%- set lr_addr  = "LR7"  -%}  {#- walking input row address -#}
{%- set lr_rc    = "LR8"  -%}  {#- R_CYCLIC read index -#}
{%- set slots    = ["LR0", "LR9", "LR10"] -%}  {#- slot indices 0, 128, 256 -#}

    SET {{lr_zero}}  CR0 ;
    SET {{lr_c}}     CR0 ;
    SET {{lr_out}}   CR0 ;;
    SET {{lr_cbase}} CR0 ;
    SET LR9 CR4 ;;                                       {#- 128 -#}
    ADD LR10 LR9 LR9 ;;                                  {#- 256 -#}

chan_loop:
    SET {{lr_y}} CR0 ;
    ADD {{lr_rowb}} {{lr_cbase}} CR0 ;;

row_loop:
    SET {{lr_t}} CR0 ;;

tile_loop:
{#- ---- preload rows 0 and 1 into slots 0 and 1 --------------------------- -#}
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{slots[0]}} ;;  {#- slot0 <- padded row y+0 -#}
    ADD {{lr_addr}} {{lr_addr}} CR5 ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{slots[1]}} ;;  {#- slot1 <- padded row y+1 -#}
{%- for dy in range(K) %}
{#- row {{dy}}: read slot {{dy % 3}}{% if dy + 2 < K %}, load row {{dy + 2}} into slot {{(dy + 2) % 3}}{% endif %} -#}
    ADD {{lr_rc}} {{slots[dy % 3]}} CR0 ;
{%- if dy + 2 < K %}
    ADD {{lr_addr}} {{lr_addr}} CR5 ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{slots[(dy + 2) % 3]}} ;
{%- endif %}
    MULT.RC.VE {{lr_rc}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; {% if dy == 0 %}ACC.MAX.FIRST{% else %}ACC.MAX{% endif %} ;;
{%- for dx in range(1, K) %}
    ADD {{lr_rc}} {{lr_rc}} CR1 ;
    MULT.RC.VE {{lr_rc}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
{%- endfor %}
{%- endfor %}

    ACTIVATE.EXACT identity CR15.L CR3 {{lr_out}} ;; {#- OUT[c, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} CR1 ;
    ADD {{lr_t}} {{lr_t}} CR1 ;;
    BLT.S {{lr_t}} CR5 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} CR5 ;
    ADD {{lr_y}} {{lr_y}} CR1 ;;
    BLT.S {{lr_y}} CR7 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} CR6 ;
    ADD {{lr_c}} {{lr_c}} CR1 ;;
    BLT.S {{lr_c}} CR8 chan_loop ;;

end:
    HALT ;;
