{#- ==========================================================================
    maxpool2d_window.asm -- stride-1 KxK windowed max, FP32 wide-vector mode

    out[c, y, x] = max over dy, dx in [0, K) of pad(in)[c, y+dy-P, x+dx-P]

    with K odd and P = K//2, so the output is the same H x W as the input and
    every output is the maximum of the window CENTRED on it. That is
    SuperPoint's `simple_nms` pool (K = 2*nms_radius+1, so K = 9 at the default
    radius 4), and any other stride-1 local-maximum pool.

    K IS A RUN-TIME BOUND, NOT AN UNROLL:
      The taps are two nested loops (dy outer, dx inner) reading CR9 = K, so one
      assembled binary serves every window size. The 3x3 convolution kernel can
      afford to unroll its nine taps because nine is fixed; here K reaches 9 or
      more and, more to the point, R_CYCLIC holds only four 128-element slots --
      so K rows cannot be resident at once for K > 4 and the rows have to be
      streamed one at a time anyway.

      Only ONE rotating slot is therefore used (slot 0): load padded row y+dy,
      take its K horizontal taps, load the next. Slot 3 holds a resident
      -FLT_MAX row (see below) and slots 1 and 2 are unused.

    SEEDING THE MAXIMUM:
      The tap loop is uniform -- every tap is an ACC.MAX -- so R_ACC has to be
      seeded before the first one rather than by peeling a tap the loop bounds
      cannot express. One resident row of -FLT_MAX, loaded into R_CYCLIC slot 3
      once at startup, gives an ACC.MAX.FIRST word per output tile that costs a
      single cycle and cannot win any subsequent maximum.

    HALO TILING (what makes the horizontal shift work at a tile edge):
      A spatial row is cut into tiles of TC = 128 - (K-1) output columns, each
      stored as one 128-element XMEM row:

        element e of tile t = input column (t*TC + e - P)

      Output lane j (0..TC-1) is column t*TC + j, and tap dx reads element
      j + dx -- so every tap of every valid lane is satisfied from within the
      one row, with no neighbouring-tile dependency. The largest element any
      valid lane reads is (TC-1) + (K-1) = 127, exactly the last element of the
      slot. Columns outside the image are stored as -FLT_MAX, which IS the
      window's border: the identity of a maximum can never win.

    VERTICAL BORDER COSTS NOTHING:
      Each input plane carries P rows of -FLT_MAX above and below (the plane is
      H + 2P spatial rows). Output row y then reads padded rows y .. y+K-1
      unconditionally -- there is no top/bottom special case anywhere.

    THE TWO LOOP BOUNDS ARE NOT THE SAME NUMBER, AND THAT IS DELIBERATE:
      BLT reads its register from the start-of-word SNAPSHOT. The dx counter is
      incremented in the same word as its own BLT, so the branch sees the
      PRE-increment value and must compare against K-1 (CR11). The dy counter is
      incremented several words before its BLT, so that branch sees the
      POST-increment value and compares against K (CR9). Making both compare
      against K would run one tap too many per row.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = SEED_ROW     (rows)      -- one row of -FLT_MAX
      CR5  = TILES_PER_ROW -- tile-loop bound AND the row step between dy taps
      CR6  = IN_PLANE_STRIDE = (H + 2P) * TILES_PER_ROW
      CR7  = HEIGHT (output rows == input rows)
      CR8  = CHANNELS                 CR9  = K   (dy bound)
      CR10 = 384 (R_CYCLIC slot-3 index, where the -FLT_MAX seed lives)
      CR11 = K - 1 (dx bound)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND. That is why a load can share a word with the LR add that positions it
    (LDR offsets read LIVE), why R_CYCLIC needs one word of separation before
    the multiply that consumes it (its CONTENTS are read from the snapshot),
    and why the store's row counter is incremented in the word AFTER the store.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- 0: mask_shift, slot-0 index, seed-row offset -#}
{%- set lr_c     = "lr1"  -%}  {#- channel counter -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_cbase = "lr3"  -%}  {#- first input row of this channel's plane -#}
{%- set lr_y     = "lr4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- input row of padded row y, tile 0 -#}
{%- set lr_t     = "lr6"  -%}  {#- tile counter AND within-row row offset -#}
{%- set lr_addr  = "lr7"  -%}  {#- walking input row address (steps by TPR per dy) -#}
{%- set lr_dy    = "lr8"  -%}  {#- vertical tap index -#}
{%- set lr_dx    = "lr9"  -%}  {#- horizontal tap index -#}
{%- set lr_rc    = "lr10" -%}  {#- R_CYCLIC read index; equals dx within slot 0 -#}
{%- set lr_seed  = "lr11" -%}  {#- 384: slot-3 index, the resident -FLT_MAX row -#}

    SET {{lr_zero}}  cr0 ;
    SET {{lr_c}}     cr0 ;
    SET {{lr_out}}   cr0 ;;
    SET {{lr_cbase}} cr0 ;
    SET {{lr_seed}}  cr10 ;;                             {#- 384 -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr4 {{lr_seed}} ;;   {#- slot3 <- -FLT_MAX, resident -#}

chan_loop:
    SET {{lr_y}} cr0 ;
    ADD {{lr_rowb}} {{lr_cbase}} cr0 ;;                  {#- rowb = this channel's plane base -#}

row_loop:
    SET {{lr_t}} cr0 ;;

tile_loop:
    ADD {{lr_addr}} {{lr_rowb}} {{lr_t}} ;
    SET {{lr_dy}} cr0 ;;                                 {#- padded row y, tile t -#}
    MULT.RC.VE {{lr_seed}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
                                                         {#- R_ACC = -FLT_MAX -#}

dy_loop:
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;
    SUB {{lr_rc}} {{lr_zero}} cr1 ;
    SET {{lr_dx}} cr0 ;;                                 {#- slot0 <- padded row y+dy -#}
    ADD {{lr_addr}} {{lr_addr}} cr5 ;
    ADD {{lr_dy}} {{lr_dy}} cr1 ;;                       {#- next row; slot0 visible NEXT word -#}
                                                         {#- rc = -1: the live +1 below lands first -#}

dx_loop:
    ADD {{lr_rc}} {{lr_rc}} cr1 ;
    ADD {{lr_dx}} {{lr_dx}} cr1 ;
    MULT.RC.VE {{lr_rc}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    BLT {{lr_dx}} cr11 dx_loop ;;                        {#- pre-increment: bound is K-1 -#}

    BLT {{lr_dy}} cr9 dy_loop ;;                         {#- post-increment: bound is K -#}

    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[c, y, tile] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_t}} {{lr_t}} cr1 ;;
    BLT {{lr_t}} cr5 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr5 ;                    {#- next spatial row: += TPR -#}
    ADD {{lr_y}} {{lr_y}} cr1 ;;
    BLT {{lr_y}} cr7 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} cr6 ;                  {#- next channel plane -#}
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    BLT {{lr_c}} cr8 chan_loop ;;

end:
    BKPT ;;
