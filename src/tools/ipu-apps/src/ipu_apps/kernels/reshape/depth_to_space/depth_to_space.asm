{#- ==========================================================================
    depth_to_space.asm -- pixel shuffle (depth-to-space), FP32 wide-vector mode

    out[r*h + a, r*w + b] = in[r*a + b, h, w]     for a, b in [0, r)

    i.e. nn.PixelShuffle(r) with one output channel. SuperPoint's detector head
    is this with r = 8: the 64 sub-grid channels left after the dustbin is
    dropped become the full-resolution heatmap.

    WHY THIS NEEDS ACC.RESHAPE:
      One output row interleaves r input planes at stride r --
      out_row[r*w + b] = plane_b[w] -- and there is no scatter-store, no
      vector shuffle and no inverse of ACC.STRIDE (which decimates, it does not
      expand). ACC.RESHAPE is the only instruction that writes MULT_RES elements
      to ARBITRARY R_ACC indices: eight per instruction, via two LRDn register
      pairs read as eight source and eight destination byte indices.

    THE SOURCE INDICES ARE ALWAYS [0..7] -- THE SHIFT IS IN THE READ:
      Output tile T covers output columns 128T..128T+127, which come from input
      columns E*T .. E*T+E-1 of each plane, where E = 128/r. Those E columns all
      live in ONE input XMEM row (E divides 128, so the offset within the row is
      a multiple of E and at most 128-E), at element offset S.

      Rather than encoding S in the ACC.RESHAPE source array -- which would need
      a fresh pair of CR constants per tile -- S is the rc_idx of the
      MULT.RC.VE that stages the row, so MULT_RES[i] is already plane[S+i] and
      the source array is the constant [0..7], stepped by +8 per instruction.
      The same trick every kernel here uses for a horizontal shift.

    THE DESTINATION INDICES:
      Element j of plane b lands at output column r*j + b, so the destination
      array for plane b, instruction k is  r*(8k + i) + b  for i in 0..7:

        seeded  [0, r, 2r, ..., 7r]   from CR14 (low four) and CR7 (high four)
        per k   += 8r                 (ADDB CR10)
        per b   -= 127                undoing the r*E = 128 total drift, +1 for b

      After r planes the array is re-seeded from the CRs rather than stepped
      back, which is why no -r constant is needed.

    THE R_ACC IS NEVER CLEARED, AND DOES NOT NEED TO BE:
      Across b in [0, r) and j in [0, E) the destinations r*j + b cover
      0..r*E-1 = 0..127 exactly once, so every lane of the output row is
      written before it is stored.

    LOOP NEST (outermost first), chosen so the output row counter is monotonic:
      h    input spatial row       -- output rows r*h + a
      a    sub-row within the cell -- output row r*h + a
      wt   input tile              -- output tiles wt*r + sub
      sub  sub-tile                -- S = sub * E
      b    input plane             -- r loads per output tile
      k    ACC.RESHAPE instruction -- E/8 per plane

    ONE OUTPUT CHANNEL. A multi-channel shuffle would add an outer loop over
    c', offsetting the plane index by c'*r*r; the harness refuses it rather
    than the kernel silently computing the first channel only.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE (rows)
      CR4  = IN_PLANE_STRIDE = H * IN_TILES_PER_ROW
             -- also the h-loop bound, since hbase ends at exactly H*IN_TPR
      CR5  = r * IN_PLANE_STRIDE      (the a-loop step)
      CR6  = IN_TILES_PER_ROW         (h-loop step AND wt-loop bound)
      CR7  = DST_HI  = bytes [4r, 5r, 6r, 7r]
      CR8  = r        (a-loop, sub-loop and plane-loop bound -- all r)
      CR9  = E = 128 / r              (the S step)
      CR10 = 8 * r  (the per-instruction destination step). ADDB reads this
             as a SIGNED byte, so 8*r must be <= 127 -- which is what caps
             the upscale factor at 8 rather than at 16.
      CR11 = E/8 - 1                  (k-loop bound; pre-increment, so E/8 - 1)
      CR12 = SRC_LO  = bytes [0, 1, 2, 3]
      CR13 = SRC_HI  = bytes [4, 5, 6, 7]
      CR14 = DST_LO  = bytes [0, r, 2r, 3r]
      CR15 = dstructure (valid_elements = 128)

    LRDn is LR(n+1):LR(n) read as eight little-endian bytes, so LRD12 is
    LR12/LR13 and LRD14 is LR14/LR15. Both ACC.RESHAPE operands and ADDB's
    implicit source read the start-of-word SNAPSHOT, so an index array is always
    stepped in a different word from the instruction that uses it.

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_out   = "lr1"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_hbase = "lr2"  -%}  {#- h * IN_TILES_PER_ROW; also the h-loop bound test -#}
{%- set lr_pl0   = "lr3"  -%}  {#- hbase + a * r * IN_PLANE_STRIDE: plane r*a+0, row h -#}
{%- set lr_a     = "lr4"  -%}  {#- sub-row within the cell -#}
{%- set lr_wt    = "lr5"  -%}  {#- input tile counter -#}
{%- set lr_addr  = "lr6"  -%}  {#- walking input row address (steps by IN_PLANE_STRIDE) -#}
{%- set lr_b     = "lr7"  -%}  {#- input plane within the cell -#}
{%- set lr_k     = "lr8"  -%}  {#- ACC.RESHAPE instruction counter -#}
{%- set lr_s     = "lr9"  -%}  {#- element offset of this output tile inside the row -#}
{%- set lr_sub   = "lr10" -%}  {#- sub-tile counter (S = sub * E) -#}

    SET {{lr_zero}}  cr0 ;
    SET {{lr_out}}   cr0 ;
    SET {{lr_hbase}} cr0 ;;

h_loop:
    SET {{lr_a}} cr0 ;
    ADD {{lr_pl0}} {{lr_hbase}} cr0 ;;                   {#- a = 0; plane0 = hbase -#}

a_loop:
    SET {{lr_wt}} cr0 ;;

wt_loop:
    SET {{lr_s}}   cr0 ;
    SET {{lr_sub}} cr0 ;;                                {#- S = 0 at the start of each input tile -#}

sub_loop:
{#- ---- one output row tile: r planes interleaved at stride r --------------- -#}
    SET {{lr_b}} cr0 ;
    ADD {{lr_addr}} {{lr_pl0}} {{lr_wt}} ;;              {#- plane r*a+0, row h, tile wt -#}
    SET lr14 cr14 ;
    SET lr15 cr7 ;;                                      {#- dst = [0, r, 2r, ..., 7r] -#}

plane_loop:
    SET lr12 cr12 ;
    SET lr13 cr13 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- src = [0..7]; slot0 <- plane b -#}
    ADD {{lr_addr}} {{lr_addr}} cr4 ;
    ADD {{lr_b}} {{lr_b}} cr1 ;
    SET {{lr_k}} cr0 ;;                                  {#- next plane; k = 0 -#}
    MULT.RC.VE {{lr_s}} cr1 0 {{lr_zero}} cr15 ;;        {#- MULT_RES[i] = plane[S + i] -#}

reshape_loop:
    ACC.RESHAPE lrd12 lrd14 0 ;;                         {#- 8 elements -> their output lanes -#}
    ADDBI lrd12 8 ;
    ADDB lrd14 cr10 ;
    ADD {{lr_k}} {{lr_k}} cr1 ;
    BLT {{lr_k}} cr11 reshape_loop ;;                    {#- pre-increment: bound E/8 - 1 -#}

    ADDBI lrd14 -127 ;;                                  {#- undo the 128 drift, +1 for the next plane -#}
    BLT {{lr_b}} cr8 plane_loop ;;

    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[r*h + a, wt*r + sub] -#}
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_s}} {{lr_s}} cr9 ;
    ADD {{lr_sub}} {{lr_sub}} cr1 ;;
    BLT {{lr_sub}} cr8 sub_loop ;;

    ADD {{lr_wt}} {{lr_wt}} cr1 ;;
    BLT {{lr_wt}} cr6 wt_loop ;;

    ADD {{lr_pl0}} {{lr_pl0}} cr5 ;                      {#- next sub-row: += r * IN_PLANE_STRIDE -#}
    ADD {{lr_a}} {{lr_a}} cr1 ;;
    BLT {{lr_a}} cr8 a_loop ;;

    ADD {{lr_hbase}} {{lr_hbase}} cr6 ;;                 {#- next input row: += IN_TILES_PER_ROW -#}
    BLT {{lr_hbase}} cr4 h_loop ;;                       {#- hbase reaches H * IN_TPR exactly -#}

end:
    BKPT ;;
