{#- ==========================================================================
    maxpool2d_stride2.asm -- 2x2 stride-2 max-pool, FP32 wide-vector mode

    out[c, y, x] = max( in[c, 2y,   2x], in[c, 2y,   2x+1],
                        in[c, 2y+1, 2x], in[c, 2y+1, 2x+1] )

    SuperPoint applies this three times (after conv1b, conv2b, conv3b). One
    launch produces the whole (C, H//2, W//2) output.

    WHY THE FOUR TAPS COST NOTHING EXTRA:
      XMEM operands are ROW numbers -- a load addresses whole 128-element rows
      and cannot shift by one element. MULT.RC.* instead reads R_CYCLIC at an
      arbitrary ELEMENT index and may cross slot boundaries, so the horizontal
      shift happens inside the register: the two vertically-neighbouring rows
      occupy R_CYCLIC slots 0 and 2, and dx becomes a +1 step on the read index.
      `MULT.RC.VE rc, cr1` is the identity move (R_CYCLIC x 1.0), so a tap is one
      MULT plus one ACC.MAX. The +1 read's final temporary position enters the
      unused next slot, but ACC.STRIDE always discards that position.

      ACC.MAX.FIRST seeds R_ACC from the first tap, so no -inf seed vector is
      needed and nothing carries over from the previous tile.

    THE DECIMATION (this is the whole difficulty):
      Four taps give the STRIDE-1 2x2 maximum at every column; a stride-2 pool
      wants only the even ones. ACC.STRIDE does exactly that decimation -- but
      it writes MULT_RES into R_ACC, OVERWRITING, so it cannot itself take a
      max and the max must already be finished. Hence the round trip:

        1. four taps -> R_ACC = stride-1 2x2 max at every column
        2. ACTIVATE identity + STR_POST_AAQ  -> a scratch row
        3. reload the scratch, MULT.RC.VE x 1.0, ACC.STRIDE 64 on off

      `ACC.STRIDE 64 on off` reads MULT_RES as two rows of 64, takes every
      second element of each, and writes the 64 survivors CONTIGUOUSLY at
      R_ACC[(offset % 4) * 32]. That is exactly positions 0,2,...,126 packed down
      to 64 -- the even columns, which are the stride-2 outputs.

      BOTH HALVES ARE STAGED BEFORE EITHER ACC.STRIDE. Half B's ACC.MAX.FIRST
      overwrites all 128 R_ACC elements, so it would destroy half A's result if
      half A had already been decimated into R_ACC[0:64]. Staging both, then
      decimating both, works because ACC.STRIDE leaves the R_ACC indices it does
      NOT write untouched: half A lands at base 0 (offset 0) and half B at base
      64 (offset 2), filling one 128-element output XMEM row between them.

    TILING:
      One output XMEM row is 128 output columns, so it spans 256 input columns
      = two full-width input tiles ("halves"). Output tile ot reads input tiles
      2*ot (half A) and 2*ot+1 (half B).

      This full-pair variant is selected only when both input tiles exist for
      every output XMEM row. The separate tail variant handles a final output
      XMEM row backed by only one input XMEM row. Both variants use exactly
      ceil(W/128) input XMEM rows per matrix row, with no guard row.

      Positions past W inside the final partly-filled XMEM row are -FLT_MAX.
      They never reach a kept output position: output element j < W//2 reads
      input elements 2j and 2j+1, both < W.

      NO VERTICAL BORDER: padding is 0, so rows 2y and 2y+1 are always real
      image rows and there is no top/bottom special case. An odd H or W simply
      drops the last row/column, which is what nn.MaxPool2d(2, 2) does.

    R_CYCLIC SLOTS (per half):
      slot 0 (idx   0) <- spatial row 2y,   current input XMEM row
      slot 2 (idx 256) <- spatial row 2y+1, current input XMEM row

      Taps read rc = 0, 1, 256, 257. LDR_CYCLIC_MULT_REG's index is restricted
      to slot boundaries {0,128,256,384}; MULT.RC.VE's rc_idx is not, which is
      what rc=1 and rc=257 rely on. Their final temporary positions enter the
      unused slots 1/3 and are discarded by ACC.STRIDE. R_CYCLIC contents are
      read from the start-of-word SNAPSHOT, so every load lands at least one
      word before the multiply that consumes it.

    CR map (set by the harness; {{cr_zero}}/{{cr_one}}are READ-ONLY hardware constants):
      CR0  = 0                        CR1 = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE   (rows)      CR3  = OUTPUT_BASE  (rows)
      CR4  = SCRATCH_BASE (rows; 2 rows)
      CR5  = IN_ROW_STRIDE = ceil(W/128) XMEM rows per spatial row
      CR6  = 2 * IN_ROW_STRIDE -- advance from spatial row 2y to 2(y+1)
      CR7  = OUT_TILES_PER_ROW        CR8  = OUT_HEIGHT = H // 2
      CR9  = CHANNELS                 CR10 = 128
      CR11 = IN_PLANE_STRIDE = H * IN_ROW_STRIDE
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND. That is why a load can share a word with the LR add that positions it
    (LDR offsets read LIVE), and why the store's row counter is incremented in
    the word AFTER the store.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}  {#- 0: mask_shift, slot-0 index, tap (0,0), scratch row A, ACC.STRIDE base 0 -#}
{%- set lr_c     = "lr1"  -%}  {#- channel counter -#}
{%- set lr_out   = "lr2"  -%}  {#- running OUTPUT row counter (one row per store) -#}
{%- set lr_cbase = "lr3"  -%}  {#- first input row of this channel's plane -#}
{%- set lr_oy    = "lr4"  -%}  {#- output spatial row -#}
{%- set lr_rowb  = "lr5"  -%}  {#- input row of spatial row 2*oy, tile 0 -#}
{%- set lr_ot    = "lr6"  -%}  {#- output tile counter -#}
{%- set lr_addr  = "lr7"  -%}  {#- walking input row address -#}
{%- set lr_s1    = "lr8"  -%}  {#- 128: R_CYCLIC slot-1 index -#}
{%- set lr_s2    = "lr9"  -%}  {#- 256: R_CYCLIC slot-2 index AND tap (1,0) -#}
{%- set lr_baddr = "lr10" -%}  {#- walking input address for spatial row 2y+1 -#}
{%- set lr_one   = "lr11" -%}  {#- 1: tap (0,1) AND scratch row B offset -#}
{%- set lr_s2p1  = "lr12" -%}  {#- 257: tap (1,1) -#}
{%- set lr_half  = "lr13" -%}  {#- 2: ACC.STRIDE offset selecting R_ACC base 64 -#}
{%- set lr_itile = "lr14" -%}  {#- address of (spatial row 2*oy, input tile 2*ot) -#}

{%- set cr_zero  = "cr0"  -%}
{%- set cr_one  = "cr1"  -%}
{%- set cr_input_base_address  = "cr2"  -%}
{%- set cr_xmem_row_stride  = "cr5"  -%}  {#- how many xmem rows needed for one matrix row -#}
{%- set cr_cyclic_slot_1  = "cr10"  -%}  {#- cr10 = 128 -#}
{%- set cr_dstructure  = "cr15"  -%} 



    SET {{lr_zero}} {{cr_zero}} ;# lr_zero = 0
    SET {{lr_s1}} {{cr_cyclic_slot_1}} ;# lr_s1 = 128
    SET {{lr_out}} {{cr_zero}} ;;# lr_out = 0
    
    SET {{lr_cbase}} {{cr_zero}} ;# lr_cbase = 0
    SET {{lr_one}} {{cr_one}} ;# lr_one = 1                      
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;# lr_s2 = 256          

    SET {{lr_c}} {{cr_zero}} ;# lr_c = 0
    ADD {{lr_s2p1}} {{lr_s2}} {{cr_one}};# lr_s2p1 = 257
    ADD {{lr_half}} {{lr_one}} {{cr_one}};;# lr_half = 2

chan_loop:
    SET {{lr_oy}} {{cr_zero}} ; # lr_oy = 0 
    ADD {{lr_rowb}} {{lr_cbase}} {{cr_zero}} ;;# lr_rowb = lr_cbase

row_loop:
    SET {{lr_ot}} {{cr_zero}} ;# lr_ot = 0
    ADD {{lr_itile}} {{lr_rowb}} {{cr_zero}} ;;# lr_itile = lr_rowb

tile_loop:
{#- ---- HALF A: input tile 2*ot --------------------------------------------- -#}
    ADD {{lr_addr}} {{lr_itile}} {{cr_zero}} ; # lr_addr = lr_itile
    ADD {{lr_baddr}} {{lr_itile}} {{cr_xmem_row_stride}};# lr_baddr = lr_itile + xmem_row_stride
    LDR_CYCLIC_MULT_REG {{lr_addr}} {{cr_input_base_address}} {{lr_zero}} ;;# R_CYCLIC[0...127] = Memory[row(input_base_address + lr_addr)]

    LDR_CYCLIC_MULT_REG {{lr_baddr}} {{cr_input_base_address}} {{lr_s2}} ;;# R_CYCLIC[256...511] = Memory[row(input_base_address + lr_baddr)]
    
    #todo: load mask. currently assuming that mask is all 1s
    MULT.RC.VE {{lr_zero}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};# MULT_RES[0...127] = R_CYCLIC[0...127] * 1
    ACC.MAX.FIRST;;# R_ACC[0...127] = MULT_RES[0...127]

    MULT.RC.VE {{lr_one}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}}; # MULT_RES[0...127] = R_CYCLIC[1...128] * 1
    ACC.MAX ;;# R_ACC[i] = max(R_ACC[i], MULT_RES[i])

    MULT.RC.VE {{lr_s2}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};# MULT_RES[0...127] = R_CYCLIC[256...383] * 1
    ACC.MAX ;; # R_ACC[i] = max(R_ACC[i], MULT_RES[i])

    MULT.RC.VE {{lr_s2p1}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};# MULT_RES[0...127] = R_CYCLIC[257...384] * 1
    ACC.MAX ;# R_ACC[i] = max(R_ACC[i], MULT_RES[i])
    ACTIVATE.QUANTIZE identity {{cr_dstructure}};
    STR_POST_AAQ_REG {{lr_zero}} cr4;;

{#- ---- HALF B: input tile 2*ot+1 ------------------------------------------- -#}
    ADD {{lr_addr}} {{lr_itile}} {{cr_one}};
    ADD {{lr_baddr}} {{lr_itile}} {{cr_xmem_row_stride}};
    LDR_CYCLIC_MULT_REG {{lr_addr}} {{cr_input_base_address}} {{lr_zero}} ;;   {#- slot0 <- row 2y,   tile it+1 -#}

    INC {{lr_baddr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_baddr}} {{cr_input_base_address}} {{lr_s2}} ;;    {#- slot2 <- row 2y+1, tile it+1 -#}

    MULT.RC.VE {{lr_zero}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.MAX.FIRST ;;

    MULT.RC.VE {{lr_one}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.MAX ;;

    MULT.RC.VE {{lr_s2}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.MAX ;;

    MULT.RC.VE {{lr_s2p1}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.MAX ;
    ACTIVATE.QUANTIZE identity {{cr_dstructure}};
    STR_POST_AAQ_REG {{lr_one}} cr4 ;;                   {#- stage half B -#}

{#- ---- decimate both halves into one output row --------------------------- -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr4 {{lr_zero}} ;;   {#- slot0 <- scratch A -#}

    LDR_CYCLIC_MULT_REG {{lr_one}} cr4 {{lr_s1}} ;;     {#- slot1 <- scratch B -#}

    MULT.RC.VE {{lr_zero}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.STRIDE 64 on off {{lr_zero}} ;;                  {#- even columns of A -> R_ACC[0:64] -#}

    MULT.RC.VE {{lr_s1}} {{cr_one}} 0 {{lr_zero}} {{cr_dstructure}};
    ACC.STRIDE 64 on off {{lr_half}} ;;                  {#- even columns of B -> R_ACC[64:128] -#}

    ACTIVATE.QUANTIZE identity {{cr_dstructure}};
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;                   {#- OUT[c, oy, ot] -#}

    ADD {{lr_out}} {{lr_out}} {{cr_one}};
    ADD {{lr_ot}} {{lr_ot}} {{cr_one}};
    INC {{lr_itile}} 2 ;;                                {#- next output tile spans two more input tiles -#}

    BLT {{lr_ot}} cr7 tile_loop ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr6 ;                    {#- next output row: += 2 * IN_ROW_STRIDE -#}
    ADD {{lr_oy}} {{lr_oy}} {{cr_one}};;

    BLT {{lr_oy}} cr8 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} cr11 ;                 {#- next channel plane -#}
    ADD {{lr_c}} {{lr_c}} {{cr_one}};;

    BLT {{lr_c}} cr9 chan_loop ;;

end:
    BKPT ;;
