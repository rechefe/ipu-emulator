{#- ==========================================================================
    l2_normalize_channels.asm -- L2 normalization down the leading axis, FP32

    out[c, n] = x[c, n] / sqrt( SUM_c x[c, n]^2 )

    over a (ROWS x COLS) matrix, where ROWS is the reduction axis. SuperPoint's
    dense descriptor normalization is this with ROWS = 256 (convDb's channels)
    and COLS = H*W.

    WHY REDUCING DOWN COLUMNS NEEDS NO AGG:
      Every column is an INDEPENDENT normalization and the datapath is 128
      lanes wide, so one pass down the rows reduces 128 columns at once with
      ACC.ADD -- the running sum of squares stays a full 128-element vector and
      never has to collapse to a scalar. There is no AGG, no fan-out, and no
      per-row bookkeeping vector, which is what makes this cheaper than the
      row-wise form (compare softmax_columns vs softmax_rows).

    1/||x|| IS AN ACTIVATION, NOT A REDUCTION POST-FUNCTION:
      The older byte-addressed kernel wrote `AGG sum inv_sqrt`. That post-fn no
      longer exists; rsqrt is one of ACTIVATE.QUANTIZE's twelve activations, so
      the reciprocal square root is taken on the way out of R_ACC:

          ACTIVATE.QUANTIZE rsqrt cr15 ; STR_POST_AAQ_REG -> the scale row

      The emulator's rsqrt is guarded (x <= 0 yields 0), so an all-zero column
      normalizes to zeros rather than producing inf or NaN.

    SEEDING THE SUM:
      The channel loop is a run-time bound, so its first iteration cannot be
      peeled to carry an ACC.ADD.FIRST. One word before the loop multiplies
      R_CYCLIC by CR0 (= 0.0) into ACC.ADD.FIRST instead, which clears R_ACC
      without touching XMEM and without needing a resident zero row.

    LAYOUT -- INPUT AND OUTPUT SHARE ONE OFFSET:
      Both regions are ROWS x TPR rows with row (c, t) at c*TPR + t, so a single
      walking register addresses the load (base CR2) and the store (base CR3).
      TPR = ceil(COLS / 128); columns past COLS are zero, whose square adds
      nothing to any real column's sum -- those lanes form their own all-zero
      columns, which rsqrt's guard sends to zero.

    PASS STRUCTURE (outer over column tile, inner over the reduction axis):
      Pass 1   R_ACC = SUM_c x[c, t]^2       -- MULT.RC.VS + ACC.ADD, 2 words/row
      Pass 2   out[c, t] = x[c, t] * rvec    -- MULT.RC.VV against R0, 3 words/row

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0 (-> the 0.0 scalar that clears R_ACC)   CR1 = 1
      CR2  = INPUT_BASE  (rows)       CR3  = OUTPUT_BASE (rows)
      CR4  = RVEC_ROW    (rows)       -- one row: 1/||x|| per column of this tile
      CR5  = TPR -- rows per matrix row, AND the tile-loop bound (same number)
      CR6  = ROWS (the reduction length)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND. That is why the walking address is advanced in the word AFTER the load
    or store that uses it, and why each loop counter is incremented one word
    BEFORE its BLT (BLT reads its register from the start-of-word snapshot, so
    an increment in the same word would run one iteration too many).
========================================================================== -#}

{%- set lr_zero = "lr0" -%}  {#- 0: mask_shift, R_CYCLIC slot-0 index, rvec offset -#}
{%- set lr_tile = "lr1" -%}  {#- column-tile counter AND the tile's row offset -#}
{%- set lr_row  = "lr2" -%}  {#- index along the reduction axis -#}
{%- set lr_addr = "lr3" -%}  {#- walking row offset: (row, tile) for BOTH regions -#}

    SET {{lr_zero}} cr0 ;
    SET {{lr_tile}} cr0 ;;

tile_loop:
{#- ---- PASS 1: R_ACC = SUM over the reduction axis of x^2 ----------------- -#}
    MULT.RC.VE {{lr_zero}} cr0 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;
    ADD {{lr_addr}} {{lr_tile}} cr0 ;
    SET {{lr_row}} cr0 ;;                                {#- R_ACC = 0; addr = tile -#}

pass1_loop:
    ADD {{lr_row}} {{lr_row}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- x[row, tile] -#}
    ADD {{lr_addr}} {{lr_addr}} cr5 ;
    MULT.RC.VS {{lr_zero}} 0 {{lr_zero}} cr15 ; ACC.ADD ;
    BLT {{lr_row}} cr6 pass1_loop ;;                     {#- += x^2, lane-wise -#}

    ACTIVATE.QUANTIZE rsqrt cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr4 ;;                  {#- rvec = 1/||x[:, col]|| -#}
    LDR_MULT_REG r0 {{lr_zero}} cr4 ;
    ADD {{lr_addr}} {{lr_tile}} cr0 ;
    SET {{lr_row}} cr0 ;;                                {#- R0 = rvec (visible NEXT word) -#}

{#- ---- PASS 2: out = x * rvec, lane-wise -------------------------------- -#}
pass2_loop:
    ADD {{lr_row}} {{lr_row}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- x[row, tile] -#}
    MULT.RC.VV {{lr_zero}} r0 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_addr}} cr3 ;;                  {#- OUT[row, tile] -#}
    ADD {{lr_addr}} {{lr_addr}} cr5 ;
    BLT {{lr_row}} cr6 pass2_loop ;;

    ADD {{lr_tile}} {{lr_tile}} cr1 ;;
    BLT {{lr_tile}} cr5 tile_loop ;;

end:
    BKPT ;;
