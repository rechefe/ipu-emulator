{#-
================================================================================
 Split to the x and z branches -- MambaVisionMixer
================================================================================

     x, z = xz.chunk(2, dim=1)

 At this point in MambaVisionMixer.forward the tensor is in CHANNEL VIEW,
 (B, d_inner, L), because the rearrange to "b d l" ran just before. The
 split is therefore along the *line* axis of the XMEM layout: the first
 half of the channel lines becomes x, the second half becomes z.

     input   xz : d_inner lines of L elements, line c at SRC_BASE  + c*LPC
     output  x  : half    lines of L elements, line c at DSTX_BASE + c*LPC
     output  z  : half    lines of L elements, line c at DSTZ_BASE + c*LPC

     LPC       = ceil(L/128)   rows per channel
     HALF_ROWS = half * LPC    rows each copy loop moves

 Because the two halves are contiguous, the kernel is two back-to-back row
 copies over one monotonically increasing source pointer: rows
 [0, HALF_ROWS) go to x, rows [HALF_ROWS, 2*HALF_ROWS) go to z. The x loop's
 last prefetch is already z's first row, so the hand-off costs one word.

 NOTE on why this is a copy at all. On silicon the split is free: x and z
 are disjoint address ranges of the same cache table, so the RISC core can
 hand the two conv1d kernels different base addresses and no data moves.
 This kernel exists because the downstream kernels take independent buffers,
 and because it gives the zero-copy version a measured baseline.

--------------------------------------------------------------------------------
 How a row copy is expressed with a multiplier
--------------------------------------------------------------------------------
 The IPU has no move instruction: everything reaches R_ACC through the
 multiplier. A copy is therefore "multiply by one". MULT.RC.VE's scalar
 operand is an LcrIdx, so naming a CR takes that register's low byte
 directly as the scalar -- and CR1 is permanently 1. No constants row, no
 R0 load:

     MULT.RC.VE lr15 cr1 0 lr15 cr15    ->   1 * R_CYCLIC window

 then ACC.ADD.FIRST, ACTIVATE.QUANTIZE identity and STR_POST_AAQ_REG put it
 back in XMEM as INT8. Four VLIW words per 128-element row.

--------------------------------------------------------------------------------
 Register map
--------------------------------------------------------------------------------
 CR1  1 (locked)  -- the multiply-by-one scalar
 CR2  SRC_BASE_ROW  (xz, channel view)       CR15  dstructure (valid_elements=128)
 CR3  DSTX_BASE_ROW
 CR4  DSTZ_BASE_ROW
 CR5  HALF_ROWS = half * ceil(L/128)

 LR4  running source row (0 .. 2*HALF_ROWS-1, never reset)
 LR5  loop bound (HALF_ROWS)      LR7   destination row
 LR6  loop counter                LR15  constant 0
-#}

{#- ---------------------------------------------------------------- prologue #}
    SET                 lr15 cr0 ;
    SET                 lr4 cr0 ;
    SET                 lr5 cr5 ;;
    SET                 lr6 cr0 ;
    SET                 lr7 cr0 ;;

    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;              {#- prime with xz row 0 -#}

{#- ------------------------------------- copy the x half: rows [0, HALF_ROWS) #}
x_loop:
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;              {#- prefetch the next row -#}

    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 x_loop ;;

{#- The last x iteration already prefetched source row HALF_ROWS, which is
    z's first row, so the z loop can start straight away. -#}
    SET                 lr6 cr0 ;
    SET                 lr7 cr0 ;;

{#- ------------------------- copy the z half: rows [HALF_ROWS, 2*HALF_ROWS) #}
z_loop:
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;

    STR_POST_AAQ_REG    lr7 cr4 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 z_loop ;;

end:
    BKPT ;;
