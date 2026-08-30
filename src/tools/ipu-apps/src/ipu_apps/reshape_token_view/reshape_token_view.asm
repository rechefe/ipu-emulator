{#-
================================================================================
 Reshape to token view -- MambaVisionMixer layout change
================================================================================

 Implements both directions of the layout change the mixer does twice:

     xz = rearrange(xz, "b l d -> b d l")    token view  -> channel view
     y  = rearrange(y,  "b d l -> b l d")    channel view -> token view

 Both are the same operation: transpose an INT8 matrix that is stored as
 row-padded 128-element lines.

     input :  M lines of N elements, line m at row  SRC_BASE + m*SPL
     output:  N lines of M elements, line n at row  DST_BASE + n*DPL

     SPL = ceil(N/128)   source rows per line
     DPL = ceil(M/128)   destination rows per line

 The caller must zero-fill input lines M .. 128*DPL-1, because the inner
 loop is unconditionally 128 iterations long; those zero lines land in the
 padding elements of the output, which is exactly where they belong.

--------------------------------------------------------------------------------
 How a transpose is done on a machine with no cross-element shuffle
--------------------------------------------------------------------------------
 The multiplier produces a 128-element vector per cycle: one scalar times a
 128-element window of R_CYCLIC. There is no permute/gather network, so the
 only way to move element `i` of one line to element `j` of another is to
 multiply it by the one-hot vector e_j and accumulate.

 Output row (n, b) collects M-block b of output line n:

     R_ACC = sum over j in [0,128) of  in[128*b + j][n] * e_j

 which is 128 multiply-accumulate steps per output row. A naive version
 would keep 128 one-hot rows in XMEM and load one per step, but there is
 only one load slot per VLIW word and the source data already needs it.

 Instead the kernel exploits R_CYCLIC being a 512-element *cyclic* buffer
 that MULT.RC.VE reads as a wrapping 128-element window starting at an
 element index. Load it once, in the prologue, with a buffer that is zero
 everywhere except element 128:

     R_CYCLIC = [ 0 x128 , 1 , 0 x383 ]

 Then the window that starts at element (128 - j) has its single 1 at
 position j -- it *is* e_j -- for every j in [0,128). All 128 one-hot
 vectors come from four prologue loads.

     rc window start   128 127 126 ...   2   1
     one-hot position    0   1   2  ... 126 127

--------------------------------------------------------------------------------
 Register map
--------------------------------------------------------------------------------
 CR2  SRC_BASE_ROW           CR9   128
 CR3  DST_BASE_ROW           CR10  128 * SPL   (source stride between blocks)
 CR4  DPL                    CR11  ONEHOT_BASE_ROW
 CR5  SPL                    CR13  256         (R_CYCLIC load index)
 CR6  129                    CR14  384         (R_CYCLIC load index)
 CR7  N   (output lines)     CR15  dstructure  (valid_elements = 128)
 CR8  DPL (output blocks)

 LR0  n, output line          LR6   j, inner counter
 LR1  b, output block         LR7   destination row  = n*DPL + b
 LR2  nRow = n / 128          LR8   source block base = b * 128 * SPL
 LR3  nIdx = n % 128          LR9   n * DPL
 LR4  running source row      LR10  LR11  prologue scratch
 LR5  R_CYCLIC window start   LR15  constant 0
-#}

{#- ---------------------------------------------------------------- prologue #}
    SET                 lr15 cr0 ;
    SET                 lr10 cr0 ;
    SET                 lr11 cr0 ;;

    {#- Four loads fill all 512 elements of R_CYCLIC with the one-hot table.
        Element 128 is the single 1; every window start in [1,128] therefore
        yields a distinct one-hot vector. R_CYCLIC is never written again, so
        the inner loop's single load slot stays free for source data. -#}
    LDR_CYCLIC_MULT_REG lr11 cr11 lr10 ;;
    ADD                 lr11 lr11 cr1 ;
    SET                 lr10 cr9 ;;
    LDR_CYCLIC_MULT_REG lr11 cr11 lr10 ;;
    ADD                 lr11 lr11 cr1 ;
    SET                 lr10 cr13 ;;
    LDR_CYCLIC_MULT_REG lr11 cr11 lr10 ;;
    ADD                 lr11 lr11 cr1 ;
    SET                 lr10 cr14 ;;
    LDR_CYCLIC_MULT_REG lr11 cr11 lr10 ;;

    SET                 lr0 cr0 ;
    SET                 lr2 cr0 ;
    SET                 lr3 cr0 ;;
    SET                 lr9 cr0 ;;

{#- ------------------------------------------------- loop over output lines #}
line_loop:
    SET                 lr1 cr0 ;
    SET                 lr8 cr0 ;;

{#- ------------------------------- loop over the DPL rows of one output line #}
block_loop:
    ADD                 lr7 lr9 lr1 ;                {#- dest row  = n*DPL + b -#}
    ADD                 lr4 lr8 lr2 ;                {#- src  row  = b*128*SPL + nRow -#}
    SET                 lr5 cr6 ;;                   {#- window start, pre-incremented -#}
    SET                 lr6 cr0 ;;

    {#- MULT reads R0 from the start-of-cycle snapshot (issue #157), so the
        row consumed by a multiply must be loaded at least one word earlier.
        Prime element j = 0 here, then every loop body loads the NEXT
        element's row while multiplying the one loaded last cycle. -#}
    LDR_MULT_REG        r0 lr4 cr2 ;;

    ADD                 lr4 lr4 cr5 ;                {#- -> row of element 1 -#}
    SUB                 lr5 lr5 cr1 ;                {#- -> 128, the window for e_0 -#}
    ADD                 lr6 lr6 cr1 ;;

    MULT.RC.VE          lr5 lr3 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    LDR_MULT_REG        r0 lr4 cr2 ;
    BNE                 lr6 cr9 element_loop_pre ;;
    B                   after_element_loop ;;

element_loop_pre:
    ADD                 lr4 lr4 cr5 ;
    SUB                 lr5 lr5 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

element_loop:
    MULT.RC.VE          lr5 lr3 0 lr15 cr15 ;
    ACC.ADD ;
    LDR_MULT_REG        r0 lr4 cr2 ;
    BNE                 lr6 cr9 element_loop_pre ;;

after_element_loop:
    {#- R_ACC now holds one full 128-element output row as INT32. Identity
        activation + INT8 quantization hands it to POST_AAQ_REG; the store
        writes all 512 bytes of that register but successive stores are one
        row apart, so each overwrites the three zero rows the previous one
        left and the buffer ends up packed. Only the 3 rows past the end of
        the buffer are clobbered -- allocate that slack. -#}
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr1 lr1 cr1 ;
    ADD                 lr8 lr8 cr10 ;;
    BLT                 lr1 cr8 block_loop ;;

    {#- n += 1, and keep (nRow, nIdx) = divmod(n, 128) without a divider. -#}
    ADD                 lr0 lr0 cr1 ;
    ADD                 lr9 lr9 cr4 ;;
    INCR_MOD_POW2       lr3 cr1 7 ;;
    BNE                 lr3 cr0 next_line ;;
    ADD                 lr2 lr2 cr1 ;;

next_line:
    BLT                 lr0 cr7 line_loop ;;

end:
    BKPT ;;
