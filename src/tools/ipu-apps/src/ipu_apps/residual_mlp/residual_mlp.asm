{#-
================================================================================
 Residual connection of the MLP after the mixer -- Block.forward
================================================================================

     x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

 The second residual of the same Block, the one that closes the MLP.
 Structurally identical to residual_mixer; it is a separate kernel because
 it is a separate point in the schedule, with different buffers, a different
 gamma (gamma_2), and its own latency budget.

     input   skip   : L lines of d_model elements at SKIP_BASE   + t*RPT
                      (the tensor produced by the mixer residual)
     input   branch : L lines of d_model elements at BRANCH_BASE + t*RPT
                      (timm Mlp output: fc1 -> GELU -> fc2, hidden = 4*d_model)
     output         : L lines of d_model elements at OUT_BASE    + t*RPT

     RPT   = ceil(d_model/128)
     NROWS = L * RPT

 The MLP itself is not part of this kernel: fc1 and fc2 are fully-connected
 layers (the fully_connected app), and GELU is an ACTIVATE.QUANTIZE
 activation code, so only the residual needed writing.

 Note that the *branch* here is the wide-then-narrow MLP output while the
 *skip* is the block input, so the two addends often have very different
 dynamic ranges. Everything is summed in INT32 inside R_ACC and clamped
 once, on the way out, by ACTIVATE.QUANTIZE -- there is no intermediate
 rounding.

--------------------------------------------------------------------------------
 Register map
--------------------------------------------------------------------------------
 CR1  1 (locked)  -- the multiply-by-one scalar for the skip term
 CR2  SKIP_BASE_ROW                          CR6   gamma_2 (low byte)
 CR3  OUT_BASE_ROW                           CR15  dstructure (valid_elements=128)
 CR4  BRANCH_BASE_ROW
 CR5  NROWS = L * ceil(d_model/128)

 LR4  running row offset, shared by both inputs   LR7  destination row
 LR5  loop bound (NROWS)                          LR15 constant 0
 LR6  loop counter
-#}

{#- ---------------------------------------------------------------- prologue #}
    SET                 lr15 cr0 ;
    SET                 lr4 cr0 ;
    SET                 lr5 cr5 ;;
    SET                 lr6 cr0 ;
    SET                 lr7 cr0 ;;

    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;              {#- prime with skip row 0 -#}

{#- --------------------------------------------------------------- main loop #}
residual_loop:
    {#- word 1: skip row * 1 -> R_ACC, fetch the matching MLP-output row. -#}
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    LDR_CYCLIC_MULT_REG lr4 cr4 lr15 ;;

    {#- word 2: MLP row * gamma_2 accumulated on top, then advance. -#}
    MULT.RC.VE          lr15 cr6 0 lr15 cr15 ;
    ACC.ADD ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    {#- word 3: quantize the INT32 sum and prefetch the next skip row. -#}
    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;

    {#- word 4: commit the cache line. -#}
    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 residual_loop ;;

end:
    BKPT ;;
