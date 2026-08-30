{#-
================================================================================
 Residual connection of the MambaVision mixer -- Block.forward
================================================================================

     x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))

 At inference DropPath is nn.Identity, and mamba_vision_T passes no
 layer_scale, so Block.__init__ leaves gamma_1 as the Python int 1 -- the
 operation is a plain element-wise add of two TOKEN VIEW tensors,
 (B, L, d_model).

     input   skip   : L lines of d_model elements at SKIP_BASE   + t*RPT
     input   branch : L lines of d_model elements at BRANCH_BASE + t*RPT
     output         : L lines of d_model elements at OUT_BASE    + t*RPT

     RPT   = ceil(d_model/128)  rows per token
     NROWS = L * RPT            rows the loop walks

 The gamma_1 path is kept live even though mamba_vision_T does not use it.
 MULT.RC.VE's scalar operand is an LcrIdx, so the skip row is multiplied by
 CR1 (permanently 1) and the branch row by CR6, whose low byte holds
 gamma_1. A layer-scaled variant (mamba_vision_B / _L, layer_scale=1e-5)
 only changes that CR -- no change to the kernel, no extra cycles, and no
 constants row in XMEM.

--------------------------------------------------------------------------------
 Numerics
--------------------------------------------------------------------------------
 Both addends arrive as INT8 and are summed in the INT32 accumulator, so the
 add itself never overflows. ACTIVATE.QUANTIZE then brings the result back
 to INT8; in the emulator's INT8 mode that is a direct clamp to [-128, 127].
 The golden data deliberately makes about a fifth of the sums saturate so
 the tests cover that path.

--------------------------------------------------------------------------------
 Register map
--------------------------------------------------------------------------------
 CR1  1 (locked)  -- the multiply-by-one scalar for the skip term
 CR2  SKIP_BASE_ROW                          CR6   gamma_1 (low byte)
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
    {#- word 1: skip row * 1 -> R_ACC, and fetch the matching branch row.
        MULT reads R_CYCLIC from the start-of-cycle snapshot, so the row it
        consumes here is the one loaded in the previous word. -#}
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    LDR_CYCLIC_MULT_REG lr4 cr4 lr15 ;;

    {#- word 2: branch row * gamma_1 accumulated on top, then advance. -#}
    MULT.RC.VE          lr15 cr6 0 lr15 cr15 ;
    ACC.ADD ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    {#- word 3: quantize the INT32 sum and prefetch the next skip row. -#}
    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;

    {#- word 4: commit the cache line. Consecutive destinations are one row
        apart, so the 512-byte store's tail is overwritten by the next one
        and the output buffer stays packed (3 rows of slack at the end). -#}
    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 residual_loop ;;

end:
    BKPT ;;
