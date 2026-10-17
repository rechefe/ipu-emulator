{#- ==========================================================================
    residual_add.asm -- y = a + b, FP32, one channel per row

    Adds two FP32 tensors element by element: num_channels rows of 128 FP32
    elements each (a [num_channels, 128] tensor, e.g. a CHW tensor whose H*W
    is 128), the CHW residual of the convolution stages (MobileViT-S:
    64x64x64, 32x32x96, 16x16x128, 8x8x160). FP32 only: each channel is one
    whole XMEM row and ST.ACC stores the whole 128-element accumulator.

    Algorithm: MULT.RC.VE with the scalar CR1 (1.0) copies a row into
    MULT_RES, and the accumulator sums both operands:
      ACC.ADD.FIRST  <- 1 * a[i]
      ACC.ADD        <- 1 * b[i]
      y[i]           <- R_ACC = a[i] + b[i]

    Pipeline note (issue #157): MULT reads R_CYCLIC as it was at the start
    of the cycle, so an LD.MULT.CYCLIC in the same instruction word is not
    visible to the MULT that uses it. Each load is issued one cycle before
    its MULT.

    XMEM operands (LD.MULT.CYCLIC's offset and base, ST.ACC) are row
    numbers. LD.MULT.CYCLIC's index (LR0, always 0: FP32 fills the whole
    R_CYCLIC from one slot) is in R_CYCLIC element space.

    ABI (kernel.toml [abi]; a, b and y at fixed rows 0, 1024 and 2048):
      CR0 = 0 (hardwired: starts the LRs)
      CR1 = 1 (hardwired: the identity scalar)
      CR2 = a, the base row          CR3 = b, the base row
      CR4 = y, the base row          CR5 = step, rows per channel (1)
      CR6 = num_channels

    LRs (set by the program):
      LR0  = 0 (R_CYCLIC index; mask shift 0, the mask offset is a literal 0)
      LR2  = the input row (0, 1, 2, ...)
      LR3  = the output row (0, 1, 2, ...)
      LR11 = num_channels (a copy of CR6, for BLT.S)
========================================================================== -#}

# ===========================================================================
# Initialization
# ===========================================================================

    SET     LR0 CR0;
    SET     LR2 CR0;;

    SET     LR3 CR0;;

    ADD     LR11 LR0 CR6;;

# Prime the pipeline: load channel 0 of input A into r_cyclic.
    LD.MULT.CYCLIC LR2 CR2 LR0;;

# ===========================================================================
# Main loop: one iteration per 512-byte channel chunk
# ===========================================================================

loop:
# A is already in r_cyclic (loaded last cycle). Copy A -> r_acc, and load
# B for this same channel into r_cyclic (lands next cycle).
    LD.MULT.CYCLIC LR2 CR3 LR0;
    MULT.RC.VE          LR0 CR1 0 LR0 CR15;
    ACC.ADD.FIRST;;

# B is now in r_cyclic. Accumulate B -> r_acc = A + B. Pre-load A of the
# NEXT channel so it is ready for the next iteration's acc.first.
    ADD     LR2 LR2 CR5;;

    LD.MULT.CYCLIC LR2 CR2 LR0;
    MULT.RC.VE          LR0 CR1 0 LR0 CR15;
    ACC.ADD;;

# Store FP32 accumulator (512 bytes per chunk). The output-pointer bump must
# NOT share this word: LR runs before XMEM, so it would store to the wrong
# address. Advance lr3 in the branch word instead (after the store).
    ST.ACC         LR3 CR4;;

    ADD     LR3 LR3 CR5;
    BLT.S     LR2 LR11 loop;;

end:
    HALT;;
