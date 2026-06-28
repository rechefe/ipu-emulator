# Universal Residual Add (wide-vector FP32 debug mode): FP32 + FP32 -> FP32.
#
# Adds two FP32 tensors element-wise, one channel per 512-byte chunk
# (128 FP32 lanes each). Wide-vector debug mode widens every lane to 4
# bytes, so a load brings in a full 512-byte channel and STR_ACC_REG
# writes the full 512-byte r_acc.
#
# Algorithm: use MULT.RC.VE with scalar = 1.0 (cr1 low byte = 1) to copy
# each FP32 chunk into mult_res, then accumulate both into r_acc:
#   acc.first  <- 1 * A[i]   (FP32)
#   acc        <- 1 * B[i]
#   result     <- r_acc = A[i] + B[i]   (FP32)
#
# Pipeline note (master ISA, issue #157): MULT reads r_cyclic from the
# cycle-start SNAPSHOT, so a same-cycle LDR_CYCLIC_MULT_REG is NOT visible
# to the consuming mult. Each load is therefore issued one cycle before the
# mult that uses it.
#
# CR registers:
#   cr0 = 0  (reserved config register; reads as 0 -> used to init LRs)
#   cr1 = 1  (reserved config register; reads as 1 -> identity scalar)
#   cr2 = input A base address      (set by harness)
#   cr3 = input B base address      (set by harness)
#   cr4 = output base address       (set by harness)
#   cr5 = chunk step (512)          (set by harness)
#   cr6 = total input bytes         (set by harness)
#
# LR registers:
#   lr0  = 0  (cyclic index; also mask_shift=0. mask_offset is a literal 0)
#   lr2  = input chunk offset  (0, 512, 1024, ...)
#   lr3  = output chunk offset (0, 512, 1024, ...)
#   lr11 = total input bytes (copy of cr6, for blt comparison)

# ===========================================================================
# Initialization
# ===========================================================================

    SET     lr0 cr0;
    SET     lr2 cr0;;

    SET     lr3 cr0;;

    add     lr11 lr0 cr6;;

# Prime the pipeline: load channel 0 of input A into r_cyclic.
    ldr_cyclic_mult_reg lr2 cr2 lr0;;

# ===========================================================================
# Main loop: one iteration per 512-byte channel chunk
# ===========================================================================

loop:
# A is already in r_cyclic (loaded last cycle). Copy A -> r_acc, and load
# B for this same channel into r_cyclic (lands next cycle).
    ldr_cyclic_mult_reg lr2 cr3 lr0;
    mult.rc.ve          lr0 cr1 0 lr0;
    acc.first;;

# B is now in r_cyclic. Accumulate B -> r_acc = A + B. Pre-load A of the
# NEXT channel so it is ready for the next iteration's acc.first.
    add     lr2 lr2 cr5;;

    ldr_cyclic_mult_reg lr2 cr2 lr0;
    mult.rc.ve          lr0 cr1 0 lr0;
    acc;;

# Store FP32 accumulator (512 bytes per chunk). The output-pointer bump must
# NOT share this word: LR runs before XMEM, so it would store to the wrong
# address. Advance lr3 in the branch word instead (after the store).
    str_acc_reg         lr3 cr4;;

    add     lr3 lr3 cr5;
    blt     lr2 lr11 loop;;

end:
    bkpt;;
