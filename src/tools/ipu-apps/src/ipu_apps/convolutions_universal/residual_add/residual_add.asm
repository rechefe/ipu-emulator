# Universal Residual Add: INT8 + INT8 -> INT32, flexible channel count.
#
# Adds two INT8 tensors element-wise and stores the result as INT32.
# Both inputs use the same paired-channel 128-byte chunk layout as the
# other convolutions_universal apps (2 channels per 128-byte chunk).
#
# Algorithm: use mult.ve.cr with scalar=1 (cr5) to sign-extend each
# INT8 chunk to INT32 in mult_res, then accumulate both into r_acc.
#   acc.first  <- 1 * A[i]  (sign-extended INT32)
#   acc        <- 1 * B[i]
#   result     <- r_acc = A[i] + B[i]  (INT32)
#
# CR registers (set by harness):
#   cr0 = input A base address
#   cr1 = input B base address
#   cr2 = output base address
#   cr4 = total_bytes (= num_channels / 2 * 128)
#   cr5 = 1  (identity scalar for mult.ve.cr)
#
# LR registers:
#   lr0  = 0  (cyclic index, mask_offset=0, mask_shift=0)
#   lr2  = input chunk offset (0, 128, 256, ...)
#   lr3  = output chunk offset (0, 512, 1024, ...)
#   lr11 = total_bytes (copy of cr4, for blt comparison)

# ===========================================================================
# Initialization
# ===========================================================================

    set     lr0 0;
    set     lr2 0;;

    set     lr3 0;;

    add     lr11 cr4 lr0;;

# ===========================================================================
# Main loop: one iteration per 128-byte input chunk
# ===========================================================================

loop:
# Load chunk from input A into r_cyclic[0], multiply by 1 -> r_acc = A[i]
    ldr_cyclic_mult_reg lr2 cr0 lr0;
    mult.ve.cr          lr0 lr0 lr0 cr5;
    acc.first;;

# Load chunk from input B into r_cyclic[0], multiply by 1 -> r_acc += B[i]
    ldr_cyclic_mult_reg lr2 cr1 lr0;
    mult.ve.cr          lr0 lr0 lr0 cr5;
    acc;;

# Store INT32 accumulator (512 bytes per chunk)
    str_acc_reg         lr3 cr2;;

    incr    lr2 128;
    incr    lr3 512;;

    blt     lr2 lr11 loop;;

end:
    bkpt;;
