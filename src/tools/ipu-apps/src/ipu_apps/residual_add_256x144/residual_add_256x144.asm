# Residual add: C[r] = A[r] + B[r]  for r = 0..287
#
# A, B: interleaved channel-major [256 tokens, 144 channels] = 288 rows × 128 bytes
#   Row r at A_BASE + r*128  (resp. B_BASE + r*128)
# C: FP32 accumulator output, 288 rows × 512 bytes
#   Row r at OUTPUT_BASE + r*512
#
# Strategy: no dedicated element-wise add instruction exists.
#   Load 128 bytes of dtype-1.0 into r_cyclic[0..127] once at startup.
#   Per row:
#     Cycle 1: mult.ee(A[r], 1.0) → acc.first   (r_acc = A[r])
#     Cycle 2: mult.ee(B[r], 1.0) → acc          (r_acc += B[r])
#     Cycle 3: str_acc_reg → store result; incr counter
#     Cycle 4: incr output ptr; blt → loop
#   Total: 4 cycles × 288 rows = 1152 cycles.
#
# IMPORTANT — LR/XMEM interaction:
#   ldr_mult_reg reads its offset LR as "live" (after LR runs in the same cycle).
#   So ptrs lr2/lr3 must start at -128 (startup offset): incr adds 128 → first load=0.
#   str_acc_reg ALSO reads its offset LR as "live", so lr4 must NOT be incremented
#   in the same cycle as str_acc_reg (it would overshoot by 512).
#   blt reads from snapshot, so incr of loop counter lr5 must happen in the cycle
#   BEFORE blt so that blt sees the already-incremented value.
#
# CRs:
#   cr0 = A_BASE     = 0x00000
#   cr1 = B_BASE     = 0x10000
#   cr2 = ONES_BASE  = 0x20000  (128 bytes of dtype-1.0, loaded by harness)
#   cr3 = OUTPUT_BASE = 0x30000
#
# LRs:
#   lr0 = 0    (const: r_cyclic base offset, points to 1.0 bytes at position 0)
#   lr1 = 0    (const: mask_offset = 0, mask_shift = 0, no masking)
#   lr2 = -128 (ptr into A; startup offset: incr +128 before ldr reads live lr2,
#              so first load = A_BASE + (-128+128) = A_BASE+0 = A[0])
#   lr3 = -128 (ptr into B; same startup offset pattern)
#   lr4 = 0    (ptr into C, +512 per row; incremented in cycle separate from str_acc_reg)
#   lr5 = 0    (row counter, 0..287; incremented in cycle before blt)
#   lr6 = 288  (total rows, loop bound)
#
# STARTUP OFFSET PATTERN (ldr_mult_reg reads offset as "live"):
#   The LR slot (incr) runs before XMEM, and XMEM reads the LR as "live" (post-increment).
#   So to load from address 0 on the first iteration, initialise to -stride:
#     lr2 = -128:  incr → 0,   ldr reads 0   → A[0]
#     lr3 = -128:  incr → 0,   ldr reads 0   → B[0]

    set lr0 0;;
    set lr1 0;;
    set lr2 -128;;
    set lr3 -128;;
    set lr4 0;;
    set lr5 0;;
    set lr6 288;;
    ldr_cyclic_mult_reg lr0 cr2 lr0;;   # r_cyclic[0..127] = ONES_BASE[0..127]

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0
    ldr_mult_reg mem_bypass lr2 cr0; incr lr2 128; mult.ee mem_bypass lr0 lr1 lr1; acc.first;;
    # Cycle 2: r_acc += B[r] × 1.0
    ldr_mult_reg mem_bypass lr3 cr1; incr lr3 128; mult.ee mem_bypass lr0 lr1 lr1; acc;;
    # Cycle 3: store result; advance row counter (do NOT incr lr4 here: str_acc_reg reads lr4 live)
    str_acc_reg lr4 cr3; incr lr5 1;;
    # Cycle 4: advance output ptr; branch (blt reads lr5 from snapshot = already-incremented value)
    incr lr4 512; blt lr5 lr6 row_loop;;

end:
    bkpt;;
