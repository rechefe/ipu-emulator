# Residual add: C[r] = A[r] + B[r]  for r = 0..287
#
# A, B: interleaved channel-major [256 tokens, 144 channels] = 288 rows × 128 bytes
#   Row r at A_BASE + r*128  (resp. B_BASE + r*128)
# C: FP32 accumulator output, 288 rows × 512 bytes
#   Row r at OUTPUT_BASE + r*512
#
# New kernel: MULT.EE r0 replaces MULT.EE mem_bypass (mem_bypass removed in PR #69).
#   Strategy: preload 128 bytes of dtype-1.0 into r0 once before the row loop.
#   Per row:
#     Cycle 1: LDR_CYCLIC_MULT_REG → r_cyclic = A[r]; MULT.EE r0 → A[r]×1=A[r]; ACC.FIRST
#     Cycle 2: LDR_CYCLIC_MULT_REG → r_cyclic = B[r]; MULT.EE r0 → B[r]×1=B[r]; ACC
#     Cycle 3: STR_ACC_REG → store A[r]+B[r]; advance row counter
#     Cycle 4: advance output ptr; BLT → loop
#   Total: 4 cycles × 288 rows = 1152 cycles.
#   r0 holds ones for the entire loop (one XMEM slot per cycle → no conflict).
#
# IMPORTANT — live-LR semantics:
#   ADD fires before XMEM; ptrs init at -128 so first live = 0 after ADD.
#   STR_ACC_REG reads lr4 live; do NOT ADD lr4 in the same cycle as STR_ACC_REG.
#   BLT reads snapshot; ADD lr5 must happen the cycle before BLT.
#
# CRs:
#   cr0 = A_BASE      = 0x00000
#   cr9 = B_BASE      = 0x10000   (moved off read-only CR1)
#   cr2 = ONES_BASE   = 0x20000  (128 bytes of dtype-1.0, loaded by harness)
#   cr3 = OUTPUT_BASE = 0x30000
#   cr4 = 0    (const zero)
#   cr5 = -128 (startup init for A/B row ptrs)
#   cr6 = 288  (loop bound = N_ROWS)
#   cr7 = 128  (row stride for A and B)
#   cr8 = 512  (output stride)
#
# LRs:
#   lr0 = 0    (const: r_cyclic slot-0 base offset; mask_shift=0)
#   lr1 = 0    (const: mask_offset=0 for MULT.EE — same value as lr0, kept separate for clarity)
#   lr2 = -128 (ptr into A; startup: ADD +128 fires first → first live = 0)
#   lr3 = -128 (ptr into B; same startup pattern)
#   lr4 = 0    (ptr into C, +512 per row; incremented in separate cycle from STR_ACC_REG)
#   lr5 = 0    (row counter, 0..287)
#   lr6 = 288  (loop bound)
#   lr7 = 128  (row stride for A and B)
#   lr8 = 512  (output stride)

    SET                 lr0 cr4;;
    SET                 lr1 cr4;;
    SET                 lr2 cr5;;
    SET                 lr3 cr5;;
    SET                 lr4 cr4;;
    SET                 lr5 cr4;;
    SET                 lr6 cr6;;
    SET                 lr7 cr7;;
    SET                 lr8 cr8;;
    LDR_MULT_REG        r0 lr0 cr2;;      # r0 = ONES_BASE[0..127] = dtype-1.0 × 128

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0  (live ADD lr2 fires first → live lr2 = r*128)
    #   MULT.RC.VE r_cyclic[0] × CR10(=dtype 1.0) → A[r] passed through (replaces MULT.EE ones).
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.RC.VE lr0 cr10 0 lr0; ACC.FIRST;;
    # Cycle 2: r_acc += B[r] × 1.0
    LDR_CYCLIC_MULT_REG lr3 cr9 lr0; ADD lr3 lr3 lr7; MULT.RC.VE lr0 cr10 0 lr0; ACC;;
    # Cycle 3: store (do NOT ADD lr4 here: STR_ACC_REG reads lr4 live)
    STR_ACC_REG         lr4 cr3; ADD lr5 lr5 cr1;;
    # Cycle 4: advance output ptr; BLT reads snap lr5 = already-incremented
    ADD                 lr4 lr4 lr8; BLT lr5 lr6 row_loop;;

end:
    BKPT;;
