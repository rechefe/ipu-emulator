# RF Feedback Test Kernel
#
# Tests the round-trip: XMEM → mult → r_acc → agg → aaq → mult.ve.aaq → r_acc → XMEM
#
# New kernel: MULT.EE r0 replaces MULT.EE mem_bypass (mem_bypass removed in PR #69).
#   Load SCALAR_BASE row into r0 via ldr_mult_reg r0; load ones into r_cyclic first.
#   MULT.EE r0 lr0 0 lr0: r0[i] × r_cyclic[i] = scalar_row[i] × 1.0 = scalar_row[i].
#
# Memory layout:
#   SCALAR_BASE = 0x00000   128 bytes: known nonzero value at byte 0, rest zeros
#   DATA_BASE   = 0x00080   128 bytes: row of known test values
#   ONES_BASE   = 0x00100   128 bytes: all dtype(1.0)
#   OUTPUT_BASE = 0x00180   512 × 3 = 1536 bytes: three output rows (value, inv, inv_sqrt)
#
# CR assignments (loaded by harness):
#   cr0  = SCALAR_BASE = 0x00000
#   cr1  = DATA_BASE   = 0x00080
#   cr2  = ONES_BASE   = 0x00100
#   cr3  = OUTPUT_BASE = 0x00180
#   cr15 = dtype (set by harness)
#
# LRs:
#   lr0 = 0   (cyclic slot 0, const-zero mask offset/shift)
#   lr1 = 512 (output stride: str_acc_reg writes 512 bytes per row)

    SET lr0 0;;
    SET lr1 512;;
    SET lr2 128;;   # valid_elements for AGG (all 128 lanes)

# ============================================================
# PHASE 1: Load scalar into aaq0/aaq1/aaq2 via agg post-fns
# ============================================================
#
# Step 1: Load ones into r_cyclic[0..127]
    LDR_CYCLIC_MULT_REG lr0 cr2 lr0;;
# Step 2: Load SCALAR_BASE row into r0; MULT.EE r0 × ones → mult_res; ACC.FIRST
    LDR_MULT_REG r0 lr0 cr0; MULT.EE r0 lr0 0 lr0; ACC.FIRST;;
# Note: MULT.EE r0 lr0 0 lr0 = r0[i] × r_cyclic[lr0 + i], mask_offset=0 (immediate), mask_shift=lr0=0
# Step 3: Store three agg variants
    AGG SUM VALUE   lr2 cr0 aaq0;;   # aaq0 = scalar (raw)
    AGG SUM INV     lr2 cr0 aaq1;;   # aaq1 = 1/scalar
    AGG SUM INV_SQRT lr2 cr0 aaq2;;  # aaq2 = 1/sqrt(scalar)

# ============================================================
# PHASE 2a: mult.ve.aaq with aaq0 (value) → OUTPUT_BASE
# ============================================================
    LDR_CYCLIC_MULT_REG lr0 cr1 lr0;;   # r_cyclic = DATA_BASE row
    MULT.VE.AAQ lr0 0 lr0 aaq0; ACC.FIRST;;
    STR_ACC_REG lr0 cr3;;               # store at OUTPUT_BASE + 0*512

# ============================================================
# PHASE 2b: mult.ve.aaq with aaq1 (inv) → OUTPUT_BASE+512
# ============================================================
    LDR_CYCLIC_MULT_REG lr0 cr1 lr0;;   # reload DATA_BASE row
    MULT.VE.AAQ lr0 0 lr0 aaq1; ACC.FIRST;;
    STR_ACC_REG lr1 cr3;;               # store at OUTPUT_BASE + 1*512

# ============================================================
# PHASE 2c: mult.ve.aaq with aaq2 (inv_sqrt) → OUTPUT_BASE+1024
# ============================================================
    SET lr2 1024;;
    LDR_CYCLIC_MULT_REG lr0 cr1 lr0;;   # reload DATA_BASE row
    MULT.VE.AAQ lr0 0 lr0 aaq2; ACC.FIRST;;
    STR_ACC_REG lr2 cr3;;               # store at OUTPUT_BASE + 2*512

end:
    BKPT;;
