# Native vector tour: 128-element INT8 / FP8 vectors (no emulator-only format).
#
# A curated conformance program for the Cycle Inspector: it walks the MULT and
# ACC slots once each on real 8-bit data. It loads a mask, two R_CYCLIC slots
# and R0/R1, runs MULT.VE, MULT.RC.VV/VS/VE and MULT.EE into ACC.ADD/SUB/MAX,
# two masked multiplies, and a 4-row matrix-vector loop with AGG.SUM.FIRST.
# The harness is vector_tour/app.py; cases.py checks every store with NumPy.
#
#   CR2  = A base row (4 rows)     CR3  = w row        CR4 = c row
#   CR5  = mask row                CR6  = 128 (second R_CYCLIC slot)
#   CR7  = output base row         CR8  = 4 (rows one R_ACC store spans)
#   CR9  = scalar 2 in the active format (low byte)
#   CR11 = ROWS - 1 (BLT.S reads the counter from cycle start)
#   CR12 = 1 in INT8 (take the ACTIVATE.QUANTIZE tail), else 0
#
# Output rows from CR7: 0-3, 4-7 and 8-11 are three R_ACC stores; in INT8,
# 12-15 hold POST_AAQ_REG. ACTIVATE.QUANTIZE needs INT8, so in a float format
# the results leave through ST.ACC only.
    SET                 LR0 CR0;
    SET                 LR3 CR6;
    SET                 LR1 CR0;;
    LD.MASK             LR0 CR5;
    SET                 LR4 CR8;;                                 # LR4 = 4: an R0 element used as a scalar below
    LD.MULT.CYCLIC      LR0 CR3 LR0;;                             # R_CYCLIC[0..127]   = w
    LD.MULT.CYCLIC      LR0 CR4 LR3;;                             # R_CYCLIC[128..255] = c
    LD.MULT             R0 LR0 CR2;;                              # R0 = A row 0
    LD.MULT             R1 LR0 CR3;;                              # R1 = w
    MULT.VE             LR0 CR9 0 LR0 CR15;  ACC.ADD.FIRST;;      # R_ACC  = 2 * A0
    MULT.RC.VV          LR0 R0 0 LR0 CR15;   ACC.ADD;;            # R_ACC += w * A0
    MULT.RC.VS          LR3 0 LR0 CR15;      ACC.SUB;;            # R_ACC -= c * c
    MULT.RC.VE          LR3 LR4 0 LR0 CR15;  ACC.MAX;;            # R_ACC  = max(R_ACC, c * A0[4])
    ST.ACC              LR0 CR7;;                                 # output rows 0-3
    MULT.RC.VV          LR0 R0 1 LR0 CR15;   ACC.ADD.FIRST;;      # mask slot 1: odd lanes off
    MULT.EE             LR4 CR9 2 LR0 CR15;  ACC.ADD;;            # broadcast 2 * A0[4]; mask slot 2: upper half off
    ADD                 LR5 LR0 CR8;;
    ST.ACC              LR5 CR7;;                                 # output rows 4-7
matvec:                                                           # R_ACC[r] = A[r] . w
    LD.MULT             R0 LR1 CR2;;
    MULT.RC.VV          LR0 R0 0 LR0 CR15;   AGG.SUM.FIRST LR1 CR15;;
    ADD                 LR1 LR1 CR1;
    BLT.S               LR1 CR11 matvec;;                         # LR1 before this word's ADD
    MULT.RC.VV          LR0 R0 0 LR0 CR15;   AGG.MAX.FIRST LR1 CR15;;   # R_ACC[4] = max(A[3] * w)
    ADD                 LR5 LR5 CR8;;
    ST.ACC              LR5 CR7;;                                 # output rows 8-11
    BEQ                 CR12 CR0 end;;
    ACTIVATE.QUANTIZE   relu CR15;;                               # INT8 only: clamp(relu(R_ACC))
    ADD                 LR5 LR5 CR8;;
    ST.AAQ              LR5 CR7;;                                 # output rows 12-15
end:
    HALT;;
