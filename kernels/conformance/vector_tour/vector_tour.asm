{#- vector_tour.asm -- native vector tour: 128-element NX8 rows (the hardware
    mode, ADR 0007).

    A curated conformance program for the Cycle Inspector: it walks the MULT,
    ACC and AaQ slots once each on real NX8 data. It loads a mask, two
    R_CYCLIC slots and R0/R1, runs MULT.VE, MULT.RC.VV/VS/VE and MULT.EE into
    ACC.ADD/SUB/MAX, two masked multiplies, and a 4-row matrix-vector loop with
    AGG.SUM.FIRST. kernel.toml's [abi] says which CR holds what.

    Design notes
    - The registers are written literally (CR2, not a template binding), so the
      file is plain assembly: the Inspector shows it as source, and its labels
      are the source's own.
    - XMEM from row 0: the four rows of a, then w, c and the R_MASK row (three
      16-byte slots: all lanes, even lanes, the lower half). acc starts at
      row 8, as the old harness placed it.
    - Every MULT reads its vectors with the case's QF half in CR14.L, each
      element scaled by the metadata of the row it was loaded from.
    - acc is three ACTIVATE.QUANTIZE identity stores of R_ACC, one NX8 row
      each (rows 0-2 from CR7); relu follows at row 3: ACTIVATE.QUANTIZE relu
      of the third vector. Each store quantizes its row with its own scale.
    - CR9 is the MULT constant 2.0 (the NX8 constant layout); CR8 is lane 4,
      the R0 element MULT.RC.VE and MULT.EE use as a scalar; CR11 is ROWS - 1
      because BLT.S reads the counter from the start of the cycle.
    - The data (gen.py) is small: multiples of 0.25 in [-2, 2], which every
      variant holds exactly. Every product and sum is then exact in the FP32
      accumulator, and the golden is quantized row by row like the program's
      stores, so every variant is compared EXACT.
-#}
    SET                 LR0 CR0;
    SET                 LR3 CR6;
    SET                 LR1 CR0;;
    LD.MASK             LR0 CR5;
    SET                 LR4 CR8;;                                 # LR4 = 4: an R0 element used as a scalar below
    LD.MULT.CYCLIC      LR0 CR3 LR0;;                             # R_CYCLIC[0..127]   = w
    LD.MULT.CYCLIC      LR0 CR4 LR3;;                             # R_CYCLIC[128..255] = c
    LD.MULT             R0 LR0 CR2;;                              # R0 = A row 0
    LD.MULT             R1 LR0 CR3;;                              # R1 = w
    MULT.VE             LR0 CR9 0 LR0 CR15.L CR14.L;  ACC.ADD.FIRST;;   # R_ACC  = 2 * A0
    MULT.RC.VV          LR0 R0 0 LR0 CR15.L CR14.L;   ACC.ADD;;   # R_ACC += w * A0
    MULT.RC.VS          LR3 0 LR0 CR15.L CR14.L;      ACC.SUB;;   # R_ACC -= c * c
    MULT.RC.VE          LR3 LR4 0 LR0 CR15.L CR14.L;  ACC.MAX;;   # R_ACC  = max(R_ACC, c * A0[4])
    ACTIVATE.QUANTIZE   identity CR15.L CR14.L CR7 LR0;;          # output row 0
    MULT.RC.VV          LR0 R0 1 LR0 CR15.L CR14.L;   ACC.ADD.FIRST;;   # mask slot 1: odd lanes off
    MULT.EE             LR4 CR9 2 LR0 CR15.L CR14.L;  ACC.ADD;;   # broadcast 2 * A0[4]; mask slot 2: upper half off
    ADD                 LR5 LR0 CR1;;
    ACTIVATE.QUANTIZE   identity CR15.L CR14.L CR7 LR5;;          # output row 1
matvec:                                                           # R_ACC[r] = A[r] . w
    LD.MULT             R0 LR1 CR2;;
    MULT.RC.VV          LR0 R0 0 LR0 CR15.L CR14.L;   AGG.SUM.FIRST LR1 CR15.L;;
    ADD                 LR1 LR1 CR1;
    BLT.S               LR1 CR11 matvec;;                         # LR1 before this word's ADD
    MULT.RC.VV          LR0 R0 0 LR0 CR15.L CR14.L;   AGG.MAX.FIRST LR1 CR15.L;;   # R_ACC[4] = max(A[3] * w)
    ADD                 LR5 LR5 CR1;;
    ACTIVATE.QUANTIZE   identity CR15.L CR14.L CR7 LR5;;          # output row 2
    ADD                 LR5 LR5 CR1;;
    ACTIVATE.QUANTIZE   relu CR15.L CR14.L CR7 LR5;;              # relu row: relu(R_ACC), quantized
    HALT;;
