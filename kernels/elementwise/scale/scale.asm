{#- scale.asm -- y = x * s, one XMEM row (128 elements) per iteration.

    Design notes
    - MULT.VE multiplies the row in R0 by the MULT constant in {{ cr.s }}
      (float32 bits in FP32; ADR 0007), reading R0 with the QF half in
      {{ cr.qf }}.L (FP32 ignores it).
    - The product goes through ACC (first add), and ACTIVATE.EXACT identity
      stores the accumulator vector: one FP32 row per iteration.
    - Region bases and the row count are in rows, not bytes.
-#}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.EXACT identity CR15.L {{ cr.y }} LR2 ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
