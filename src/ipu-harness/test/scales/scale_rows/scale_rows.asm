{#- scale_rows.asm -- y = x * s, one XMEM row (128 elements) per iteration.

    Design notes
    - MULT.VE multiplies the row in R0 by the MULT constant in {{ cr.s }}
      (NX8: the constant layout; FP32: float32 bits; ADR 0007), reading R0
      with the QF half in {{ cr.qf }}.L.
    - The product goes through ACC (first add), and ACTIVATE.QUANTIZE
      (identity) stores one row: quantized to the case's QF in NX8, the 128
      FP32 values in FP32. One row per iteration in both modes.
    - Region bases and the row count are in rows, not bytes.
-#}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR2 ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
