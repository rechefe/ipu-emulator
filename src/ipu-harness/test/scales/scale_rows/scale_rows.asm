{#- scale_rows.asm -- y = x * s, one XMEM row (128 elements) per iteration.

    Design notes
    - MULT.VE multiplies the row in R0 by the element scalar in {{ cr.s }}.
      The scalar is decoded in the active element format (Q49), so the same
      program runs in FP32, INT8 and E4.
    - The product goes through ACC (first add), and ST.ACC stores the
      accumulator vector: y is an accumulator output, FP32 in FP32 and E4,
      INT32 in INT8. One accumulator vector is one row in FP32 and four rows
      in an 8-bit format, hence acc_rows.
    - Region bases and the row count are in rows, not bytes.
-#}
{%- set acc_rows = 1 if format in ("fp32", "int32") else 4 %}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ST.ACC LR2 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 {{ acc_rows }} ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
