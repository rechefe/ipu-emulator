{#- Dense 8x descriptor interpolation, FP32, channels contiguous (H,W,256).

    O[8i+a,8j+b,c] = sum_dy,dx K[i,j,dy,dx,a,b] D[i+dy-1,j+dx-1,c].
    Shared mode has one K[3,3,8,8]; stock mode has one K per coarse cell.
    All nine coefficients are consumed, including arbitrary caller weights.
    No channel mixing, bias, activation or L2 normalization.

    XMEM addresses are ROWS (128 FP32 elements = 512 bytes).
    Input: zero-padded (H+2,W+2,256), two rows per descriptor.
    Weights: nine rows per table, dy/dx order, phases 8*a+b in lanes 0..63.
    Output: (8*band_height,8*W,256), directly in raster order.

    CR0=0, CR1=1 (hardware constants)
    CR2=input base at band's top halo row; CR3=output base
    CR4=weight base for band's first cell
    CR5=padded input row stride=2*(W+2)
    CR6=CR5-4, next tap-row address delta
    CR7=W; CR8=band coarse height; CR9=2 channel tiles; CR10=8 phases/axis
    CR11=16*W-16 output phase-row gap; CR12=7*16*W output cell-row gap
    CR13=weight cell step (0 shared, 9 stock); CR15=128-lane dstructure

    LR0=zero/cyclic slot; LR1=coarse row; LR2=coarse column
    LR3=a; LR4=b; LR5=channel tile; LR6=top-left neighbor row offset
    LR7=cell weight offset; LR8=output cell origin; LR9=phase scalar index
    LR10=output walking address; LR11=tap descriptor address; LR12=tap weight

    MULT reads register snapshots. Prefetch the NEXT weight while multiplying
    the current tap, then load the next descriptor in a separate word.
    Address arithmetic is live for loads/stores; BLT sees start-of-word LRs.
    The final tap never prefetches beyond the input/weights allocation.
-#}
    SET lr0 cr0 ; SET lr1 cr0 ; SET lr6 cr0 ;;
    SET lr7 cr0 ; SET lr8 cr0 ;;
row_loop:
    SET lr2 cr0 ;;
cell_loop:
    SET lr3 cr0 ; SET lr9 cr0 ; ADD lr10 lr8 cr0 ;;
phase_row_loop:
    SET lr4 cr0 ;;
pixel_loop:
    SET lr5 cr0 ;;
channel_loop:
    ADD lr11 lr6 lr5 ; ADD lr12 lr7 cr0 ;;
    LDR_MULT_REG r0 lr12 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr2 lr0 ;;
{% for tap in range(8) %}
    ADD lr11 lr11 {{ 'cr6' if tap in (2,5) else 'cr9' }} ;
    ADD lr12 lr12 cr1 ;
    LDR_MULT_REG r0 lr12 cr4 ;
    MULT.RC.VE lr0 lr9 0 lr0 cr15 ;
    {{ 'ACC.ADD.FIRST' if tap == 0 else 'ACC.ADD' }} ;;
    LDR_CYCLIC_MULT_REG lr11 cr2 lr0 ;;
{% endfor %}
    ADD lr5 lr5 cr1 ;
    MULT.RC.VE lr0 lr9 0 lr0 cr15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity cr15 ; STR_POST_AAQ_REG lr10 cr3 ;;
    ADD lr10 lr10 cr1 ; BLT lr5 cr9 channel_loop ;;
    ADD lr4 lr4 cr1 ; ADD lr9 lr9 cr1 ;;
    BLT lr4 cr10 pixel_loop ;;
    ADD lr3 lr3 cr1 ; ADD lr10 lr10 cr11 ;;
    BLT lr3 cr10 phase_row_loop ;;
    ADD lr2 lr2 cr1 ; ADD lr6 lr6 cr9 ; ADD lr7 lr7 cr13 ;;
    INC lr8 16 ; BLT lr2 cr7 cell_loop ;;
    ADD lr1 lr1 cr1 ; INC lr6 4 ; ADD lr8 lr8 cr12 ;;
    BLT lr1 cr8 row_loop ;;
    BKPT ;;
