{#- Dense 8x descriptor interpolation, FP32, channels contiguous (H,W,256).

    O[8i+a,8j+b,c] = sum_dy,dx K[i,j,dy,dx,a,b] D[i+dy-1,j+dx-1,c].
    Shared mode has one K[3,3,8,8]; stock mode has one K per coarse cell.
    All nine coefficients are consumed, including arbitrary caller weights.
    No channel mixing, bias, activation or L2 normalization.

    XMEM addresses are ROWS (128 FP32 elements = 512 bytes).
    Input: zero-padded (H+2,W+2,256), two rows per descriptor
      (ipu_kernels.sampling.pack, hwc_halo).
    Weights: nine rows per table, dy/dx order, phases 8*a+b in lanes 0..63
      (../tables.py): one table (scratch k, stock = false) or one per cell
      (k_cells, stock = true); CR13 steps the table per cell.
    Output: (8*band_height,8*W,256), directly in raster order.

    ABI (kernel.toml [abi]); CR0=0, CR1=1 (hardware constants)
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
    Address arithmetic is live for loads/stores; BLT.S sees start-of-word LRs.
    The final tap never prefetches beyond the input/weights allocation.
-#}
    SET LR0 CR0 ; SET LR1 CR0 ; SET LR6 CR0 ;;
    SET LR7 CR0 ; SET LR8 CR0 ;;
row_loop:
    SET LR2 CR0 ;;
cell_loop:
    SET LR3 CR0 ; SET LR9 CR0 ; ADD LR10 LR8 CR0 ;;
phase_row_loop:
    SET LR4 CR0 ;;
pixel_loop:
    SET LR5 CR0 ;;
channel_loop:
    ADD LR11 LR6 LR5 ; ADD LR12 LR7 CR0 ;;
    LD.MULT R0 LR12 CR4 ;;
    LD.MULT.CYCLIC LR11 CR2 LR0 ;;
{% for tap in range(8) %}
    ADD LR11 LR11 {{ 'CR6' if tap in (2,5) else 'CR9' }} ;
    ADD LR12 LR12 CR1 ;
    LD.MULT R0 LR12 CR4 ;
    MULT.RC.VE LR0 LR9 0 LR0 CR15 ;
    {{ 'ACC.ADD.FIRST' if tap == 0 else 'ACC.ADD' }} ;;
    LD.MULT.CYCLIC LR11 CR2 LR0 ;;
{% endfor %}
    ADD LR5 LR5 CR1 ;
    MULT.RC.VE LR0 LR9 0 LR0 CR15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity CR15 ; ST.AAQ LR10 CR3 ;;
    ADD LR10 LR10 CR1 ; BLT.S LR5 CR9 channel_loop ;;
    ADD LR4 LR4 CR1 ; ADD LR9 LR9 CR1 ;;
    BLT.S LR4 CR10 pixel_loop ;;
    ADD LR3 LR3 CR1 ; ADD LR10 LR10 CR11 ;;
    BLT.S LR3 CR10 phase_row_loop ;;
    ADD LR2 LR2 CR1 ; ADD LR6 LR6 CR9 ; ADD LR7 LR7 CR13 ;;
    INC LR8 16 ; BLT.S LR2 CR7 cell_loop ;;
    ADD LR1 LR1 CR1 ; INC LR6 4 ; ADD LR8 LR8 CR12 ;;
    BLT.S LR1 CR8 row_loop ;;
    HALT ;;
