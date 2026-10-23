{#- ==========================================================================
    softmax_columns.asm -- column softmax (down the rows), FP32

    y[:, col] = softmax(x[:, col]) for a (rows, width) matrix with width >= 65:

      softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))
      c = log2(e), so 2^(c*d) == e^d and the IPU's native exp2 applies directly.

    This is the TRANSPOSE of softmax_rows: the reduction runs ACROSS rows, within
    each element. Each element is an independent column, so the per-column max/sum
    are full 128-element vectors -- NO AGG, NO fan-out, and no group cap on the
    rows (rows is an ordinary loop bound). The running reduce down the rows is
    ACC.MAX (Pass 1) / ACC.ADD (Pass 3); the subtract (Pass 2) and normalise
    (Pass 4) are element-aligned vector ops against the resident cmax/rvec
    chunks:

      Pass 1 (reduce): cmax[c]  = max_r (c * x[r, c])          -> max rows
      Pass 2 (trip):   num[r,c] = 2^(c * x[r, c] - cmax[c])    -> num region
      Pass 3 (reduce): sum[c]   = SUM_r num[r, c]; rvec = 1/sum -> rvec rows
      Pass 4 (trip):   y[r,c]   = num[r, c] * rvec[c]          -> y region

    The first row (r=0) of each chunk-column is PEELED so it runs the *.FIRST
    accumulate (clean init); the loop then runs r=1..rows-1 with the running
    accumulate. This avoids a per-iteration first/running branch.

    Issue #157: MULT reads its Ra/R_CYCLIC DATA at cycle start, so a value
    LOADED this cycle is only visible to MULT NEXT cycle. Every loop therefore
    keeps the LD.MULT in its own instruction word, one word before the MULT that consumes it
    (exactly as softmax_rows does).

    LAYOUT (cpr = ceil(width / 128) chunks per row):
      row r, chunk-column c at <region> + r * cpr + c. The harness packs x
      row-major, each row zero-padded to cpr rows: a padding element is a column
      of its own (never part of a real column's reduce), computed harmlessly and
      dropped on unpack. At width 65..127, 128 - width lanes of the one chunk
      are padding, so such a width runs at the cost of width 128 (below 65
      several whole rows fit one vector: softmax_columns_packed).
      Loop OUTER over chunk-column c (0..cpr-1), INNER over rows r (0..rows-1).
      cmax[c] / rvec[c] are resident full chunks at cr.max + c / cr.rvec + c.

    Registers. All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The
    harness places the regions and sets the CRs from kernel.toml ([abi]):
      cr.x      x, the input logits         cr.y     y, the output
      cr.num    num, the numerators         cr.log2e C_VEC: log2(e) in every lane
      cr.max    cmax, cpr rows              cr.rvec  rvec, cpr rows
      cr.row_stride        = 1 (one row per chunk)
      cr.row_stride_chunks = cpr (XMEM rows per matrix row: the stride within a chunk-column)
      cr.rows              = rows (the row loop bound)
      cr.cpr               = cpr (the chunk-column loop bound)
      CR0 = 0 and CR1 = 1 (-> the 1.0 scalar, and the +1 step) are read-only;
      CR15.L holds the default DS (128 valid elements: every element live).

    LR uses 3 sub-slots; ";;" ends a instruction word, ";" separates sub-instructions.
    R_CYCLIC loads use index 0 (a whole row).
========================================================================== -#}

{%- set lr_ioff   = "LR0" -%}  {#- working x / num / y row offset (steps by cpr) -#}
{%- set lr_row    = "LR1" -%}  {#- inner row index r -#}
{%- set lr_cyc    = "LR2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_woff   = "LR3" -%}  {#- working write offset (num / y), steps by cpr -#}
{%- set lr_chunk  = "LR4" -%}  {#- outer chunk-column index c -#}
{%- set lr_coff   = "LR5" -%}  {#- chunk row offset = c (start of this chunk-column) -#}
{%- set lr_svoff  = "LR6" -%}  {#- resident scalar-vector row offset = c (cmax[c]/rvec[c]) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- cmax[c] = max_r (c * x[r,c]).  Running ACC.MAX down rows.     -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}}   CR0 ;
    SET {{lr_chunk}} CR0 ;
    SET {{lr_coff}}  CR0 ;;
    SET {{lr_svoff}} CR0 ;;
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;       {#- R_CYCLIC = C_VEC -#}

p1_chunk:
    ADD {{lr_ioff}} {{lr_coff}} CR0 ;                       {#- input offset = c (r=0) -#}
    SET {{lr_row}}  CR1 ;;                                  {#- inner index starts at r=1 (r=0 peeled) -#}

    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;                     {#- R0 = x[0,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;          {#- MULT_RES = c*x[0,c] -#}
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;;                      {#- advance to r=1 -#}
    ACC.MAX.FIRST ;;                                        {#- R_ACC = c*x[0,c] -#}

p1_row:
    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;                     {#- R0 = x[r,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;          {#- MULT_RES = c*x[r,c] -#}
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}}  {{lr_row}}  CR1 ;;
    ACC.MAX ;;                                              {#- R_ACC = max(R_ACC, c*x[r,c]) -#}
    BLT.S {{lr_row}} {{ cr.rows }} p1_row ;;

    ACTIVATE.EXACT identity CR15.L {{ cr.max }} {{lr_svoff}} ;;  {#- reads R_ACC forwarded; CMAX[c] <- per-column max -#}

    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;
    ADD {{lr_coff}}  {{lr_coff}}  {{ cr.row_stride }} ;
    ADD {{lr_svoff}} {{lr_svoff}} {{ cr.row_stride }} ;;
    BLT.S {{lr_chunk}} {{ cr.cpr }} p1_chunk ;;

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,c] = 2^(c*x[r,c] - cmax[c]).  Writes NUM region.        -#}
{#-   k:   R0=x[r,c]; MULT_RES=c*x[r,c]; ACC.ADD.FIRST                      -#}
{#-   k+1: R_CYCLIC=cmax[c]; MULT.RC.VE x CR1(=1.0); ACC.SUB               -#}
{#-   k+2: ACTIVATE exp2 (reads R_ACC forwarded); store num[r,c]            -#}
{#- The cmax[c] subtract is a FULL element-aligned vector (no element select).-#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} CR0 ;
    SET {{lr_coff}}  CR0 ;;
    SET {{lr_svoff}} CR0 ;;

p2_chunk:
    ADD {{lr_ioff}} {{lr_coff}} CR0 ;
    ADD {{lr_woff}} {{lr_coff}} CR0 ;
    SET {{lr_row}}  CR0 ;;

p2_row:
    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;                     {#- R0 = x[r,c] (visible next cycle) -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;        {#- R_CYCLIC = C_VEC (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;          {#- MULT_RES = c*x[r,c] -#}
    ACC.ADD.FIRST ;;                                        {#- R_ACC = c*x[r,c] -#}

    LD.MULT.CYCLIC {{lr_svoff}} {{ cr.max }} {{lr_cyc}} ;;      {#- R_CYCLIC = cmax[c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;         {#- MULT_RES = cmax[c]*1.0 -#}
    ACC.SUB ;                                               {#- R_ACC = c*x[r,c] - cmax[c] -#}
    ACTIVATE.EXACT exp2 CR15.L {{ cr.num }} {{lr_woff}} ;;  {#- reads R_ACC forwarded; NUM[r,c] = 2^(...) -#}

    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_woff}} {{lr_woff}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}}  {{lr_row}}  CR1 ;;
    BLT.S {{lr_row}} {{ cr.rows }} p2_row ;;

    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;
    ADD {{lr_coff}}  {{lr_coff}}  {{ cr.row_stride }} ;
    ADD {{lr_svoff}} {{lr_svoff}} {{ cr.row_stride }} ;;
    BLT.S {{lr_chunk}} {{ cr.cpr }} p2_chunk ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sum[c] = SUM_r num[r,c]; rvec[c] = 1/sum.  Running ACC.ADD.   -#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} CR0 ;
    SET {{lr_coff}}  CR0 ;;
    SET {{lr_svoff}} CR0 ;;

p3_chunk:
    ADD {{lr_ioff}} {{lr_coff}} CR0 ;                       {#- num offset = c (r=0) -#}
    SET {{lr_row}}  CR1 ;;                                  {#- r=0 peeled; loop from r=1 -#}

    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;       {#- R_CYCLIC = num[0,c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;         {#- MULT_RES = num[0,c]*1.0 -#}
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;;                      {#- advance to r=1 -#}
    ACC.ADD.FIRST ;;                                        {#- R_ACC = num[0,c] -#}

p3_row:
    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;       {#- R_CYCLIC = num[r,c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;         {#- MULT_RES = num[r,c]*1.0 -#}
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}}  {{lr_row}}  CR1 ;;
    ACC.ADD ;;                                              {#- R_ACC += num[r,c] -#}
    BLT.S {{lr_row}} {{ cr.rows }} p3_row ;;

    ACTIVATE.EXACT reciprocal CR15.L {{ cr.rvec }} {{lr_svoff}} ;;  {#- reads R_ACC forwarded; RVEC[c] <- 1/sum -#}

    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;
    ADD {{lr_coff}}  {{lr_coff}}  {{ cr.row_stride }} ;
    ADD {{lr_svoff}} {{lr_svoff}} {{ cr.row_stride }} ;;
    BLT.S {{lr_chunk}} {{ cr.cpr }} p3_chunk ;;

{#- ===================================================================== -#}
{#- PASS 4 -- out[r,c] = num[r,c] * rvec[c].  Writes OUTPUT region.         -#}
{#-   R_CYCLIC = num[r,c]; R1 = rvec[c]; element-aligned vector x vector.      -#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} CR0 ;
    SET {{lr_coff}}  CR0 ;;
    SET {{lr_svoff}} CR0 ;;

p4_chunk:
    LD.MULT R1 {{lr_svoff}} {{ cr.rvec }} ;;                     {#- R1 = rvec[c] (resident full chunk, visible next cycle) -#}
    ADD {{lr_ioff}} {{lr_coff}} CR0 ;
    SET {{lr_row}}  CR0 ;;

p4_row:
    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;       {#- R_CYCLIC = num[r,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R1 0 {{lr_cyc}} CR15.L {{ cr.qf }}.L ;          {#- MULT_RES = num[r,c] * rvec[c] -#}
    ACC.ADD.FIRST ;
    ACTIVATE.EXACT identity CR15.L {{ cr.y }} {{lr_ioff}} ;;  {#- reads R_ACC forwarded; OUT[r,c] = softmax -#}

    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}}  {{lr_row}}  CR1 ;;
    BLT.S {{lr_row}} {{ cr.rows }} p4_row ;;

    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;
    ADD {{lr_coff}}  {{lr_coff}}  {{ cr.row_stride }} ;
    ADD {{lr_svoff}} {{lr_svoff}} {{ cr.row_stride }} ;;
    BLT.S {{lr_chunk}} {{ cr.cpr }} p4_chunk ;;

end:
    HALT ;;
