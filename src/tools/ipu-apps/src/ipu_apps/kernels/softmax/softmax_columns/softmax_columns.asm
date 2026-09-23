{#- ==========================================================================
    softmax_columns.asm -- column-softmax (down the rows), FP32 wide-vector mode

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))
    c = log2(e), so 2^(c*d) == e^d and the IPU's native exp2 applies directly.

    This is the TRANSPOSE of softmax_rows: the reduction runs ACROSS rows, in
    element. Each element is an independent column, so the per-column max/sum are full
    128-element vectors -- NO AGG, NO fan-out. The running reduce down the rows is
    ACC.MAX (Pass 1) / ACC.ADD (Pass 3); the subtract (Pass 2) and normalise
    (Pass 4) are element-aligned vector ops against the resident cmax/rvec chunks.

    The first row (r=0) of each chunk-column is PEELED so it runs the *.FIRST
    accumulate (clean init); the loop then runs r=1..rows-1 with the running
    accumulate. This avoids a per-iteration first/running branch.

    Issue #157: MULT reads its Ra/r_cyclic DATA from the snapshot, so a value
    LOADED this cycle is only visible to MULT NEXT cycle. Every loop therefore
    keeps the LDR in its own VLIW word, one word before the MULT that consumes it
    (exactly as softmax_rows does).

    LAYOUT (width >= 128, padded to a multiple of 128; cpr = ceil(W/128)):
      row r, chunk-column c at byte (r*cpr + c)*512 = r*row_stride + c*512.
      Loop OUTER over chunk-column c (0..cpr-1), INNER over rows r (0..rows-1).
      cmax[c] / rvec[c] are resident full chunks at BASE + c*512.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                       CR1  = 1  (-> 1.0 scalar; +1 increment)
      CR2  = OUTPUT_BASE             CR3  = CVEC_ADDR        CR4  = NUM_BASE
      CR5  = CMAX_ADDR               CR6  = RVEC_ADDR        CR7  = CHUNK_BYTES (512)
      CR9  = ROW_BYTES (cpr*512, the per-row stride within a chunk-column)
      CR10 = INPUT_BASE              CR11 = ROWS (row loop bound)
      CR13 = CHUNKS_PER_ROW (cpr, chunk-column loop bound)
      CR15 = dstructure, valid_elements = 128 (every element live)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    r_cyclic loads use index 0 in wide mode (full 512B load).
========================================================================== -#}

{%- set lr_ioff   = "lr0" -%}  {#- working input/num/out byte offset (steps by row_stride) -#}
{%- set lr_row    = "lr1" -%}  {#- inner row index r -#}
{%- set lr_cyc    = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_woff   = "lr3" -%}  {#- working write offset (NUM / OUT), steps by row_stride -#}
{%- set lr_chunk  = "lr4" -%}  {#- outer chunk-column index c -#}
{%- set lr_coff   = "lr5" -%}  {#- chunk byte offset = c*512 (start of this chunk-column) -#}
{%- set lr_svoff  = "lr6" -%}  {#- resident scalar-vector offset = c*512 (cmax[c]/rvec[c]) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- cmax[c] = max_r (c * x[r,c]).  Running ACC.MAX down rows.     -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}}   cr0 ;
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}}  cr0 ;;
    SET {{lr_svoff}} cr0 ;;
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;       {#- r_cyclic = C_VEC -#}

p1_chunk:
    ADD {{lr_ioff}} {{lr_coff}} cr0 ;                       {#- input offset = c*512 (r=0) -#}
    SET {{lr_row}}  cr1 ;;                                  {#- inner index starts at r=1 (r=0 peeled) -#}

    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;                     {#- R0 = x[0,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;          {#- mult_res = c*x[0,c] -#}
    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;;                      {#- advance to r=1 -#}
    ACC.MAX.FIRST ;;                                        {#- r_acc = c*x[0,c] -#}

p1_row:
    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;                     {#- R0 = x[r,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;          {#- mult_res = c*x[r,c] -#}
    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;
    ADD {{lr_row}}  {{lr_row}}  cr1 ;;
    ACC.MAX ;;                                              {#- r_acc = max(r_acc, c*x[r,c]) -#}
    BLT {{lr_row}} cr11 p1_row ;;

    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_svoff}} cr5 ;;                    {#- CMAX[c] <- per-column max -#}

    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;
    ADD {{lr_coff}}  {{lr_coff}}  cr7 ;
    ADD {{lr_svoff}} {{lr_svoff}} cr7 ;;
    BLT {{lr_chunk}} cr13 p1_chunk ;;

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,c] = 2^(c*x[r,c] - cmax[c]).  Writes NUM region.        -#}
{#-   k:   R0=x[r,c]; mult_res=c*x[r,c]; ACC.ADD.FIRST                      -#}
{#-   k+1: r_cyclic=cmax[c]; MULT.RC.VE x CR1(=1.0); ACC.SUB               -#}
{#-   k+2: ACTIVATE exp2 (reads r_acc snapshot); store num[r,c]            -#}
{#- The cmax[c] subtract is a FULL element-aligned vector (no element select).-#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}}  cr0 ;;
    SET {{lr_svoff}} cr0 ;;

p2_chunk:
    ADD {{lr_ioff}} {{lr_coff}} cr0 ;
    ADD {{lr_woff}} {{lr_coff}} cr0 ;
    SET {{lr_row}}  cr0 ;;

p2_row:
    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;                     {#- R0 = x[r,c] (visible next cycle) -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;        {#- r_cyclic = C_VEC (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;          {#- mult_res = c*x[r,c] -#}
    acc.add.first ;;                                        {#- r_acc = c*x[r,c] -#}

    LDR_CYCLIC_MULT_REG {{lr_svoff}} cr5 {{lr_cyc}} ;;      {#- r_cyclic = cmax[c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;         {#- mult_res = cmax[c]*1.0 -#}
    acc.sub ;                                               {#- r_acc = c*x[r,c] - cmax[c] -#}
    ACTIVATE.QUANTIZE exp2 cr15 ;                            {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_woff}} cr4 ;;                     {#- NUM[r,c] = 2^(...) -#}

    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;
    ADD {{lr_woff}} {{lr_woff}} cr9 ;
    ADD {{lr_row}}  {{lr_row}}  cr1 ;;
    BLT {{lr_row}} cr11 p2_row ;;

    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;
    ADD {{lr_coff}}  {{lr_coff}}  cr7 ;
    ADD {{lr_svoff}} {{lr_svoff}} cr7 ;;
    BLT {{lr_chunk}} cr13 p2_chunk ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sum[c] = SUM_r num[r,c]; rvec[c] = 1/sum.  Running ACC.ADD.   -#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}}  cr0 ;;
    SET {{lr_svoff}} cr0 ;;

p3_chunk:
    ADD {{lr_ioff}} {{lr_coff}} cr0 ;                       {#- num offset = c*512 (r=0) -#}
    SET {{lr_row}}  cr1 ;;                                  {#- r=0 peeled; loop from r=1 -#}

    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;       {#- r_cyclic = num[0,c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;         {#- mult_res = num[0,c]*1.0 -#}
    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;;                      {#- advance to r=1 -#}
    acc.add.first ;;                                        {#- r_acc = num[0,c] -#}

p3_row:
    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;       {#- r_cyclic = num[r,c] (visible next cycle) -#}
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;         {#- mult_res = num[r,c]*1.0 -#}
    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;
    ADD {{lr_row}}  {{lr_row}}  cr1 ;;
    acc.add ;;                                              {#- r_acc += num[r,c] -#}
    BLT {{lr_row}} cr11 p3_row ;;

    ACTIVATE.QUANTIZE reciprocal cr15 ;                     {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_svoff}} cr6 ;;                    {#- RVEC[c] <- 1/sum -#}

    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;
    ADD {{lr_coff}}  {{lr_coff}}  cr7 ;
    ADD {{lr_svoff}} {{lr_svoff}} cr7 ;;
    BLT {{lr_chunk}} cr13 p3_chunk ;;

{#- ===================================================================== -#}
{#- PASS 4 -- out[r,c] = num[r,c] * rvec[c].  Writes OUTPUT region.         -#}
{#-   r_cyclic = num[r,c]; R1 = rvec[c]; element-aligned vector x vector.      -#}
{#- ===================================================================== -#}
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}}  cr0 ;;
    SET {{lr_svoff}} cr0 ;;

p4_chunk:
    LDR_MULT_REG r1 {{lr_svoff}} cr6 ;;                     {#- R1 = rvec[c] (resident full chunk, visible next cycle) -#}
    ADD {{lr_ioff}} {{lr_coff}} cr0 ;
    SET {{lr_row}}  cr0 ;;

p4_row:
    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;       {#- r_cyclic = num[r,c] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r1 0 {{lr_cyc}} cr15 ;          {#- mult_res = num[r,c] * rvec[c] -#}
    acc.add.first ;
    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_ioff}} cr2 ;;                     {#- OUT[r,c] = softmax -#}

    ADD {{lr_ioff}} {{lr_ioff}} cr9 ;
    ADD {{lr_row}}  {{lr_row}}  cr1 ;;
    BLT {{lr_row}} cr11 p4_row ;;

    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;
    ADD {{lr_coff}}  {{lr_coff}}  cr7 ;
    ADD {{lr_svoff}} {{lr_svoff}} cr7 ;;
    BLT {{lr_chunk}} cr13 p4_chunk ;;

end:
    BKPT ;;
