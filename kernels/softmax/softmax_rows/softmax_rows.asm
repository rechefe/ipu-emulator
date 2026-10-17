{#- ==========================================================================
    softmax_rows.asm -- numerically stable row softmax, FP32

    y[r, :] = softmax(x[r, :]) for a (rows, 128) matrix, any rows >= 1:

      softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

    so that 2^(c*d) == e^d and the IPU's native exp2 activation applies
    directly. All scaling is on the FP32 vector path (MULT.RC.VV / MULT.RC.VE):
    a CR scalar is an integer even in the FP32 format
    (docs/content/explanation/element-formats.md), so c is the resident
    128-element vector log2e, while the integer scalar 1 (-> 1.0) rides in the
    read-only CR1.

    Passes (per group, below):
      Pass 1 (reduction): maxvec[r] = max_j (c * x[r, j])         -> max row
      Pass 2 (trip):      num[r, j] = 2^(c * x[r, j] - maxvec[r])  -> num region
      Pass 3 (reduction): sumvec[r] = SUM_j num[r, j]; rvec = 1/sumvec -> rvec row
      Pass 4 (trip):      y[r, j]   = num[r, j] * rvec[r]          -> y region
    Only Passes 2 and 4 write a whole matrix; Passes 1 and 3 each leave one
    128-element vector (one scalar per row) in R_ACC and store it once.

    ROW GROUPING (exact, no padding):
      maxvec[r] / rvec[r] pack one scalar per row into a single 128-element
      vector, so the chip holds the per-row bookkeeping for at most 128 rows at
      once. All four passes therefore run once per GROUP of up to 128 rows. Each
      group is processed completely (Pass 1..4) before the next, because
      maxvec / rvec are single shared vectors that the next group overwrites.

      The group size is EXACT, not padded: at the top of each group
        rows_this_group = min(128, total_rows - rows_done)
      so the last (partial) group runs only the rows that remain. A 7-row input
      runs exactly 7 rows; a 130-row input runs 128 then 2. The four passes all
      branch against lr_bound (this group's size), not a fixed constant. The
      loop ends when rows_done reaches total_rows. Groups of 128 run at about
      18 cycles per row (bazel run //kernels/softmax/softmax_rows -- --benchmark).

      lr_gbase = rows_done is the group's first row in the x / num / y regions;
      each pass starts its working row offset there, so row r (0..bound-1) of the
      current group is at <region> + lr_gbase + r.

    Registers. All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The
    harness places the regions and sets the CRs from kernel.toml ([abi]):
      cr.x      x, the input logits         cr.y     y, the output
      cr.num    num, the numerators         cr.log2e C_VEC: log2(e) in every lane
      cr.max    maxvec row                  cr.rvec  rvec row (1 / sum)
      cr.row_stride = 1  (one XMEM row per matrix row)
      cr.group_cap  = 128 (the group cap, and maxvec's index base in R1: R1's
                    element 128 + r is maxvec[r])
      cr.rows       = total rows
      CR0 = 0 and CR1 = 1 (-> the 1.0 scalar, and the +1 step) are read-only;
      CR15 holds the default dstructure (128 valid elements).

    LR uses 3 sub-slots; ";;" ends a instruction word, ";" separates sub-instructions.
    R_CYCLIC loads use index 0 (a whole row).
========================================================================== -#}

{%- set lr_off       = "LR0" -%}  {#- working x / num / y row offset (steps by 1) -#}
{%- set lr_row       = "LR1" -%}  {#- per-group row index r = AGG dest slot / element select -#}
{%- set lr_cyc       = "LR2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_wr        = "LR3" -%}  {#- working write offset for the NUM region -#}
{%- set lr_max_idx   = "LR4" -%}  {#- element index 128+r into R1 = maxvec[r] -#}
{%- set lr_done      = "LR5" -%}  {#- rows completed so far -#}
{%- set lr_gbase     = "LR6" -%}  {#- group base row = rows_done -#}
{%- set lr_bound     = "LR7" -%}  {#- this group's exact row count = min(128, remaining) -#}
{%- set lr_total     = "LR8" -%}  {#- total_rows (LR copy; SUB needs an LR src_a) -#}

    SET {{lr_done}}  CR0 ;
    SET {{lr_gbase}} CR0 ;
    SET {{lr_cyc}}   CR0 ;;
    SET {{lr_total}} {{ cr.rows }} ;;                             {#- lr_total = total_rows (for SUB src_a) -#}

group_loop:
{#- ---- exact group size: lr_bound = min(128, total_rows - rows_done) ------ -#}
    SUB {{lr_bound}} {{lr_total}} {{lr_done}} ;;         {#- bound = remaining = total - done (>=1) -#}
    BLT.S {{lr_bound}} {{ cr.group_cap }} group_size_set ;;               {#- if remaining < 128, keep it ... -#}
    SET {{lr_bound}} {{ cr.group_cap }} ;;                              {#- ... else cap bound = 128 -#}
group_size_set:

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[r] = max_j (c * x[r, j]).  No row written.            -#}
{#- ===================================================================== -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;   {#- R_CYCLIC = C_VEC -#}
    ADD {{lr_off}} {{lr_gbase}} CR0 ;                   {#- working offset = group base (SET takes CR only) -#}
    SET {{lr_row}} CR0 ;;

pass1_loop:
    LD.MULT  R0 {{lr_off}} {{ cr.x }} ;;                 {#- R0 = x[r] (read at cycle start: visible NEXT cycle, issue #157) -#}
    MULT.RC.VV    {{lr_cyc}} R0 0 {{lr_cyc}} CR15 ;          {#- MULT_RES = c*x[r] -#}
    AGG.MAX.FIRST {{lr_row}} CR15 ;;                       {#- R_ACC[r] = max element -#}
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} pass1_loop ;;

    ACTIVATE.QUANTIZE identity CR15;                     {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.max }} ;;                  {#- cr.max <- maxvec -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[r] = 2^(c*x[r] - maxvec[r]).  Writes NUM region.         -#}
{#-   k:   R0=x[r]; R1=maxvec row; MULT_RES=c*x[r]; ACC.ADD.FIRST          -#}
{#-   k+1: MULT_RES = maxvec[r] (x CR1=1.0); ACC.SUB -> R_ACC - maxvec[r]  -#}
{#-   k+2: ACTIVATE exp2 (reads R_ACC forwarded); store num[r]              -#}
{#- ===================================================================== -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;   {#- R_CYCLIC = C_VEC -#}
    LD.MULT R1 {{lr_cyc}} {{ cr.max }} ;;                   {#- R1 = maxvec row (resident; all rows) -#}
    ADD {{lr_off}} {{lr_gbase}} CR0 ;
    ADD {{lr_wr}}  {{lr_gbase}} CR0 ;
    SET {{lr_row}} CR0 ;;
    SET {{lr_max_idx}} {{ cr.group_cap }} ;;                           {#- lr_max_idx = 128 (R1[0] = maxvec[0]); cr.group_cap = 128 -#}

pass2_loop:
    LD.MULT R0 {{lr_off}} {{ cr.x }} ;;                   {#- R0 = x[r] (read at cycle start: visible NEXT cycle, issue #157) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15 ;           {#- MULT_RES = c*x[r] -#}
    ACC.ADD.FIRST ;;                                         {#- R_ACC = c*x[r] -#}

    MULT.EE {{lr_max_idx}} CR1 0 {{lr_cyc}} CR15 ;           {#- MULT_RES = maxvec[r]*1.0 -#}
    ACC.SUB ;                                                {#- R_ACC = c*x[r] - maxvec[r] -#}
    ACTIVATE.QUANTIZE exp2 CR15;                             {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_wr}} {{ cr.num }} ;;                    {#- NUM[r] = 2^(...) -#}

    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_wr}} {{lr_wr}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    ADD {{lr_max_idx}} {{lr_max_idx}} CR1 ;;             {#- next maxvec element -#}
    BLT.S {{lr_row}} {{lr_bound}} pass2_loop ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM_j num[r,j]; rvec = 1/sumvec.  Staged.        -#}
{#-   R_CYCLIC = num[r]; MULT.RC.VE x CR1(=1.0); AGG.SUM.FIRST into R_ACC[r]-#}
{#- ===================================================================== -#}
    ADD {{lr_off}} {{lr_gbase}} CR0 ;
    SET {{lr_row}} CR0 ;;

pass3_loop:
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;    {#- R_CYCLIC = num[r] (read at cycle start: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} CR1 0 {{lr_cyc}} CR15 ;         {#- MULT_RES = num[r]*1.0 -#}
    AGG.SUM.FIRST {{lr_row}} CR15 ;;                        {#- R_ACC[r] = SUM num[r] -#}
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} pass3_loop ;;

    ACTIVATE.QUANTIZE reciprocal CR15;                   {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.rvec }} ;;                   {#- cr.rvec <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[r] = num[r] * rvec[r].  Writes OUTPUT region.            -#}
{#-   R_CYCLIC = num[r]; R0 = rvec row; broadcast rvec[r] via LR-element.  -#}
{#- ===================================================================== -#}
    LD.MULT R0 {{lr_cyc}} {{ cr.rvec }} ;;                    {#- R0 = rvec row (resident; elem r = rvec[r]) -#}
    ADD {{lr_off}} {{lr_gbase}} CR0 ;
    SET {{lr_row}} CR0 ;;

pass4_loop:
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;    {#- R_CYCLIC = num[r] (read at cycle start: visible NEXT cycle) -#}
    MULT.RC.VE   {{lr_cyc}} {{lr_row}} 0 {{lr_cyc}} CR15 ;   {#- MULT_RES = num[r]*rvec[r] -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE identity CR15;                     {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_off}} {{ cr.y }} ;;                   {#- OUT[r] = softmax row -#}

    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} pass4_loop ;;

{#- ---- advance: rows_done += bound; loop while done < total -------------- -#}
{#- After Pass 4, lr_off = gbase + bound (it stepped once per row), which -#}
{#- is exactly the next group's base row -- no multiply needed (the LR slot   -#}
{#- has no multiply). Copy it into lr_gbase. -#}
    ADD {{lr_done}} {{lr_done}} {{lr_bound}} ;            {#- rows_done += this group -#}
    ADD {{lr_gbase}} {{lr_off}} CR0 ;;                   {#- next gbase = end of this group -#}
    BLT.S {{lr_done}} {{lr_total}} group_loop ;;

end:
    HALT ;;
