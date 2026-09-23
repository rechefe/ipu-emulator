{#- ==========================================================================
    softmax_rows_long.asm -- row-softmax for N>128, N%128 != 0, FP32 wide mode

    Mix of softmax_rows (full 128-wide rows) and softmax_rows_partial (sub-128
    tail). Each row = full_chunks (CR12) full 128-element chunks + one tail of
    `tail` (CR8) elements. Row r, chunk c lives at byte BASE + r*CR14 + c*512
    (CR14 = cpr*512 row stride; CR13 = cpr).

      softmax(x_i) = 2^(c*(x_i - xmax)) / SUM 2^(c*(x_j - xmax)),  c = log2(e)

    Per-row scalars maxvec[r]/rvec[r] hold one value/row in a 128-element vector, so
    all four passes run once per GROUP of up to 128 rows: each group is processed
    completely (Pass 1..4) before the next, because maxvec/rvec are single shared
    vectors the next group overwrites. The group size is exact, not padded --
      rows_this_group = min(128, total_rows - rows_done)
    so the last group runs only the rows that remain, and every pass branches
    against lr_bound (this group's size) rather than the total. This keeps
    lr_row in 0..127 within a group, which matters beyond loop bookkeeping:
    lr_row is MULT.RC.VE's `src` scalar-select (Pass 4) and AGG's dest slot, and
    an index >=128 would silently select R1 instead of R0.

    CROSS-CHUNK REDUCTION (the novelty): a row's max (Pass 1) and sum (Pass 3)
    span cpr chunks. Each pass keeps a RUNNING per-row scalar: AGG.*.FIRST on the
    row's first full chunk, AGG.* (running, combines with r_acc[row]) on the rest,
    and finally the tail chunk with valid_elements=tail. Full chunks name CR15
    (valid_elements=128); the tail chunk names CR8 (valid_elements=tail).

    CR map (CR0=0, CR1=1 read-only):
      CR2=OUT_BASE  CR3=CVEC  CR4=NUM_BASE  CR5=MAXVEC  CR6=RVEC
      CR7=512 (chunk stride)  CR8=tail (also tail dstructure: valid_elements=tail)
      CR9=128 (R1 maxvec byte base; also GROUP_CAP)  CR10=INPUT_BASE
      CR11=total rows (group loop bound)
      CR12=full_chunks (full-chunk loop bound)  CR13=cpr  CR14=cpr*512 (row stride)
      CR15=full-chunk dstructure (valid_elements=128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    r_cyclic loads use index 0 in wide mode (full 512B load).
========================================================================== -#}

{%- set lr_off     = "lr0" -%}  {#- working chunk byte offset (input/num/out) -#}
{%- set lr_row     = "lr1" -%}  {#- row index r = AGG dest slot / rvec element select -#}
{%- set lr_cyc     = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_rbase   = "lr3" -%}  {#- row byte base = r * cpr * 512 -#}
{%- set lr_max_idx = "lr4" -%}  {#- R1 byte index 128+r = maxvec[r] -#}
{%- set lr_chunk   = "lr5" -%}  {#- inner chunk counter (0..full_chunks-1) -#}
{%- set lr_wr      = "lr6" -%}  {#- working write offset (NUM region, Pass 2) -#}
{%- set lr_done    = "lr7" -%}  {#- rows completed by previous groups -#}
{%- set lr_bound   = "lr8" -%}  {#- this group's exact row count = min(128, remaining) -#}
{%- set lr_gbase   = "lr9" -%}  {#- group byte base = rows_done * cpr * 512 -#}
{%- set lr_total   = "lr10" -%} {#- total rows (SUB needs an LR src_a) -#}

    SET {{lr_cyc}}   cr0 ;
    SET {{lr_done}}  cr0 ;
    SET {{lr_gbase}} cr0 ;;
    SET {{lr_total}} cr11 ;;                           {#- SUB needs an LR src_a -#}

group_loop:
{#- ---- exact group size: lr_bound = min(128, total_rows - rows_done) ------ -#}
    SUB {{lr_bound}} {{lr_total}} {{lr_done}} ;;       {#- remaining = total - done (>=1) -#}
    BLT {{lr_bound}} cr9 group_size_set ;;             {#- remaining < 128 -> keep it ... -#}
    SET {{lr_bound}} cr9 ;;                            {#- ... else cap at 128 -#}
group_size_set:

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[r] = max over the row's N elements (cross-chunk).      -#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;   {#- r_cyclic = C_VEC -#}
    SET {{lr_row}}   cr0 ;
    ADD {{lr_rbase}} {{lr_gbase}} cr0 ;;               {#- start at this group's base -#}

p1_row:
    {#- first full chunk: AGG.MAX.FIRST seeds r_acc[row] -#}
    ADD {{lr_off}} {{lr_rbase}} cr0 ;;
    LDR_MULT_REG  r0 {{lr_off}} cr10 ;;                {#- R0 = x[r,chunk0] (snapshot) -#}
    MULT.RC.VV    {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;     {#- c*x -#}
    AGG.MAX.FIRST {{lr_row}} cr15 ;;                    {#- r_acc[r] = max chunk0 -#}
    ADD {{lr_off}} {{lr_off}} cr7 ;
    SET {{lr_chunk}} cr1 ;;                             {#- chunk = 1 (chunk0 done) -#}

p1_full:
    {#- remaining full chunks: running AGG.MAX into r_acc[row] -#}
    BGE {{lr_chunk}} cr12 p1_tail ;;                    {#- no more full chunks -> tail -#}
    LDR_MULT_REG r0 {{lr_off}} cr10 ;;
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;
    AGG.MAX      {{lr_row}} cr15 ;;                     {#- running max -#}
    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    B p1_full ;;

p1_tail:
    {#- tail chunk: running AGG.MAX over the first `tail` elements (CR8) -#}
    LDR_MULT_REG r0 {{lr_off}} cr10 ;;
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr8 ;
    AGG.MAX      {{lr_row}} cr8 ;;                      {#- running max, valid_elements=tail -#}

    ADD {{lr_rbase}} {{lr_rbase}} cr14 ;               {#- next row base -#}
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT {{lr_row}} {{lr_bound}} p1_row ;;

    ACTIVATE.QUANTIZE identity cr15 ;                  {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr5 ;;

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,c] = 2^(c*x - maxvec[r]).  Writes NUM region.          -#}
{#-   maxvec[r] is the SAME for every chunk of row r (subtract via ACC.SUB).-#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;   {#- r_cyclic = C_VEC -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr5 ;;                  {#- R1 = maxvec (elem r) -#}
    SET {{lr_row}}   cr0 ;
    ADD {{lr_rbase}} {{lr_gbase}} cr0 ;;
    SET {{lr_max_idx}} cr9 ;;                          {#- 128 + 0 = maxvec[0] -#}

p2_row:
    ADD {{lr_off}} {{lr_rbase}} cr0 ;
    ADD {{lr_wr}}  {{lr_rbase}} cr0 ;
    SET {{lr_chunk}} cr0 ;;

p2_chunk:
    BGE {{lr_chunk}} cr13 p2_row_done ;;               {#- all cpr chunks done -#}
    LDR_MULT_REG r0 {{lr_off}} cr10 ;;                 {#- R0 = x[r,c] (snapshot) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;     {#- c*x -#}
    acc.add.first ;;
    MULT.EE {{lr_max_idx}} cr1 0 {{lr_cyc}} cr15 ;     {#- maxvec[r]*1.0 -#}
    acc.sub ;                                          {#- r_acc = c*x - maxvec[r] -#}
    ACTIVATE.QUANTIZE exp2 cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_wr}} cr4 ;;

    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_wr}}  {{lr_wr}}  cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    B p2_chunk ;;

p2_row_done:
    ADD {{lr_rbase}} {{lr_rbase}} cr14 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;
    ADD {{lr_max_idx}} {{lr_max_idx}} cr1 ;;
    BLT {{lr_row}} {{lr_bound}} p2_row ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM over row's N num values; rvec = 1/sum.        -#}
{#- ===================================================================== -#}
    SET {{lr_row}}   cr0 ;
    ADD {{lr_rbase}} {{lr_gbase}} cr0 ;;

p3_row:
    ADD {{lr_off}} {{lr_rbase}} cr0 ;;
    {#- first full chunk: AGG.SUM.FIRST seeds r_acc[row] -#}
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;   {#- r_cyclic = num[r,0] -#}
    MULT.RC.VE    {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;   {#- num*1.0 -#}
    AGG.SUM.FIRST {{lr_row}} cr15 ;;
    ADD {{lr_off}} {{lr_off}} cr7 ;
    SET {{lr_chunk}} cr1 ;;

p3_full:
    BGE {{lr_chunk}} cr12 p3_tail ;;
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;
    MULT.RC.VE {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;
    AGG.SUM    {{lr_row}} cr15 ;;                       {#- running sum -#}
    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    B p3_full ;;

p3_tail:
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;
    MULT.RC.VE {{lr_cyc}} cr1 0 {{lr_cyc}} cr8 ;
    AGG.SUM    {{lr_row}} cr8 ;;                        {#- running sum, valid_elements=tail -#}

    ADD {{lr_rbase}} {{lr_rbase}} cr14 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT {{lr_row}} {{lr_bound}} p3_row ;;

    ACTIVATE.QUANTIZE reciprocal cr15 ;                {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;

{#- ===================================================================== -#}
{#- PASS 4 -- out[r,c] = num[r,c] * rvec[r].  Writes OUTPUT region.         -#}
{#- ===================================================================== -#}
    LDR_MULT_REG r0 {{lr_cyc}} cr6 ;;                  {#- R0 = rvec (elem r) -#}
    SET {{lr_row}}   cr0 ;
    ADD {{lr_rbase}} {{lr_gbase}} cr0 ;;

p4_row:
    ADD {{lr_off}} {{lr_rbase}} cr0 ;
    SET {{lr_chunk}} cr0 ;;

p4_chunk:
    BGE {{lr_chunk}} cr13 p4_row_done ;;
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;   {#- r_cyclic = num[r,c] -#}
    MULT.RC.VE   {{lr_cyc}} {{lr_row}} 0 {{lr_cyc}} cr15 ;  {#- num*rvec[r] -#}
    acc.add.first ;
    ACTIVATE.QUANTIZE identity cr15 ;                   {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_off}} cr2 ;;

    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    B p4_chunk ;;

p4_row_done:
    ADD {{lr_rbase}} {{lr_rbase}} cr14 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT {{lr_row}} {{lr_bound}} p4_row ;;

{#- ---- advance: rows_done += bound; loop while done < total --------------- -#}
{#- Pass 4 stepped lr_rbase once per row, so it now equals gbase + bound*CR14 -#}
{#- -- exactly the next group's base. Copy it; the LR slot has no multiply.   -#}
    ADD {{lr_done}}  {{lr_done}} {{lr_bound}} ;
    ADD {{lr_gbase}} {{lr_rbase}} cr0 ;;
    BLT {{lr_done}} {{lr_total}} group_loop ;;

end:
    BKPT ;;
