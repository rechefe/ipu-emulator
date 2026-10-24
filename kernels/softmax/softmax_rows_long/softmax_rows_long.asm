{#- ==========================================================================
    softmax_rows_long.asm -- row softmax for n > 128, FP32

    y[r, :] = softmax(x[r, :]) for a (rows, n) matrix with n > 128. A mix of
    softmax_rows (whole 128-wide rows) and softmax_rows_partial (a sub-128
    tail): each row is full_chunks = n // 128 whole 128-element chunks and a
    tail chunk of n % 128 elements, cpr = ceil(n / 128) chunks in all. Row r,
    chunk c is at <region> + r * cpr + c; the tail chunk's lanes past the tail
    are zero (the harness packs x row-major, each row zero-padded to cpr rows).

      softmax(x_i) = 2^(c*(x_i - xmax)) / SUM 2^(c*(x_j - xmax)),  c = log2(e)

    so the native exp2 activation applies directly; c is the resident vector
    log2e and the 1.0 scalar the read-only CR1.

    When 128 divides n there is no tail chunk: the tail block still runs, with
    valid_elements = 0 (cr.tail), which makes its running AGG.MAX / AGG.SUM
    exact no-ops, so the same program covers that shape.

    Per-row scalars maxvec[r]/rvec[r] hold one value per row in a 128-element
    vector, so all four passes run once per GROUP of up to 128 rows: each group
    is processed completely (Pass 1..4) before the next, because maxvec/rvec are
    single shared vectors the next group overwrites. The group size is exact,
    not padded --
      rows_this_group = min(128, total_rows - rows_done)
    so the last group runs only the rows that remain, and every pass branches
    against lr_bound (this group's size) rather than the total. This keeps
    lr_row in 0..127 within a group, which matters beyond loop bookkeeping:
    lr_row is MULT.RC.VE's `src` scalar-select (Pass 4) and AGG's dest slot, and
    an index >= 128 would silently select R1 instead of R0. A row costs about
    48 cycles at n = 200 (bazel run //kernels/softmax/softmax_rows_long --
    --benchmark).

    CROSS-CHUNK REDUCTION (the novelty): a row's max (Pass 1) and sum (Pass 3)
    span cpr chunks. Each pass keeps a RUNNING per-row scalar: AGG.*.FIRST on the
    row's first full chunk, AGG.* (running, combines with R_ACC[row]) on the rest,
    and finally the tail chunk with valid_elements = tail. Full chunks name cr.ds_qf.L
    (valid_elements = 128); the tail chunk names cr.tail.L.

    Registers. All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The
    harness places the regions and sets the CRs from kernel.toml ([abi]):
      cr.x      x, the input logits         cr.y     y, the output
      cr.num    num, the numerators         cr.log2e C_VEC: log2(e) in every lane
      cr.max    maxvec row                  cr.rvec  rvec row (1 / sum)
      cr.row_stride        = 1 (one row per chunk)
      cr.tail              = the tail DS: valid_elements = n % 128
      cr.ds_qf             LOW: the full-chunk DS (valid_elements = 128); HIGH:
                             the case's QF (every MULT reads it; FP32 ignores it)
      cr.group_cap         = 128 (the group cap, and maxvec's index base in R1)
      cr.rows              = total rows (the group loop bound)
      cr.full_chunks       = n // 128 (the full-chunk loop bound)
      cr.cpr               = chunks per row
      cr.row_stride_chunks = cpr (XMEM rows per matrix row: the row stride)
      CR0 = 0 and CR1 = 1 are read-only; cr.ds_qf.L holds the full-chunk DS
      (128 valid elements, the default).

    LR uses 3 sub-slots; ";;" ends a instruction word, ";" separates sub-instructions.
    R_CYCLIC loads use index 0 (a whole row).
========================================================================== -#}

{%- set lr_off     = "LR0" -%}  {#- working chunk row offset (x / num / y) -#}
{%- set lr_row     = "LR1" -%}  {#- row index r = AGG dest slot / rvec element select -#}
{%- set lr_cyc     = "LR2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_rbase   = "LR3" -%}  {#- row base = r * cpr -#}
{%- set lr_max_idx = "LR4" -%}  {#- R1 element index 128+r = maxvec[r] -#}
{%- set lr_chunk   = "LR5" -%}  {#- inner chunk counter (0..full_chunks-1) -#}
{%- set lr_wr      = "LR6" -%}  {#- working write offset (NUM region, Pass 2) -#}
{%- set lr_done    = "LR7" -%}  {#- rows completed by previous groups -#}
{%- set lr_bound   = "LR8" -%}  {#- this group's exact row count = min(128, remaining) -#}
{%- set lr_gbase   = "LR9" -%}  {#- group base row = rows_done * cpr -#}
{%- set lr_total   = "LR10" -%} {#- total rows (SUB needs an LR src_a) -#}

    SET {{lr_cyc}}   CR0 ;
    SET {{lr_done}}  CR0 ;
    SET {{lr_gbase}} CR0 ;;
    SET {{lr_total}} {{ cr.rows }} ;;                           {#- SUB needs an LR src_a -#}

group_loop:
{#- ---- exact group size: lr_bound = min(128, total_rows - rows_done) ------ -#}
    SUB {{lr_bound}} {{lr_total}} {{lr_done}} ;;       {#- remaining = total - done (>=1) -#}
    BLT.S {{lr_bound}} {{ cr.group_cap }} group_size_set ;;             {#- remaining < 128 -> keep it ... -#}
    SET {{lr_bound}} {{ cr.group_cap }} ;;                            {#- ... else cap at 128 -#}
group_size_set:

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[r] = max over the row's N elements (cross-chunk).      -#}
{#- ===================================================================== -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;   {#- R_CYCLIC = C_VEC -#}
    SET {{lr_row}}   CR0 ;
    ADD {{lr_rbase}} {{lr_gbase}} CR0 ;;               {#- start at this group's base -#}

p1_row:
    {#- first full chunk: AGG.MAX.FIRST seeds R_ACC[row] -#}
    ADD {{lr_off}} {{lr_rbase}} CR0 ;;
    LD.MULT  R0 {{lr_off}} {{ cr.x }} ;;                {#- R0 = x[r,chunk0] (read at cycle start: visible next cycle) -#}
    MULT.RC.VV    {{lr_cyc}} R0 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;     {#- c*x -#}
    AGG.MAX.FIRST {{lr_row}} {{ cr.ds_qf }}.L ;;                    {#- R_ACC[r] = max chunk0 -#}
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    SET {{lr_chunk}} CR1 ;;                             {#- chunk = 1 (chunk0 done) -#}

p1_full:
    {#- remaining full chunks: running AGG.MAX into R_ACC[row] -#}
    BGE.S {{lr_chunk}} {{ cr.full_chunks }} p1_tail ;;                    {#- no more full chunks -> tail -#}
    LD.MULT R0 {{lr_off}} {{ cr.x }} ;;
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;
    AGG.MAX      {{lr_row}} {{ cr.ds_qf }}.L ;;                     {#- running max -#}
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    B p1_full ;;

p1_tail:
    {#- tail chunk: running AGG.MAX over the first `tail` elements (cr.tail) -#}
    LD.MULT R0 {{lr_off}} {{ cr.x }} ;;
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} {{ cr.tail }}.L {{ cr.ds_qf }}.H ;
    AGG.MAX      {{lr_row}} {{ cr.tail }}.L ;;                      {#- running max, valid_elements=tail -#}

    ADD {{lr_rbase}} {{lr_rbase}} {{ cr.row_stride_chunks }} ;               {#- next row base -#}
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} p1_row ;;

    ACTIVATE.EXACT identity {{ cr.ds_qf }}.L {{ cr.max }} {{lr_cyc}} ;;  {#- reads R_ACC forwarded -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,c] = 2^(c*x - maxvec[r]).  Writes NUM region.          -#}
{#-   maxvec[r] is the SAME for every chunk of row r (subtract via ACC.SUB).-#}
{#- ===================================================================== -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;   {#- R_CYCLIC = C_VEC -#}
    LD.MULT R1 {{lr_cyc}} {{ cr.max }} ;;                  {#- R1 = maxvec (elem r) -#}
    SET {{lr_row}}   CR0 ;
    ADD {{lr_rbase}} {{lr_gbase}} CR0 ;;
    SET {{lr_max_idx}} {{ cr.group_cap }} ;;                          {#- 128 + 0 = maxvec[0] -#}

p2_row:
    ADD {{lr_off}} {{lr_rbase}} CR0 ;
    ADD {{lr_wr}}  {{lr_rbase}} CR0 ;
    SET {{lr_chunk}} CR0 ;;

p2_chunk:
    BGE.S {{lr_chunk}} {{ cr.cpr }} p2_row_done ;;               {#- all cpr chunks done -#}
    LD.MULT R0 {{lr_off}} {{ cr.x }} ;;                 {#- R0 = x[r,c] (read at cycle start: visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;     {#- c*x -#}
    ACC.ADD.FIRST ;;
    MULT.EE {{lr_max_idx}} CR1 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;     {#- maxvec[r]*1.0 -#}
    ACC.SUB ;                                          {#- R_ACC = c*x - maxvec[r] -#}
    ACTIVATE.EXACT exp2 {{ cr.ds_qf }}.L {{ cr.num }} {{lr_wr}} ;;  {#- reads R_ACC forwarded -#}

    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_wr}}  {{lr_wr}}  {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    B p2_chunk ;;

p2_row_done:
    ADD {{lr_rbase}} {{lr_rbase}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;
    ADD {{lr_max_idx}} {{lr_max_idx}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} p2_row ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM over row's N num values; rvec = 1/sum.        -#}
{#- ===================================================================== -#}
    SET {{lr_row}}   CR0 ;
    ADD {{lr_rbase}} {{lr_gbase}} CR0 ;;

p3_row:
    ADD {{lr_off}} {{lr_rbase}} CR0 ;;
    {#- first full chunk: AGG.SUM.FIRST seeds R_ACC[row] -#}
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;   {#- R_CYCLIC = num[r,0] -#}
    MULT.RC.VE    {{lr_cyc}} CR1 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;   {#- num*1.0 -#}
    AGG.SUM.FIRST {{lr_row}} {{ cr.ds_qf }}.L ;;
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    SET {{lr_chunk}} CR1 ;;

p3_full:
    BGE.S {{lr_chunk}} {{ cr.full_chunks }} p3_tail ;;
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;
    MULT.RC.VE {{lr_cyc}} CR1 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;
    AGG.SUM    {{lr_row}} {{ cr.ds_qf }}.L ;;                       {#- running sum -#}
    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    B p3_full ;;

p3_tail:
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;
    MULT.RC.VE {{lr_cyc}} CR1 0 {{lr_cyc}} {{ cr.tail }}.L {{ cr.ds_qf }}.H ;
    AGG.SUM    {{lr_row}} {{ cr.tail }}.L ;;                        {#- running sum, valid_elements=tail -#}

    ADD {{lr_rbase}} {{lr_rbase}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} p3_row ;;

    ACTIVATE.EXACT reciprocal {{ cr.ds_qf }}.L {{ cr.rvec }} {{lr_cyc}} ;;  {#- reads R_ACC forwarded -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[r,c] = num[r,c] * rvec[r].  Writes OUTPUT region.         -#}
{#- ===================================================================== -#}
    LD.MULT R0 {{lr_cyc}} {{ cr.rvec }} ;;                  {#- R0 = rvec (elem r) -#}
    SET {{lr_row}}   CR0 ;
    ADD {{lr_rbase}} {{lr_gbase}} CR0 ;;

p4_row:
    ADD {{lr_off}} {{lr_rbase}} CR0 ;
    SET {{lr_chunk}} CR0 ;;

p4_chunk:
    BGE.S {{lr_chunk}} {{ cr.cpr }} p4_row_done ;;
    LD.MULT.CYCLIC {{lr_off}} {{ cr.num }} {{lr_cyc}} ;;   {#- R_CYCLIC = num[r,c] -#}
    MULT.RC.VE   {{lr_cyc}} {{lr_row}} 0 {{lr_cyc}} {{ cr.ds_qf }}.L {{ cr.ds_qf }}.H ;  {#- num*rvec[r] -#}
    ACC.ADD.FIRST ;
    ACTIVATE.EXACT identity {{ cr.ds_qf }}.L {{ cr.y }} {{lr_off}} ;;  {#- reads R_ACC forwarded -#}

    ADD {{lr_off}} {{lr_off}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    B p4_chunk ;;

p4_row_done:
    ADD {{lr_rbase}} {{lr_rbase}} {{ cr.row_stride_chunks }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} p4_row ;;

{#- ---- advance: rows_done += bound; loop while done < total --------------- -#}
{#- Pass 4 stepped lr_rbase once per row, so it now equals gbase + bound*cpr -#}
{#- -- exactly the next group's base. Copy it; the LR slot has no multiply.   -#}
    ADD {{lr_done}}  {{lr_done}} {{lr_bound}} ;
    ADD {{lr_gbase}} {{lr_rbase}} CR0 ;;
    BLT.S {{lr_done}} {{lr_total}} group_loop ;;

end:
    HALT ;;
