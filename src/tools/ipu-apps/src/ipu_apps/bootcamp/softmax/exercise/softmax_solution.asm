{#- =============================================================================
    EXERCISE SOLUTION -- softmax_solution.asm  (based on ../softmax.asm)
    =============================================================================
    Answer key for exercise/softmax_unstable_stub.asm: re-adds Pass 1 (the
    max reduction) and Pass 2's max-subtract step, restoring the exact
    numerically-stable 4-pass structure of ../softmax.asm (this file is
    logically identical to it). Do not read this until you've attempted
    softmax_unstable_stub.asm yourself -- see that file for the task
    description and self-check instructions.
    =============================================================================

    WHAT THIS PROGRAM DOES
    -----------------------
    Numerically-stable softmax over `rows` independent rows of up to 128 FP32
    elements each (rows <= 128 -- this drill deliberately stops at ONE tile;
    see "WHY <=128 ROWS ONLY" below):

        softmax(x)_j = exp(x_j - max(x)) / sum_k exp(x_k - max(x))

    WHY FP32 WIDE-VECTOR MODE (not INT8)
    ---------------------------------------
    Softmax needs a real exponential and a real division -- computing those
    accurately in INT8 requires a careful fixed-point scale/requantize scheme
    that is a whole topic on its own (see project memory on quantized
    softmax). This drill runs in the emulator's FP32 "wide-vector debug mode"
    instead (each lane is 4 bytes, holding an IEEE float), so the ISA
    concepts -- multi-pass reduction, AGG registers, ACTIVATE -- can be
    taught without also teaching INT8 requantization. The harness
    (__init__.py) builds this mode for you automatically.

    WHY BASE-2 (exp2), NOT BASE-e (exp)
    ---------------------------------------
    The hardware's activation table only offers exp2 (2^x), not exp (e^x).
    softmax is reformulated using the identity 2^(c*d) == e^d for c=log2(e):

        softmax(x)_j = 2^(c*(x_j - xmax)) / SUM_k 2^(c*(x_k - xmax))

    so multiplying every logit by the constant c = log2(e) up front lets
    ACTIVATE's exp2 stand in for e^x exactly. c is supplied as a resident
    128-lane FP32 vector C_VEC (a CR scalar can't hold a fraction).

    WHY <=128 ROWS ONLY (the one simplification vs. a "real" softmax app)
    -----------------------------------------------------------------------
    Passes 1 and 3 below produce one scalar per ROW (row max, row sum) and
    park them in R_ACC -- which is only 128 slots wide. A production softmax
    kernel loops over GROUPS of <=128 rows to handle arbitrarily many rows
    (see project memory: softmax_rows). This drill fixes `rows` at <=128 so
    there is exactly one group and no group loop -- the point here is the
    4-pass structure itself, not the chunking logic layered on top of it in
    a real kernel.

    THE FOUR PASSES
    -----------------
    Pass 1 (reduce):  maxvec[r]  = max_j (c * x[r,j])          -> staged XMEM
    Pass 2 (map):     num[r,j]   = 2^(c*x[r,j] - maxvec[r])    -> NUM region
    Pass 3 (reduce):  sumvec[r]  = SUM_j num[r,j]; rvec = 1/sumvec -> staged
    Pass 4 (map):     out[r,j]   = num[r,j] * rvec[r]          -> OUTPUT region

    Passes 1 and 3 are REDUCTIONS: many elements (one row) collapse to one
    number, using the AGG.MAX/AGG.SUM instructions, which write into ONE slot
    of R_ACC per row (that's the LR "dest_slot" operand) instead of every
    lane getting its own independent result like ACC.ADD does. Passes 2 and
    4 are ordinary elementwise maps (one output row per input row), like
    drills 1 and 2's ACC.ADD-based loops.

    NEW ISA SURFACE: AAQ (ACTIVATE) AND AGG
    -------------------------------------------
    - ACTIVATE.QUANTIZE reads r_acc LIVE (not snapshot -- an upstream fix;
      contrast with MULT reading R_CYCLIC from the snapshot) and writes into
      POST_AAQ_REG, which STR_POST_AAQ_REG then drains to XMEM. This is a
      different output path than STR_ACC_REG (drills 1-2's simulation-only
      accumulator dump) -- ACTIVATE+STR_POST_AAQ_REG is the real hardware
      path for producing final results.
    - AGG.MAX.FIRST / AGG.SUM.FIRST / AGG.MAX / AGG.SUM reduce ACROSS the 128
      lanes of one MULT_RES into a SINGLE R_ACC slot chosen by an LR register
      (`dest_slot`) -- this is different from ACC.ADD/ACC.ADD.FIRST (drill
      1-2), which write independently into all 128 R_ACC lanes.

    CR MAP (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      cr0  = 0  (zero source)         cr1  = 1  (-> 1.0 scalar, Pass 3 identity)
      cr2  = OUTPUT_BASE row          cr3  = CVEC_ADDR row     cr4 = NUM_BASE row
      cr5  = MAXVEC_ADDR row          cr6  = RVEC_ADDR row     cr7 = 1 (row step)
      cr9  = 128 (R1 byte-index base for maxvec element select)
      cr10 = INPUT_BASE row           cr13 = rows (<=128, exact -- no padding)

    LR REGISTERS:
      lr0 = working input/num/out row offset (steps by cr7=1 each element)
      lr1 = per-row-group row index r (also the AGG dest slot / element select)
      lr2 = cyclic index (always 0 -- wide mode fills the whole 512 B slot)
      lr3 = working write row offset for the NUM region
      lr4 = byte index 128+r into R1 = maxvec[r] (R1 is a combined R0++R1 space)
      lr7 = rows (bound for BLT, as an LR)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    r_cyclic loads use index 0 in wide mode (a load always fills the whole
    512-byte register in one shot).
============================================================================= -#}

{%- set lr_off      = "lr0" -%}  {#- working input/num/out row offset -#}
{%- set lr_row      = "lr1" -%}  {#- row index r (AGG dest slot / element select) -#}
{%- set lr_cyc      = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_wr       = "lr3" -%}  {#- working write row offset for the NUM region -#}
{%- set lr_max_idx  = "lr4" -%}  {#- byte index 128+r into R1 = maxvec[r] -#}
{%- set lr_rows     = "lr7" -%}  {#- rows, as an LR (BLT bound) -#}

    SET {{lr_cyc}} cr0;
    SET {{lr_rows}} cr13;;

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[r] = max_j (c * x[r, j]).  No output row written.     -#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}};;    {#- r_cyclic = C_VEC (log2(e), resident) -#}
    SET {{lr_off}} cr0;
    SET {{lr_row}} cr0;;

pass1_loop:
    LDR_MULT_REG  r0 {{lr_off}} cr10;;                  {#- R0 = x[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VV    {{lr_cyc}} r0 0 {{lr_cyc}} cr15;
    AGG.MAX.FIRST {{lr_row}} cr15;;                     {#- r_acc[r] = max lane of (c * x[r]) -#}
    ADD {{lr_off}} {{lr_off}} cr7;
    ADD {{lr_row}} {{lr_row}} cr1;;
    BLT {{lr_row}} {{lr_rows}} pass1_loop;;

    ACTIVATE.QUANTIZE identity cr15;                    {#- reads r_acc LIVE -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr5;;                   {#- MAXVEC_ADDR <- maxvec (one row, 128 lanes = 128 rows' worth of maxes) -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,j] = 2^(c*x[r,j] - maxvec[r]).  Writes the NUM region. -#}
{#-   step k:   R0=x[r]; R1=maxvec row (resident); mult_res=c*x[r]; ACC.ADD.FIRST -#}
{#-   step k+1: mult_res = maxvec[r] (x CR1=1.0 identity); ACC.SUB -> r_acc-maxvec[r]-#}
{#-   step k+2: ACTIVATE exp2 (reads r_acc LIVE); store num[r]              -#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}};;    {#- r_cyclic = C_VEC again -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr5;;                    {#- R1 = maxvec row (resident: all r's maxes, one per lane) -#}
    SET {{lr_off}} cr0;
    SET {{lr_wr}} cr0;
    SET {{lr_row}} cr0;;
    SET {{lr_max_idx}} cr9;;                            {#- lr_max_idx = 128 (R1[0] = maxvec[0], combined R0++R1 space) -#}

pass2_loop:
    LDR_MULT_REG r0 {{lr_off}} cr10;;                   {#- R0 = x[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15;
    ACC.ADD.FIRST;;                                     {#- r_acc = c*x[r] -#}

    MULT.EE {{lr_max_idx}} cr1 0 {{lr_cyc}} cr15;       {#- mult_res = maxvec[r] * 1.0, broadcast -#}
    ACC.SUB;                                            {#- r_acc = c*x[r] - maxvec[r] -#}
    ACTIVATE.QUANTIZE exp2 cr15;                        {#- reads r_acc LIVE: r_acc <- unmodified, post_aaq_reg <- 2^(...) -#}
    STR_POST_AAQ_REG {{lr_wr}} cr4;;                    {#- NUM[r] = 2^(c*x[r] - maxvec[r]) -#}

    ADD {{lr_off}} {{lr_off}} cr7;
    ADD {{lr_wr}} {{lr_wr}} cr7;
    ADD {{lr_row}} {{lr_row}} cr1;;
    ADD {{lr_max_idx}} {{lr_max_idx}} cr1;;             {#- next row's maxvec element -#}
    BLT {{lr_row}} {{lr_rows}} pass2_loop;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM_j num[r,j]; rvec[r] = 1 / sumvec[r].         -#}
{#-   r_cyclic = num[r]; MULT.RC.VE x CR1(=1.0) identity; AGG.SUM.FIRST    -#}
{#- ===================================================================== -#}
    SET {{lr_off}} cr0;
    SET {{lr_row}} cr0;;

pass3_loop:
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}};;    {#- r_cyclic = num[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} cr1 0 {{lr_cyc}} cr15;
    AGG.SUM.FIRST {{lr_row}} cr15;;                     {#- r_acc[r] = SUM_j num[r,j] -#}
    ADD {{lr_off}} {{lr_off}} cr7;
    ADD {{lr_row}} {{lr_row}} cr1;;
    BLT {{lr_row}} {{lr_rows}} pass3_loop;;

    ACTIVATE.QUANTIZE reciprocal cr15;                  {#- reads r_acc LIVE: post_aaq_reg <- 1/sumvec -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6;;                   {#- RVEC_ADDR <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[r,j] = num[r,j] * rvec[r].  Writes the OUTPUT region.    -#}
{#- ===================================================================== -#}
    LDR_MULT_REG r0 {{lr_cyc}} cr6;;                    {#- R0 = rvec row (resident: element r = rvec[r]) -#}
    SET {{lr_off}} cr0;
    SET {{lr_row}} cr0;;

pass4_loop:
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}};;    {#- r_cyclic = num[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE   {{lr_cyc}} {{lr_row}} 0 {{lr_cyc}} cr15;   {#- mult_res = num[r] * rvec[r] (LR selects R0[r]) -#}
    ACC.ADD.FIRST;
    ACTIVATE.QUANTIZE identity cr15;                    {#- reads r_acc LIVE -#}
    STR_POST_AAQ_REG {{lr_off}} cr2;;                   {#- OUT[r] = softmax row -#}

    ADD {{lr_off}} {{lr_off}} cr7;
    ADD {{lr_row}} {{lr_row}} cr1;;
    BLT {{lr_row}} {{lr_rows}} pass4_loop;;

end:
    BKPT;;
