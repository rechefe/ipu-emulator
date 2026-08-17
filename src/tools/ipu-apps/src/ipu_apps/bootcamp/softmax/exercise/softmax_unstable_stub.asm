{#- =============================================================================
    EXERCISE -- softmax_unstable_stub.asm  (based on ../softmax.asm)
    =============================================================================

    YOUR TASK
    ---------
    This is a copy of softmax.asm with the numerical-stability machinery
    STUBBED OUT:

      - Pass 1 (max reduction: maxvec[r] = max_j (c * x[r,j])) is REMOVED.
      - Pass 2's max-subtract step (r_acc = c*x[r] - maxvec[r]) is REMOVED --
        Pass 2 now feeds c*x[r,j] DIRECTLY into ACTIVATE's exp2, computing
        2^(c*x[r,j]) instead of 2^(c*x[r,j] - maxvec[r]).

    Everything else (Pass 3's sum-and-reciprocal, Pass 4's normalize) is
    unchanged and still correct GIVEN Pass 2's (now unstable) numerators.

    Mathematically, softmax(x) = softmax(x - max(x)) for any constant shift,
    so this "unstable" version computes the exact same probabilities *in
    infinite precision*. But 2^(c*x_j) can overflow FP32 (or underflow to 0)
    for even moderately large |x_j|, while 2^(c*(x_j - xmax)) never exceeds
    2^0 = 1 by construction. Try feeding this program logits with
    scale=50 (see the self-check test) and see the output diverge from a
    numpy softmax reference -- that's the bug you're meant to fix.

    YOUR JOB: re-add the numerical-stability machinery so this program
    matches ../softmax.asm exactly:
      1. Re-add Pass 1: loop over rows computing maxvec[r] = max_j(c*x[r,j])
         via AGG.MAX.FIRST, then ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
         to stage maxvec to XMEM (see ../softmax.asm's Pass 1 for the exact
         instructions -- MULT.RC.VV + AGG.MAX.FIRST per row, cr15 valid_elements).
      2. Re-add Pass 2's subtract: after computing r_acc = c*x[r] (ACC.ADD.FIRST),
         broadcast maxvec[r] via MULT.EE and ACC.SUB it out of r_acc BEFORE
         the ACTIVATE.QUANTIZE exp2 call.

    Find the "# TODO" markers below.

    HOW TO CHECK YOUR WORK
    -----------------------
      PYTHONPATH=... python -m pytest \
        src/tools/ipu-apps/test/test_bootcamp_softmax_exercise.py -v
    test_softmax_solution_matches_numpy exercises the answer key
    (exercise/softmax_solution.asm, logically identical to ../softmax.asm)
    against a numpy softmax reference, including a large-magnitude
    (scale=50) case. test_stub_diverges_on_large_logits documents the
    failure mode of the unfixed stub on that same large-magnitude case, so
    you can see exactly what "broken" looks like before and after your fix.

    See ../softmax.asm for the full CR/LR register map and the 4-pass
    walkthrough -- it is not repeated in full here.
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
{#- PASS 1 -- STUBBED OUT. The real softmax.asm computes                  -#}
{#-   maxvec[r] = max_j (c * x[r, j])  via AGG.MAX.FIRST, staged to XMEM. -#}
{#-   As shipped, this stub SKIPS that pass entirely -- no maxvec is ever -#}
{#-   computed or written to MAXVEC_ADDR (cr5). Pass 2 below will end up  -#}
{#-   reading garbage/zero from that address if you try to subtract it   -#}
{#-   without first fixing this pass.                                    -#}
{#- ===================================================================== -#}
    {#- TODO: re-add Pass 1 here. See ../softmax.asm's Pass 1 section:
        - LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}};; (r_cyclic = C_VEC)
        - SET {{lr_off}} cr0; SET {{lr_row}} cr0;;
        - a pass1_loop: LDR_MULT_REG r0 {{lr_off}} cr10;; MULT.RC.VV {{lr_cyc}} r0 0 {{lr_cyc}} cr15;
          AGG.MAX.FIRST {{lr_row}} cr15;; then bump lr_off/lr_row and BLT back to pass1_loop
        - ACTIVATE.QUANTIZE identity cr15; STR_POST_AAQ_REG {{lr_cyc}} cr5;;
          to stage maxvec to XMEM (MAXVEC_ADDR = cr5)
    -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[r,j] = 2^(c*x[r,j]).  Writes the NUM region.            -#}
{#-   UNSTABLE: no max-subtract before the exp2 (that's the point of this -#}
{#-   exercise -- see the TODO below for what's missing).                 -#}
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

    {#- TODO: this is where the numerical-stability subtract belongs. The
        real softmax.asm does, right here (see its Pass 2):
          MULT.EE {{lr_max_idx}} cr1 0 {{lr_cyc}} cr15;   (mult_res = maxvec[r] * 1.0, broadcast)
          ACC.SUB;                                        (r_acc = c*x[r] - maxvec[r])
        As shipped, this stub skips straight to ACTIVATE.QUANTIZE exp2 below
        on the RAW r_acc = c*x[r] -- i.e. it computes 2^(c*x[r,j]) instead of
        2^(c*x[r,j] - maxvec[r]). That's numerically unstable for large |x|.
    -#}
    ACTIVATE.QUANTIZE exp2 cr15;                        {#- reads r_acc LIVE: r_acc <- unmodified, post_aaq_reg <- 2^(...) -#}
    STR_POST_AAQ_REG {{lr_wr}} cr4;;                    {#- NUM[r] = 2^(c*x[r]) -- UNSTABLE, see TODO above -#}

    ADD {{lr_off}} {{lr_off}} cr7;
    ADD {{lr_wr}} {{lr_wr}} cr7;
    ADD {{lr_row}} {{lr_row}} cr1;;
    ADD {{lr_max_idx}} {{lr_max_idx}} cr1;;             {#- next row's maxvec element -#}
    BLT {{lr_row}} {{lr_rows}} pass2_loop;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM_j num[r,j]; rvec[r] = 1 / sumvec[r].         -#}
{#-   Unchanged from softmax.asm -- correct given whatever Pass 2 wrote.  -#}
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
{#-   Unchanged from softmax.asm.                                         -#}
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
