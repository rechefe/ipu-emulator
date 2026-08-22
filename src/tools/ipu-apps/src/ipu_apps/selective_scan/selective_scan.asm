{#- ==========================================================================
    selective_scan.asm -- Mamba / MambaVision selective scan, FP32 wide-vector

      delta_t = softplus(dt_t + delta_bias)     <- ON DEVICE (native activation)
      Abar[n] = exp2(delta_t * cA[n])            cA = log2(e) * A, see below
      h[n]    = Abar[n] * h[n] + B[n,t] * (delta_t * u_t)
      y_t     = SUM_n C[n,t] * h[n] + D_skip * u_t

    Everything that depends on the INPUT is computed here. The one host-side
    transform, cA = log2(e)*A, is a constant refold of a WEIGHT: A itself is
    derived from the checkpoint's stored A_log as A = -exp(A_log), so
    cA = -log2(e)*exp(A_log) is just a different constant folding of the same
    parameter -- computed once per model, never per inference. It buys the
    single native exp2 in place of exp(delta*A), which has no native form.

    STATE-MAJOR LAYOUT: lane = channel d, one 128-lane row per state index n.
    This is forced, not chosen -- multiply masks are DISABLED in wide mode
    (docs/content/wide-vector-debug-mode.md), so the state-minor alternative
    (lane = d*N+n) cannot do its segmented SUM_n reduction at all. Here SUM_n
    is just N vector-x-scalar multiplies accumulating in R_ACC: no reduction
    instruction, no masks, no partitions.

    B[n,t] / C[n,t] are channel-independent scalars. One BC row per timestep
    holds B at elements 0..N-1 and C at elements N..2N-1; both are broadcast
    with MULT.RC.VE using an LR-selected element of R1 (LR value 128+k selects
    R1[k]). A CR cannot carry them -- CR scalars are integer low-byte only in
    wide mode.

    INSTANCES: the kernel walks a flat list of (batch, channel-tile) pairs. For
    instance i every region is addressed uniformly:
        U / DT / BC / Y     ->  row i*L + t        (lr_row, monotonic)
        CA / H / ABAR       ->  row i*N + n        (lr_ld / lr_st)
        DSKIP               ->  row i              (lr_i)
        SCRATCH             ->  row 0 = du_t (lr_z), row 1 = delta_t (lr_one)
    H is per-instance, so each instance starts on zero-initialised XMEM and
    h[.,.,-1] = 0 needs no zeroing prologue.

    R_CYCLIC SLOTS -- all four are live at once, which is what keeps the load
    count down. LDR_CYCLIC_MULT_REG accepts index 0/128/256/384 in BOTH modes
    (ipu.py execute_ldr_cyclic_mult_reg); the "index must be 0 in wide mode"
    claims in wide-vector-debug-mode.md and softmax_rows.asm are stale -- the
    softmax kernels simply never needed more than slot 0. See STATUS.md.
        slot 0   = du_t = delta_t * u_t   (once per timestep)
        slot 128 = dt_t, then delta_t     (once per timestep)
        slot 256 = D_skip, then h[n]      (output loop)
        slot 384 = Abar[n]                (state loop)

    ------------------------------------------------------------------------
    SOFTWARE PIPELINING -- why the loops look "off by one"
    ------------------------------------------------------------------------
    Slot order in a word is LR -> load -> mult -> acc -> aaq -> store -> cond
    (ipu.py execute_vliw_cycle), and:

      * MULT's Ra (R0/R1) and R_CYCLIC are read from the START-OF-CYCLE
        SNAPSHOT (_debug_ra_lane_vals / _debug_rb_lane_vals). A load in the
        SAME word therefore does NOT disturb that word's multiply -- it stages
        the operand for the NEXT iteration. That is what lets one R0 and one
        r_cyclic slot serve a whole loop with no copies.
      * LR sub-slots run BEFORE load/mult/store, and load offsets, MULT.RC.VE's
        LR scalar-select and store offsets are all read LIVE. So an increment
        placed in a word takes effect for THAT word's load/store/select.

      * BLT's OPERANDS ARE SNAPSHOT (instruction_spec.py marks both reg1 and
        reg2 "read": "snapshot"), unlike everything else in a word. So an ADD
        on the counter in the SAME word as its BLT is INVISIBLE to that BLT --
        the branch compares the PRE-increment value and the loop runs one extra
        time. Both single-word loops here (abar_loop, out_loop) and the
        timestep epilogue have exactly that shape, so their counters are SEEDED
        TO 1 instead of 0. state_loop needs no seed: it is two words, and its
        ADD sits in the first while its BLT sits in the second, so the branch
        already sees the incremented value.

        The extra pass was not merely wasted, it was load-bearing on an
        invariant. out_loop's spare iteration accumulated h[row i*N+N] scaled
        by R1[128+2N] into y_t, and abar_loop's spare iteration STORED an
        Abar to row i*N+N -- one row past this instance's region, i.e. the
        FIRST row of instance i+1. Both were harmless only because instances
        run in ascending order, so instance i+1's H rows are still the zeroed
        XMEM they started as when instance i reads them: the accumulate was
        always (zero vector) * scalar. Note the scalar itself is out of range
        too -- at N=64 lr_csel reaches 256, which wraps mod 2*128 to R0[0] =
        u_t[0] rather than BC-row padding -- so below N=64 the term was doubly
        dead and at N=64 singly. Seeding to 1 deletes both passes, which drops
        the reliance on instance ordering and saves 2 cycles per timestep.

    Both facts together give the loop shape used below: each iteration issues
    the load for n+1 while multiplying the operands loaded for n, and the
    pointer registers are pre-biased so that after the in-word increment they
    name the right rows:

        lr_ld enters iteration n holding n    -> increments to n+1 -> LOADS n+1
        lr_st enters iteration n holding n-1  -> increments to n   -> STORES n
        lr_bsel / lr_csel are biased by -1 the same way

    lr_st starts at (i*N - 1), which for i=0 wraps to 0xFFFFFFFF; the first
    in-word increment wraps it back to 0. LR arithmetic is uint32 (execute_lr_add
    masks with 0xFFFFFFFF), so this is exact, not luck.

    Each loop's last iteration issues one load past the end of its region
    (row i*N + N). The value is never consumed, and every region is 64 KiB
    aligned with slack, so the address is always in bounds.

    Measured cost per timestep per 128-channel tile: 15 + 4N cycles -- 47 at
    N=8, 79 at N=16 (benchmark/results.md). The naive unpipelined version this
    replaced cost 12 + 12N = 108 at N=8, at 31% mult utilisation.

    The floor is the LOAD slot: one load per word, and a timestep issues
    4N + 10 loads (dt, BC, u_t, the delta_t and du_t reloads, three loop primes,
    a second u_t for the D_skip term, plus 4 per n) = 42 at N=8. The remaining
    5 cycles are the three loop prologues, which have no load to hide behind --
    the timestep epilogue now has no cycles of its own at all.

    The on-device softplus costs exactly 1 of those 15: it rides in the u_t load
    word and adds only the delta_t reload word.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0  (zero source)        CR1  = 1  (row stride / counter step)
      CR2  = U_BASE                  CR3  = DT_BASE  (dt + delta_bias)
      CR4  = CA_BASE                 CR5  = BC_BASE
      CR6  = DSKIP_BASE              CR7  = H_BASE
      CR8  = ABAR_BASE               CR9  = Y_BASE
      CR10 = SCRATCH_BASE            CR11 = N   (d_state)
      CR12 = L   (seq_len)           CR13 = NUM_INSTANCES
      CR14 = 128 (lane count; also the R1 element-select base)
      CR15 = dstructure (valid_elements = 128)

    All shape parameters arrive in CRs, never as Jinja constants: the assembler
    renders templates with no variables (lark_tree.py), so Jinja here is only
    for readable register aliases.

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
========================================================================== -#}

{%- set lr_i     = "lr0"  -%}  {#- instance index i (also the DSKIP row)          -#}
{%- set lr_t     = "lr1"  -%}  {#- timestep t within the instance                 -#}
{%- set lr_n     = "lr2"  -%}  {#- state index n within the timestep              -#}
{%- set lr_z     = "lr3"  -%}  {#- constant 0: rc slot 0, mask_shift, scratch row -#}
{%- set lr_s1    = "lr4"  -%}  {#- constant 128: rc slot 128                      -#}
{%- set lr_s2    = "lr5"  -%}  {#- constant 256: rc slot 256                      -#}
{%- set lr_s3    = "lr6"  -%}  {#- constant 384: rc slot 384                      -#}
{%- set lr_row   = "lr7"  -%}  {#- i*L + t -- U/DELTA/BC/Y row, monotonic         -#}
{%- set lr_ld    = "lr8"  -%}  {#- CA/H/ABAR LOAD row, runs one ahead of lr_st    -#}
{%- set lr_bsel  = "lr9"  -%}  {#- 128 + n     -> R1[n]   = B[n,t]                -#}
{%- set lr_csel  = "lr10" -%}  {#- 128 + N + n -> R1[N+n] = C[n,t]                -#}
{%- set lr_cm1   = "lr11" -%}  {#- 127 + N -- pre-biased reset value for lr_csel  -#}
{%- set lr_ib    = "lr12" -%}  {#- i*N -- instance base for lr_ld / lr_st         -#}
{%- set lr_st    = "lr13" -%}  {#- CA/H/ABAR STORE row (the current n)            -#}
{%- set lr_one   = "lr14" -%}  {#- constant 1 (SUB src_b; scratch row for delta)  -#}
{%- set lr_bm1   = "lr15" -%}  {#- 127 -- pre-biased reset value for lr_bsel      -#}

{#- ---- init: constants and monotonic counters ---------------------------- -#}
    SET {{lr_z}}    cr0 ;
    SET {{lr_i}}    cr0 ;
    SET {{lr_row}}  cr0 ;;
    SET {{lr_ib}}   cr0 ;
    SET {{lr_one}}  cr1 ;
    SET {{lr_s1}}   cr14 ;;
    ADD {{lr_s2}}   {{lr_s1}} cr14 ;                     {#- 256 -#}
    SET {{lr_bm1}}  cr14 ;
    SUB {{lr_row}}  {{lr_row}} {{lr_one}} ;;             {#- -1: t_loop pre-increments -#}
    ADD {{lr_s3}}   {{lr_s2}} cr14 ;                     {#- 384 -#}
    SUB {{lr_bm1}}  {{lr_bm1}} {{lr_one}} ;;             {#- 127 -#}
    ADD {{lr_cm1}}  {{lr_bm1}} cr11 ;;                   {#- 127 + N -#}

instance_loop:
    SET {{lr_t}} cr1 ;;                                  {#- 1, not 0: see BLT rule -#}

{#- ===================================================================== -#}
{#- TIMESTEP t -- softplus dt_t into delta_t, then du_t = delta_t * u_t.   -#}
{#- The word that computes du_t also primes the Abar loop's pointers, and  -#}
{#- the word after it primes R0 with cA[0]: LR and load slots are free     -#}
{#- there, so the prologue costs no extra cycles.                          -#}
{#- ===================================================================== -#}
t_loop:
    ADD {{lr_row}} {{lr_row}} cr1 ;                      {#- live: this word's load sees it -#}
    LDR_CYCLIC_MULT_REG {{lr_row}} cr3 {{lr_s1}} ;;      {#- slot128 = dtb_t (raw) -#}
    LDR_MULT_REG r1 {{lr_row}} cr5 ;;                    {#- R1 = BC row t (B | C) -#}

{#- delta_t = softplus(dtb_t). The x1.0 is a CR scalar: in FP32 wide mode a CR -#}
{#- src to MULT.RC.VE contributes its low byte as a float, so cr1 is exactly   -#}
{#- 1.0 (ipu.py _mult_resolve_lcr_scalar_wide). Costs no load and no LR, and   -#}
{#- rides in the same word as the u_t load -- +1 cycle per timestep in total.  -#}
    LDR_MULT_REG r0 {{lr_row}} cr2 ;                     {#- R0 = u_t -#}
    MULT.RC.VE {{lr_s1}} cr1 0 {{lr_z}} cr15 ;           {#- dtb_t * 1.0 -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE softplus cr15 ;
    STR_POST_AAQ_REG {{lr_one}} cr10 ;;                  {#- DU row 1 <- delta_t -#}

    LDR_CYCLIC_MULT_REG {{lr_one}} cr10 {{lr_s1}} ;;     {#- slot128 = delta_t -#}

    MULT.RC.VV {{lr_s1}} r0 0 {{lr_z}} cr15 ;            {#- delta_t * u_t -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_z}} cr10 ;                     {#- DU row 0 <- du_t -#}
    ADD {{lr_ld}} {{lr_ib}} cr0 ;
    SUB {{lr_st}} {{lr_ib}} {{lr_one}} ;
    SET {{lr_n}} cr1 ;;                                  {#- 1: BLT sees the snapshot -#}

    LDR_CYCLIC_MULT_REG {{lr_z}} cr10 {{lr_z}} ;;        {#- slot0 = du_t -#}
    LDR_MULT_REG r0 {{lr_ld}} cr4 ;;                     {#- prime R0 = cA[0] -#}

{#- ---- Abar loop: 1 word per n ------------------------------------------ -#}
{#-   loads cA[n+1] while multiplying the cA[n] already in R0 (snapshot).   -#}
abar_loop:
    LDR_MULT_REG r0 {{lr_ld}} cr4 ;                      {#- R0 <- cA[n+1] -#}
    MULT.RC.VV {{lr_s1}} r0 0 {{lr_z}} cr15 ;            {#- delta_t * cA[n] -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE exp2 cr15 ;                        {#- 2^(delta*cA) == e^(delta*A) -#}
    STR_POST_AAQ_REG {{lr_st}} cr8 ;                     {#- ABAR[n] <- Abar -#}
    ADD {{lr_ld}} {{lr_ld}} cr1 ;
    ADD {{lr_st}} {{lr_st}} cr1 ;
    ADD {{lr_n}}  {{lr_n}}  cr1 ;
    BLT {{lr_n}} cr11 abar_loop ;;

{#- ---- state loop: h[n] <- Abar[n]*h[n] + B[n,t]*du_t, 2 words per n ----- -#}
    ADD {{lr_ld}} {{lr_ib}} cr0 ;
    SUB {{lr_st}} {{lr_ib}} {{lr_one}} ;
    SET {{lr_n}} cr0 ;;
    LDR_CYCLIC_MULT_REG {{lr_ld}} cr8 {{lr_s3}} ;        {#- prime slot384 = Abar[0] -#}
    ADD {{lr_bsel}} {{lr_bm1}} cr0 ;;                    {#- lr_bsel = 127 (pre-biased) -#}
    LDR_MULT_REG r0 {{lr_ld}} cr7 ;;                     {#- prime R0 = h[0] -#}

state_loop:
    LDR_CYCLIC_MULT_REG {{lr_ld}} cr8 {{lr_s3}} ;        {#- slot384 <- Abar[n+1] -#}
    MULT.RC.VV {{lr_s3}} r0 0 {{lr_z}} cr15 ;            {#- Abar[n] * h[n] (both snapshot) -#}
    ACC.ADD.FIRST ;
    ADD {{lr_ld}} {{lr_ld}} cr1 ;
    ADD {{lr_st}} {{lr_st}} cr1 ;
    ADD {{lr_n}}  {{lr_n}}  cr1 ;;

    LDR_MULT_REG r0 {{lr_ld}} cr7 ;                      {#- R0 <- h[n+1] -#}
    MULT.RC.VE {{lr_z}} {{lr_bsel}} 0 {{lr_z}} cr15 ;    {#- du_t * B[n,t] -#}
    ACC.ADD ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_st}} cr7 ;                     {#- H[n] <- new state -#}
    ADD {{lr_bsel}} {{lr_bsel}} cr1 ;
    BLT {{lr_n}} cr11 state_loop ;;

{#- ---- output loop: y_t = D_skip*u_t + SUM_n C[n,t]*h[n], 1 word per n --- -#}
{#-   The D_skip term seeds R_ACC with ACC.ADD.FIRST before the loop, so    -#}
{#-   every n uses a plain ACC.ADD -- no peeled iteration, no branch.       -#}
    LDR_CYCLIC_MULT_REG {{lr_i}} cr6 {{lr_s2}} ;;        {#- slot256 = D_skip -#}
    LDR_MULT_REG r0 {{lr_row}} cr2 ;;                    {#- R0 = u_t -#}

    MULT.RC.VV {{lr_s2}} r0 0 {{lr_z}} cr15 ;            {#- D_skip * u_t -#}
    ACC.ADD.FIRST ;
    ADD {{lr_ld}} {{lr_ib}} cr0 ;
    SET {{lr_n}} cr1 ;                                   {#- 1: BLT sees the snapshot -#}
    ADD {{lr_csel}} {{lr_cm1}} cr0 ;;                    {#- lr_csel = 127 + N (pre-biased) -#}

    LDR_CYCLIC_MULT_REG {{lr_ld}} cr7 {{lr_s2}} ;;       {#- prime slot256 = h[0] -#}

out_loop:
    LDR_CYCLIC_MULT_REG {{lr_ld}} cr7 {{lr_s2}} ;        {#- slot256 <- h[n+1] -#}
    MULT.RC.VE {{lr_s2}} {{lr_csel}} 0 {{lr_z}} cr15 ;   {#- h[n] * C[n,t] -#}
    ACC.ADD ;
    ADD {{lr_ld}}   {{lr_ld}}   cr1 ;
    ADD {{lr_csel}} {{lr_csel}} cr1 ;
    ADD {{lr_n}}    {{lr_n}}    cr1 ;
    BLT {{lr_n}} cr11 out_loop ;;

{#- ---- store y_t and close the timestep, all in ONE word ---------------- -#}
{#-   The store reads lr_row LIVE and nothing here touches it; the branch    -#}
{#-   reads lr_t from the SNAPSHOT, so the ADD beside it is invisible to it  -#}
{#-   -- which is exactly why lr_t was seeded to 1 instead of 0.             -#}
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_row}} cr9 ;                    {#- Y[i*L+t] <- y_t -#}
    ADD {{lr_t}} {{lr_t}} cr1 ;
    BLT {{lr_t}} cr12 t_loop ;;

{#- ---- advance instance ------------------------------------------------- -#}
{#-   This one CANNOT collapse the way the timestep epilogue did: lr_i is    -#}
{#-   not just a counter, it is also the DSKIP row, so it cannot be seeded   -#}
{#-   to 1, and no CR is free to hold NUM_INSTANCES-1 (all 13 writable CRs   -#}
{#-   are in use). It costs 2 words per INSTANCE, not per timestep.          -#}
    ADD {{lr_ib}} {{lr_ib}} cr11 ;
    ADD {{lr_i}}  {{lr_i}}  cr1 ;;
    BLT {{lr_i}} cr13 instance_loop ;;

end:
    BKPT ;;
