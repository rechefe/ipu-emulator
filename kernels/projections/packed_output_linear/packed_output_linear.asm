{#- ==========================================================================
    packed_output_linear.asm -- one packed output row of a projection,
    y = activation(a @ b), in the MobileViT packed layout at tokens = 16, FP32.
    The template of packed_output_linear{,_silu}_240 (../registry.toml,
    E07-23): the activation selects the final ACTIVATE.EXACT (none or
    silu); E02 digests at k = 240 and 960. chunk_widths is computed here from
    k: 128 a weight-chunk, the last one k % 128.

    ABI (kernel.toml's [abi]; cr.<name> below):
      b             the (k, 16) data as its packed rows, 8 channels a row
      a             the call's 8 output-major weight rows per weight-chunk
                    (pack weight_chunks, ../pack.py)
      y             the one packed output row: y[o, t] at lane 16 * o + t
      mask          one R_MASK row: slot p enables output p's window
      dstruct_mult  the MULT's DS in .L: ds(16, 8, 8), partition P8
      ds_qf         the store's DS in .L (ds(128), every lane) and the case's
                    QF in .H (no CR is free for a QF of its own)
      seed0..seed7  CR8..CR15: R_CYCLIC index seeds, (512 - 16 (p + 1)) % 512
    The program sets every LR it reads.

    N_OUT is fixed at 8 per call: with N_OUT=8 the
    8 R_MASK slots map 1:1 onto p_out, which is the alignment the construction
    depends on (the 8 R_MASK slots map 1:1 onto the 8 p_out values -- an
    alignment that holds only at P8, which is L5). For N_OUT>8
    (out-proj=240, FFN1=960, QKV=720) this kernel is called N_OUT/8 times by
    its caller, each call producing ONE packed row of 8 output channels --
    still zero host-side data conversion: each call's output is a complete,
    valid packed row on its own; nothing converts data BETWEEN calls, only the
    caller chooses which 8-channel weight slice (a) and output row (y) a call
    takes, through their [abi] bases.

      OUT[o, t] = sum_k W[o, k] * X[k, t]  for o = 0..7 (this call's 8 outputs), k = 0..K-1

      rc_idx = 16*(p_in - p_out) mod 512

    Prerequisites (R_CYCLIC replicated into all 4 slots per packed row;
    R_MASK's 8 slots pre-built with mask_offset=p_out => bits
    [16p_out,16p_out+16) set, loaded once via LD.MASK). Same
    IPC-fix-style pre-increment bias for the in-partition accumulation
    (rc_idx_reg/k_idx seeded one step behind so the co-issued advance ADDs are
    visible to the SAME word's MULT).

    RUNTIME ROW LOOP (not fully unrolled): a first cut fully unrolled the
    packed-row loop -- at K=240 (30 packed rows) that produced 3130+ static
    instructions, blowing the 1024-instruction program-memory ceiling ~3x
    over. The 8x8=64 MULT+ACC-per-row body (unavoidable: this construction
    visits each row once per OUTPUT partition, not once total, to land all 8
    outputs packed -- same 1920 total MULT+ACC ops as path(b)/unpacked at
    240->8, just organized differently) is small and stays unrolled per row;
    what cannot be unrolled at real K is the outer row-count loop itself.

    PEELED FIRST ROW: ACC.ADD.FIRST must fire exactly once for the WHOLE
    kernel run (row 0, p_out=0, p_in=0), not once per row -- a runtime
    loop body re-executes the same static instructions every iteration, so a
    compile-time "p_out==0 and p_in==0" guard inside the loop body would
    incorrectly re-fire .FIRST (and reset the accumulator) on EVERY row.
    Fixed the way an earlier unpacked linear program handled it ("row 0:
    prime + peel"): weight-chunk 0's packed row 0 is statically unrolled
    ONCE outside any loop with .FIRST at its one true first position; TWO
    separate runtime loops then cover everything else -- weight-chunk 0's
    REMAINING packed rows (pc=1..bound0, only entered if bound0>0), then
    weight-chunks 1..W_CHUNKS-1 in full (pc=0..bound_i each). This avoids the
    awkward "which bound applies, and did we already consume row 0"
    branching a single merged loop would need.

    SEED LOOKUP TABLE (8 CRs, one per p_out) for rc_idx: kernel.toml [derived]
    constants, bound in [abi].

    Weight layout: 8 rows (one per p_out in this call), each split across
    ceil(K/128) weight-chunks -- same convention as every other linear kernel
    here. K assumed a multiple of 8 (packing group size); the L5 layer's
    shapes (240, 960) both qualify.

    CROSS-KERNEL R_MASK HAZARD: this kernel loads a one-hot 8-slot R_MASK
    (one 16-lane window per output partition) via LD.MASK and never
    restores an all-ones R_MASK before halting. A downstream kernel in the
    same IpuState that relies on R_MASK's all-ones default (e.g.
    residual_add_rows.asm (packed_residual_add_240x16)) will silently read back zeros in
    partitions 1-7 unless the caller explicitly reloads R_MASK first --
    encountered directly when chaining this kernel into residual add for the
    full-L5-layer test.
========================================================================== -#}
{%- set chunk_widths = [128] * (k // 128) + ([k % 128] if k % 128 else []) %}

{% set rc_idx_reg   = "LR0" %}
{% set w_ptr        = "LR1" %}
{% set out_ptr      = "LR2" %}
{% set k_idx        = "LR3" %}
{% set mask_off_lr  = "LR4" %}
{% set slot_lr      = "LR5" %}
{% set chunk_base   = "LR6" %}
{% set chunk_ctr    = "LR7" %}   {# running packed-row counter across the WHOLE kernel, drives chunk_base #}
{% set pc_idx       = "LR8" %}   {# packed-row index within the current weight-chunk's runtime loop #}
{% set pc_bound     = "LR9" %}   {# this weight-chunk's pc_idx upper bound (inclusive) #}
{% set wc_idx       = "LR10" %}  {# weight-chunk index, runtime counter #}
{% set wc_bound     = "LR12" %}  {# W_CHUNKS (chunk_widths|length), a compile-time constant loaded into an LR via INC immediates since no CR slot is free for it #}
{% set k_base       = "LR11" %}  {# 8*pc_idx: this packed row's channel offset WITHIN the current weight-chunk's R0 row -- k_idx = k_base + p_in, NOT just p_in, whenever a weight-chunk spans >1 packed row (K>8) #}
{% set w_row_ptr    = "LR13" %}  {# walks the current weight-chunk's 8 per-p_out rows, reset to w_ptr at the start of every packed row #}

{#- CR0/CR1 are HARDWIRED read-only (0 and 1 respectively) -- confirmed via
    state.regfile.get_cr(1) returning 1 even after set_cr(1, 0). An earlier
    draft assigned DATA_BASE=CR1 assuming it was a normal writable CR; the
    host's set_cr(1, data_base_row) silently had NO EFFECT, leaving
    chunk_base permanently seeded at 1 instead of the real data base row --
    this produced the multi-row correctness failures (K=16 err=4.09,
    K=240 err=25.09/18.37) that looked like addressing/loop bugs but were a
    register-allocation mistake. Fixed: only CR0 is used as a genuine
    constant (ZERO); every real per-call value lives in CR2-CR15. #}
{% set ZERO          = "CR0"  %}
{% set DATA_BASE     = cr.b  %}
{% set WEIGHTS_BASE  = cr.a  %}
{% set OUTPUT_BASE   = cr.y  %}
{% set MASK_BASE     = cr.mask  %}
{% set DSTRUCT_MULT  = cr.dstruct_mult ~ ".L"  %}
{% set DSTRUCT_STORE = cr.ds_qf ~ ".L"  %}
{% set QF            = cr.ds_qf ~ ".H"  %}
{% set SEED_CR = [cr.seed0, cr.seed1, cr.seed2, cr.seed3, cr.seed4, cr.seed5, cr.seed6, cr.seed7] %}
{#- SEED_CR[p_out] holds (512 - 16*p_out - 16) % 512, the pre-increment-biased
    rc_idx seed for output partition p_out (a kernel.toml [derived] constant, bound in [abi]).
    W_CHUNKS is NOT a CR -- chunk_widths' length is a compile-time (Jinja
    render-time) constant, baked directly into the BLT.S comparisons below, so
    no CR slot is spent on it (CR budget is otherwise exactly full: 14
    values in CR2-CR15). #}

{% macro replicate_chunk() %}
    SET {{ chunk_base }} {{ DATA_BASE }};;
    ADD {{ chunk_base }} {{ chunk_base }} {{ chunk_ctr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    LD.MULT.CYCLIC {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    INC {{ slot_lr }} 128;;
    LD.MULT.CYCLIC {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    INC {{ slot_lr }} 255;;
    INC {{ slot_lr }} 1;;
    LD.MULT.CYCLIC {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    INC {{ slot_lr }} 255;;
    INC {{ slot_lr }} 129;;
    LD.MULT.CYCLIC {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
{% endmacro %}

{% macro accumulate_chunk(first) %}
    {#- Each output partition p_out has ITS OWN weight row (W[p_out, ...]),
        so R0 must be reloaded for every p_out within every packed row --
        R0 only ever holds one row, and unlike the unpacked/path(b) kernels
        (which visit a whole weight-chunk's worth of k for ONE output before
        moving on), this construction visits one packed-row's worth of k
        for ALL 8 outputs before moving to the next row. w_row_ptr walks
        the CURRENT weight-chunk's 8 rows (p_out=0..7), reset to w_ptr (this
        weight-chunk's row 0, i.e. output 0's row) at the start of every
        packed row. An earlier draft loaded weights only once per
        WEIGHT-CHUNK (matching path (b)'s cadence) and reused that single
        row for all 8 p_out -- silently correct arithmetic for p_out=0 only,
        with every other p_out's MULT reading p_out=0's weights instead of
        its own (all 8 output partitions converged to the same value: traced
        via r_acc lane dumps at K=16, all partitions equal to partition 0's
        running sum after their own p_out's accumulation). #}
    ADD {{ w_row_ptr }} {{ w_ptr }} {{ ZERO }};;
    {%- for p_out in range(8) %}
    LD.MULT R0 {{ w_row_ptr }} {{ ZERO }};;
    SET {{ rc_idx_reg }} {{ SEED_CR[p_out] }};;
    ADD {{ k_idx }} {{ k_base }} {{ ZERO }};;
    DEC {{ k_idx }} 1;;
    {%- for p_in in range(8) %}
    INC {{ rc_idx_reg }} 16;
    INC {{ k_idx }} 1;
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} {{ p_out }} {{ mask_off_lr }} {{ DSTRUCT_MULT }} {{ QF }};
    ACC.ADD{{ ".FIRST" if (first and p_out == 0 and p_in == 0) else "" }};;
    {%- endfor %}
    {%- if p_out < 7 %}
    INC {{ w_row_ptr }} 1;;
    {%- endif %}
    {%- endfor %}
{% endmacro %}

{% set chunk0_width = chunk_widths[0] %}
{% set chunk0_pc_count = chunk0_width // 8 %}

    SET {{ mask_off_lr }} {{ ZERO }};;
    LD.MASK {{ mask_off_lr }} {{ MASK_BASE }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;
    SET {{ w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ chunk_ctr }} {{ ZERO }};;
    SET {{ k_base }} {{ ZERO }};;

    {#- ---- peeled: weight-chunk 0's packed row 0, the one true .FIRST position ---- #}
    {{ replicate_chunk() }}
    {{ accumulate_chunk(true) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 8;;

    {%- if chunk0_pc_count > 1 %}
    {#- ---- weight-chunk 0's REMAINING packed rows (pc=1..chunk0_pc_count-1) ---- #}
    SET {{ pc_idx }} {{ ZERO }};;
    INC {{ pc_idx }} 1;;
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ chunk0_pc_count - 1 }};;
wc0_pc_loop:
    {{ replicate_chunk() }}
    {{ accumulate_chunk(false) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 8;;
    BLT.S {{ pc_idx }} {{ pc_bound }} wc0_pc_loop_inc;;
    B wc0_pc_loop_exit;;
wc0_pc_loop_inc:
    INC {{ pc_idx }} 1;;
    B wc0_pc_loop;;
wc0_pc_loop_exit:
    {%- endif %}

    {%- if chunk_widths | length > 1 %}
    {#- ---- weight-chunks 1..W_CHUNKS-1, each in full (pc=0..bound_i) ---- #}
    SET {{ wc_idx }} {{ ZERO }};;
    INC {{ wc_idx }} 1;;
    INC {{ w_ptr }} 8;;
    {#- +8, not +1: each weight-chunk now occupies 8 rows (one per p_out),
        weight-chunk-major layout (row = c*8 + p_out) since accumulate_chunk
        reloads R0 per p_out from w_row_ptr = w_ptr + p_out. #}
    SET {{ wc_bound }} {{ ZERO }};;
    INC {{ wc_bound }} {{ chunk_widths | length }};;
    {#- wc_bound = W_CHUNKS (not W_CHUNKS-1): the loop-continue check below
        is "BLT.S wc_idx wc_bound", testing whether there's still a
        weight-chunk left to process AFTER incrementing wc_idx post-loop-body, so
        it must compare against the total count, not count-1 -- an earlier
        draft used W_CHUNKS-1 (matching a "last valid index" convention that
        doesn't apply to a pre-increment continue-check) and silently
        dropped the LAST weight-chunk (K=480's tail, 12 packed rows)
        entirely: chunk_ctr topped out at 48/59, discovered by tracing the
        weight-chunk counter directly since the resulting error (9.14) didn't
        obviously point at a dropped weight-chunk. #}
    {#- wc_bound (a compile-time constant, chunk_widths|length - 1) loaded
        into an LR via INC immediate: BLT.S's reg2 operand is LcrIdx
        (LR/CR-only, no raw immediate), and every CR slot (CR2-CR15) is
        already spoken for by DATA_BASE/WEIGHTS_BASE/.../8 SEED_CRs -- see
        the CR budget note above. wc_num (a per-weight-chunk compile-time
        index, used below in the pc_bound_skip comparison) has the SAME
        constraint: an earlier draft passed the raw Jinja integer wc_num
        directly as BLT.S's reg2, which only happened not to fail because
        chunk_widths[1:] had exactly one element (K in {16,240}, 2 total
        weight-chunks) so that branch's guard never actually rendered --
        K=480 (4 weight-chunks) would have hit it. Fixed by loading each
        wc_num into wc_bound (reused as scratch here, safe since it's
        re-set before its OTHER use above) via INC immediates. #}

wc_loop:
    SET {{ k_base }} {{ ZERO }};;
    {%- for chunk_width in chunk_widths[1:] %}
    {%- set wc_num = loop.index %}
    {%- if not loop.last %}
    {#- Every entry except the last: check "is this the current
        weight-chunk?" (wc_idx == wc_num) and branch past the remaining
        candidates if so. The LAST entry needs no check at all -- reaching
        it by fall-through already proves it's the only one left, so it
        unconditionally sets its own bound (avoids a BLT.S whose "taken" path
        would otherwise skip past setting pc_bound entirely, the bug an
        earlier draft had: routing the last entry's match case straight to
        pc_bound_done without ever executing its own SET/INC). #}
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ wc_num }};;
    BGT.S {{ wc_idx }} {{ pc_bound }} pc_bound_skip_{{ wc_num }};;
    {%- endif %}
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ (chunk_width // 8) - 1 }};;
    {%- if not loop.last %}
    B pc_bound_done;;
pc_bound_skip_{{ wc_num }}:
    {%- endif %}
    {%- endfor %}
pc_bound_done:
    SET {{ pc_idx }} {{ ZERO }};;

wc_pc_loop:
    {{ replicate_chunk() }}
    {{ accumulate_chunk(false) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 8;;
    BLT.S {{ pc_idx }} {{ pc_bound }} wc_pc_loop_inc;;
    B wc_pc_loop_exit;;
wc_pc_loop_inc:
    INC {{ pc_idx }} 1;;
    B wc_pc_loop;;
wc_pc_loop_exit:

    INC {{ wc_idx }} 1;;
    BLT.S {{ wc_idx }} {{ wc_bound }} wc_loop_continue;;
    B wc_loop_exit;;
wc_loop_continue:
    INC {{ w_ptr }} 8;;
    B wc_loop;;
wc_loop_exit:
    {%- endif %}

    ACTIVATE.EXACT {{ "identity" if activation == "none" else activation }} {{ DSTRUCT_STORE }} {{ ZERO }} {{ out_ptr }};;

end:
    HALT;;
