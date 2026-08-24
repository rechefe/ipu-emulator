# 1-SLOT REPLICATION VARIANT (task item 4): checks whether rc_idx =
# 16*((p_in-p_out) mod 8) -- always in [0,112] -- ever needs a read window
# beyond R_CYCLIC element 127, i.e. whether replicating the packed chunk
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed (input, output) -- scatter-on-write construction
# Shape:   Kch->8ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_output_linear_generic.asm (4-slot baseline this optimizes);
#          asm_packed_output_linear_tiny.asm (original construction, K=8)
# Tests:   test_packed_output_linear_1slot_replication.py
# into all 4 R_CYCLIC slots (asm_packed_output_linear_generic.asm's
# replicate_chunk(), 4 loads/chunk) is more than the construction actually
# needs. Verified by direct enumeration before touching this file: for
# every (p_in, p_out) pair in 0..7, rc_idx in [0,112] and the 16-lane read
# window [rc_idx, rc_idx+15] has max index 127 -- NEVER crossing into a
# second slot at all. This variant replicates into slot 0 ONLY (1 load,
# not 2 or 4) to test that finding directly; see the task report for the
# reconciliation against the brief's own "slots 0 and 1" framing (the
# brief's window bound "r+i <= 239" is a looser/safer bound than the
# tighter one actually achieved by this specific rc_idx formula).
#
# Packed-OUTPUT linear layer, GENERIC (K, N_OUT=8 fixed) -- extends
# asm_packed_output_linear_tiny.asm's validated construction to K=240 (30
# packed chunks) and beyond. N_OUT is fixed at 8 per call: with N_OUT=8 the
# 8 R_MASK slots map 1:1 onto p_out, which is the alignment the construction
# depends on (per the brief: "the 8 R_MASK slots map 1:1 onto the 8 p_out
# values -- the alignment that holds only at P8, which is L5"). For N_OUT>8
# (out-proj=240, FFN1=480, QKV=720) this kernel is called N_OUT/8 times by
# the harness, each call producing ONE packed row of 8 output channels --
# still zero host-side data conversion: each call's output is a complete,
# valid packed row on its own; nothing converts data BETWEEN calls, only the
# harness chooses (at load time) which 8-channel weight slice a given call
# targets, exactly like "the harness may choose what CRs/LRs to load."
#
#   OUT[o, t] = sum_k W[o, k] * X[k, t]  for o = 0..7 (this call's 8 outputs), k = 0..K-1
#
#   rc_idx = 16*(p_in - p_out) mod 512
#
# See asm_packed_output_linear_tiny.asm for the full derivation and the two
# prerequisites (R_CYCLIC replicated into all 4 slots per packed chunk;
# R_MASK's 8 slots pre-built with mask_offset=p_out => bits
# [16p_out,16p_out+16) set, loaded once via LDR_MULT_MASK_REG). Same
# IPC-fix-style pre-increment bias for the in-partition accumulation
# (rc_idx_reg/k_idx seeded one step behind so the co-issued advance ADDs are
# visible to the SAME bundle's MULT).
#
# RUNTIME CHUNK LOOP (not fully unrolled): a first cut fully unrolled the
# packed-chunk loop -- at K=240 (30 chunks) that produced 3130+ static
# instructions, blowing the 1024-instruction program-memory ceiling ~3x
# over. The 8x8=64 MULT+ACC-per-chunk body (unavoidable: this construction
# visits each chunk once per OUTPUT partition, not once total, to land all 8
# outputs packed -- same 1920 total MULT+ACC ops as path(b)/unpacked at
# 240->8, just organized differently) is small and stays unrolled per chunk;
# what cannot be unrolled at real K is the outer chunk-count loop itself.
#
# PEELED FIRST CHUNK: ACC.ADD.FIRST must fire exactly once for the WHOLE
# kernel run (chunk 0, p_out=0, p_in=0), not once per chunk -- a runtime
# loop body re-executes the same static instructions every iteration, so a
# compile-time "p_out==0 and p_in==0" guard inside the loop body would
# incorrectly re-fire .FIRST (and reset the accumulator) on EVERY chunk.
# Fixed the same way asm_unpacked_linear_240to8.asm handles it ("chunk 0:
# prime + peel"): weight-chunk 0's packed chunk 0 is statically unrolled
# ONCE outside any loop with .FIRST at its one true first position; TWO
# separate runtime loops then cover everything else -- weight-chunk 0's
# REMAINING packed chunks (pc=1..bound0, only entered if bound0>0), then
# weight-chunks 1..W_CHUNKS-1 in full (pc=0..bound_i each). This avoids the
# awkward "which bound applies, and did we already consume chunk 0"
# branching a single merged loop would need.
#
# SEED LOOKUP TABLE (8 CRs, one per p_out) for rc_idx: computed once by the
# harness at load time (legal -- "harness may choose what CRs to load").
#
# Weight layout: 8 rows (one per p_out in this call), each split across
# ceil(K/128) weight-chunks -- same convention as every other linear kernel
# here. K assumed a multiple of 8 (packing group size); this session's real
# shapes (240, 480) both qualify.
#
# CROSS-KERNEL R_MASK HAZARD: this kernel loads a one-hot 8-slot R_MASK
# (one 16-lane window per output partition) via LDR_MULT_MASK_REG and never
# restores an all-ones R_MASK before halting. A downstream kernel in the
# same IpuState that relies on R_MASK's all-ones default (e.g.
# asm_packed_residual_add_240x16.asm) will silently read back zeros in
# partitions 1-7 unless the caller explicitly reloads R_MASK first. See
# docs/isa_friction_log.md's "Cross-kernel R_MASK state bleed" entry.

{% set rc_idx_reg   = "lr0" %}
{% set w_ptr        = "lr1" %}
{% set out_ptr      = "lr2" %}
{% set k_idx        = "lr3" %}
{% set mask_off_lr  = "lr4" %}
{% set slot_lr      = "lr5" %}
{% set chunk_base   = "lr6" %}
{% set chunk_ctr    = "lr7" %}   {# running packed-chunk counter across the WHOLE kernel, drives chunk_base #}
{% set pc_idx       = "lr8" %}   {# packed-chunk index within the current weight-chunk's runtime loop #}
{% set pc_bound     = "lr9" %}   {# this weight-chunk's pc_idx upper bound (inclusive) #}
{% set wc_idx       = "lr10" %}  {# weight-chunk index, runtime counter #}
{% set wc_bound     = "lr12" %}  {# W_CHUNKS (chunk_widths|length), a compile-time constant loaded into an LR via INC immediates since no CR slot is free for it #}
{% set k_base       = "lr11" %}  {# 8*pc_idx: this packed chunk's channel offset WITHIN the current weight-chunk's R0 row -- k_idx = k_base + p_in, NOT just p_in, whenever a weight-chunk spans >1 packed chunk (K>8) #}
{% set w_row_ptr    = "lr13" %}  {# walks the current weight-chunk's 8 per-p_out rows, reset to w_ptr at the start of every packed chunk #}

{#- CR0/CR1 are HARDWIRED read-only (0 and 1 respectively) -- confirmed via
    state.regfile.get_cr(1) returning 1 even after set_cr(1, 0). An earlier
    draft assigned DATA_BASE=cr1 assuming it was a normal writable CR;
    the;
    harness's set_cr(1, data_base_row) silently had NO EFFECT, leaving
    chunk_base permanently seeded at 1 instead of the real data base row --
    this produced the multi-chunk correctness failures (K=16 err=4.09,
    K=240 err=25.09/18.37) that looked like addressing/loop bugs but were a
    register-allocation mistake. Fixed: only cr0 is used as a genuine
    constant (ZERO);
    every real per-call value lives in cr2-cr15.;  #}
{% set ZERO          = "cr0"  %}
{% set DATA_BASE     = "cr2"  %}
{% set WEIGHTS_BASE  = "cr3"  %}
{% set OUTPUT_BASE   = "cr4"  %}
{% set MASK_BASE     = "cr5"  %}
{% set DSTRUCT_MULT  = "cr6"  %}
{% set DSTRUCT_STORE = "cr7"  %}
{% set SEED_CR = ["cr8", "cr9", "cr10", "cr11", "cr12", "cr13", "cr14", "cr15"] %}
{#- SEED_CR[p_out] holds (512 - 16*p_out - 16) % 512, the pre-increment-biased
    rc_idx seed for output partition p_out (harness-loaded constant).
    W_CHUNKS is NOT a CR -- chunk_widths' length is a compile-time (Jinja
    render-time) constant, baked directly into the BLT comparisons below, so
    no CR slot is spent on it (CR budget is otherwise exactly full: 14
    values in cr2-cr15). #}

{% macro replicate_chunk() %}
    SET {{ chunk_base }} {{ DATA_BASE }};;
    ADD {{ chunk_base }} {{ chunk_base }} {{ chunk_ctr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    LDR_CYCLIC_MULT_REG {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
{% endmacro %}

{% macro accumulate_chunk(first) %}
    {#- Each output partition p_out has ITS OWN weight row (W[p_out, ...]),
        so R0 must be reloaded for every p_out within every packed chunk --
        R0 only ever holds one row, and unlike the unpacked/path(b) kernels
        (which visit a whole weight-chunk's worth of k for ONE output before
        moving on), this construction visits one packed-chunk's worth of k
        for ALL 8 outputs before moving to the next chunk. w_row_ptr walks
        the CURRENT weight-chunk's 8 rows (p_out=0..7), reset to w_ptr (this
        weight-chunk's row 0, i.e. output 0's row) at the start of every
        packed chunk. An earlier draft loaded weights only once per
        WEIGHT-CHUNK (matching path (b)'s cadence) and reused that single
        row for all 8 p_out -- silently correct arithmetic for p_out=0 only,
        with every other p_out's MULT reading p_out=0's weights instead of
        its own (all 8 output partitions converged to the same value: traced
        via r_acc lane dumps at K=16, all partitions equal to partition 0's
        running sum after their own p_out's accumulation). #}
    ADD {{ w_row_ptr }} {{ w_ptr }} {{ ZERO }};;
    {%- for p_out in range(8) %}
    LDR_MULT_REG r0 {{ w_row_ptr }} {{ ZERO }};;
    SET {{ rc_idx_reg }} {{ SEED_CR[p_out] }};;
    ADD {{ k_idx }} {{ k_base }} {{ ZERO }};;
    DEC {{ k_idx }} 1;;
    {%- for p_in in range(8) %}
    INC {{ rc_idx_reg }} 16;
    INC {{ k_idx }} 1;
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} {{ p_out }} {{ mask_off_lr }} {{ DSTRUCT_MULT }};
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
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;
    SET {{ w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ chunk_ctr }} {{ ZERO }};;
    SET {{ k_base }} {{ ZERO }};;

    {#- ---- peeled: weight-chunk 0's packed chunk 0, the one true .FIRST position ---- #}
    {{ replicate_chunk() }}
    {{ accumulate_chunk(true) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 8;;

    {%- if chunk0_pc_count > 1 %}
    {#- ---- weight-chunk 0's REMAINING packed chunks (pc=1..chunk0_pc_count-1) ---- #}
    SET {{ pc_idx }} {{ ZERO }};;
    INC {{ pc_idx }} 1;;
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ chunk0_pc_count - 1 }};;
wc0_pc_loop:
    {{ replicate_chunk() }}
    {{ accumulate_chunk(false) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 8;;
    BLT {{ pc_idx }} {{ pc_bound }} wc0_pc_loop_inc;;
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
        is "BLT wc_idx wc_bound", testing whether there's still a weight-
        chunk left to process AFTER incrementing wc_idx post-loop-body, so
        it must compare against the total count, not count-1 -- an earlier
        draft used W_CHUNKS-1 (matching a "last valid index" convention that
        doesn't apply to a pre-increment continue-check) and silently
        dropped the LAST weight-chunk (K=480's tail, 12 packed chunks)
        entirely: chunk_ctr topped out at 48/59, discovered by tracing the
        chunk counter directly since the resulting error (9.14) didn't
        obviously point at a dropped chunk. #}
    {#- wc_bound (a compile-time constant, chunk_widths|length - 1) loaded
        into an LR via INC immediate: BLT's reg2 operand is LcrIdx
        (LR/CR-only, no raw immediate), and every CR slot (cr2-cr15) is
        already spoken for by DATA_BASE/WEIGHTS_BASE/.../8 SEED_CRs -- see
        the CR budget note above. wc_num (a per-weight-chunk compile-time
        index, used below in the pc_bound_skip comparison) has the SAME
        constraint: an earlier draft passed the raw Jinja integer wc_num
        directly as BLT's reg2, which only happened not to fail because
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
        unconditionally sets its own bound (avoids a BLT whose "taken" path
        would otherwise skip past setting pc_bound entirely, the bug an
        earlier draft had: routing the last entry's match case straight to
        pc_bound_done without ever executing its own SET/INC). #}
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ wc_num }};;
    BGT {{ wc_idx }} {{ pc_bound }} pc_bound_skip_{{ wc_num }};;
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
    BLT {{ pc_idx }} {{ pc_bound }} wc_pc_loop_inc;;
    B wc_pc_loop_exit;;
wc_pc_loop_inc:
    INC {{ pc_idx }} 1;;
    B wc_pc_loop;;
wc_pc_loop_exit:

    INC {{ wc_idx }} 1;;
    BLT {{ wc_idx }} {{ wc_bound }} wc_loop_continue;;
    B wc_loop_exit;;
wc_loop_continue:
    INC {{ w_ptr }} 8;;
    B wc_loop;;
wc_loop_exit:
    {%- endif %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT_STORE }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;

end:
    BKPT;;
