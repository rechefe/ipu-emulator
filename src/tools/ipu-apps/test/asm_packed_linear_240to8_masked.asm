# Packed linear layer, path (b): PACKED DATA / UNPACKED WEIGHTS
# (memory-optimal). 240 input channels -> 8 output channels, 16 tokens
# (L5 packed-viability task, kernel C, the "hard case").
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed (input) -> unpacked (output)
# Shape:   240ch->8ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_linear_240to8_replicated.asm (superseded path (a) sibling);
#          asm_packed_linear_masked_generic.asm (generic K/N_OUT template of this path)
# Tests:   test_packed_linear_240to8.py, test_l5_real_size_packed_b.py
#
#   OUT[o, t] = sum_k W[o, k] * X[k, t]   for o = 0..7, k = 0..239
#
# Same weight layout as asm_unpacked_linear_240to8.asm (W_CHUNKS=2 rows of
# up to 128 scalars each, LDR_MULT_REG + R0 index, NO replication). Only
# the data side changes: X is packed 8 channels/row (16 lanes each, 30 rows
# total) instead of one channel/row.
#
# The two reload periods align WITHOUT runtime branching: a packed chunk
# covers 8 channels, a weight-chunk row covers <=128 channels, and 8
# divides 128 exactly -- 16 packed-chunk reloads per full-width weight
# row, 14 for the width-112 tail row (112/8=14). So the loop nests
# statically as: weight-chunk (2, matches unpacked) -> packed-chunk (16 or
# 14) -> partition (8, fully unrolled) -- channel order 0,1,...,7,8,9,...
# is identical to the unpacked kernel's k=0..239 walk, just grouped.
#
# rc_idx = 16*p selects partition p's 16 lanes from the currently-loaded
# packed chunk (same read this session already validated in primitive A
# and path (a)); src = k_idx selects the weight scalar from R0 by index,
# exactly as in the unpacked kernel. No masking, no combine: every k's
# MULT.RC.VE lands its 16 useful values at mult_res lanes 0..15 already,
# so ACC.ADD accumulates directly -- valid_elements=16 crops the rest at
# the final store.
#
# This measures the "activation-memory win only, no compute win" path the
# session's math check predicted: 240 MULT/ACC per output (same as
# unpacked), only X's XMEM footprint shrinks (30 packed rows vs 240).
# Weight-load instruction count is reported as its own line (2/output,
# same as unpacked) so it is never conflated with the packed-specific data
# reload count (30/output) in the headline comparison.
#
# MULT SNAPSHOT CONTRACT (issue #157): applies to R_CYCLIC (rc_idx) and R0
# (src as an LR index into R0) -- both "read: snapshot" fields resolved
# from loads one bundle earlier, as in the unpacked kernel and path (a).

{% set data_ptr    = "lr0"  %}  {# packed-chunk row pointer, walks 0..29, RESET per output #}
{% set rc_idx_reg  = "lr1"  %}  {# rc_idx = 16*p (READ side, MULT.RC.VE), pre-increment biased to -16 at chunk start (see IPC fix note below) #}
{% set k_idx       = "lr2"  %}  {# weight-scalar select into R0, pre-increment biased to (base-1) at chunk start #}
{% set chunk_w_ptr = "lr4"  %}  {# weight row pointer: o*W_CHUNKS + chunk_idx #}
{% set out_ptr     = "lr6"  %}  {# output row pointer, += 1 per o #}
{% set o_idx       = "lr7"  %}  {# output-channel counter 0..7 #}
{% set weight_row_off = "lr8" %} {# o*W_CHUNKS, += W_CHUNKS per o #}
{% set load_idx    = "lr9"  %}  {# LDR_CYCLIC_MULT_REG's index operand (WRITE side) -- always 0, kept separate from rc_idx_reg because that varies 0..112 between loads #}

{% set ZERO         = "cr0"  %}  {# const 0 #}
{% set ONE          = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE    = "cr2"  %}  {# packed X base row MINUS ONE (pre-increment bias) #}
{% set WEIGHTS_BASE = "cr3"  %}  {# W base row (unpacked layout, identical to asm_unpacked_linear_240to8.asm) #}
{% set OUTPUT_BASE  = "cr4"  %}  {# OUT base row #}
{% set N_OUT        = "cr9"  %}  {# 8 #}
{% set W_CHUNKS     = "cr8"  %}  {# 2 #}
{% set SIXTEEN      = "cr10" %}  {# 16: rc_idx step per partition #}
{% set NEG_SIXTEEN  = "cr11" %}  {# -16: rc_idx_reg pre-increment seed bias #}
{% set NEG_ONE      = "cr12" %}  {# -1: k_idx pre-increment seed bias #}
{% set DSTRUCT      = "cr15" %}  {# valid_elements=16 #}

{# IPC FIX (item 2): the original version issued the per-partition advance
   ADDs (rc_idx_reg += 16, k_idx += 1) in a SEPARATE bundle after the
   MULT.RC.VE+ACC.ADD bundle that used their pre-advance value -- 2 cycles
   per partition-step instead of 1, because MULT.RC.VE's rc_idx and (via
   _mult_resolve_lcr_scalar_wide's live regfile.get_lr read) src/k_idx are
   BOTH live-read operands (instruction_spec.py:591,594;
   ipu.py:777) and;
   LR-slot writes dispatch before load/mult within a bundle (same rule
   documented in reference_vliw_bundle_semantics.md / SKILL.md, and used
   throughout this session for data_ptr's pre-increment bias). That means
   the advance ADDs CAN be co-issued in the SAME bundle as the MULT that is
   meant to consume the advanced value, exactly like data_ptr's existing
   bias -- seed rc_idx_reg/k_idx one step BEHIND the first real index, and
   let each MULT's own bundle carry its own advance. #}

{# W_CHUNKS' width (128 then 112) is baked directly into the unrolled
   Jinja loop counts below (16 then 14 packed-chunks) -- no runtime bound
   register needed for either loop, unlike the unpacked kernel's FULL_BOUND
   / TAIL_BOUND (which existed because ITS inner loop ran per-k, needing a
   k-count comparison;
   here the inner loop is per-packed-chunk, and both;
   packed-chunk counts are compile-time constants). #}

    SET {{ o_idx }} {{ ZERO }};;
    SET {{ weight_row_off }} {{ ZERO }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;

o_loop:
    SET {{ data_ptr }} {{ DATA_BASE }};;
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ k_idx }} {{ NEG_ONE }};;

    {%- for pc in range(16) %}
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ load_idx }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    SET {{ rc_idx_reg }} {{ NEG_SIXTEEN }};;
    {%- for p in range(8) %}
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} 0 {{ load_idx }} {{ DSTRUCT }};
    ACC.ADD{{ ".FIRST" if pc == 0 and p == 0 else "" }};
    ADD {{ rc_idx_reg }} {{ rc_idx_reg }} {{ SIXTEEN }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;
    {%- endfor %}
    {%- endfor %}

    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ONE }};;
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ k_idx }} {{ NEG_ONE }};;

    {%- for pc in range(14) %}
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ load_idx }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    SET {{ rc_idx_reg }} {{ NEG_SIXTEEN }};;
    {%- for p in range(8) %}
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} 0 {{ load_idx }} {{ DSTRUCT }};
    ACC.ADD;
    ADD {{ rc_idx_reg }} {{ rc_idx_reg }} {{ SIXTEEN }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;
    {%- endfor %}
    {%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_CHUNKS }};
    INC {{ o_idx }} 1;;
    BLT {{ o_idx }} {{ N_OUT }} o_loop;;

end:
    BKPT;;
