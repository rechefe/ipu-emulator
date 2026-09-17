# Residual add (L5): C[ch] = A[ch] + B[ch]  for ch = 0..239
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   16tok x 240chan (240 rows)
# Status:  validated
# Related: L5 port of residual_add_256x144 (L3) via residual_add_64x192 (L4);
#          consumes layernorm_16x240's output (full-row, uncropped, per the
#          producer-emits-full-rows / final-consumer-crops convention)
# Tests:   test_residual_add_16x240_wide (src/tools/ipu-apps/BUILD.bazel)
#
# A, B: channel-major [240 channels, 16 tokens], ONE CHANNEL PER ROW.
#   Row ch at A_BASE + ch  (resp. B_BASE + ch).
#   A channel's 16 FP32 tokens occupy 64 of a row's 512 bytes; the rest is
#   padding. Rows are NEVER shared between channels.
# C: FP32 output, 240 rows of 512 bytes; row ch at OUTPUT_BASE + ch.
#   The harness crops each row to its 16-token prefix in teardown().
#
# The lane count is a property of the row, not of the loop: every row is
# processed as a full 128-lane vector and only the first 16 lanes carry
# meaning, so no loop bound or stride here depends on N_TOK.
#
# There is no add instruction in the vector path, so each addend is passed
# through the multiplier against a constant 1.0 and summed in the accumulator.
# The scalar 1.0 comes from a CR (DTYPE_ONE), so no register holds ones.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot, so it cannot consume a row loaded by
# LDR_CYCLIC_MULT_REG in its own bundle -- it would see the previous r_cyclic.
# `;;` ends one VLIW word = one cycle = one snapshot, so a load and a MULT in
# the same bundle always execute in the same cycle regardless of textual order;
# co-issuing them is fine, consuming the same-cycle load is not.
# Each load therefore runs one bundle ahead of the MULT that consumes it. The
# two loads are pipelined into cycles that were already doing other work, so
# the steady state stays 4 cycles per row:
#   Per row:
#     Cycle 1: MULT.RC.VE x 1.0 on A[ch] (loaded last row's cycle 3); ACC.ADD.FIRST;
#              LDR_CYCLIC_MULT_REG -> r_cyclic = B[ch]
#     Cycle 2: MULT.RC.VE x 1.0 on B[ch]; ACC.ADD
#     Cycle 3: ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG -> stage and store
#              A[ch]+B[ch] in ONE word; advance row counter;
#              LDR_CYCLIC_MULT_REG -> r_cyclic = A[ch+1]  (prefetch for next row)
#     Cycle 4: advance output ptr; BLT -> loop
#   Total: 4 cycles x 240 rows = 960 cycles, plus one priming load at startup.
#
# IMPORTANT -- live-LR semantics:
#   ADD fires before XMEM; ptrs init at -1 row so first live = 0 after ADD.
#   Each a_ptr/b_ptr ADD stays co-issued with the load it feeds, so the load
#   sees the already-advanced pointer -- that is what makes the prefetch in
#   cycle 3 address row ch+1 rather than row ch.
#   STR_POST_AAQ_REG reads out_ptr live; do NOT ADD out_ptr in the same cycle as it.
#   BLT reads snapshot; ADD row_index must happen the cycle before BLT.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "lr1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "lr2"  %}  {# row offset into A (startup -1) #}
{% set b_ptr      = "lr3"  %}  {# row offset into B (startup -1) #}
{% set out_ptr    = "lr4"  %}  {# row offset into C, += 1 per row #}
{% set row_index  = "lr5"  %}  {# row counter 0..239 #}
{% set row_limit  = "lr6"  %}  {# 240 = N_ROWS #}
{% set row_stride = "lr7"  %}  {# 1 = row stride for A and B #}
{% set out_stride = "lr8"  %}  {# 1 = output row stride #}

{% set A_BASE     = "cr0"  %}  {# base of A #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE   = "cr3"  %}  {# output base #}
{% set ZERO       = "cr4"  %}  {# const 0 #}
{% set PTR_START  = "cr5"  %}  {# -1 row startup init for the A/B row pointers #}
{% set ROW_COUNT  = "cr6"  %}  {# 240 = N_ROWS #}
{% set ROW_STRIDE = "cr7"  %}  {# 1 = A/B row stride #}
{% set OUT_STRIDE = "cr8"  %}  {# 1 = output row stride #}
{% set B_BASE     = "cr9"  %}  {# base of B #}
{% set DTYPE_ONE  = "cr10" %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set DSTRUCT    = "cr15" %}  {# reserved dstructure register #}

    SET                 {{ rc_slot0 }} {{ ZERO }};;
    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ a_ptr }} {{ PTR_START }};;
    SET                 {{ b_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ ROW_COUNT }};;
    SET                 {{ row_stride }} {{ ROW_STRIDE }};;
    SET                 {{ out_stride }} {{ OUT_STRIDE }};;
    # Prime the pipeline: load A[0] so row 0's cycle 1 has it in the snapshot.
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;

row_loop:
    # Cycle 1: r_acc = A[ch] x 1.0  (A[ch] was loaded a cycle earlier)
    #   Co-issued load fetches B[ch] for cycle 2.
    LDR_CYCLIC_MULT_REG {{ b_ptr }} {{ B_BASE }} {{ rc_slot0 }};
    ADD {{ b_ptr }} {{ b_ptr }} {{ row_stride }};
    MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;
    # Cycle 2: r_acc += B[ch] x 1.0
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;;
    # Cycle 3: stage r_acc into post_aaq_reg for the hardware store path.
    #   ACTIVATE.QUANTIZE and STR_POST_AAQ_REG co-issue in ONE VLIW word: AaQ
    #   and STR are consecutive pipeline stages within a word, and the in-word
    #   slot order is CTRL -> MULT -> ACC -> AaQ -> STR, so STR consumes this
    #   cycle's AaQ result. This is why the store is free and the loop stays
    #   4 cycles per row.
    #   Co-issued load prefetches A[ch+1] for the next row's cycle 1.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};
    ADD {{ row_index }} {{ row_index }} {{ ONE }};
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;
    # Cycle 4: advance output ptr; BLT reads snap row_index = already-incremented
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ out_stride }};
    BLT {{ row_index }} {{ row_limit }} row_loop;;

end:
    BKPT;;
