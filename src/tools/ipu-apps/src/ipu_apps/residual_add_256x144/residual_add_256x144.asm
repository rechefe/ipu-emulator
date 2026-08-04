# Residual add: C[r] = A[r] + B[r]  for r = 0..287
#
# A, B: interleaved channel-major [256 tokens, 144 channels] = 288 rows × 128 bytes
#   Row r at A_BASE + r*128  (resp. B_BASE + r*128)
# C: FP32 accumulator output, 288 rows × 512 bytes
#   Row r at OUTPUT_BASE + r*512
#
# There is no add instruction in the vector path, so each addend is passed
# through the multiplier against a constant 1.0 and summed in the accumulator.
# The scalar 1.0 comes from a CR (DTYPE_ONE), so no register holds ones.
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
#     Cycle 1: MULT.RC.VE × 1.0 on A[r] (loaded last row's cycle 3); ACC.ADD.FIRST;
#              LDR_CYCLIC_MULT_REG → r_cyclic = B[r]
#     Cycle 2: MULT.RC.VE × 1.0 on B[r]; ACC.ADD
#     Cycle 3: ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG → stage and store
#              A[r]+B[r] in ONE word; advance row counter;
#              LDR_CYCLIC_MULT_REG → r_cyclic = A[r+1]  (prefetch for next row)
#     Cycle 4: advance output ptr; BLT → loop
#   Total: 4 cycles × 288 rows = 1152 cycles, plus one priming load at startup.
#
# IMPORTANT — live-LR semantics:
#   ADD fires before XMEM; ptrs init at -128 so first live = 0 after ADD.
#   Each a_ptr/b_ptr ADD stays co-issued with the load it feeds, so the load
#   sees the already-advanced pointer -- that is what makes the prefetch in
#   cycle 3 address row r+1 rather than row r.
#   STR_POST_AAQ_REG reads out_ptr live; do NOT ADD out_ptr in the same cycle as it.
#   BLT reads snapshot; ADD row_index must happen the cycle before BLT.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "lr1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "lr2"  %}  {# byte offset into A (startup -128) #}
{% set b_ptr      = "lr3"  %}  {# byte offset into B (startup -128) #}
{% set out_ptr    = "lr4"  %}  {# byte offset into C, += 512 per row #}
{% set row_index  = "lr5"  %}  {# row counter 0..287 #}
{% set row_limit  = "lr6"  %}  {# 288 = N_ROWS #}
{% set row_stride = "lr7"  %}  {# 128 = row stride for A and B #}
{% set out_stride = "lr8"  %}  {# 512 = output row stride #}

{% set A_BASE     = "cr0"  %}  {# base of A #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set ONES_BASE  = "cr2"  %}  {# 128 bytes of dtype-1.0 #}
{% set OUT_BASE   = "cr3"  %}  {# output base #}
{% set ZERO       = "cr4"  %}  {# const 0 #}
{% set PTR_START  = "cr5"  %}  {# -128 startup init for the A/B row pointers #}
{% set ROW_COUNT  = "cr6"  %}  {# 288 = N_ROWS #}
{% set ROW_STRIDE = "cr7"  %}  {# 128 = A/B row stride #}
{% set OUT_STRIDE = "cr8"  %}  {# 512 = output row stride #}
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
    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ ONES_BASE }};;  # r0 = ONES_BASE[0..127] = dtype-1.0 × 128
    # Prime the pipeline: load A[0] so row 0's cycle 1 has it in the snapshot.
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }}; ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0  (A[r] was loaded a cycle earlier)
    #   MULT.RC.VE r_cyclic[0] × DTYPE_ONE(=dtype 1.0) → A[r] passed through.
    #   Co-issued load fetches B[r] for cycle 2.
    LDR_CYCLIC_MULT_REG {{ b_ptr }} {{ B_BASE }} {{ rc_slot0 }}; ADD {{ b_ptr }} {{ b_ptr }} {{ row_stride }}; MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;;
    # Cycle 2: r_acc += B[r] × 1.0
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;;
    # Cycle 3: stage r_acc into post_aaq_reg for the hardware store path.
    #   ACTIVATE.QUANTIZE and STR_POST_AAQ_REG co-issue in ONE VLIW word: AaQ
    #   and STR are consecutive pipeline stages within a word, and the in-word
    #   slot order is CTRL -> MULT -> ACC -> AaQ -> STR (docs/content/specs/
    #   stage-aaq-str.md §7.0), so STR consumes this cycle's AaQ result. This
    #   is why the store is free and the loop stays 4 cycles per row.
    #   `identity` + DSTRUCT's default valid_elements=128 makes this a
    #   lane-for-lane FP32 copy of all 128 lanes, i.e. byte-identical to the
    #   STR_ACC_REG it replaces in wide mode.
    #   Co-issued load prefetches A[r+1] for the next row's cycle 1.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT }}; STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }}; ADD {{ row_index }} {{ row_index }} {{ ONE }}; LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }}; ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;
    # Cycle 4: advance output ptr; BLT reads snap row_index = already-incremented
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ out_stride }}; BLT {{ row_index }} {{ row_limit }} row_loop;;

end:
    BKPT;;
