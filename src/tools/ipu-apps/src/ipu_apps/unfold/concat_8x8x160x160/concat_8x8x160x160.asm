# Concat 8x8x160x160: channel-axis concat of two same-spatial-shape tensors (L5)
#
# Layer:   L5
# Scope:   single-stream (two independent copy passes into one output buffer)
# Layout:  unpacked
# Shape:   two 160-channel inputs, 8x8 spatial -> one 320-channel output
# Status:  new
# Related: MobileViT-S ends every transformer block with
#          torch.cat((residual, features), dim=channel); this is the general
#          (non-pre-offset) kernel version -- see header note "WHY A REAL
#          COPY KERNEL" below.
#
# Op:      output[0:C_A]      = A[0:C_A]     (channel-for-channel copy)
#          output[C_A:C_A+C_B] = B[0:C_B]     (channel-for-channel copy)
#
# A, B: channel-major, ONE CHANNEL PER ROW, same convention every kernel in
#   this family uses (fold's output / conv's raw output layout). In
#   wide-vector FP32 debug mode an XMEM row is unconditionally 128 lanes x 4B
#   = 512 bytes; this kernel treats "one row" as the addressable unit (like
#   fold/unfold/residual_add) and moves ROWS wholesale -- it never interprets
#   what is inside a row, so it is correct for any per-channel spatial
#   packing upstream chose to put there (the 8x8=64 spatial elements of a
#   real channel are whatever the producer laid out; this kernel treats each
#   channel as exactly one opaque row, the same simplification the rest of
#   this row-per-channel family makes).
#   Concretely: input A occupies C_A rows at A_BASE (row i = channel i's
#   data), input B occupies C_B rows at B_BASE, and the output occupies
#   C_A + C_B rows at OUT_BASE.
#
# Output: C_A + C_B rows at OUT_BASE.
#   Row i (0 <= i < C_A):        copied from A row i.
#   Row C_A + j (0 <= j < C_B):  copied from B row j.
#
# WHY A REAL COPY KERNEL (not just picking base-row offsets):
#   On this ISA, every producer's store address is base_row + walking_offset,
#   where base_row is a compile-time/harness constant -- so two producers
#   COULD write directly into disjoint channel-offset regions of one shared
#   output buffer with zero data movement, if the caller pre-arranges it.
#   That is NOT what this kernel does. This is the general-purpose registry
#   kernel: it assumes A and B already live in their own (potentially
#   non-adjacent) memory regions with their own base addresses -- the safe
#   assumption for a kernel other code can call without assuming its callers
#   pre-arranged memory -- and it physically copies both into one contiguous
#   output buffer. A genuine copy, not a no-op.
#
# There is no direct copy/move instruction in the vector path (same
# constraint residual_add documents): every value must pass through the
# multiplier against a constant 1.0 before landing in r_acc, then through
# ACTIVATE.QUANTIZE (identity) to stage it for the store. Unlike
# residual_add, concat performs NO cross-input arithmetic: each row is
# ACC.ADD.FIRST'd alone (r_acc := row x 1.0, no second addend), so this is
# structurally two independent copy loops back to back, not one add loop.
#
# ACTIVATE.QUANTIZE cannot stage directly from a freshly-loaded register --
# checked against ipu.py's execute_activate_quantize: it reads only the live
# r_acc register file (never r0/r1/r_cyclic directly), so a load-only path
# with no MULT/ACC step is not available. The load -> MULT x1.0 ->
# ACC.ADD.FIRST -> ACTIVATE.QUANTIZE -> STR chain is the minimum, exactly
# matching residual_add's per-operand pipeline stage (just without the
# second MULT/ACC.ADD for a second addend).
#
# MULT SNAPSHOT CONTRACT (issue #157, same as fold/unfold/residual_add):
# MULT.RC.VE reads its r_cyclic DATA from the start-of-cycle snapshot, so it
# cannot consume a row loaded by LDR_CYCLIC_MULT_REG issued in the SAME
# bundle -- `;;` ends one VLIW word = one cycle = one snapshot, so a load and
# a MULT co-issued in one bundle both see that cycle's PRE-load r_cyclic
# state. Each load therefore runs one bundle ahead of the MULT that consumes
# it; verified empirically against this exact kernel (see gen_debug_data.py /
# the throwaway verification script) before trusting the pipelining below.
#
# STR_POST_AAQ_REG reads its offset operand LIVE -- never co-issue an ADD on
# the store's offset LR in the same bundle as the store itself, or the row
# lands one slot late (same hazard documented in fold_16x16x192.asm and
# residual_add_16x240.asm).
#
# CR0 and CR1 are BOTH read-only hardwired constants (0 and 1 respectively;
# writes are silently dropped -- CR_READ_ONLY_INITIAL_VALUES). A_BASE_ROW is
# 0 in this kernel's memory map, so naming it cr0 is a harmless no-op (same
# convention residual_add_16x240 uses); every OTHER nonzero/non-one base or
# count avoids cr0/cr1.
#
# Two independent BLT-driven loops (NOT hand-unrolled -- C_A/C_B are up to
# 160 in this family, unlike fold's small fixed 4x4 structure): loop 1 copies
# A's C_A rows to output rows [0, C_A); loop 2 copies B's C_B rows to output
# rows [C_A, C_A+C_B). The two loops share the register file (loop 2 reuses
# loop 1's registers with B's/second-half's bases and counts) but are
# textually and functionally independent -- neither reads the other's state.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base #}
{% set src_ptr    = "lr1"  %}  {# row offset into the active input (startup -1) #}
{% set out_ptr    = "lr2"  %}  {# row offset into output, += 1 per row #}
{% set row_index  = "lr3"  %}  {# row counter within the active loop #}
{% set row_limit  = "lr4"  %}  {# C_A then C_B: loop bound for the active loop #}
{% set row_stride = "lr5"  %}  {# 1 = row stride (both inputs, output) #}

{% set ZERO       = "cr0"  %}  {# hardwired read-only 0 #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set PTR_START  = "cr2"  %}  {# -1 row startup init for src_ptr #}
{% set ROW_STRIDE = "cr3"  %}  {# 1 = row stride #}
{% set DTYPE_ONE  = "cr4"  %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set A_BASE     = "cr5"  %}  {# base row of input A #}
{% set B_BASE     = "cr6"  %}  {# base row of input B #}
{% set OUT_A_BASE = "cr7"  %}  {# output base row for A's copy (== 0 conceptually; may be nonzero) #}
{% set OUT_B_BASE = "cr8"  %}  {# output base row for B's copy (== C_A) #}
{% set C_A_COUNT  = "cr9"  %}  {# channel count of A #}
{% set C_B_COUNT  = "cr10" %}  {# channel count of B #}
{% set DSTRUCT    = "cr15" %}  {# reserved dstructure register #}

# ---------------------------------------------------------------------------
# Pass 1: copy A's C_A rows -> output rows [OUT_A_BASE, OUT_A_BASE + C_A)
#
# out_ptr is a 0-based ROW COUNTER for this pass, not an absolute row number:
# STR_POST_AAQ_REG's address is offset + base (ipu.py execute_str_post_aaq_reg),
# so the pass's absolute output base lives entirely in the STR's `base`
# operand (OUT_A_BASE here, OUT_B_BASE in pass 2) and out_ptr only ever
# counts 0, 1, 2, ... within the pass -- no separate ADD needed to fold a
# nonzero base into it at startup.
# ---------------------------------------------------------------------------
    SET                 {{ rc_slot0 }} {{ ZERO }};;
    SET                 {{ src_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ C_A_COUNT }};;
    SET                 {{ row_stride }} {{ ROW_STRIDE }};;
    # Prime the pipeline: load A[0] so row 0's cycle 1 has it in the snapshot.
    LDR_CYCLIC_MULT_REG {{ src_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ src_ptr }} {{ src_ptr }} {{ row_stride }};;

a_loop:
    # Cycle 1: r_acc = A[ch] x 1.0 (A[ch] was loaded a cycle earlier).
    #   Co-issued load prefetches A[ch+1] for the next iteration's cycle 1.
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ src_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ src_ptr }} {{ src_ptr }} {{ row_stride }};;
    # Cycle 2: stage r_acc into post_aaq_reg and store it (ACTIVATE.QUANTIZE
    #   identity + STR_POST_AAQ_REG co-issue in one VLIW word; in-word slot
    #   order CTRL -> MULT -> ACC -> AaQ -> STR means STR consumes this
    #   cycle's AaQ result -- same free-store pattern residual_add_16x240
    #   uses). Advance the row counter in the same cycle (it does not feed
    #   this cycle's STR offset, which is out_ptr, not row_index).
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_A_BASE }};
    ADD {{ row_index }} {{ row_index }} {{ ONE }};;
    # Cycle 3: advance output ptr; BLT reads the snapshot row_index, which is
    #   already incremented (set the cycle before, per the live/snapshot
    #   convention every kernel in this family follows).
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ row_stride }};
    BLT {{ row_index }} {{ row_limit }} a_loop;;

# ---------------------------------------------------------------------------
# Pass 2: copy B's C_B rows -> output rows [OUT_B_BASE, OUT_B_BASE + C_B)
# ---------------------------------------------------------------------------
    SET                 {{ src_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ C_B_COUNT }};;
    # Prime the pipeline: load B[0].
    LDR_CYCLIC_MULT_REG {{ src_ptr }} {{ B_BASE }} {{ rc_slot0 }};
    ADD {{ src_ptr }} {{ src_ptr }} {{ row_stride }};;

b_loop:
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ src_ptr }} {{ B_BASE }} {{ rc_slot0 }};
    ADD {{ src_ptr }} {{ src_ptr }} {{ row_stride }};;
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_B_BASE }};
    ADD {{ row_index }} {{ row_index }} {{ ONE }};;
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ row_stride }};
    BLT {{ row_index }} {{ row_limit }} b_loop;;

end:
    BKPT;;
