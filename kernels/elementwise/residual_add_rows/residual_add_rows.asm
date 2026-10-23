{#- ==========================================================================
    residual_add_rows.asm -- y = a + b over `rows` whole XMEM rows, FP32.

    The template of the shape-named residual adds (kernels/elementwise/
    registry.toml): residual_add_16x240 (L5), residual_add_64x192 (L4),
    residual_add_256x144 (L3); E07-23 adds the MobileViT packed programs
    (240x16, 192x64), the same binary. One program serves them all: only CR6
    (rows) differs.

    Design notes
    - FP32 only: an element is 4 bytes and a row is 128 lanes (512 bytes).
      There is no narrow path.
    - The lane count is a property of the row, not of the loop: every row is
      processed as a full 128-lane vector, so no loop bound or stride depends on
      tokens. The layouts differ only in which lanes carry data:
        one channel per row (16x240, 64x192): tokens in lanes 0..tokens-1, the
          rest zero padding, rows never shared between channels. Padding adds to
          zero; the consumer crops it (producer emits full rows, final consumer
          crops).
        two rows per channel (256x144): tokens > 128, each channel's 256 tokens
          over 2 consecutive full rows (channel-major, 2 rows per channel),
          288 rows, no padding.
        packed (per_row > 1): per_row channels side by side in one row
          (240x16: 8 per row, 30 rows; 192x64: 2 per row, 96 rows). Elementwise
          add is partition-local, so packing changes only the row count.
      kernel.toml declares each tensor as [channels / per_row, per_row * tokens]
      with the default row-major pack, which gives all three layouts.
    - Regions follow each other: a at row 0, b at row `rows`, y at 2 * `rows`.
      The program reads a's base through CR0 (hardwired 0), so a is pinned at
      row 0 and [abi] binds it to CR0.
    - There is no add in the vector path: each addend passes through the
      multiplier against the scalar 1.0 (DTYPE_ONE, a CR; Q49 decodes it in the
      active format) and the accumulator sums them.
    - MULT start-of-cycle contract (issue #157): MULT.RC.VE reads R_CYCLIC as
      it was at the start of the cycle, so it cannot consume a row loaded by
      LD.MULT.CYCLIC in its own word. Each load runs one word ahead of the MULT
      that consumes it; the loads co-issue with other work, so the steady state
      is 4 cycles per row:
        Cycle 1: MULT.RC.VE x 1.0 on A[r] (loaded in the previous row's cycle 3);
                 ACC.ADD.FIRST; LD.MULT.CYCLIC -> r_cyclic = B[r]
        Cycle 2: MULT.RC.VE x 1.0 on B[r]; ACC.ADD
        Cycle 3: ACTIVATE.EXACT identity stores A[r]+B[r];
                 advance the row counter; LD.MULT.CYCLIC -> r_cyclic = A[r+1]
        Cycle 4: advance the output pointer; BLT.S -> loop
      Total: 4 cycles x rows, plus the setup words and one priming load.
    - LR forwarding: ADD fires before XMEM, so the A/B pointers start at -1
      (CR5) and read as 0 after their first ADD. Each pointer ADD co-issues with
      the load it feeds, so the cycle-3 prefetch addresses row r+1. ACTIVATE.EXACT reads
      out_ptr's forwarded value: do not ADD out_ptr in its word. BLT.S reads
      the value at the start of the cycle, so row_index advances the word before
      BLT.S.
    - ACTIVATE.EXACT activates and stores in one instruction (ADR 0007).
      identity with DSTRUCT's default DS (ds(128)) is a lane-for-lane FP32
      copy of all 128 lanes.

    Registers are named in the block below. The CRs [abi] binds come from
    kernel.toml (cr.<name>), so the program and its ABI cannot drift.
    Jinja runs before comment stripping: comments must not hold Jinja delimiters.
========================================================================== -#}

# ---------------------------------------------------------------------------
# Register names (Jinja2; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0   = "LR0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "LR1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "LR2"  %}  {# row offset into A (startup -1) #}
{% set b_ptr      = "LR3"  %}  {# row offset into B (startup -1) #}
{% set out_ptr    = "LR4"  %}  {# row offset into y, += 1 per row #}
{% set row_index  = "LR5"  %}  {# row counter 0..rows-1 #}
{% set row_limit  = "LR6"  %}  {# rows #}
{% set row_stride = "LR7"  %}  {# 1 = row stride for A and B #}
{% set out_stride = "LR8"  %}  {# 1 = output row stride #}

{% set A_BASE     = cr.a          %}  {# CR0: base row of a, hardwired 0 (a is pinned at row 0) #}
{% set ONE        = "CR1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE   = cr.y          %}  {# CR3: base row of y #}
{% set ZERO       = cr.zero       %}  {# CR4: const 0 #}
{% set PTR_START  = cr.ptr_start  %}  {# CR5: -1, startup value of the A/B row pointers #}
{% set ROW_COUNT  = cr.rows       %}  {# CR6: rows #}
{% set ROW_STRIDE = cr.row_stride %}  {# CR7: 1, A/B row stride #}
{% set OUT_STRIDE = cr.out_stride %}  {# CR8: 1, output row stride #}
{% set B_BASE     = cr.b          %}  {# CR9: base row of b #}
{% set DTYPE_ONE  = cr.one        %}  {# CR10: 1.0 in the active format, the pass-through multiplier #}
{% set DSTRUCT    = "CR15" %}  {# dstructure: default valid_elements = 128 #}

    SET                 {{ rc_slot0 }} {{ ZERO }};;
    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ a_ptr }} {{ PTR_START }};;
    SET                 {{ b_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ ROW_COUNT }};;
    SET                 {{ row_stride }} {{ ROW_STRIDE }};;
    SET                 {{ out_stride }} {{ OUT_STRIDE }};;
    # Prime the pipeline: load A[0] so row 0's cycle 1 sees it.
    LD.MULT.CYCLIC {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0  (A[r] was loaded a cycle earlier)
    #   MULT.RC.VE r_cyclic[0] × DTYPE_ONE (1.0) → A[r] passed through.
    #   Co-issued load fetches B[r] for cycle 2.
    LD.MULT.CYCLIC {{ b_ptr }} {{ B_BASE }} {{ rc_slot0 }};
    ADD {{ b_ptr }} {{ b_ptr }} {{ row_stride }};
    MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}.L {{ cr.qf }}.L;
    ACC.ADD.FIRST;;
    # Cycle 2: r_acc += B[r] × 1.0
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}.L {{ cr.qf }}.L;
    ACC.ADD;;
    # Cycle 3: ACTIVATE.EXACT identity stores r_acc: it activates and
    #   stores in one instruction, so the loop stays 4 cycles per row.
    #   `identity` + DSTRUCT's default DS (ds(128)) makes this a
    #   lane-for-lane FP32 copy of all 128 lanes.
    #   Co-issued load prefetches A[r+1] for the next row's cycle 1.
    ACTIVATE.EXACT   identity {{ DSTRUCT }}.L {{ OUT_BASE }} {{ out_ptr }};
    ADD {{ row_index }} {{ row_index }} {{ ONE }};
    LD.MULT.CYCLIC {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;
    # Cycle 4: advance output ptr; BLT.S reads the cycle-start row_index = already-incremented
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ out_stride }};
    BLT.S {{ row_index }} {{ row_limit }} row_loop;;

end:
    HALT;;
