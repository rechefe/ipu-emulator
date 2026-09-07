# Packed LayerNorm, L5 shape: 240 channels x 16 tokens, PACKED 8 channels
# per 128-lane row (channel p's 16 tokens at lanes [16p, 16p+16)), 30 packed
# rows (240/8, exact). Row r holds channels [8r, 8r+8).
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed
# Shape:   240ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_layernorm_192x64.asm (L4 counterpart, same algorithm shape)
# Tests:   test_packed_layernorm_240x16.py
#
# Unlike the elementwise packed kernels (residual add), LayerNorm reduces
# over the CHANNEL axis per token -- in packed layout that is a reduction
# ACROSS ROWS *and* across the 8 partitions within each row. Six steps,
# mirroring layernorm_16x240.asm's algorithm shape but each cross-partition
# step re-expressed with the masked MULT.RC.VE gather/scatter construction
# validated by asm_packed_output_linear_generic.asm (rc_idx picks the READ
# window, mask_offset picks the WRITE window; ACC.ADD without reset lets
# disjoint windows share one r_acc).
#
#   Step 1 (SUM):     for chunk c=0..29, p_in=0..7: gather packed chunk c's
#                     partition p_in (16 tokens) into lanes 0-15 via
#                     rc_idx=16*p_in, mask_offset=0 (one-hot lanes 0-15),
#                     scaled by -1/240 via MULT.RC.VV against an all-lanes
#                     -1/240 row loaded into R0 (MULT.RC.VE's CR-scalar path
#                     truncates to a signed BYTE -- ipu.py
#                     _wide_cr_scalar_byte_as_int32 -- so a fractional
#                     constant must come from an XMEM row, same as
#                     layernorm_16x240.asm's NEG_INV_N_BASE). ACC.ADD
#                     (.FIRST once, at c=0,p_in=0) accumulates all 240
#                     channels' contributions into r_acc lanes 0-15. Store
#                     (valid_elements=16) -> NEG_MEAN row.
#
#   Step 2 (BCAST1):  broadcast NEG_MEAN's 16 values into a 128-lane tile
#                     (every 16-lane window p=0..7 holds the same 16 values)
#                     via rc_idx=(-16*p) mod 512, mask_offset=p, ACC.ADD
#                     (.FIRST at p=0). This is NOT rc_idx=(-16*p) mod 128 --
#                     that formula (the original task brief's wording) is
#                     WRONG; see docs/isa_friction_log.md for the
#                     derivation. Store (valid_elements=128) -> NEG_MEAN_TILE.
#
#   Step 3 (CENTER):  centered[c] = x[c] + NEG_MEAN_TILE, partition-local
#                     elementwise. NEG_MEAN_TILE is loaded ONCE into R_CYCLIC
#                     slot 128 before the loop (constant across all 30
#                     chunks); x[c] cycles through slot 0. Two MULT.RC.VE(x1)
#                     + ACC.ADD/.FIRST per chunk (one per slot), same 2-term
#                     shape as asm_packed_residual_add_240x16.asm's A+B, one
#                     store per chunk.
#
#   Step 4 (SUMSQ):   same cross-partition-gather shape as step 1, but the
#                     source is `centered` and the per-lane op is
#                     MULT.RC.VS (square in place, no scalar) instead of
#                     MULT.RC.VV. Raw (unscaled) sum of squares accumulates
#                     into r_acc lanes 0-15, stored to TEMP. A separate
#                     single MULT.RC.VV by an all-lanes 1/240 row + rsqrt
#                     activation turns the raw sum into 1/sigma -- same
#                     split as layernorm_16x240.asm's step3+step4. Store ->
#                     INVSTD row.
#
#   Step 5 (BCAST2):  broadcast INVSTD the same way as step 2.
#                     -> INVSTD_TILE (128 lanes).
#
#   Step 6 (AFFINE):  output[c] = centered[c]*INVSTD_TILE*GAMMA_TILE[c] +
#                     BETA_TILE[c], partition-local elementwise.
#                     GAMMA_TILE/BETA_TILE are PRE-REPLICATED by the harness
#                     at load time (240 scalars each, tiled 16x per channel
#                     into the matching packed-row/window layout) -- legal
#                     per the brief ("harness may choose what to load"); the
#                     per-channel gamma/beta values never change within a
#                     run, so no masked pass is needed here.
#
# All six steps operate purely on XMEM rows already resident before the
# kernel starts, or produced by an EARLIER step of the SAME kernel run --
# no host-side conversion between steps, no numpy in the loop.
#
# RUNTIME CHUNK LOOPS (not fully unrolled): a first cut fully unrolled every
# per-chunk step (Jinja `for c in range(30)`) -- at 240 channels that
# produced 1954 static instructions, blowing the 1024-instruction
# program-memory ceiling ~1.9x over (same lesson
# asm_packed_output_linear_generic.asm already logged for its own
# per-chunk loop). Steps 1/3/4a/6 are therefore genuine runtime BLT loops
# over the chunk counter; only the 8-way partition unroll WITHIN one
# chunk (steps 1/4a) stays statically unrolled, since it is fixed-size
# and small (8 x ~2 static instructions per chunk-loop body).
#
# PEELED FIRST CHUNK (steps 1, 3, 4a, 6): ACC.ADD.FIRST must fire exactly
# once for the whole kernel run, not once per loop iteration -- a runtime
# loop body re-executes the same static instructions every pass, so a
# compile-time "c==0" guard inside the loop body would incorrectly
# re-fire .FIRST (and reset the accumulator) on EVERY chunk. Fixed the
# same way asm_packed_output_linear_generic.asm handles it: chunk 0 is
# unrolled once outside any loop with .FIRST at its one true first
# position; a runtime loop then covers chunks 1..29.
#
# VLIW rules: same MULT-snapshot contract as every other kernel here (a
# same-bundle load is not visible to that bundle's MULT; every loop primes
# its first row before entry and prefetches the next row from inside the
# body). Loop counters seeded to 1, tested with BLT against the
# pre-increment snapshot -- see layernorm_16x240.asm's header for the
# shared rationale, unchanged here.
#
# CR budget: 16 total (cr0-cr15), all committed -- see the register-name
# block below. NEG_INV_N_BASE and INV_N_BASE reuse NEG_MEAN_TILE's and
# INVSTD_TILE's XMEM rows AS SCRATCH before those rows hold their real
# tiled broadcast content: step 1 needs -1/240 before NEG_MEAN_TILE exists
# (step 2 produces it), and step 4b needs 1/240 after the sum-of-squares
# scratch is consumed but before INVSTD_TILE exists (step 5 produces it)
# -- both windows are safe because nothing else reads those rows' PRIOR
# content at that point. SCRATCH16 similarly consolidates every
# 16-value-wide intermediate (-mean, raw sum-of-squares, 1/sigma) into one
# physical row -- see the register-name block's note for the full
# lifetime audit. Only ONE 128-wide dstructure register (DSTRUCT128) is
# needed alongside DSTRUCT16: mask_shift is held at 0 throughout (only
# mask_offset, an immediate 0-7 selecting an R_MASK slot, varies -- a
# different mechanism), and ipu.py's _mult_mask_and_shift only reads
# CR[cr_idx].partition when shift != 0, so `partition` never matters here.

{% set rc_idx_reg   = "lr0"  %}  {# r_cyclic base ELEMENT offset for masked gather/scatter/multiply #}
{% set mask_off_lr  = "lr1"  %}  {# mask_offset carrier / LDR_MULT_MASK_REG offset / LDR_CYCLIC_MULT_REG index carrier #}
{% set read_ptr     = "lr2"  %}  {# per-step row read pointer (offset from a ZERO base -- absolute row number) #}
{% set write_ptr    = "lr3"  %}  {# per-step row write pointer #}
{% set gb_ptr       = "lr4"  %}  {# gamma/beta row pointer in step 6, recomputed each chunk from base + running chunk offset #}
{% set row_idx      = "lr5"  %}  {# packed-row / chunk counter (runtime loop induction variable) #}
{% set row_limit    = "lr6"  %}  {# runtime loop bound, set per step #}
{% set row_stride   = "lr7"  %}  {# 1 = row stride #}
{% set mask_shift0  = "lr8"  %}  {# const 0: mask_shift (no partition-boundary shifting used) #}
{% set slot0_lr     = "lr9"  %}  {# const 0: LDR_CYCLIC_MULT_REG index for slot 0 #}
{% set slot128_lr   = "lr10" %}  {# const 128: LDR_CYCLIC_MULT_REG index for slot 128 #}

{#- SCRATCH16 is ONE physical row shared by every 16-value-wide intermediate
    (-mean, raw sum-of-squares, 1/sigma): each write fully supersedes the
    previous occupant before the next read of a DIFFERENT logical value
    happens -- step1 writes it as NEG_MEAN, step2 reads+exhausts it;
    step4a writes it as raw sumsq, step4b reads+exhausts it THEN writes it
    again as INVSTD;
    step5 reads+exhausts it. This consolidation is what;
    frees the CR budget for a SECOND dstructure register (DSTRUCT16/
    DSTRUCT128) -- ACTIVATE.QUANTIZE/MULT.RC.* read a genuine CR
    (DstructureCrIdx), which no instruction can rewrite at runtime, so two
    different valid_elements settings alive in the same kernel need two
    distinct CRs, not one reused with a different runtime value. #}
{% set ZERO            = "cr0"  %}  {# const 0 #}
{% set ONE             = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE       = "cr2"  %}  {# base row of X, 30 packed rows #}
{% set ROW_COUNT       = "cr3"  %}  {# 30 #}
{% set ROW_STRIDE      = "cr4"  %}  {# 1 #}
{% set SCRATCH16       = "cr5"  %}  {# shared 16-value scratch row -- see note above #}
{% set NEG_MEAN_TILE   = "cr6"  %}  {# NEG_MEAN_TILE row (128 lanes); also NEG_INV_N scratch pre-step-2 #}
{% set CENTERED_BASE   = "cr7"  %}  {# 30 packed rows, centered #}
{% set INVSTD_TILE     = "cr8"  %}  {# 1/sigma tile (128 lanes); also INV_N scratch pre-step-5 #}
{% set GAMMA_TILE_BASE = "cr9"  %}  {# 30 packed rows, gamma pre-replicated #}
{% set BETA_TILE_BASE  = "cr10" %}  {# 30 packed rows, beta pre-replicated #}
{% set OUTPUT_BASE     = "cr11" %}  {# 30 packed rows, output #}
{% set MASK_BASE       = "cr12" %}  {# R_MASK source row: 8 one-hot 16-lane slots #}
{% set DSTRUCT16       = "cr13" %}  {# dstructure: valid_elements=16 #}
{% set DSTRUCT128      = "cr14" %}  {# dstructure: valid_elements=128 (shared by every masked and unmasked 128-lane step -- see header note) #}
{% set ALLONES_MASK_BASE = "cr15" %}  {# R_MASK source row: all 1024 bits set -- see note below #}

{#- R_MASK is a single 1024-bit register (8 x 128-bit slots); LDR_MULT_MASK_REG
    replaces ALL of it in one call, and there is no default it reverts to
    between calls within a kernel run (only IpuState's own INITIAL value is
    all-ones -- ipu.py's RegFile.__init__ -- which this kernel's own
    LDR_MULT_MASK_REG for the one-hot mask overwrites early on). Steps 1/2/
    4a/5 need the one-hot 8-slot mask (MASK_BASE) to select a 16-lane
    gather/scatter window;
    steps 3/6 are ordinary full-row elementwise;
    passes that need EVERY lane active, which the one-hot mask's slot 0
    (bits [0,16) only) does NOT provide -- an earlier draft used MASK_BASE's
    slot 0 for steps 3/6 too and silently zeroed partitions 1-7 of every
    elementwise result (only partition 0's window passed the mask), caught
    by comparing packed step-3 output against numpy per-channel. Fixed by
    maintaining a SECOND mask row (ALLONES_MASK_BASE, all 1024 bits set)
    and reloading R_MASK from it at the start of steps 3 and 6. #}

{% set NEG_MEAN_BASE  = SCRATCH16 %}
{% set TEMP_BASE      = SCRATCH16 %}
{% set INVSTD_BASE    = SCRATCH16 %}
{% set NEG_INV_N_BASE = NEG_MEAN_TILE %}
{% set INV_N_BASE     = INVSTD_TILE %}
{% set N_PACKED_ROWS  = 30 %}
{% set SLOT_STRIDE    = 128 %}

{% macro inc(dest, n) %}
{%- set n = n % 512 -%}
{%- if n != 0 -%}
{%- for chunk in ((n // 255) * [255] + ([n % 255] if n % 255 else [])) %}
    INC {{ dest }} {{ chunk }};;
{%- endfor -%}
{%- endif -%}
{% endmacro %}

    SET {{ mask_off_lr }} {{ ZERO }};;
    SET {{ mask_shift0 }} {{ ZERO }};;
    SET {{ row_stride }} {{ ROW_STRIDE }};;
    SET {{ slot0_lr }} {{ ZERO }};;
    SET {{ slot128_lr }} {{ ZERO }};;
    {{- inc(slot128_lr, SLOT_STRIDE) }}
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;

# -----------------------------------------------------------------------
# Step 1: -mean[t] = sum over 240 channels of x[ch,t] * (-1/240)
#
# Chunk 0 unrolled/peeled outside the loop (the one true ACC.ADD.FIRST
# position); a runtime loop then covers chunks 1..29.
# -----------------------------------------------------------------------

    LDR_MULT_REG r0 {{ mask_off_lr }} {{ NEG_INV_N_BASE }};;   # r0 <- -1/240 (every lane)

    SET {{ read_ptr }} {{ DATA_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime chunk 0 into slot 0

{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    {%- set is_first = (p_in == 0) %}
    {%- set is_last_p = (p_in == 7) %}
    {%- if is_last_p %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1: row_idx enters the loop already at 1 with no BLT check performed by the peel itself (unlike layernorm_16x240.asm, whose peel's own trailing bundle performs the FIRST ch_index/BLT check before the loop is ever entered) -- see the loop-count derivation in docs/isa_friction_log.md

step1_loop:
{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    {%- if loop.last %}
    {#- row_idx's ADD and the BLT must be in the SAME bundle as this trailing
        load, mirroring layernorm_16x240.asm's step1 loop exactly (line 98/
        102: 'LDR_CYCLIC...;
        ADD read_ptr...;
        ADD ch_index...;
        BLT ch_index;
        ch_limit') -- BLT's operands are snapshot reads (instruction_spec.py),
        so it sees row_idx's value from BEFORE this bundle's own ADD. An
        earlier draft put row_idx's ADD/BLT in a SEPARATE bundle one cycle
        after this load: that shifts the snapshot BLT sees by one full
        cycle relative to the reference kernel's fused structure, making
        the loop run ONE EXTRA iteration (248 MULT.RC.VV calls observed for
        30 chunks x 8 partitions = 240 expected) -- it silently accumulated
        a nonexistent 31st packed chunk (uninitialized/zero XMEM), caught
        by comparing the dynamic MULT call count against the expected
        240, not by the numeric result alone (zero-valued garbage made the
        error small enough to look like float noise at this step, though
        it corrupted a real neighboring row via an out-of-bounds STORE
        later in step 3). #}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step1_loop;;
    {%- else %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {%- endif %}
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ NEG_MEAN_BASE }};;

# -----------------------------------------------------------------------
# Step 2: broadcast NEG_MEAN (16 values) into a 128-lane tile, one 16-lane
# window per partition p=0..7. rc_idx = (-16*p) mod 512 (NOT mod 128 --
# see header note / docs/isa_friction_log.md). Fixed 8-way unroll, no
# runtime loop needed (8 static instructions, negligible).
# -----------------------------------------------------------------------

    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ NEG_MEAN_BASE }} {{ slot0_lr }};;

{%- for p in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - 16 * p) % 512) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if p == 0 else "" }};;
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ NEG_MEAN_TILE }};;

# -----------------------------------------------------------------------
# Step 3: centered[c] = x[c] + NEG_MEAN_TILE, partition-local elementwise.
# NEG_MEAN_TILE loaded once into slot 128 (constant); x[c] cycles slot 0.
# Chunk 0 peeled; runtime loop covers chunks 1..29.
# -----------------------------------------------------------------------

    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ ALLONES_MASK_BASE }};;   # unmasked full-row pass -- see header note on the two mask rows
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ NEG_MEAN_TILE }} {{ slot128_lr }};;

    SET {{ read_ptr }} {{ DATA_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: STR_POST_AAQ_REG's offset is a LIVE read and a same-bundle ADD (lr slot) dispatches before store, so the store sees the POST-increment value -- the -1 startup offset compensates, same convention layernorm_16x240.asm uses throughout
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime chunk 0 into slot 0

    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    MULT.RC.VE {{ slot128_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {#- read_ptr's advance MUST co-issue in the SAME bundle as the load that
        reads it here (like the priming load two lines up), not the
        following bundle -- LR-slot writes dispatch before load, so a
        same-bundle ADD is what makes THIS load see the ADVANCED offset.
        An earlier draft put this ADD in the NEXT bundle (mirroring
        write_ptr's own advance, which correctly happens in the STORE's
        bundle since store's live-read wants the POST-increment value --
        see the note above): that left read_ptr one step behind for every
        LOAD, so this load kept re-reading chunk 0's row on what should
        have been chunk 1's iteration, silently shifting every subsequent
        chunk by one and never loading the true last chunk. Caught by
        comparing packed step-3 output row-by-row against numpy (every row
        but the first was wrong by roughly one full row's worth of error).
        The "lr" slot allows 3 independent sub-slots per cycle
        (instruction_spec.py's SLOT_COUNT), so this fits alongside
        write_ptr's advance in the same store-bundle. #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1: row_idx enters the loop already at 1 with no BLT check performed by the peel itself (unlike layernorm_16x240.asm, whose peel's own trailing bundle performs the FIRST ch_index/BLT check before the loop is ever entered) -- see the loop-count derivation in docs/isa_friction_log.md

step3_loop:
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    MULT.RC.VE {{ slot128_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {#- write_ptr, read_ptr, AND row_idx's advances, plus the BLT, must ALL
        land in this ONE bundle -- exactly layernorm_16x240.asm's step1
        pattern (load + ch_index-ADD + BLT fused in one bundle). Splitting
        row_idx's ADD/BLT into a separate trailing bundle (an earlier
        draft's mistake) inserts an extra cycle between the load and the
        loop-exit check that BLT's SNAPSHOT read doesn't compensate for,
        making the loop run ONE EXTRA pass -- caught by counting STR_POST_
        AAQ_REG calls to CENTERED_BASE (31 observed for 30 expected rows;
        the 31st write landed on row CENTERED_BASE+30, silently corrupting
        whatever tensor's row happened to sit there next, in this case
        INVSTD_TILE). The "lr" slot has exactly 3 sub-slots per cycle
        (instruction_spec.py SLOT_COUNT) -- write_ptr+read_ptr+row_idx
        uses all three, with no room to spare. #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step3_loop;;

# -----------------------------------------------------------------------
# Step 4a: raw sum of squares (unscaled) -- same gather shape as step 1,
# source = centered, op = MULT.RC.VS (square in place). Same peel/loop
# split as step 1.
# -----------------------------------------------------------------------

    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;   # back to the one-hot mask -- step 3 switched to all-ones
    SET {{ read_ptr }} {{ CENTERED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime chunk 0 into slot 0

{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    {%- set is_first = (p_in == 0) %}
    {%- set is_last_p = (p_in == 7) %}
    {%- if is_last_p %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1: row_idx enters the loop already at 1 with no BLT check performed by the peel itself (unlike layernorm_16x240.asm, whose peel's own trailing bundle performs the FIRST ch_index/BLT check before the loop is ever entered) -- see the loop-count derivation in docs/isa_friction_log.md

step4a_loop:
{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    {%- if loop.last %}
    {#- read_ptr's advance AND row_idx's ADD+BLT fused into this ONE
        bundle -- see step1's identical note on why splitting them causes
        an off-by-one extra loop pass. #}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step4a_loop;;
    {%- else %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {%- endif %}
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};;

# -----------------------------------------------------------------------
# Step 4b: 1/sigma = rsqrt((1/240) * sum_of_squares)
# -----------------------------------------------------------------------

    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ INV_N_BASE }} {{ slot0_lr }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;

    ACTIVATE.QUANTIZE rsqrt {{ DSTRUCT16 }};;
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ INVSTD_BASE }};;

# -----------------------------------------------------------------------
# Step 5: broadcast INVSTD (16 values) into a 128-lane tile, same
# construction as step 2.
# -----------------------------------------------------------------------

    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ INVSTD_BASE }} {{ slot0_lr }};;

{%- for p in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - 16 * p) % 512) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if p == 0 else "" }};;
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ INVSTD_TILE }};;

# -----------------------------------------------------------------------
# Step 6: output[c] = centered[c] * INVSTD_TILE * GAMMA_TILE[c] + BETA_TILE[c]
#
# The two MULTIPLICATIONS (by invstd_tile, by gamma_tile[c]) both use
# MULT.RC.VV(rc_idx=slot0, ra=R0/R1) -- R_CYCLIC[i]*Ra[i] in one
# instruction -- NOT two MULT.RC.VE(x1)+ACC.ADD passes: an earlier draft
# used the latter for the invstd multiply (copying step 3's construction,
# which is correct there only because step 3's operation IS an addition,
# x+neg_mean_tile) and got x+invstd instead of x*invstd (caught by
# comparing r_acc directly against the expected product: kernel gave
# -1.9153+0.5930=-1.3223, not -1.9153*0.5930=-1.1358). INVSTD_TILE is
# loaded ONCE into R1 (a full 128-lane mult-stage register, via
# LDR_MULT_REG -- constant across chunks, so R_CYCLIC slot 128 is not
# needed for it at all in this step); normalized[c] is staged through
# SCRATCH16 into R0 for the gamma multiply exactly as before. The final
# ADDITION (+ beta_tile[c]) correctly stays a MULT.RC.VE(x1)+ACC.ADD pass.
# centered[c], gamma_tile[c], beta_tile[c] all vary per chunk and share
# slot 0 sequentially within one chunk's own bundle sequence, each
# consumed one bundle after its own load, same snapshot-timed ping-pong
# asm_packed_residual_add_240x16.asm uses for its two varying inputs.
# SCRATCH16 (dead after step 4b) is reused as a single scratch row,
# overwritten and immediately reconsumed every chunk iteration (never
# read across iterations, so one row suffices, not 30). gb_ptr is
# recomputed from a base CR (GAMMA_TILE_BASE / BETA_TILE_BASE) plus
# row_idx (the SAME counter driving the loop -- row_idx already equals
# the current chunk index at every point BEFORE its own trailing-bundle
# increment, so no separate offset register is needed; an earlier draft
# used a dedicated gb_chunk_off register incremented independently, which
# either needed a 4th "lr" sub-slot in the final bundle (over the 3-slot
# budget) or, when moved to an earlier bundle to make room, incremented
# too soon and fed the WRONG offset into the very iteration meant to
# consume it). Per-chunk bundle sequence:
#   1. MULT.RC.VV(slot0=centered[c], ra=r1=invstd_tile); ACC.ADD.FIRST -> normalized[c]
#   2. AAQ store r_acc -> SCRATCH16;           SET gb_ptr <- GAMMA_TILE_BASE
#   3. ADD gb_ptr, gb_ptr, row_idx
#   4. LDR_CYCLIC(gb_ptr -> slot0)  [gamma_tile[c]]
#   5. LDR_MULT_REG r0 <- SCRATCH16 (normalized[c])
#   6. MULT.RC.VV(slot0=gamma_tile[c], ra=r0=normalized[c]); ACC.ADD.FIRST -> normalized*gamma
#   7. AAQ store r_acc -> SCRATCH16;           SET gb_ptr <- BETA_TILE_BASE
#   8. ADD gb_ptr, gb_ptr, row_idx
#   9. LDR_CYCLIC(gb_ptr -> slot0)  [beta_tile[c]]
#  10. MULT.RC.VE(slot0=beta_tile[c]) x1;      ACC.ADD          -> + beta = output[c]
#  11. AAQ store r_acc -> OUTPUT_BASE[c]; advance pointers, row_idx, and
#      branch, all fused in one bundle (see step1/step3/step4a's note on
#      why the loop counter's ADD and BLT must share the closing load's
#      bundle -- splitting them causes an extra pass).
#
# Chunk 0 peeled (row_idx == 0 throughout the peel, set explicitly before
# it and only advanced to 1 once entering the loop, so its gamma/beta
# pointers land exactly at the tile bases); runtime loop covers chunks
# 1..29.
# -----------------------------------------------------------------------

    {#- CORRECTION: normalized[c] = centered[c] * invstd_tile is a genuine
        elementwise TENSOR PRODUCT, not a sum -- an earlier draft computed
        it as two MULT.RC.VE(x1)+ACC.ADD passes (one per operand), which
        is A+B, not A*B (ACC.ADD always ADDS mult_res into r_acc;
        there is;
        no elementwise-multiply accumulate). Caught by comparing r_acc
        directly against the expected product after the first chunk:
        kernel gave centered[0,0]+invstd[0] = -1.9153+0.5930 = -1.3223,
        not centered[0,0]*invstd[0] = -1.1358. Fixed by loading
        INVSTD_TILE into R1 (a full 128-lane mult-stage register, via
        LDR_MULT_REG, not R_CYCLIC) ONCE before the loop -- it is constant
        across all chunks -- then using MULT.RC.VV(rc_idx=slot0,
        ra=r1)+ACC.ADD.FIRST, which computes R_CYCLIC[i]*R1[i] directly in
        ONE instruction per chunk (the same mechanism GAMMA_TILE[c]'s
        multiply already used correctly, since that one multiplies
        R_CYCLIC by R0 via MULT.RC.VV rather than by two ACC.ADD passes).
        slot 128 of R_CYCLIC is no longer needed in this step. #}
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ ALLONES_MASK_BASE }};;   # unmasked full-row pass -- see header note on the two mask rows
    LDR_MULT_REG r1 {{ mask_off_lr }} {{ INVSTD_TILE }};;

    SET {{ read_ptr }} {{ CENTERED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: see step 3's note on STR_POST_AAQ_REG's live-read/same-bundle-ADD timing
    SET {{ row_idx }} {{ ZERO }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime centered[0] into slot 0

    # ---- chunk 0 (peeled): row_idx == 0, pointers land at the tile bases ----
    MULT.RC.VV {{ slot0_lr }} r1 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ GAMMA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ BETA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {#- read_ptr's advance co-issues with the load that reads it (same fix
        as step 3). row_idx advances to 1 here (its own dedicated bundle,
        not fused with anything else -- the peel is a one-off, not a loop
        iteration, so there is no BLT to fuse it with). #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;

    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1: row_idx enters the loop already at 1 with no BLT check performed by the peel itself (unlike layernorm_16x240.asm, whose peel's own trailing bundle performs the FIRST ch_index/BLT check before the loop is ever entered) -- see the loop-count derivation in docs/isa_friction_log.md

step6_loop:
    MULT.RC.VV {{ slot0_lr }} r1 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ GAMMA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ BETA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {#- write_ptr, read_ptr, AND row_idx's advances plus BLT all fused
        into this one bundle -- exactly 3 "lr" ops, the per-cycle maximum
        -- see step1/step3/step4a's identical note on why row_idx's own
        ADD must share the closing load's bundle with the BLT that reads
        its pre-increment snapshot. #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step6_loop;;

end:
    BKPT;;
