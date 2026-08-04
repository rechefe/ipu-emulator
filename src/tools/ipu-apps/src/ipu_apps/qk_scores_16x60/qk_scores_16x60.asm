# QKᵀ scores (Layer 5), one attention head:
#   S[i, s] = sum_c Q[i, c] * K[s, c]   contraction over head_dim c = 0..59
#
# One head: N = 16 tokens (queries = keys), head_dim D = 60.
# This is the L5 port of qk_scores_256x36. The MAPPING is unchanged; only the
# loop counts move:
#   * D 36 -> 60. head_dim is a contraction LOOP COUNT here (chan_bound = D-2),
#     not a lane count, so 60 needs no padding.
#   * N 256 -> 16. 256 keys spanned two 128-lane key groups; 16 keys fit in ONE
#     group, so the entire g=1 half of the L3 kernel is gone.
#
# ONE CHANNEL PER ROW: a K channel column is 16 live FP32 elements inside a
# WHOLE 512-B row, never packed with a neighbouring channel.
#
#   Q, K logically channel-major. K is loaded channel-major as given:
#     K[s, c] at K_BASE + c*K_STRIDE rows (a full channel column, 16 keys,
#     loaded as ONE chunk into r_cyclic).
#   Q is STAGED query-major by the harness (gather of the strided channels):
#     QROW[i] = Q[i, 0..59] contiguous at QROW_BASE + i*QROW_STRIDE.
#     This lets r0 hold one query's 60 channels with a single LDR_MULT_REG,
#     exactly the matmul broadcast template (scalar from r0 indexed by c).
#
# Broadcast template (mirrors matmul k-loop1, width 60):
#   per query i:  r0 = QROW[i]              (60 scalars Q[i, 0..59])
#   per channel:  r_cyclic = K[0..15, c]        (16 keys' channel-c column)
#                 MULT.RC.VE: scalar Q[i,c] (= r0[c]) x vector r_cyclic
#                   -> mult_res[s] = Q[i,c] * K[s,c]
#                 ACC.ADD.FIRST (c=0) / ACC.ADD  -> R_ACC[s] += Q[i,c] * K[s,c]
#   after 60 channels R_ACC[s] = S[i, s] for the 16 keys.
#   ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#                 -> S[i] (one whole row, first 16 lanes live, query-major)
#
# No AGG. The store goes through the standard quantize boundary, same as every
# other kernel; `identity` is the right activation here -- softmax applies its
# own nonlinearity downstream. cr15.valid_elements = 16 narrows the AAQ window
# to the live keys.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume a K column loaded by LDR_CYCLIC_MULT_REG in its own bundle -- it
# would see the previous column. `;;` ends one VLIW word = one cycle = one
# snapshot, so a load and a MULT in the same bundle always run in the same
# cycle regardless of textual order; co-issuing is fine, consuming the
# same-cycle load is not.
#   The key group therefore primes c=0's column, then every loop body
#   multiplies the column loaded last cycle while prefetching the next one.
#   k_ptr (the load offset) is read LIVE by the load, so its ADD stays
#   co-issued with the load it feeds -- and because the load now runs one
#   bundle ahead, k_ptr naturally addresses the NEXT channel's column.
#   chan_index (the r0 scalar selector) is read LIVE by MULT and must name the
#   channel just loaded, so its ADD moves into a c_loop_pre block that falls
#   through into the loop body. That changes what BLT sees: with the ADD in the
#   same word, BLT read the pre-ADD snapshot; with the ADD one word earlier it
#   reads the already-incremented value and would exit a channel early. The
#   loop bound is therefore raised by one, in-kernel, into chan_last
#   (= chan_bound + 1 = D-1 = 59), leaving the harness's lr6 untouched.
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
{% set rc_slot0    = "lr0"  %}  {# const 0: r_cyclic write-index / mask_shift #}
{% set k_stride    = "lr2"  %}  {# rows per K channel column #}
{% set out_stride  = "lr3"  %}  {# rows per query in S (1 key group x 1 row) #}
{% set k_ptr       = "lr4"  %}  {# row offset into K, walks head channels c #}
{% set chan_index  = "lr5"  %}  {# head-channel index c -> selects Q[i,c] in r0 #}
{% set chan_bound  = "lr6"  %}  {# 58 = contraction bound (first=0, width=60) #}
{% set chan_last   = "lr11" %}  {# 59 = chan_bound + 1; BLT bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = "lr7"  %}  {# row offset into S, += out_stride per query #}
{% set q_ptr       = "lr8"  %}  {# row offset into staged QROW, += qrow_stride #}
{% set q_index     = "lr9"  %}  {# query counter i #}
{% set q_limit     = "lr10" %}  {# 16 = N queries #}
{% set qrow_stride = "lr12" %}  {# 1 = QROW stride in rows per query #}

{% set K_BASE      = "cr0"  %}  {# channel-major K base #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = "cr3"  %}  {# S base (single key group) #}
{% set K_START     = "cr5"  %}  {# -1 row startup skew (first live = row 0) #}
{% set CHAN_START  = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set QROW_BASE   = "cr9"  %}  {# staged query-major Q base #}
{% set DSTRUCT     = "cr15" %}  {# dstructure register: valid_elements = 16 #}

    # chan_last = chan_bound + 1 (= D-1 = 59). See the snapshot note in the
    # header: chan_index's ADD now sits one word ahead of the BLT, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

q_loop:
    LDR_MULT_REG r0 {{ q_ptr }} {{ QROW_BASE }};;                                               # r0 = QROW[i] = Q[i, 0..59] (rest pad)

    # -- the single key group (keys 0..15) -----------------------------------
    SET {{ k_ptr }} {{ K_START }};;                                                             # K-data startup: -1 row
    SET {{ chan_index }} {{ CHAN_START }};;                                                     # channel fixed_idx startup: -1

    # Prime c=0's K column: MULT reads r_cyclic from the snapshot, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }}; ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }}; ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }}; ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }}; STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE }};;# store one row -> S[i, keys 0..15]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;                                          # advance output ptr (+1 row)

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }}; ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query: Q ptr += 1 row, i++
    BLT {{ q_index }} {{ q_limit }} q_loop;;

end:
    BKPT;;
