# QKᵀ scores (Agent C), one attention head:
#
# Layer:   L3
# Scope:   single-stream
# Layout:  unpacked
# Shape:   256tok (2 groups x 128) x 36chan (head_dim), one head
# Status:  validated
# Related: qk_scores_64x48 (L4) / qk_scores_16x60 (L5) are ports of this
#          kernel; feeds attn_v_256x36 (query-major chain); attn_scores_km_256x36
#          is the key-major sibling for the same layer
# Tests:   test_qk_scores_256x36_wide (src/tools/ipu-apps/BUILD.bazel)
#
#   S[i, s] = sum_c Q[i, c] * K[s, c]   contraction over head_dim c = 0..35
#
# One head: N = 256 tokens (queries = keys), head_dim D = 36.
#   Q, K logically channel-major. K is loaded channel-major as given:
#     K[s, c] at K_BASE + c*256 + s  (a full channel column = 256 contiguous
#     bytes, loaded as two 128-key chunks: g=0 → +0, g=1 → +128).
#   Q is STAGED query-major by the harness (gather of the strided channels):
#     QROW[i] = Q[i, 0..35] contiguous at QROW_BASE + i*QROW_STRIDE.
#     This lets r0 hold one query's 36 channels with a single LDR_MULT_REG,
#     exactly the matmul broadcast template (scalar from r0 indexed by c).
#
# Broadcast template (mirrors matmul_144x144_x128 k-loop1, width 36):
#   per query i:  r0 = QROW[i]              (36 scalars Q[i, 0..35])
#   per channel:  r_cyclic = K[g*128:+128, c]   (128 keys' channel-c column)
#                 MULT.RC.VE: scalar Q[i,c] (= r0[c]) × vector r_cyclic
#                   → mult_res[s] = Q[i,c] * K[s,c]
#                 ACC.ADD.FIRST (c=0) / ACC.ADD      → R_ACC[s] += Q[i,c] * K[s,c]
#   after 36 channels R_ACC[s] = S[i, s] for the 128 keys of group g.
#   ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#                 → S[i, g] (512 B / 128 FP32|INT32 lanes, query-major)
#
# No AGG. The store goes through the standard quantize boundary, same as every
# other kernel: XMEM is 8-bit, so the downstream softmax reads these scores back
# through that boundary no matter what this kernel does. `identity` is the right
# activation here -- softmax applies its own nonlinearity.
#
# (An earlier version of this header claimed this kernel was special, on the
# grounds that enabling quantization would change its MEANING rather than just
# its precision. That was wrong: every kernel writes to the same 8-bit XMEM, and
# attn_scores_km_256x36 computes the same QK^T scores for the same softmax with
# no such caveat. Real per-lane requantize is not implemented yet -- see
# instruction_spec.py on ACTIVATE.QUANTIZE -- so precision behaviour here is
# simply whatever that scheme turns out to be, for this kernel and all others.)
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume a K column loaded by LDR_CYCLIC_MULT_REG in its own bundle -- it
# would see the previous column. `;;` ends one VLIW word = one cycle = one
# snapshot, so a load and a MULT in the same bundle always run in the same
# cycle regardless of textual order; co-issuing is fine, consuming the
# same-cycle load is not.
#   Each key group therefore primes c=0's column, then every loop body
#   multiplies the column loaded last cycle while prefetching the next one.
#   k_ptr (the load offset) is read LIVE by the load, so its ADD stays
#   co-issued with the load it feeds -- and because the load now runs one
#   bundle ahead, k_ptr naturally addresses the NEXT channel's column.
#   chan_index (the r0 scalar selector) is read LIVE by MULT and must name the
#   channel just loaded, so its ADD moves into a c_loop_*_pre block that falls
#   through into the loop body. That changes what BLT sees: with the ADD in the
#   same word, BLT read the pre-ADD snapshot; with the ADD one word earlier it
#   reads the already-incremented value and would exit a channel early. The
#   loop bound is therefore raised by one, in-kernel, into chan_last
#   (= chan_bound + 1 = D-1 = 35), leaving the harness's lr6 untouched.
#   The two key groups are independent contractions (each re-SETs k_ptr and
#   chan_index), so no pipeline state crosses the g=0/g=1 boundary.
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
{% set k_stride    = "lr2"  %}  {# 256 = bytes per K channel column #}
{% set out_stride  = "lr3"  %}  {# 1024 = bytes per query (2 key-groups x 512) #}
{% set k_ptr       = "lr4"  %}  {# byte offset into K, walks head channels c #}
{% set chan_index  = "lr5"  %}  {# head-channel index c -> selects Q[i,c] in r0 #}
{% set chan_bound  = "lr6"  %}  {# 34 = contraction bound (first=0, width=36) #}
{% set chan_last   = "lr11" %}  {# 35 = chan_bound + 1; BLT bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = "lr7"  %}  {# byte offset into S, += out_stride per query #}
{% set q_ptr       = "lr8"  %}  {# byte offset into staged QROW, += qrow_stride #}
{% set q_index     = "lr9"  %}  {# query counter i #}
{% set q_limit     = "lr10" %}  {# 256 = N queries #}
{% set qrow_stride = "lr12" %}  {# 512 = QROW_STRIDE bytes per query #}

{% set K_BASE      = "cr0"  %}  {# channel-major K base #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE_G0   = "cr3"  %}  {# S base, key group 0 #}
{% set S_BASE_G1   = "cr4"  %}  {# S base + 512, key group 1 #}
{% set K_START_G0  = "cr5"  %}  {# -256 startup skew (first live = 0) #}
{% set K_START_G1  = "cr6"  %}  {# -128 startup skew (first live = 128) #}
{% set CHAN_START  = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set QROW_BASE   = "cr9"  %}  {# staged query-major Q base #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 35). See the snapshot note in the
    # header: chan_index's ADD now sits one word ahead of the BLT, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

q_loop:
    LDR_MULT_REG r0 {{ q_ptr }} {{ QROW_BASE }};;                                               # r0 = QROW[i] = Q[i, 0..35] (rest pad)

    # -- key group 0 ---------------------------------------------------------
    SET {{ k_ptr }} {{ K_START_G0 }};;                                                          # g=0 K-data startup: -256
    SET {{ chan_index }} {{ CHAN_START }};;                                                     # channel fixed_idx startup: -1

    # Prime c=0's K column: MULT reads r_cyclic from the snapshot, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;
    B after_c_g0;;

c_loop_g0_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop_g0:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;

after_c_g0:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE_G0 }};;  # store 512B → S[i, keys 0..127]

    # -- key group 1 ---------------------------------------------------------
    SET {{ k_ptr }} {{ K_START_G1 }};;                                                          # g=1 K-data startup: -128
    SET {{ chan_index }} {{ CHAN_START }};;                                                     # channel fixed_idx startup: -1

    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;
    B after_c_g1;;

c_loop_g1_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop_g1:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;

after_c_g1:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE_G1 }};;  # store 512B → S[i, keys 128..255]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;                                          # advance output ptr (+1024)

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }};
    ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query: Q ptr += 512, i++
    BLT {{ q_index }} {{ q_limit }} q_loop;;

end:
    BKPT;;
