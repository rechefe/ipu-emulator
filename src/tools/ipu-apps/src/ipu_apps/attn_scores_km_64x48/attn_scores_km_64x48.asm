# kQ^T -> key-major attention scores (Layer 4), one selected head, all streams
#
# Per stream p in [0,4), for the head the harness selected:
#   S[p, i, s] = sum_c Q[p, i, c] * K[p, s, c]     i,s in [0,64), c in [0,48)
#     Lanes = QUERIES (i); outer loop = key (s); contraction = head channel (c).
#
# Layer 4: d = 192, N = 64 tokens/stream, P = 4 streams, h = 4 heads,
# head_dim D = 48. head_dim = 48 needs NO padding: it is the contraction LOOP
# COUNT (bound chan_bound = D-2 = 46), not a lane count.
#
# This is the L4 port of attn_scores_km_256x36. The mapping is carried over
# unchanged -- broadcast template, no AGG, no MULT.RC.VV score form:
#   Q channel-major: Q[p, i, c] on row (p*48 + c), queries 0..63 in the first 64
#     lanes -> loads straight into R_CYCLIC (the vector operand).
#   K key-major scratch (built by the harness): K[p, s, 0:48] on row (p*64 + s)
#     -> one LDR_MULT_REG pulls a key's whole head-channel vector into R0 (the
#     scalar operand, indexed by the contraction channel c).
#   S key-major: key (p, s) owns row (p*64 + s); its leading 64 lanes are the
#     query scores S[p, 0:64, s]. The harness crops them in teardown -- rows are
#     NEVER shared between keys.
#
# N = 64 <= LANES = 128 is the one structural difference from L3: a query axis
# fits in a SINGLE row, so the L3 g=0/g=1 query-group split collapses to one
# group (N_TG = 1) and each key needs exactly one MULT/ACC chain and one store.
#
# Compute (matmul template: scalar from R0 indexed by c x vector in R_CYCLIC):
#   MULT.RC.VE rc_idx=0, src=chan_index (= c -> R0[c])
#       -> mult_res[i] = Q[i,c] * K[s,c]
#   c==0: ACC.ADD.FIRST  else: ACC.ADD   -> R_ACC[i] += Q[i,c] * K[s,c]
#   After 48 channels: R_ACC[i] = S[p, i, s] for the 64 queries of the stream.
#   ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG -> one whole 512 B row.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume a Q column loaded by LDR_CYCLIC_MULT_REG in its own bundle -- it would
# see the previous column. `;;` ends one VLIW word = one cycle = one snapshot,
# so a load and a MULT in the same bundle always run in the same cycle
# regardless of textual order; co-issuing is fine, consuming the same-cycle load
# is not.
#   Each key therefore primes c=0's column, then every loop body multiplies the
#   column loaded last cycle while prefetching the next one.
#   q_ptr (the load offset) is read LIVE by the load, so its ADD stays co-issued
#   with the load it feeds -- and because the load now runs one bundle ahead,
#   q_ptr naturally addresses the NEXT channel's column.
#   chan_index (the r0 scalar selector) is read LIVE by MULT and must name the
#   channel just loaded, so its ADD moves into a c_loop_pre block that falls
#   through into the loop body. That changes what BLT sees: with the ADD in the
#   same word, BLT read the pre-ADD snapshot; with the ADD one word earlier it
#   reads the already-incremented value and would drop the last channel. The
#   loop bound is therefore raised by one, in-kernel, into chan_last
#   (= chan_bound + 1 = D-1 = 47), leaving the harness's lr6 untouched.
#   Each key re-SETs q_ptr and chan_index, so no pipeline state crosses a key or
#   stream boundary.
#
# Loop nest:  for p in 0..3 { for s in 0..63 {
#                 load K[p,s] -> r0; prime c=0
#                 for c in 0..47: MULT.RC.VE + ACC
#                 ACTIVATE.QUANTIZE + STR_POST_AAQ_REG
#             }}
#
# Counts: 4 streams x 64 keys x 48 channels = 12288 MULT+ACC bundles + 256
#         stores.
#
# Addressing is split offset(LR) + base(CR) and all .asm operands are ROW
# numbers (issue #179). key_ptr and out_ptr advance monotonically across all 4
# streams (key (p,s) is row p*64 + s in both regions), so only q_stream_off
# needs a per-stream step (+48 rows).
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
{% set rc_slot0     = "lr0"  %}  {# const 0: r_cyclic write/read index #}
{% set chan_stride  = "lr2"  %}  {# 1 = rows between consecutive Q channels #}
{% set out_stride   = "lr3"  %}  {# 1 = rows per stored key row #}
{% set q_ptr        = "lr4"  %}  {# row offset into Q, walks head channels c #}
{% set chan_index   = "lr5"  %}  {# head-channel index c -> selects K[s,c] in r0 #}
{% set chan_bound   = "lr6"  %}  {# 46 = c-loop bound (first=0, width=48) #}
{% set chan_last    = "lr11" %}  {# 47 = chan_bound + 1; BLT bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr      = "lr7"  %}  {# row offset into S, += out_stride per store #}
{% set key_ptr      = "lr8"  %}  {# row offset into key-major K, += key_stride #}
{% set key_index    = "lr9"  %}  {# key counter s within the stream #}
{% set key_limit    = "lr10" %}  {# 64 = N keys per stream #}
{% set key_stride   = "lr12" %}  {# 1 = rows per key in the K scratch #}
{% set q_stream_off = "lr13" %}  {# Q row offset of the current stream, += 48 #}
{% set stream_index = "lr14" %}  {# stream counter p #}
{% set q_start      = "lr15" %}  {# q_stream_off + Q_START; per-key q_ptr seed #}

{% set ZERO         = "cr0"  %}  {# hardwired read-only 0 #}
{% set ONE          = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE       = "cr2"  %}  {# key-major score output base #}
{% set Q_BASE       = "cr3"  %}  {# channel-major Q base #}
{% set Q_START      = "cr5"  %}  {# -1 row startup skew (first live = stream base) #}
{% set CHAN_START   = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set CHAN_BOUND   = "cr8"  %}  {# 46 = D - 2 #}
{% set K_BASE_KM    = "cr9"  %}  {# key-major K scratch base #}
{% set STREAM_LIMIT = "cr10" %}  {# 4 = P streams #}
{% set Q_STREAM     = "cr11" %}  {# 48 = Q rows per stream #}
{% set DSTRUCT      = "cr15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 47). See the snapshot note in the
    # header: chan_index's ADD now sits one word ahead of the BLT, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

p_loop:
    SET {{ key_index }} {{ ZERO }};;                        # key counter = 0
    ADD {{ q_start }} {{ q_stream_off }} {{ Q_START }};;    # per-key q_ptr seed = stream base - 1 row

s_loop:
    LDR_MULT_REG r0 {{ key_ptr }} {{ K_BASE_KM }};;         # r0 = K[p, s, 0:48] (rest pad)

    ADD {{ q_ptr }} {{ q_start }} {{ ZERO }};;              # Q-data startup: stream base - 1 row (LR copy)
    SET {{ chan_index }} {{ CHAN_START }};;                 # fixed_idx c startup: -1

    # Prime c=0's Q column: MULT reads r_cyclic from the snapshot, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;       # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }}; STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE }};;# store S[p, 0:64, s] (key-major row)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;      # advance output row

    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }}; ADD {{ key_index }} {{ key_index }} {{ ONE }};;  # next key
    BLT {{ key_index }} {{ key_limit }} s_loop;;

    ADD {{ q_stream_off }} {{ q_stream_off }} {{ Q_STREAM }};;  # next stream: Q base += 48 rows
    ADD {{ stream_index }} {{ stream_index }} {{ ONE }};;
    BLT {{ stream_index }} {{ STREAM_LIMIT }} p_loop;;

end:
    BKPT;;
