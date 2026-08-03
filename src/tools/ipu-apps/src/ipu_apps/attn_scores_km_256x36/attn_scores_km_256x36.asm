# Agent D — kQᵀ → key-major attention scores, one head (D=36, N=256)
#
# S[i, s] = sum_c Q[i, c] * K[s, c]      i,s in [0,256), c in [0,36)
#   Lanes = queries (i); outer loop = key (s); contraction = head channel (c).
#
# Memory layout (one head; harness offsets by head h before load):
#   Q: channel-major     Q[i, c] at QBASE + c*256 + i
#      A channel column (128 queries, one channel c, group g) is contiguous:
#      QBASE + c*256 + g*128  → loads straight into R_CYCLIC (vector operand).
#   K: key-major scratch K[s, 0:35] at KBASE_KM + s*128 (36 ch padded to 128 B)
#      Loaded once per key s into R0 (scalar operand, indexed by c).
#   S: key-major words   S[i, s] at SBASE + (s*256 + i)*4
#      (s,g=0) → SBASE + s*1024;  (s,g=1) → SBASE + s*1024 + 512
#      One running 512-B-stride pointer covers g=0, g=1, next key contiguously.
#
# Compute (matmul template: scalar from R0 indexed by c × vector in R_CYCLIC):
#   MULT.RC.VE rc_idx=0, src=chan_index (=c → R0[c]) → mult_res[i] = Q[i,c]·K[s,c]
#   c==0: ACC.FIRST  else: ACC   → R_ACC[i] += Q[i,c]·K[s,c]
#   After 36 channels: R_ACC[i] = S[i,s] for the 128 queries of this group.
#   STR_ACC_REG → 512 B (128 × int32/fp32) key-major score row.  No AGG.
#
# Counts/head: 256 keys × 2 groups × 36 channels = 18432 MULT+ACC bundles
#              + 512 stores.
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
{% set rc_slot0    = "lr0"  %}  {# const 0: r_cyclic write/read index #}
{% set chan_stride = "lr2"  %}  {# 256 = bytes between consecutive Q channels #}
{% set out_stride  = "lr3"  %}  {# 512 = bytes per stored (key, group) row #}
{% set q_ptr       = "lr4"  %}  {# byte offset into Q, walks head channels c #}
{% set chan_index  = "lr5"  %}  {# head-channel index c -> selects K[s,c] in r0 #}
{% set chan_bound  = "lr6"  %}  {# 34 = c-loop bound (first=0, width=36) #}
{% set out_ptr     = "lr7"  %}  {# byte offset into S, += 512 per store #}
{% set key_ptr     = "lr8"  %}  {# byte offset into key-major K, += key_stride #}
{% set key_index   = "lr9"  %}  {# key counter s #}
{% set key_limit   = "lr10" %}  {# 256 = N keys #}
{% set key_stride  = "lr12" %}  {# 128 = bytes per key row in the K scratch #}

{% set Q_BASE      = "cr0"  %}  {# channel-major Q base #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = "cr2"  %}  {# key-major score output base #}
{% set Q_START_G0  = "cr5"  %}  {# -256 startup skew (first live = 0) #}
{% set Q_START_G1  = "cr6"  %}  {# -128 startup skew (first live = 128) #}
{% set CHAN_START  = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set K_BASE_KM   = "cr9"  %}  {# key-major K scratch base #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

s_loop:
    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }};;  # key byte offset += 128 (first live = 0)
    LDR_MULT_REG r0 {{ key_ptr }} {{ K_BASE_KM }};;     # r0[0..127] = K[s, 0:35] + zeros

    # -- query group 0 (queries 0..127) -------------------------------------
    SET {{ q_ptr }} {{ Q_START_G0 }};;                  # channel-column startup: -256
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ chan_index }} {{ chan_bound }} c_loop_g0;;
    B after_c_g0;;

c_loop_g0:
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ chan_index }} {{ chan_bound }} c_loop_g0;;

after_c_g0:
    STR_ACC_REG {{ out_ptr }} {{ S_BASE }};;            # store S[0:128, s] (key-major row, g=0)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # output ptr += 512

    # -- query group 1 (queries 128..255) -----------------------------------
    SET {{ q_ptr }} {{ Q_START_G1 }};;                  # g=1 channel-column startup: -128 → first live = 128
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ chan_index }} {{ chan_bound }} c_loop_g1;;
    B after_c_g1;;

c_loop_g1:
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }}; ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }}; ADD {{ chan_index }} {{ chan_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ chan_index }} {{ chan_bound }} c_loop_g1;;

after_c_g1:
    STR_ACC_REG {{ out_ptr }} {{ S_BASE }};;            # store S[128:256, s] (key-major row, g=1)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # output ptr += 512

    ADD {{ key_index }} {{ key_index }} {{ ONE }}; BLT {{ key_index }} {{ key_limit }} s_loop;; # next key

end:
    BKPT;;
