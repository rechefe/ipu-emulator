{#- matmul_x128_2tg_c2.asm: the template of the registry.toml entries
    matmul_144x144_x128, matmul_432x144_x128 and matmul_288x144_x128 (SiLU:
    the activation parameter renders ACTIVATE.EXACT). The notes below are
    matmul_144x144_x128's; the shape reaches the program through
    kernel.toml's [abi] (rows and indices, not bytes; the byte figures below
    are the old harness's).

    Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]

    Layer:   L3
    Scope:   single-stream
    Layout:  unpacked
    Shape:   M=256 tokens (2 token groups x 128), K=144, N=144
    Related: L3 OutProj (single-stream). One of 4 near-identical L3 matmuls —
             matmul_288x144_x128 (FFN1, K=144->288), matmul_432x144_x128
             (fused QKV, K=144->432), matmul_144x288_x128 (FFN2, K=288->144) —
             all share this j_loop/k-loop template. Consumed by
             proj_outproj_144_p4, which wraps the same arithmetic across all 4
             pixel streams in one invocation.
    Tests:   //kernels/matmul:matmul_144x144_x128 (the registry entry)

    Computes the L3 OutProj matmul: 144 input channels contracted down to 144
    output channels, over 256 tokens (handled as two 128-token groups, tg=0/1,
    since one XMEM row holds 128 lanes). Because K=144 spans two SIMD-width
    registers, each output channel's contraction is split into two k-loops —
    k=0..127 read their weight scalar from R0, k=128..143 from R1 — with the
    first iteration of the first k-loop peeled to seed r_acc via ACC.ADD.FIRST
    (see the "Peeled first k-iter" comment below), and the rest accumulated
    with ACC.ADD, once per token group.

    D: interleaved channel-major [K=144 channels, 2 tg, 128 tokens]
       Row (k, tg) at DATA_BASE + k*256 + tg*128
    W: output-major [144 out_ch, 144 in_ch], NO transposition
       W[j, 0..127]   at WEIGHTS_BASE + j*256
       W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
    C: grouped channel-major [2 tg, 144 out_ch, 128 tokens], FP32 accumulators
       Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512

    Algorithm: load W[j,:] into R0/R1 (once per j); load D[k,tg] into r_cyclic per k.
      MULT.RC.VE: R0[fixed_idx] x r_cyclic[:] → 128 outputs per cycle.
      k=0..127 uses R0[k] (fixed_idx=k); k=128..143 uses R1[k-128].

    Registers are referred to below by the symbolic names defined in the register-name
    block. The assembler's Jinja2 preprocessor substitutes them before parsing, so the
    emitted binary is byte-identical to the raw-register form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not contain
    Jinja delimiters -- the preprocessor would try to execute them.

    Register assignments and their meanings are listed in the register-name
    block below -- it is the single source of truth for this kernel.

    One-cycle startup offset pattern (pipeline alignment):
      k-loop startup: data_ptr = first_real_addr - stride, k_index = -1
      First cycle: XMEM loads from (DATA_BASE + data_ptr start) which may be negative/invalid
                   but r_cyclic[k_index start = -1 mod 512 = 511] is unused slot → harmless
      After LR increments: data_ptr = first_real_addr, k_index = 0 → k=0 mult uses correct data
      tg=0: data_ptr start=-256 → first real load at 0 (D[k=0,tg=0])
      tg=1: data_ptr start=-128 → first real load at 128 (D[k=0,tg=1])
      k-loop2 startup: data_ptr continues from k-loop1 end (naturally at k=128 addr)
                       k_index reset to 127 → first live fixed_idx=128 (reads R1[0])
-#}
{%- set act = "silu" if activation == "silu" else "identity" -%}

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0      = "LR0" %}   {# const 0: r_cyclic write-index / mask_shift #}
{% set data_ptr      = "LR4" %}   {# byte offset into D, walks channels k       #}
{% set k_index       = "LR5" %}   {# contraction index k → selects W[j,k]       #}
{% set data_stride   = "LR2" %}   {# 256 = bytes per input channel              #}
{% set out_stride    = "LR3" %}   {# 512 = bytes per output row                 #}
{% set k_bound_r0    = "LR6" %}   {# 126 = k-loop1 bound (k=0..127 from R0)     #}
{% set k_bound_r1    = "LR11" %}  {# 142 = k-loop2 bound (k=128..143 from R1)   #}
{% set out_ptr       = "LR7" %}   {# byte offset into C, += out_stride per j    #}
{% set w_ptr         = "LR8" %}   {# byte offset into W, += w_stride per j      #}
{% set j_index       = "LR9" %}   {# output-channel counter j                   #}
{% set j_limit       = "LR10" %}  {# 144 = N_OUT                                #}
{% set w_stride      = "LR12" %}  {# 256 = W_STRIDE bytes per output channel    #}

{% set DATA_BASE     = "CR0" %}   {# base of D                                  #}
{% set ONE           = "CR1" %}   {# hardwired read-only 1                      #}
{% set W_BASE_HI     = "CR2" %}   {# WEIGHTS_BASE + 128 (W[j,128..143])         #}
{% set OUT_BASE_TG0  = "CR3" %}
{% set OUT_BASE_TG1  = "CR4" %}
{% set DATA_START_TG0 = "CR5" %}  {# -256 startup skew                          #}
{% set DATA_START_TG1 = "CR6" %}  {# -128 startup skew                          #}
{% set K_START_R0    = "CR7" %}   {# -1  → first live k = 0                     #}
{% set K_START_R1    = "CR8" %}   {# 127 → first live k = 128 (R1[0])           #}
{% set W_BASE_LO     = "CR9" %}   {# WEIGHTS_BASE (W[j,0..127])                 #}
{% set DSTRUCT       = "CR15.L" %}  {# the default DS half (ds(128)) #}
{% set QF            = cr.qf ~ ".L" %}  {# the QF half (case_qf; FP32 ignores it) #}

    # MULT.RC.VE reads its r_cyclic DATA from the start-of-cycle snapshot while
    # keeping the LR index live (issue #157), so a MULT cannot consume the chunk
    # that LD.MULT.CYCLIC loads in its own bundle. `;;` ends one VLIW word
    # = one cycle = one snapshot, so a load and a MULT in the same bundle always
    # run in the same cycle regardless of textual order -- co-issuing is fine,
    # consuming the same-cycle load is not.
    #
    # Each token group therefore primes k=0's chunk ONCE, before k_loop1, and
    # every loop body multiplies the chunk loaded last cycle while prefetching
    # the next. The critical detail is the k_loop1 -> k_loop2 boundary: data_ptr
    # walks CONTINUOUSLY across it (it is never re-SET at after_k_*) while
    # k_index RESETS via SET K_START_R1. So when k_loop2 is entered the chunk
    # for its first iteration is ALREADY IN FLIGHT from k_loop1's last prefetch.
    # k_loop2 must NOT re-prime -- doing so would advance data_ptr an extra step
    # per output channel and drop real work.
    #
    # k_index keeps its ADD CO-ISSUED with the MULT and the BLT.S, exactly as
    # before the fix -- moving it into a separate *_pre word would cost a whole
    # extra cycle per contraction step (measured: 43345 -> 84819 cycles), which
    # is the trap this transform has to avoid.
    #
    # Co-issuing still works because both readers keep their original
    # relationship: MULT reads k_index LIVE (post-ADD) and BLT.S reads the
    # SNAPSHOT (pre-ADD), so the harness's LR6/LR11 bounds stay valid unchanged.
    # The only adjustment is the STARTING value. The load now runs one bundle
    # ahead, so on the cycle a chunk is multiplied, k_index must name THAT chunk
    # rather than the one being fetched -- i.e. one lower than before at every
    # point. Each SET of k_index is therefore followed by a SUB of 1, which sits
    # outside the k-loops and so costs nothing per contraction step. CR7/CR8 are
    # read as-is; no harness constant changes.

j_loop:
    LD.MULT R0 {{ w_ptr }} {{ W_BASE_LO }};;       # R0[0..127] = W[j, 0..127]
    LD.MULT R1 {{ w_ptr }} {{ W_BASE_HI }};;       # R1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG0 }};;           # tg=0 startup offset: -256
    SET {{ k_index }} {{ K_START_R0 }};;
    SUB {{ k_index }} {{ k_index }} {{ ONE }};;         # k-loop1 fixed_idx startup: -1 - 1 = -2 (biased)

    # Prime k=0's chunk for this token group (see the note above j_loop).
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;
    B after_k_tg0;;

k_loop1_tg0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;

after_k_tg0:
    # No re-prime: k=128's chunk is already in flight from k_loop1's last
    # prefetch, and data_ptr already points past it.
    SET {{ k_index }} {{ K_START_R1 }};;                # k-loop2 fixed_idx startup: 127 -> first live k=128.
    # NOT biased down like k_loop1's: k_loop1's trailing prefetch already
    # advanced the pipeline one step, so k_loop2 inherits the correct phase.

k_loop2_tg0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r1 }} k_loop2_tg0;;

    # Store (ADR 0007): ACTIVATE.EXACT activates R_ACC's active lanes and
    # stores the row as 128 FP32 values in the same word; the DS half
    # (ds(128), every lane) makes `identity` a lane-for-lane copy of r_acc.
    ACTIVATE.EXACT {{ act }} {{ DSTRUCT }} {{ OUT_BASE_TG0 }} {{ out_ptr }};;  # store 512B → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG1 }};;           # tg=1 startup offset: -128
    SET {{ k_index }} {{ K_START_R0 }};;
    SUB {{ k_index }} {{ k_index }} {{ ONE }};;         # k-loop1 fixed_idx startup: -1 - 1 = -2 (biased)

    # Prime k=0's chunk for this token group.
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;
    B after_k_tg1;;

k_loop1_tg1:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;

after_k_tg1:
    # No re-prime: k=128's chunk is already in flight (see tg=0).
    SET {{ k_index }} {{ K_START_R1 }};;                # k-loop2 fixed_idx startup: 127 -> first live k=128.
    # NOT biased down like k_loop1's: k_loop1's trailing prefetch already
    # advanced the pipeline one step, so k_loop2 inherits the correct phase.

k_loop2_tg1:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }};
    ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT.S {{ k_index }} {{ k_bound_r1 }} k_loop2_tg1;;

    ACTIVATE.EXACT {{ act }} {{ DSTRUCT }} {{ OUT_BASE_TG1 }} {{ out_ptr }};;  # store 512B → OUTPUT[j, tg=1]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # advance output ptr

    ADD {{ w_ptr }} {{ w_ptr }} {{ w_stride }};
    ADD {{ j_index }} {{ j_index }} {{ ONE }};;  # next j
    BLT.S {{ j_index }} {{ j_limit }} j_loop;;

end:
    HALT;;
