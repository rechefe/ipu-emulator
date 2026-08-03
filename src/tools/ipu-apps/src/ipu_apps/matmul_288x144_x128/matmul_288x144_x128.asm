# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=144 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [288 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 288 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# (No activation applied — Swish to be added when ISA support is available)
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Algorithm: load W[j,:] into r0/r1 (once per j); load D[k,tg] into r_cyclic per k.
#   MULT.RC.VE: r0[fixed_idx] x r_cyclic[:] → 128 outputs per cycle.
#   k=0..127 uses r0[k]; k=128..143 uses r1[k-128] (fixed_idx=128..143 auto-reads r1).
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.
#
# Memory layout:
#   DATA:    2 × 144 × 128 B =  36 864 B (0x00000..0x08FFF)
#   WEIGHTS: 288 rows × 256 B =  73 728 B (0x10000..0x21FFF)
#   OUTPUT:  2 × 288 × 512 B = 294 912 B  (0x30000..0x77FFF)
#   NOTE: OUTPUT_BASE = 0x30000, not 0x20000 — weights end at 0x22000.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0      = "lr0" %}   {# const 0: r_cyclic write-index / mask_shift #}
{% set data_ptr      = "lr4" %}   {# byte offset into D, walks channels k       #}
{% set k_index       = "lr5" %}   {# contraction index k -> selects W[j,k]      #}
{% set data_stride   = "lr2" %}   {# 256 = bytes per input channel              #}
{% set out_stride    = "lr3" %}   {# 512 = bytes per output row                 #}
{% set k_bound_r0    = "lr6" %}   {# 126 = k-loop1 bound (k=0..127 from r0)     #}
{% set k_bound_r1    = "lr11" %}  {# 142 = k-loop2 bound (k=128..143 from r1)   #}
{% set out_ptr       = "lr7" %}   {# byte offset into C, += out_stride per j    #}
{% set w_ptr         = "lr8" %}   {# byte offset into W, += w_stride per j      #}
{% set j_index       = "lr9" %}   {# output-channel counter j                   #}
{% set j_limit       = "lr10" %}  {# 288 = N_OUT                                #}
{% set w_stride      = "lr12" %}  {# 256 = W_STRIDE bytes per output channel    #}

{% set DATA_BASE     = "cr0" %}   {# base of D                                  #}
{% set ONE           = "cr1" %}   {# hardwired read-only 1                      #}
{% set W_BASE_HI     = "cr2" %}   {# WEIGHTS_BASE + 128 (W[j,128..143])         #}
{% set OUT_BASE_TG0  = "cr3" %}
{% set OUT_BASE_TG1  = "cr4" %}
{% set DATA_START_TG0 = "cr5" %}  {# -256 startup skew                          #}
{% set DATA_START_TG1 = "cr6" %}  {# -128 startup skew                          #}
{% set K_START_R0    = "cr7" %}   {# -1  -> first live k = 0                    #}
{% set K_START_R1    = "cr8" %}   {# 127 -> first live k = 128 (r1[0])          #}
{% set W_BASE_LO     = "cr9" %}   {# WEIGHTS_BASE (W[j,0..127])                 #}
{% set DSTRUCT       = "cr15" %}  {# reserved dstructure register               #}

    # MULT.RC.VE reads its r_cyclic DATA from the start-of-cycle snapshot while
    # keeping the LR index live (issue #157), so a MULT cannot consume the chunk
    # that LDR_CYCLIC_MULT_REG loads in its own bundle. `;;` ends one VLIW word
    # = one cycle = one snapshot, so a load and a MULT in the same bundle always
    # run in the same cycle regardless of textual order -- co-issuing is fine,
    # consuming the same-cycle load is not.
    #
    # Each token group primes k=0's chunk ONCE before k_loop1, then every loop
    # body multiplies the chunk loaded last cycle while prefetching the next.
    # At the k_loop1 -> k_loop2 boundary data_ptr walks CONTINUOUSLY while
    # k_index RESETS, so k_loop2's first chunk is ALREADY IN FLIGHT from
    # k_loop1's trailing prefetch: k_loop2 must NOT re-prime (that would advance
    # data_ptr an extra step per output channel and drop real work).
    #
    # k_index keeps its ADD CO-ISSUED with the MULT and the BLT -- moving it to
    # a separate word costs a full cycle per contraction step (~2x on this
    # kernel family). MULT reads it LIVE (post-ADD) and BLT reads the SNAPSHOT
    # (pre-ADD), exactly as before, so the harness bounds stay valid. Only
    # k_loop1's START is biased down by one (the load now runs a bundle ahead,
    # so k_index must name the chunk being multiplied, not the one being
    # fetched). k_loop2's K_START_R1 is NOT biased: k_loop1's trailing prefetch
    # already supplied that step of phase.

j_loop:
    LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_LO }};;       # r0[0..127] = W[j, 0..127]
    LDR_MULT_REG r1 {{ w_ptr }} {{ W_BASE_HI }};;       # r1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG0 }};;           # tg=0 startup offset: -256
    SET {{ k_index }} {{ K_START_R0 }};;
    SUB {{ k_index }} {{ k_index }} {{ ONE }};;         # k-loop1 fixed_idx startup: -1 - 1 = -2 (biased: load runs a bundle ahead)

    # Prime k=0's chunk for this token group (see the note above j_loop).
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;
    B after_k_tg0;;

k_loop1_tg0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;

after_k_tg0:
    SET {{ k_index }} {{ K_START_R1 }};;  # NOT biased: k_loop1's trailing prefetch already advanced the phase                # k-loop2 fixed_idx startup: 127 → first live=128 (r1[0])

k_loop2_tg0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r1 }} k_loop2_tg0;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG0 }};;      # store 512B → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG1 }};;           # tg=1 startup offset: -128
    SET {{ k_index }} {{ K_START_R0 }};;
    SUB {{ k_index }} {{ k_index }} {{ ONE }};;         # k-loop1 fixed_idx startup: -1 - 1 = -2 (biased: load runs a bundle ahead)

    # Prime k=0's chunk for this token group (see the note above j_loop).
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;
    B after_k_tg1;;

k_loop1_tg1:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;

after_k_tg1:
    SET {{ k_index }} {{ K_START_R1 }};;  # NOT biased: k_loop1's trailing prefetch already advanced the phase

k_loop2_tg1:
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    BLT {{ k_index }} {{ k_bound_r1 }} k_loop2_tg1;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG1 }};;      # store 512B → OUTPUT[j, tg=1]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # advance output ptr

    ADD {{ w_ptr }} {{ w_ptr }} {{ w_stride }}; ADD {{ j_index }} {{ j_index }} {{ ONE }};; # next j: weight offset += W_STRIDE, j++
    BLT {{ j_index }} {{ j_limit }} j_loop;;

end:
    BKPT;;
