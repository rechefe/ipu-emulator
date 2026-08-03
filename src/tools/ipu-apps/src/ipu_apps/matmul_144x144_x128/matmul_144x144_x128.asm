# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=144 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [144 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 144 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# Algorithm: load W[j,:] into r0/r1 (once per j); load D[k,tg] into r_cyclic per k.
#   MULT.RC.VE: r0[fixed_idx] x r_cyclic[:] → 128 outputs per cycle.
#   k=0..127 uses r0[k] (fixed_idx=k); k=128..143 uses r1[k-128].
#
# Registers are referred to below by the symbolic names defined in the register-name
# block. The assembler's Jinja2 preprocessor substitutes them before parsing, so the
# emitted binary is byte-identical to the raw-register form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not contain
# Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.
#
# One-cycle startup offset pattern (pipeline alignment):
#   k-loop startup: data_ptr = first_real_addr - stride, k_index = -1
#   First cycle: XMEM loads from (DATA_BASE + data_ptr start) which may be negative/invalid
#                but r_cyclic[k_index start = -1 mod 512 = 511] is unused slot → harmless
#   After LR increments: data_ptr = first_real_addr, k_index = 0 → k=0 mult uses correct data
#   tg=0: data_ptr start=-256 → first real load at 0 (D[k=0,tg=0])
#   tg=1: data_ptr start=-128 → first real load at 128 (D[k=0,tg=1])
#   k-loop2 startup: data_ptr continues from k-loop1 end (naturally at k=128 addr)
#                    k_index reset to 127 → first live fixed_idx=128 (reads r1[0])

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0      = "lr0" %}   {# const 0: r_cyclic write-index / mask_shift #}
{% set data_ptr      = "lr4" %}   {# byte offset into D, walks channels k       #}
{% set k_index       = "lr5" %}   {# contraction index k → selects W[j,k]       #}
{% set data_stride   = "lr2" %}   {# 256 = bytes per input channel              #}
{% set out_stride    = "lr3" %}   {# 512 = bytes per output row                 #}
{% set k_bound_r0    = "lr6" %}   {# 126 = k-loop1 bound (k=0..127 from r0)     #}
{% set k_bound_r1    = "lr11" %}  {# 142 = k-loop2 bound (k=128..143 from r1)   #}
{% set out_ptr       = "lr7" %}   {# byte offset into C, += out_stride per j    #}
{% set w_ptr         = "lr8" %}   {# byte offset into W, += w_stride per j      #}
{% set j_index       = "lr9" %}   {# output-channel counter j                   #}
{% set j_limit       = "lr10" %}  {# 144 = N_OUT                                #}
{% set w_stride      = "lr12" %}  {# 256 = W_STRIDE bytes per output channel    #}

{% set DATA_BASE     = "cr0" %}   {# base of D                                  #}
{% set ONE           = "cr1" %}   {# hardwired read-only 1                      #}
{% set W_BASE_HI     = "cr2" %}   {# WEIGHTS_BASE + 128 (W[j,128..143])         #}
{% set OUT_BASE_TG0  = "cr3" %}
{% set OUT_BASE_TG1  = "cr4" %}
{% set DATA_START_TG0 = "cr5" %}  {# -256 startup skew                          #}
{% set DATA_START_TG1 = "cr6" %}  {# -128 startup skew                          #}
{% set K_START_R0    = "cr7" %}   {# -1  → first live k = 0                     #}
{% set K_START_R1    = "cr8" %}   {# 127 → first live k = 128 (r1[0])           #}
{% set W_BASE_LO     = "cr9" %}   {# WEIGHTS_BASE (W[j,0..127])                 #}
{% set DSTRUCT       = "cr15" %}  {# reserved dstructure register               #}

j_loop:
    LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_LO }};;       # r0[0..127] = W[j, 0..127]
    LDR_MULT_REG r1 {{ w_ptr }} {{ W_BASE_HI }};;       # r1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG0 }};;           # tg=0 startup offset: -256
    SET {{ k_index }} {{ K_START_R0 }};;                # k-loop1 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;
    B after_k_tg0;;

k_loop1_tg0:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg0;;

after_k_tg0:
    SET {{ k_index }} {{ K_START_R1 }};;                # k-loop2 fixed_idx startup: 127 → first live=128 (r1[0])

k_loop2_tg0:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound_r1 }} k_loop2_tg0;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG0 }};;      # store 512B → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    SET {{ data_ptr }} {{ DATA_START_TG1 }};;           # tg=1 startup offset: -128
    SET {{ k_index }} {{ K_START_R0 }};;                # k-loop1 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;
    B after_k_tg1;;

k_loop1_tg1:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound_r0 }} k_loop1_tg1;;

after_k_tg1:
    SET {{ k_index }} {{ K_START_R1 }};;

k_loop2_tg1:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound_r1 }} k_loop2_tg1;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG1 }};;      # store 512B → OUTPUT[j, tg=1]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # advance output ptr

    ADD {{ w_ptr }} {{ w_ptr }} {{ w_stride }}; ADD {{ j_index }} {{ j_index }} {{ ONE }};; # next j
    BLT {{ j_index }} {{ j_limit }} j_loop;;

end:
    BKPT;;
