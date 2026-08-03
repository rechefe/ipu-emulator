# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=288 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [144 out_ch, 288 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*384
#    W[j, 128..255] at WEIGHTS_BASE + j*384 + 128
#    W[j, 256..287] at WEIGHTS_BASE + j*384 + 256 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 144 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# Algorithm: K=288 split into three 128-element chunks per tg.
#   Per chunk: load W[j, chunk*128..(chunk+1)*128-1] into r0, run 128 k-steps.
#   MULT.RC.VE r0[k_index] x r_cyclic[:] — k_index cycles 0..127 per chunk.
#   data_ptr (data pointer) advances continuously; NOT reset between chunks.
#   k_index reset to -1 at the start of each chunk via SET k_index K_START.
#
# Loop-bound formula (do-while with live MULT/XMEM, snapshot BLT):
#   body runs for live index [k_index start + 1 … k_bound + 1]
#   for width=128 starting at index 0: k_index start = -1, k_bound = 126
#   exit cycle lands on live index 127 (the last real term), data_ptr advances 128 times total.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.
#
# Memory layout:
#   DATA:    288 × 2 × 128 B =  73 728 B (0x00000..0x11FFF)
#   WEIGHTS: 144 rows × 384 B =  55 296 B (0x20000..0x2D7FF)
#   OUTPUT:  2 × 144 × 512 B = 147 456 B  (0x40000..0x63FFF)

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0       = "lr0"  %}  {# const 0: r_cyclic write-index / mask_shift #}
{% set data_stride    = "lr2"  %}  {# 256 = bytes per input channel #}
{% set out_stride     = "lr3"  %}  {# 512 = bytes per output row #}
{% set data_ptr       = "lr4"  %}  {# byte offset into D; runs across all 3 chunks #}
{% set k_index        = "lr5"  %}  {# per-chunk scalar index 0..127 into r0 #}
{% set k_bound        = "lr6"  %}  {# 126 = per-chunk bound (width 128 -> 0+128-2) #}
{% set out_ptr        = "lr7"  %}  {# byte offset into C, += out_stride per j #}
{% set w_ptr          = "lr8"  %}  {# byte offset into W, += w_stride per j #}
{% set j_index        = "lr9"  %}  {# output-channel counter j #}
{% set j_limit        = "lr10" %}  {# 144 = N_OUT #}
{% set w_stride       = "lr12" %}  {# 384 = W_STRIDE bytes per output channel #}

{% set DATA_BASE      = "cr0"  %}  {# base of D #}
{% set ONE            = "cr1"  %}  {# hardwired read-only 1 #}
{% set W_BASE_CHUNK1  = "cr2"  %}  {# WEIGHTS_BASE + 128 (W[j,128..255]) #}
{% set W_BASE_CHUNK2  = "cr3"  %}  {# WEIGHTS_BASE + 256 (W[j,256..287]) #}
{% set OUT_BASE_TG0   = "cr4"  %}  {# OUTPUT_BASE #}
{% set OUT_BASE_TG1   = "cr5"  %}  {# OUTPUT_BASE + N_OUT*512 #}
{% set DATA_START_TG0 = "cr6"  %}  {# -256 startup skew #}
{% set DATA_START_TG1 = "cr7"  %}  {# -128 startup skew #}
{% set K_START        = "cr8"  %}  {# -1 -> first live k = 0 (per chunk) #}
{% set W_BASE_CHUNK0  = "cr9"  %}  {# WEIGHTS_BASE (W[j,0..127]) #}
{% set DSTRUCT        = "cr15" %}  {# reserved dstructure register #}

j_loop:
    SET {{ data_ptr }} {{ DATA_START_TG0 }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK0 }};;  # tg=0 startup; r0 = W[j, 0..127]
    SET {{ k_index }} {{ K_START }};;                                                           # chunk0 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ k_index }} {{ k_bound }} k_chunk0_tg0;;
    B after_chunk0_tg0;;

k_chunk0_tg0:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk0_tg0;;

after_chunk0_tg0:

    SET {{ k_index }} {{ K_START }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK1 }};;          # chunk1 startup; r0 = W[j, 128..255]

k_chunk1_tg0:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk1_tg0;;

    SET {{ k_index }} {{ K_START }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK2 }};;          # chunk2 startup; r0 = W[j, 256..287]+zeros

k_chunk2_tg0:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk2_tg0;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG0 }};;                                              # store 512B → OUTPUT[j, tg=0]

    SET {{ data_ptr }} {{ DATA_START_TG1 }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK0 }};;  # tg=1 startup; r0 = W[j, 0..127]
    SET {{ k_index }} {{ K_START }};;

    # Peeled first k-iter (k=0): ACC.ADD.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST; BLT {{ k_index }} {{ k_bound }} k_chunk0_tg1;;
    B after_chunk0_tg1;;

k_chunk0_tg1:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk0_tg1;;

after_chunk0_tg1:

    SET {{ k_index }} {{ K_START }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK1 }};;

k_chunk1_tg1:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk1_tg1;;

    SET {{ k_index }} {{ K_START }}; LDR_MULT_REG r0 {{ w_ptr }} {{ W_BASE_CHUNK2 }};;

k_chunk2_tg1:
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ data_stride }}; ADD {{ k_index }} {{ k_index }} {{ ONE }};
    MULT.RC.VE {{ rc_slot0 }} {{ k_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD; BLT {{ k_index }} {{ k_bound }} k_chunk2_tg1;;

    STR_ACC_REG {{ out_ptr }} {{ OUT_BASE_TG1 }};;                                              # store 512B → OUTPUT[j, tg=1]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;                                          # advance output ptr

    ADD {{ w_ptr }} {{ w_ptr }} {{ w_stride }}; ADD {{ j_index }} {{ j_index }} {{ ONE }};;     # next j: weight offset += W_STRIDE, j++
    BLT {{ j_index }} {{ j_limit }} j_loop;;

end:
    BKPT;;
