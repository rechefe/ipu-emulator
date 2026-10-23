{#- ==========================================================================
    layernorm.asm -- LayerNorm over the channel axis, FP32. The template of
    layernorm_{128x16,16x240,64x192,256x144} (../registry.toml, E07-10); each
    entry renders to its old binary (E02 digests).

    y[ch, t] = gamma[ch] * (x[ch, t] - mu[t]) * rsqrt(var[t]) + beta[ch]

    mu and var reduce over the CHANNELS (one XMEM row per channel), lane-wise,
    so 128 tokens are normalized at once. No eps (Q57): rsqrt maps a
    non-positive variance to 0, so a constant token column becomes beta.

    Layout (rows; kernel.toml's [abi] binds every base, cr.<name> below):
      x        channels x groups rows, row ch*groups + g, tokens in the lanes;
               x is pinned at row 0 and read through CR0 (hardwired 0)
      gamma    ceil(channels / 128) rows; beta likewise
      ones, neg_inv_n, inv_n   one constant row each: 1.0, -1/channels and
               1/channels in every lane
      neg_mean, temp, invstd   one scratch row each
      centered channels rows of scratch, reused per token group
      y        the same layout as x, after the scratch rows
      channels, row_stride (1) and lanes (128; also beta's lane offset in R1)
      are CRs too; cr.ds_qf holds the DS (LOW, 128 valid elements) and the
      case's QF (HIGH).
    Padding lanes (tokens < 128) hold x = 0: mean 0, variance 0, rsqrt 0, so
    they become beta[ch] and are not part of the output tensor.

    Passes, per token group:
      1  -mu   = SUM_ch x[ch] * (-1/channels)          MULT.RC.VV + ACC
      2  c[ch] = x[ch] * 1 + (-mu)                      -> centered
      3  s     = SUM_ch c[ch]^2                         MULT.RC.VS (squares R_CYCLIC)
      4  r     = rsqrt(s * 1/channels)                  ACTIVATE.EXACT rsqrt
      5  n[ch] = c[ch] * r                              centered, in place
      6  y[ch] = n[ch] * gamma[ch] + 1 * beta[ch]       MULT.RC.VE, the scalar
         index walking gamma (lane ch) and beta (lane 128 + ch, from R1).
         With more than 128 channels gamma/beta take two rows: sub-loop A
         covers channels 0..127, sub-loop B reloads R0/R1 from row 1 and
         covers the rest.

    Instruction word rules every pass follows:
      (a) a pointer starts at -stride: its ADD fires first, so the load in the
          same word sees the forwarded 0
      (b) a loop counter starts at 1: BLT.S reads its LR at cycle start
      (c) R0/R1 are loaded before a loop and held: a MULT reads them at cycle
          start, so it sees the previous word's load
      (d) MULT reads R_CYCLIC at cycle start too (issue #157): each pass primes
          its first row before the loop and prefetches the next row inside it;
          the load reads the offset LR's forwarded value, so its ADD co-issues
          with the load.
      Pass 6 runs 4 words per channel: word C uses all 3 LR slots (write
      pointer, gamma and beta index), so the prefetch rides in word D.
      Sub-loop B inherits sub-loop A's last prefetch and does not re-prime.
    The program sets every LR it reads, so it needs no LR preset.

    Template choices (each keeps the four old binaries):
      groups = ceil(tokens / 128) is 1 or 2. With 2, an outer loop over token
      groups strides the x and y pointers by 2 rows (data_stride) and offsets
      them by the group (tg_off); centered is reused per group.
      channels <= 128 with one group keeps layernorm_128x16's register
      allocation (row stride in LR7, gamma/beta indices in LR9/LR10); the
      other shapes use LR12..LR15. With at most 128 channels pass 6 is one
      loop, bounded by LR6 (channels).
      Sub-loop B's bound, channels - 128, is built from CR1: by doubling when
      it is a power of two (16, 64), as 128 - 2^k when that is (112). Any
      other bound renders UNSUPPORTED_CHANNEL_COUNT, which does not assemble.

    Jinja runs before comment stripping: comments must not hold Jinja delimiters.
========================================================================== -#}
{%- set groups = (tokens + 127) // 128 %}
{%- set wide = channels > 128 %}
{%- set small = (not wide) and groups == 1 %}
{%- set rc_slot0 = "LR0" %}
{%- set mask_shift = "LR1" %}
{%- set read_ptr = "LR2" %}
{%- set write_ptr = "LR3" %}
{%- set tmp_bound = "LR4" %}
{%- set ch_index = "LR5" %}
{%- set ch_limit = "LR6" %}
{%- set data_stride = "LR7" %}
{%- set tg_index = "LR9" %}
{%- set tg_off = "LR10" %}
{%- set tg_limit = "LR11" %}
{%- set row_stride = "LR7" if small else "LR12" %}
{%- set gamma_idx = "LR9" if small else "LR13" %}
{%- set beta_idx = "LR10" if small else "LR14" %}
{%- set sub_bound = "LR15" %}
{%- set data_step = data_stride if groups > 1 else row_stride %}
{%- set DATA_BASE = cr.x %}
{%- set ONE = "CR1" %}
{%- set BETA_BASE = cr.beta %}
{%- set ONES_BASE = cr.ones %}
{%- set NEG_INV_N_BASE = cr.neg_inv_n %}
{%- set INV_N_BASE = cr.inv_n %}
{%- set NEG_MEAN_BASE = cr.neg_mean %}
{%- set CENTERED_BASE = cr.centered %}
{%- set TEMP_BASE = cr.temp %}
{%- set INVSTD_BASE = cr.invstd %}
{%- set OUTPUT_BASE = cr.y %}
{%- set GAMMA_BASE = cr.gamma %}
{%- set N_CH = cr.channels %}
{%- set ROW_STRIDE = cr.row_stride %}
{%- set LANES = cr.lanes %}
{%- set DSTRUCT = cr.ds_qf ~ ".L" %}
{%- set QF = cr.ds_qf ~ ".H" %}

{%- macro build_count(dest, n) -%}
{#- dest = n, built from CR1 (no CR holds it). -#}
{%- set ns = namespace(k=0) %}
{%- for i in range(8) %}{% if 2 ** i == n %}{% set ns.k = i %}{% endif %}{% endfor %}
{%- if 2 ** ns.k == n %}
    SET     {{ dest }} {{ ONE }};;
{%- for _ in range(ns.k) %}
    ADD     {{ dest }} {{ dest }} {{ dest }};;
{%- endfor %}
{%- else %}
{%- for i in range(8) %}{% if 2 ** i == 128 - n %}{% set ns.k = i %}{% endif %}{% endfor %}
{%- if 2 ** ns.k != 128 - n %}
    UNSUPPORTED_CHANNEL_COUNT;;
{%- endif %}
    SET     {{ tmp_bound }} {{ ONE }};;
{%- for _ in range(ns.k) %}
    ADD     {{ tmp_bound }} {{ tmp_bound }} {{ tmp_bound }};;
{%- endfor %}
    SET     {{ dest }} {{ LANES }};;
    SUB     {{ dest }} {{ dest }} {{ tmp_bound }};;
{%- endif %}
{%- endmacro %}

{%- macro start(ptr, stride, grouped) -%}
    SET     {{ ptr }}  {{ DATA_BASE }};;
    SUB     {{ ptr }}  {{ ptr }}  {{ stride }};;
{%- if grouped and groups > 1 %}
    ADD     {{ ptr }}  {{ ptr }}  {{ tg_off }};;
{%- endif %}
{%- endmacro %}

{%- macro counter() -%}
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
{%- endmacro %}

{%- macro pass6_body(label, bound) %}
{{ label }}:
    MULT.RC.VE {{ rc_slot0 }} {{ gamma_idx }} 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VE {{ rc_slot0 }} {{ beta_idx }} 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;;
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ OUTPUT_BASE }} {{ write_ptr }} ;
    ADD {{ write_ptr }} {{ write_ptr }} {{ data_step }};
    ADD {{ gamma_idx }} {{ gamma_idx }} {{ ONE }};
    ADD {{ beta_idx }} {{ beta_idx }} {{ ONE }};;
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ bound }} {{ label }};;
{%- endmacro %}

    SET     {{ rc_slot0 }}  {{ DATA_BASE }};;
    SET     {{ mask_shift }}  {{ DATA_BASE }};;
    SET     {{ ch_limit }}  {{ N_CH }};;
{%- if groups > 1 %}
    SET     {{ data_stride }}  {{ ROW_STRIDE }};;
    ADD     {{ data_stride }}  {{ data_stride }}  {{ data_stride }};;
    SET     {{ tg_limit }} {{ ONE }};;
    ADD     {{ tg_limit }} {{ tg_limit }} {{ ONE }};;
{%- endif %}
    SET     {{ row_stride }} {{ ROW_STRIDE }};;
{%- if groups > 1 %}
    SET     {{ tg_index }}  {{ DATA_BASE }};;
    ADD     {{ tg_index }}  {{ tg_index }}  {{ ONE }};;
    SET     {{ tg_off }} {{ DATA_BASE }};;
tg_loop:
{%- endif %}

{#- ---- pass 1: -mu ---- #}
    LD.MULT        R0 {{ rc_slot0 }} {{ NEG_INV_N_BASE }};;
{{ start(read_ptr, data_step, true) }}
{{ counter() }}
    LD.MULT.CYCLIC {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ data_step }};;
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;;
    LD.MULT.CYCLIC {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ data_step }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step1_loop;;
    B       step1_done;;
step1_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;;
    LD.MULT.CYCLIC {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ data_step }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step1_loop;;
step1_done:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ NEG_MEAN_BASE }} {{ rc_slot0 }} ;;

{#- ---- pass 2: centered = x - mu ---- #}
    LD.MULT        R0 {{ rc_slot0 }} {{ ONES_BASE }};;
    LD.MULT        R1 {{ rc_slot0 }} {{ NEG_MEAN_BASE }};;
{{ start(read_ptr, data_step, true) }}
{{ start(write_ptr, row_stride, false) }}
{{ counter() }}
    LD.MULT.CYCLIC {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ data_step }};;
step2_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VV {{ rc_slot0 }} R1 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;;
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ CENTERED_BASE }} {{ write_ptr }} ;
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LD.MULT.CYCLIC {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ data_step }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step2_loop;;

{#- ---- pass 3: sum of squares ---- #}
{{ start(read_ptr, row_stride, false) }}
{{ counter() }}
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;;
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step3_loop;;
    B       step3_done;;
step3_loop:
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD;;
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step3_loop;;
step3_done:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ TEMP_BASE }} {{ rc_slot0 }} ;;

{#- ---- pass 4: 1/sigma ---- #}
    LD.MULT        R0 {{ rc_slot0 }} {{ TEMP_BASE }};;
    LD.MULT.CYCLIC {{ rc_slot0 }} {{ INV_N_BASE }} {{ rc_slot0 }};;
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;;
    ACTIVATE.EXACT rsqrt {{ DSTRUCT }} {{ INVSTD_BASE }} {{ rc_slot0 }};;

{#- ---- pass 5: normalized = centered * 1/sigma, in place ---- #}
    LD.MULT        R0 {{ rc_slot0 }} {{ INVSTD_BASE }};;
{{ start(read_ptr, row_stride, false) }}
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
{{ counter() }}
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
step5_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ mask_shift }} {{ DSTRUCT }} {{ QF }};
    ACC.ADD.FIRST;;
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ CENTERED_BASE }} {{ write_ptr }} ;
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD     {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT.S {{ ch_index }} {{ ch_limit }} step5_loop;;

{#- ---- pass 6: y = gamma * normalized + beta ---- #}
    LD.MULT        R0 {{ rc_slot0 }} {{ GAMMA_BASE }};;
    LD.MULT        R1 {{ rc_slot0 }} {{ BETA_BASE }};;
{{ start(read_ptr, row_stride, false) }}
{{ start(write_ptr, data_step, true) }}
{{ counter() }}
    SET     {{ gamma_idx }} {{ DATA_BASE }};;
    SET     {{ beta_idx }} {{ LANES }};;
{%- if wide %}
    SET     {{ sub_bound }} {{ LANES }};;
{%- endif %}
    LD.MULT.CYCLIC {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
{%- if wide %}
{{ pass6_body("step6A_loop", sub_bound) }}
    LD.MULT        R0 {{ row_stride }} {{ GAMMA_BASE }};;
    LD.MULT        R1 {{ row_stride }} {{ BETA_BASE }};;
{{ counter() }}
    SET     {{ gamma_idx }} {{ DATA_BASE }};;
    SET     {{ beta_idx }} {{ LANES }};;
{{ build_count(sub_bound, channels - 128) }}
{{ pass6_body("step6B_loop", sub_bound) }}
{%- else %}
{{ pass6_body("step6_loop", ch_limit) }}
{%- endif %}
{%- if groups > 1 %}
    ADD     {{ tg_off }} {{ tg_off }} {{ row_stride }};;
    ADD     {{ tg_index }}  {{ tg_index }}  {{ ONE }};
    BLT.S {{ tg_index }}  {{ tg_limit }} tg_loop;;
{%- endif %}

end:
    HALT;;
