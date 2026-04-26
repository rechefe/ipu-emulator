# Universal Standard 3x3 Convolution
#
# A single binary that handles ANY valid standard convolution configuration.
# Parameters are passed via CR registers at runtime.
#
# Supported configurations:
#   - Spatial: any rows/cols where cols is power-of-2 in [16..128]
#     and rows*cols >= 256 (at least 2 chunks)
#   - in_channels: >= 1 (any value; the last block of each filter may be
#     partial when in_channels % 14 != 0 — clamped at runtime via cr10)
#   - out_channels: >= 1
#
# The cyclic register holds 3 neighboring chunks:
#   S0 (index 0):   previous chunk
#   S1 (index 128): current chunk
#   S2 (index 256): next chunk
#
# Vertical neighbor access via universal formula:
#   kr=-1 base = 128 - cols
#   kr= 0 base = 128
#   kr=+1 base = 128 + cols
# Horizontal shift: kc offset of -1 or +1 added to base.
#
# CR register parameters (set by harness):
#   cr0 = input base address
#   cr1 = kernel base address
#   cr2 = output base address
#   cr3 = mask base address
#   cr4 = cols (spatial width)
#   cr5 = num_chunks (= rows * cols / 128)
#   cr6 = in_group_stride (= in_channels * 128)
#   cr7 = 1792 (channel group size = 14 * 128, FPB=14 constant)
#   cr8 = total_kernel_bytes (= out_channels * ceil(in_channels/14) * 128)
#   cr9 = zero region address (128 bytes of zeros for S2 in last chunk)
#
# Partial-last-block clamp:
#   Each block-entry clamps the inner-loop limit lr11 to
#   min(lr10 + cr7, in_group_stride). This handles in_channels that are
#   not a multiple of FPB=14 without per-block bookkeeping.
#
# Mask slots (precomputed by harness, depend on cols):
#   slot 0: all zeros          -> no masking (kc=0)
#   slot 1: left border        -> zero col 0 of each packed row (kc=-1)
#   slot 2: right border       -> zero last col of each packed row (kc=+1)
#   Only 3 masks needed. Bottom border is handled by loading zeros
#   into S2 for the last chunk instead of using dedicated masks.
#
# LR register allocation:
#   lr0  = 0     (zero constant, mask slot 0, mask_shift, S0 cyclic index)
#   lr1  = 1     (mask slot 1 = left border, kc offset)
#   lr2  = 2     (mask slot 2 = right border)
#   lr3  = 128 - cols  (kr=-1 cyclic base)
#   lr4  = 128   (kr=0 cyclic base, S1 cyclic index, channel stride)
#   lr5  = 128 + cols  (kr=+1 cyclic base)
#   lr6  = kernel byte index within r0 (0..125 per r0 load, FPB=14)
#   lr7  = output pointer (global, continuous)
#   lr8  = chunk base address (chunk_index * in_group_stride)
#   lr9  = chunk counter
#   lr10 = channel offset within chunk (0, 128, ..., in_group_stride-128)
#   lr11 = channel group limit (lr10 + cr7, or lr10 + cr10 on last block)
#   lr12 = kernel memory offset (0, 128, ...; reset to 0 per chunk section)
#   lr13 = total_kernel_bytes (filter loop limit, copy of cr8)
#   lr14 = temp / in_group_stride at block-entry clamp
#   lr15 = temp / chunk loop limit / S0 base in ch_loop setup / cyclic-idx 256
#
# Note: in_group_stride (cr6) is accessed directly via CR in add/sub ops.
# The filter loop is: blt lr12 lr13 (where lr13 = cr8, set once per section).

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;
    ldr_mult_mask_reg   lr0 cr3;;

    set                 lr4 128;
    set                 lr1 1;;

    set                 lr2 2;
    sub                 lr3 lr4 cr4;;

    add                 lr5 lr4 cr4;;

# ===========================================================================
# Section 1: Chunk 0 (top border)
# S0 = zeros (cyclic register initialized to 0), load S1 and S2 only.
# kr=-1 taps read from S0 = zeros -> automatic zero-padding.
# ===========================================================================

    set                 lr8 0;
    set                 lr7 0;;

    set                 lr12 0;
    add                 lr13 cr8 lr0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Clamps the last partial block when in_channels % 14 != 0.
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 g0_clamp;;
    b                   g0_ch_loop;;
g0_clamp:
    add                 lr11 lr14 lr0;;

g0_ch_loop:
    # ===== Block preamble (2 cycles): load ch 0's S1 and S2 (S0 = zeros always). =====
    # Cycle 1: lr15 = S1 base; load S1 → slot 1.
    add                 lr15 lr8 lr10;
    ldr_cyclic_mult_reg lr15 cr0 lr4;;

    # Cycle 2: lr15 = S2 base (SNAPSHOT lr15 = S1 base); lr14 = 256;
    #          load S2 → slot 2. Jump into tap body (bypasses g0_ch_loop_cont).
    add                 lr15 lr15 cr6;
    add                 lr14 lr4 lr4;
    ldr_cyclic_mult_reg lr15 cr0 lr14;
    b                   g0_tap_body;;

g0_ch_loop_cont:
    # ===== Per-channel preamble (1 cycle): load CURRENT.S1 → slot 1; advance lr6. =====
    add                 lr14 lr8 lr10;
    incr                lr6 1;
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

g0_tap_body:
    # --- tap 1: kr=-1 kc=-1.  Sub-slot 2: advance lr10 to NEXT ch. ---
    sub                 lr14 lr3 lr1;
    incr                lr10 128;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 2: kr=-1 kc=0.  Sub-slot 2: NEXT.S1 base (SNAPSHOT lr10 = NEXT ch). ---
    incr                lr6 1;
    add                 lr15 lr8 lr10;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    # --- tap 3: kr=-1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  S0 = zeros so slot 0 needs no reload; sub-slot 2 free. ---
    incr                lr6 1;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0. ---
    incr                lr6 1;
    mult.ve             r0 lr5 lr0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- Branch cycle: load NEXT.S2 → slot 2 (slots 1 & 2 dead after tap 9).
    #     lr14 = NEXT.S2 base, lr15 = 256 (cyclic index for slot 2). ---
    add                 lr14 lr15 cr6;
    add                 lr15 lr4 lr4;
    ldr_cyclic_mult_reg lr14 cr0 lr15;
    blt                 lr10 lr11 g0_ch_loop_cont;;

    # Advance kernel offset; check for more channel groups
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 g0_reload;;

    # All input channels done -- store output and advance filter
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 g0_reload_clamp;;
    b                   g0_ch_loop;;
g0_reload_clamp:
    add                 lr11 lr14 lr0;;

    b                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop)
# Load all 3 chunks (S0, S1, S2). All 9 taps with normal masks.
# ===========================================================================

main_setup:
    add                 lr8 cr6 lr0;
    set                 lr9 1;;

    sub                 lr15 cr5 lr1;;

    # Guard: if no middle chunks (num_chunks <= 2, i.e. lr15 <= 1), skip main.
    # lr9=1, lr15=cr5-1. When cr5=2: lr15=1, blt 1<1 false -> skip to gN.
    # When cr5>2: lr15>1, blt 1<lr15 true -> fall through into row_loop.
    blt                 lr9 lr15 row_loop;;

    b                   gN_section;;

row_loop:
    set                 lr12 0;;

filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 mn_clamp;;
    b                   ch_loop;;
mn_clamp:
    add                 lr11 lr14 lr0;;

ch_loop:
    # ===== Block preamble (3 cycles): load ch 0's S0/S1/S2 =====
    # Cycle 1: lr15 = S1 base; load S1 → slot 1.
    add                 lr15 lr8 lr10;
    ldr_cyclic_mult_reg lr15 cr0 lr4;;

    # Cycle 2: lr14 = S0 base, lr15 = S2 base (both read SNAPSHOT lr15 = S1 base);
    #          load S0 → slot 0 (XMEM sees LIVE lr14 = S0 base).
    sub                 lr14 lr15 cr6;
    add                 lr15 lr15 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Cycle 3: lr14 = 256 (cyclic index for slot 2);
    #          load S2 → slot 2 (XMEM sees LIVE lr15 = S2 base, LIVE lr14 = 256).
    # Jump directly to tap body (bypassing the per-channel preamble at mn_ch_loop_cont).
    add                 lr14 lr4 lr4;
    ldr_cyclic_mult_reg lr15 cr0 lr14;
    b                   mn_tap_body;;

mn_ch_loop_cont:
    # ===== Per-channel preamble (1 cycle): load CURRENT.S1 → slot 1; advance lr6. =====
    # SNAPSHOT lr10 = CURRENT ch offset (advanced at tap 1).
    # incr lr6 1: advances from tap-9 index of prev ch to tap-1 index of current ch.
    # Falls through directly into mn_tap_body (no extra branch).
    add                 lr14 lr8 lr10;
    incr                lr6 1;
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

mn_tap_body:
    # --- tap 1: kr=-1 kc=-1.  Sub-slot 2: advance lr10 to NEXT ch. ---
    sub                 lr14 lr3 lr1;
    incr                lr10 128;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 2: kr=-1 kc=0.  Sub-slot 2: NEXT.S1 base (SNAPSHOT lr10 = NEXT ch offset). ---
    incr                lr6 1;
    add                 lr15 lr8 lr10;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    # --- tap 3: kr=-1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  Slot 0 dead after tap 4; load NEXT.S0 → slot 0.
    #     lr14 = NEXT.S0 base (reads SNAPSHOT lr15 = NEXT.S1 base). ---
    incr                lr6 1;
    sub                 lr14 lr15 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0. ---
    incr                lr6 1;
    mult.ve             r0 lr5 lr0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- Branch cycle: slots 1 & 2 dead after tap 9; load NEXT.S2 → slot 2.
    #     Sub-slot 1: lr14 = NEXT.S2 base (SNAPSHOT lr15 = NEXT.S1 base).
    #     Sub-slot 2: lr15 = 256 (cyclic index for slot 2).
    #     XMEM sees LIVE lr14 = S2 base, LIVE lr15 = 256.
    #     Loop back to mn_ch_loop_cont (1-cycle preamble) for next channel. ---
    add                 lr14 lr15 cr6;
    add                 lr15 lr4 lr4;
    ldr_cyclic_mult_reg lr14 cr0 lr15;
    blt                 lr10 lr11 mn_ch_loop_cont;;

mn_after_block:
    # Advance kernel offset; check for more channel groups (Stage 4 verbatim).
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 reload;;

    # All input channels done -- store and advance output filter
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 filter_loop;;

    # All filters done for this chunk -- advance to next chunk
    add                 lr8 lr8 cr6;
    incr                lr9 1;;

    # Restore lr15 = chunk limit (trashed during S2 index computation)
    sub                 lr15 cr5 lr1;;

    blt                 lr9 lr15 row_loop;;

    b                   gN_section;;

reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 mn_reload_clamp;;
    b                   ch_loop;;
mn_reload_clamp:
    add                 lr11 lr14 lr0;;

    b                   ch_loop;;

# ===========================================================================
# Section 3: Last chunk (bottom border)
# Load S0 and S1 normally. Load S2 from zero region (cr9) so that
# kr=+1 taps read zeros — standard masks 1/0/2 suffice.
# ===========================================================================

gN_section:
    set                 lr12 0;;

gN_filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 gN_clamp;;
    b                   gN_ch_loop;;
gN_clamp:
    add                 lr11 lr14 lr0;;

gN_ch_loop:
    # ===== Block preamble (3 cycles): load ch 0's S0/S1/S2 =====
    # S2 always loaded from cr9 (zero region) with base lr0 = 0.
    # Cycle 1: lr15 = S1 base; load S1 → slot 1.
    add                 lr15 lr8 lr10;
    ldr_cyclic_mult_reg lr15 cr0 lr4;;

    # Cycle 2: lr14 = S0 base, lr15 = S2 base (both read SNAPSHOT lr15 = S1 base);
    #          load S0 → slot 0.
    sub                 lr14 lr15 cr6;
    add                 lr15 lr15 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Cycle 3: lr14 = 256 (cyclic index for slot 2);
    #          load S2 from zero region → slot 2. Jump to tap body.
    add                 lr14 lr4 lr4;
    ldr_cyclic_mult_reg lr0 cr9 lr14;
    b                   gN_tap_body;;

gN_ch_loop_cont:
    # ===== Per-channel preamble (1 cycle): load CURRENT.S1 → slot 1; advance lr6. =====
    add                 lr14 lr8 lr10;
    incr                lr6 1;
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

gN_tap_body:
    # --- tap 1: kr=-1 kc=-1.  Sub-slot 2: advance lr10 to NEXT ch. ---
    sub                 lr14 lr3 lr1;
    incr                lr10 128;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 2: kr=-1 kc=0.  Sub-slot 2: NEXT.S1 base. ---
    incr                lr6 1;
    add                 lr15 lr8 lr10;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    # --- tap 3: kr=-1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  Slot 0 dead after tap 4; load NEXT.S0 → slot 0. ---
    incr                lr6 1;
    sub                 lr14 lr15 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- tap 7: kr=+1 kc=-1. ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    # --- tap 8: kr=+1 kc=0. ---
    incr                lr6 1;
    mult.ve             r0 lr5 lr0 lr0 lr6;
    acc;;

    # --- tap 9: kr=+1 kc=+1. ---
    incr                lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- Branch cycle: load NEXT.S2 from zero region → slot 2.
    #     lr14 = 256 (cyclic index); base is always lr0 = 0 (zero region constant).
    #     lr15 = 256 recomputed via add lr15 lr4 lr4 for the index operand.
    #     But: we need both NEXT.S0-base (done at tap 5) and NEXT.S2-load here.
    #     S2 base is always 0 (zero region), so: ldr_cyclic_mult_reg lr0 cr9 lr15
    #     with lr15 = 256. Sub-slot 1 frees up since no S2 base compute needed. ---
    add                 lr15 lr4 lr4;
    ldr_cyclic_mult_reg lr0 cr9 lr15;
    blt                 lr10 lr11 gN_ch_loop_cont;;

    # Advance kernel offset; check for more channel groups
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 gN_reload;;

    # All input channels done -- store and advance
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 gN_filter_loop;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    add                 lr11 lr10 cr7;
    add                 lr14 cr6 lr0;;
    blt                 lr14 lr11 gN_reload_clamp;;
    b                   gN_ch_loop;;
gN_reload_clamp:
    add                 lr11 lr14 lr0;;

    b                   gN_ch_loop;;
