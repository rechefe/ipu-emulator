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
#   cr7 = 3584 (channel group size = 28 * 128, FPB=28 constant)
#   cr8 = total_kernel_bytes (= out_channels * ceil(in_channels/28) * 256)
#   cr9 = zero region address (128 bytes of zeros for S2 in last chunk)
#   cr11 = chunk-loop limit (= (num_chunks - 1) * in_group_stride)
#
# Partial-last-block clamp:
#   Each super-block-entry clamps the inner-loop limit lr11 to
#   min(lr10 + cr7, in_group_stride). This handles in_channels that are
#   not a multiple of FPB=28 without per-block bookkeeping.
#
# Kernel super-block layout (FPB=28):
#   Each super-block is 256 bytes loaded as TWO ldr_mult_reg calls per
#   super-block-entry: r0 <- [lr12..lr12+128) holds channels 0..13,
#   r1 <- [lr12+128..lr12+256) holds channels 14..27. The mult.ve
#   `fixed_idx` operand (0..255) addresses both halves with a shared
#   index — fixed_idx 0..127 hits R0, 128..255 hits R1.
#   lr6 sweeps 0..251 across the body (28 channels * 9 taps); lr12
#   advances by 256 between super-blocks.
#
# Mask slots (precomputed by harness, depend on cols):
#   slot 0: all zeros          -> no masking (kc=0)
#   slot 1: left border        -> zero col 0 of each packed row (kc=-1)
#   slot 2: right border       -> zero last col of each packed row (kc=+1)
#   Only 3 masks needed. Bottom border is handled by loading zeros
#   into S2 for the last chunk instead of using dedicated masks.
#
# LR register allocation:
#   lr0  = 0  (permanent constant: zero value, mask slot 0 index, mask_shift,
#              S0 cyclic index for ldr_cyclic_mult_reg)
#   lr1  = 1  (permanent constant: mask slot 1 index = left border, kc=-1 offset)
#   lr2  = 2  (permanent constant: mask slot 2 index = right border)
#   lr3  = 128 - cols  (permanent constant: kr=-1 cyclic base)
#   lr4  = 128  (permanent constant: kr=0 cyclic base, S1 cyclic slot index,
#                channel stride for ldr_cyclic_mult_reg index argument)
#   lr5  = 128 + cols  (permanent constant: kr=+1 cyclic base)
#   lr6  = shared kernel byte index across R0/R1 (0..251 in a full super-block);
#           incremented by 1 each tap (incr lr6 1);
#           values 0..127 hit R0, 128..255 hit R1 via mult.ve fixed_idx;
#           reset to 0 at each super-block entry (after both ldr_mult_reg loads)
#   lr7  = output pointer (global, monotonically increasing);
#           incremented by 512 after each filter's output is stored
#   lr8  = chunk base address = chunk_index * in_group_stride;
#           incremented by cr6 at chunk advance (main/gN);
#           compared against cr11 for chunk-loop termination
#   lr9  = 256  (permanent constant: cyclic slot-2 index for all
#                ldr_cyclic_mult_reg ... lr9 calls across g0/mn/gN)
#   lr10 = channel offset within current chunk (0, 128, ..., in_group_stride-128);
#           incremented by 128 at tap 2 of each channel iteration;
#           reset to 0 at filter_loop entry
#   lr11 = channel group limit = min(lr10 + cr7, in_group_stride);
#           set at block entry (filter_loop / reload);
#           compared against lr10 at tap 9 blt for ch_loop_cont termination
#   lr12 = kernel super-block offset within current filter (0, 256, 512, ...);
#           incremented by 256 after each 28-channel group;
#           reset to 0 at each section entry (g0/main/gN)
#   lr13 = THIS.S1 base carrier (cross-channel in mn 9-cyc body):
#           [g0/gN preamble cycle 1] = lr8 + lr10  (THIS.S1 base; load S1)
#           [g0/gN tap 5]             = lr8 + lr10 (NEXT.S1 base; SNAPSHOT lr10=NEXT)
#           [g0/gN tap 8]            -> used as input to lr14 = lr13 + cr6 (NEXT.S2 base addr)
#           [mn preamble cycle 1]     = lr8 + lr10  (ch0.S1 base; load S1)
#           [mn tap 1]               -> XMEM addr for NEW.S1 load (= this ch's S1 base)
#           [mn tap 4]               -> input to lr14 = lr13 - cr6 (THIS.S0 base addr)
#           [mn tap 5]                = lr8 + lr10 (NEXT.S1 base; overwrites THIS.S1)
#           [mn tap 8]               -> input to lr14 = lr13 + cr6 (NEXT.S2 base addr)
#           [gN tap 1]                = lr14 - cr6 (THIS.S0 base; XMEM addr; old 10-cyc role)
#   lr14 = multi-role scratch (written and consumed within 1-2 cycles):
#           [g0/gN ch_loop_cont] THIS.S1 base = lr8 + lr10 (XMEM addr same cycle)
#           [mn preamble cy 2]   S0 base = lr13 - cr6 (XMEM addr)
#           [mn preamble cy 3]   S2 base = lr13 + cr6 (XMEM addr)
#           [mn tap 4]           THIS.S0 base = lr13 - cr6 (XMEM addr same cycle)
#           [mn tap 8 / g0/gN tap 8] NEXT.S2 base = lr13 + cr6 (XMEM addr)
#           [tap 1 LR2]    cyclic offset for kr=+1 kc=-1 = lr5 - lr1
#           [tap 3 LR2]    cyclic offset for kr=+1 kc=+1 = lr5 + lr1
#           [g0/gN tap 4]  cyclic offset for kr=0  kc=-1 = lr4 - lr1
#                          (mn tap 4 uses lr15=127 directly instead)
#           [tap 6 LR2]    cyclic offset for kr=0  kc=+1 = lr4 + lr1
#           [tap 7 LR2]    cyclic offset for kr=-1 kc=-1 = lr3 - lr1
#           [tap 9 LR2]    cyclic offset for kr=-1 kc=+1 = lr3 + lr1
#   lr15 = 127  (permanent constant: kr=0 kc=-1 cyclic offset = lr4-lr1).
#           Used at mn tap 4 to free the LR sub-slot that previously held
#           `sub lr14 lr4 lr1`, enabling the 9 cyc/ch fold (S0 address compute
#           occupies that freed sub-slot).
#
# Note: branches now accept CR operands directly (LcrIdx). We use:
#   filter loop:        blt lr12 cr8
#   chunk advance:      blt lr8  cr11
#   reload check:       blt lr10 cr6
#   clamp check:        blt cr6  lr11

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;
    set                 lr4 128;
    ldr_mult_mask_reg   lr0 cr3;;

    set                 lr2 2;
    set                 lr1 1;;

    sub                 lr15 lr4 lr1;     # lr15 = 127 (permanent: kr=0 kc=-1 cyclic offset)
    sub                 lr3 lr4 cr4;;

    add                 lr5 lr4 cr4;
    add                 lr9 lr4 lr4;;              # lr9 = 256 (cyclic-2 index, permanent)



# ===========================================================================
# Section 1: Chunk 0 (top border)
# S0 = zeros (cyclic register initialized to 0), load S1 and S2 only.
# kr=-1 taps read from S0 = zeros -> automatic zero-padding.
# ===========================================================================

    set                 lr8 0;
    set                 lr7 0;;

    set                 lr12 0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Clamps the last partial super-block when in_channels % 28 != 0.
    # Same cycle: lr14 = lr12+128 (R1 kernel half offset); load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 g0_clamp;;
    b                   g0_ch_loop;;
g0_clamp:
    add                 lr11 lr0 cr6;;

g0_ch_loop:
    # ===== Block preamble (2 cycles): load ch 0's S1 and S2 (S0 = zeros always). =====
    # 9 cyc/ch body: NO ch_loop_cont. Tap 1 loads NEW.S1 (same-cycle visibility).
    # S0 stays zero throughout g0 (never loaded).
    # lr13 carries NEXT.S1 base across channel boundary (set at tap 5).
    #
    # Cycle 1: lr13 = S1 base; load S1 → slot 1.
    add                 lr13 lr8 lr10;
    ldr_cyclic_mult_reg lr13 cr0 lr4;;

    # Cycle 2: lr14 = S2 base (= lr13 + cr6); load S2 → slot 2; set lr6 = -1; branch.
    set                 lr6 -1;
    add                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    b                   g0_tap_body;;

g0_tap_body:
    # ===== 9 cyc/ch body. lr15 = 127 permanent constant for tap 4 cyclic offset. =====
    # Slot 0 = zeros (never loaded in g0), read at taps 7-9
    # Slot 1 holds THIS.S1, read every tap; loaded at tap 1 (same-cycle, slot 1 frozen at fresh data)
    # Slot 2 holds THIS.S2, read at taps 1-3; loaded at tap 8 (prev ch's body)
    # lr13 (cross-channel): NEXT.S1 base. Set at prev ch's tap 5; preamble cy 1 (for ch0).

    # --- tap 1: kr=+1 kc=-1.  XMEM loads NEW.S1 → slot 1 (lr13 LIVE = NEW.S1 base).
    #     Same-cycle write/read of slot 1: MULT reads freshly-loaded NEW.S1 (correct). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr5 lr1;
    ldr_cyclic_mult_reg lr13 cr0 lr4;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 2: kr=+1 kc=0.  Sub-slot 2: incr lr10 to NEXT ch. ---
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr5 0 lr0 lr6;
    acc;;

    # --- tap 3: kr=+1 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1.  Use lr15=127 as cyclic offset; sub-slot 2 free.
    #     (S0 is always zero in g0 — no S0 load needed.) ---
    add                 lr6 lr6 1;
    mult.ve.cyclic      lr15 1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  Sub-slot 2: lr13 = NEXT.S1 base (SNAPSHOT lr10 = NEXT). ---
    add                 lr6 lr6 1;
    add                 lr13 lr8 lr10;
    mult.ve.cyclic      lr4 0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=-1 kc=-1.  Slot 0 = zeros; reads zero. ---
    add                 lr6 lr6 1;
    sub                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 8: kr=-1 kc=0.  Sub-slot 2: lr14 = NEXT.S2 base = lr13 + cr6.
    #     XMEM: load NEXT.S2 → slot 2. Slot 2 dead since tap 4.
    #     lr14 reused (tap 8 mult uses lr3 directly, not lr14). ---
    add                 lr6 lr6 1;
    add                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=-1 kc=+1.  Branch DIRECTLY to tap 1 (no ch_loop_cont). ---
    add                 lr6 lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;
    blt                 lr10 lr11 g0_tap_body;;

    # Advance kernel offset; check for more channel groups
    add                 lr12 lr12 cr13;

    blt                 lr10 cr6 g0_reload;;

    # All input channels done -- store output and advance filter
    str_acc_reg         lr7 cr2;;

    add                 lr7 lr7 cr13;;
    add                 lr7 lr7 cr13;

    blt                 lr12 cr8 g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Same cycle: lr14 = lr12+128; load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 g0_reload_clamp;;
    b                   g0_ch_loop;;
g0_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop)
# Load all 3 chunks (S0, S1, S2). All 9 taps with normal masks.
# ===========================================================================

main_setup:
    # Guard: if no middle chunks (num_chunks <= 2), skip main.
    # cr11 = (cr5-1)*cr6 = chunk_limit. When cr5=2: cr11=cr6, blt lr8<cr11
    # is blt cr6<cr6 = false -> skip to gN. When cr5>2: cr11 > cr6 -> enter row_loop.
    add                 lr8 lr0 cr6;;
    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

row_loop:
    set                 lr12 0;;

filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Same cycle: lr14 = lr12+128; load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 mn_clamp;;
    b                   ch_loop;;
mn_clamp:
    add                 lr11 lr0 cr6;;

ch_loop:
    # ===== Block preamble (3 cycles): load ch 0's S0/S1/S2 =====
    # 9 cyc/ch body: NO ch_loop_cont. Tap 1 loads NEW.S1, tap 4 loads THIS.S0,
    # tap 8 loads NEXT.S2. lr13 carries NEXT.S1 base across channel boundary.
    #
    # Cycle 1: lr13 = S1 base; load S1 → slot 1.
    add                 lr13 lr8 lr10;
    ldr_cyclic_mult_reg lr13 cr0 lr4;;

    # Cycle 2: lr14 = S0 base (= lr13 - cr6); load S0 → slot 0.
    sub                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Cycle 3: lr14 = S2 base (= lr13 + cr6); load S2 → slot 2; branch into body.
    # Also set lr6 = -1 so tap 1's `incr lr6 1` brings it to 0 for ch0's first byte.
    set                 lr6 -1;
    add                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    b                   mn_tap_body;;

mn_tap_body:
    # ===== 9 cyc/ch body. lr15 = 127 permanent constant for tap 4 cyclic offset. =====
    # Slot 0 holds THIS.S0, read at taps 7-9 ; loaded at tap 4 (same-cycle visibility OK)
    # Slot 1 holds THIS.S1, read every tap   ; loaded at tap 1 (same-cycle, slot 1 frozen at fresh data)
    # Slot 2 holds THIS.S2, read at taps 1-3 ; loaded at tap 8 (prev channel's body)
    # lr13 (cross-channel): NEXT.S1 base. Set at prev ch's tap 5; preamble's cycle 1
    #                       (for ch0). Used at this ch's tap 1 (XMEM addr) and as the
    #                       base for THIS.S0 (tap 4) and NEXT.S2 (tap 8).

    # --- tap 1: kr=+1 kc=-1.  XMEM loads NEW.S1 → slot 1 (lr13 LIVE = NEW.S1 base).
    #     Same-cycle write/read of slot 1: MULT reads freshly-loaded NEW.S1 (correct). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr5 lr1;
    ldr_cyclic_mult_reg lr13 cr0 lr4;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 2: kr=+1 kc=0.  Sub-slot 2: incr lr10 to NEXT ch. ---
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr5 0 lr0 lr6;
    acc;;

    # --- tap 3: kr=+1 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1. cyclic_offset = lr15 = 127 (permanent constant).
    #     LR2 free → lr14 = THIS.S0 base = lr13 - cr6.
    #     XMEM loads THIS.S0 → slot 0.
    #     Same-cycle: MULT lane 0 reads slot 0 lane 127 = freshly-loaded THIS.S0
    #     last byte (correct kc=-1 wrap value for kr=0). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;
    mult.ve.cyclic      lr15 1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  Sub-slot 2: lr13 = NEXT.S1 base (SNAPSHOT lr10 = NEXT).
    #     This overwrites lr13 (was THIS.S1 base) → tap 8 uses lr13 = NEXT.S1 base. ---
    add                 lr6 lr6 1;
    add                 lr13 lr8 lr10;
    mult.ve.cyclic      lr4 0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=-1 kc=-1.  Slot 0 = THIS.S0 (loaded at tap 4). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 8: kr=-1 kc=0.  Sub-slot 2: lr14 = NEXT.S2 base = lr13 + cr6.
    #     XMEM: load NEXT.S2 → slot 2. Slot 2 dead since tap 4 — safe.
    #     lr14 reused (mult uses lr3 directly, not lr14). ---
    add                 lr6 lr6 1;
    add                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=-1 kc=+1.  Branch DIRECTLY to tap 1 (no ch_loop_cont). ---
    add                 lr6 lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;
    blt                 lr10 lr11 mn_tap_body;;

mn_after_block:
    # Advance kernel offset; check for more channel groups.
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 reload;;

    # All input channels done -- store and advance output filter
    str_acc_reg         lr7 cr2;;

    add                 lr7 lr7 cr13;;
    add                 lr7 lr7 cr13;
    blt                 lr12 cr8 filter_loop;;

    # All filters done for this chunk -- advance to next chunk.
    # Compare lr8 (chunk base addr) directly against cr11 = (cr5-1)*cr6.
    add                 lr8 lr8 cr6;;

    blt                 lr8 cr11 row_loop;;

    b                   gN_section;;

reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Same cycle: lr14 = lr12+128; load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 mn_reload_clamp;;
    b                   ch_loop;;
mn_reload_clamp:
    add                 lr11 lr0 cr6;
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
    # Same cycle: lr14 = lr12+128; load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 gN_clamp;;
    b                   gN_ch_loop;;
gN_clamp:
    add                 lr11 lr0 cr6;;

gN_ch_loop:
    # ===== Block preamble (3 cycles): load ch 0's S0/S1/S2 =====
    # 9 cyc/ch body: NO ch_loop_cont. Tap 1 loads NEW.S1, tap 4 loads THIS.S0,
    # tap 8 loads zero-region S2. lr13 carries NEXT.S1 base across channel boundary.
    # S2 always loaded from cr9 (zero region).
    #
    # Cycle 1: lr13 = S1 base; load S1 → slot 1.
    add                 lr13 lr8 lr10;
    ldr_cyclic_mult_reg lr13 cr0 lr4;;

    # Cycle 2: lr14 = S0 base (= lr13 - cr6); load S0 → slot 0.
    sub                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Cycle 3: load S2 from zero region → slot 2. Set lr6=-1; branch.
    set                 lr6 -1;
    ldr_cyclic_mult_reg lr0 cr9 lr9;
    b                   gN_tap_body;;

gN_tap_body:
    # ===== 9 cyc/ch body. lr15 = 127 permanent constant for tap 4 cyclic offset. =====
    # Slot 0 holds THIS.S0 (real data), read at taps 7-9; loaded at tap 4 (same-cycle)
    # Slot 1 holds THIS.S1, read every tap; loaded at tap 1 (same-cycle visibility)
    # Slot 2 holds zeros (from cr9), read at taps 1-3; loaded at tap 8 (prev ch's body)
    # lr13 (cross-channel): NEXT.S1 base. Set at prev ch's tap 5; preamble cy 1 (for ch0).

    # --- tap 1: kr=+1 kc=-1.  XMEM loads NEW.S1 → slot 1 (lr13 LIVE = NEW.S1 base).
    #     Same-cycle write/read of slot 1: MULT reads freshly-loaded NEW.S1 (correct). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr5 lr1;
    ldr_cyclic_mult_reg lr13 cr0 lr4;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 2: kr=+1 kc=0.  Sub-slot 2: incr lr10 to NEXT ch. ---
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;
    mult.ve.cyclic      lr5 0 lr0 lr6;
    acc;;

    # --- tap 3: kr=+1 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 4: kr=0 kc=-1. cyclic_offset = lr15 = 127 (permanent constant).
    #     LR2 free → lr14 = THIS.S0 base = lr13 - cr6.
    #     XMEM loads THIS.S0 → slot 0 (same-cycle visibility OK). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr13 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;
    mult.ve.cyclic      lr15 1 lr0 lr6;
    acc;;

    # --- tap 5: kr=0 kc=0.  Sub-slot 2: lr13 = NEXT.S1 base (SNAPSHOT lr10 = NEXT). ---
    add                 lr6 lr6 1;
    add                 lr13 lr8 lr10;
    mult.ve.cyclic      lr4 0 lr0 lr6;
    acc;;

    # --- tap 6: kr=0 kc=+1. ---
    add                 lr6 lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- tap 7: kr=-1 kc=-1.  Slot 0 = THIS.S0 (loaded at tap 4). ---
    add                 lr6 lr6 1;
    sub                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    # --- tap 8: kr=-1 kc=0.  XMEM: load zero region → slot 2 (lr0=0 base, lr9=256 idx). ---
    add                 lr6 lr6 1;
    ldr_cyclic_mult_reg lr0 cr9 lr9;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    # --- tap 9: kr=-1 kc=+1.  Branch DIRECTLY to tap 1 (no ch_loop_cont). ---
    add                 lr6 lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;
    blt                 lr10 lr11 gN_tap_body;;

    # Advance kernel offset; check for more channel groups
    add                 lr12 lr12 cr13;
    blt                 lr10 cr6 gN_reload;;

    # All input channels done -- store and advance
    str_acc_reg         lr7 cr2;;

    add                 lr7 lr7 cr13;;
    add                 lr7 lr7 cr13;
    blt                 lr12 cr8 gN_filter_loop;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    # Inner-loop limit: min(lr10+cr7, in_group_stride).
    # Same cycle: lr14 = lr12+128; load R1.
    add                 lr14 lr12 lr4;
    add                 lr11 lr10 cr7;
    ldr_mult_reg        r1 lr14 cr1;;
    blt                 cr6 lr11 gN_reload_clamp;;
    b                   gN_ch_loop;;
gN_reload_clamp:
    add                 lr11 lr0 cr6;
    b                   gN_ch_loop;;
