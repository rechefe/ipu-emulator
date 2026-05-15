# Hardcoded First-Layer Conv: 256x256x3 -> 128x128x16, stride 2, INT8
#
# Standard 3x3 convolution: 3 input channels -> 16 output filters.
# Each filter has 3x3x3 = 27 kernel weights.
#
# Input: 256x256x3 in CHW layout (channel-first, row-major).
#   Channel c, row r, col c_pos: INPUT_BASE + c*65536 + r*256 + c_pos
#   Left half  (cols 0..127):  offset + 0
#   Right half (cols 128..255): offset + 128
#
# Output: 128x128x16 in interleaved 128-byte chunks (INT8 via aaq).
#   Row r, filter f: OUTPUT_BASE + (r*16 + f) * 128
#
# Kernel: 16 filters x 128 bytes each.
#   Per filter: ch0 taps[0..8], ch1 taps[9..17], ch2 taps[18..26], padding.
#   Tap order per channel: (kr=-1,kc=-1), (kr=-1,kc=0), (kr=-1,kc=+1),
#                           (kr=0, kc=-1), (kr=0, kc=0), (kr=0, kc=+1),
#                           (kr=+1,kc=-1), (kr=+1,kc=0), (kr=+1,kc=+1)
#
# Processing: For each output row, each filter:
#   1. LEFT HALF: 3 channels x 9 taps -> aaq -> temp[0]
#   2. RIGHT HALF: same with addr+128 -> aaq -> temp[128]
#   3. STRIDE: acc.stride 64 on off packs 64+64 -> 128 output elements
#   4. aaq -> store output
#
# Boundary handling:
#   - Top border (row 0): skip kr=-1 taps
#   - Left border: mask slot 1 zeros position 0 for kc=-1
#   - Right border: mask slot 2 zeros position 127 for kc=+1
#   - Internal boundary (col 127/128): right-half position 0 kc=-1 masked
#     (same slot 1 mask). Affects output col 64 only.
#
# Loads one row at a time into cyclic[0], applies 3 kc taps, then next row.
# (Same approach as conv_128x128_4to8.)
#
# CR registers:
#   cr0  = INPUT_BASE (channel 0)
#   cr1  = INPUT_BASE + 65536 (channel 1)
#   cr2  = INPUT_BASE + 131072 (channel 2)
#   cr3  = KERNEL_BASE
#   cr4  = MASK_BASE
#   cr5  = OUTPUT_BASE
#   cr6  = TEMP_BASE (256 bytes: temp_left + temp_right)
#   cr9  = 1 (identity scalar for mult.ve.cr)
#   cr12 = 128  (step constant: kernel/output advance)
#
# LR registers:
#   lr0  = 0     (zero, mask slot 0, cyclic index, kc=0 offset)
#   lr1  = 1     (kc=+1 offset, mask slot 1)
#   lr2  = 2     (mask slot 2, stride offset for right half)
#   lr3  = 0..26 (kernel byte index / tap counter)
#   lr4  = 128   (right-half offset, temp_right offset)
#   lr5  = row offset (2 * output_row * 256, advances by 512)
#   lr6  = 256   (row stride for address computation)
#   lr7  = output pointer (advances by 128 per filter store)
#   lr8  = filter kernel offset (0, 128, ..., 1920)
#   lr9  = output row counter
#   lr10 = 511   (kc=-1 cyclic offset)
#   lr11 = 2048  (total kernel bytes, filter loop limit)
#   lr12 = 512   (row advance = 2 * 256)
#   lr13 = 128   (main loop row limit)
#   lr14 = temp
#   lr15 = 4     (channel skip: advance lr3 by 4 between channels)

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr0 cr0;
    ldr_mult_mask_reg   lr0 cr4;;

    SET                 lr4 cr12;
    SET                 lr1 cr9;;

    add                 lr2 lr0 2;
    SET                 lr6 cr7;;

    SET                 lr10 cr8;
    SET                 lr11 cr11;;

    SET                 lr12 cr10;
    SET                 lr13 cr12;;

    add                 lr15 lr0 4;;

# ===========================================================================
# Row 0 (top border): skip kr=-1 taps (6 taps/channel instead of 9)
# Center row = 0, need rows -1, 0, 1. Row -1 doesn't exist -> skip.
# ===========================================================================

    SET                 lr5 cr0;
    SET                 lr7 cr0;;

    SET                 lr8 cr0;;

row0_filter_loop:
    ldr_mult_reg        r0 lr8 cr3;
    add                 lr3 lr0 3;;

    reset_acc;;

    # --- LEFT HALF: channel 0, kr=0 and kr=+1 ---

    # kr=0: load row 0 left, channel 0
    ldr_cyclic_mult_reg lr5 cr0 lr0;;

    # Tap 3 (kr=0, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    # Tap 4 (kr=0, kc=0)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    # Tap 5 (kr=0, kc=+1)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: load row 1 left, channel 0
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Tap 6 (kr=+1, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    # Tap 7 (kr=+1, kc=0)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    # Tap 8 (kr=+1, kc=+1)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # --- LEFT HALF: channel 1, kr=0 and kr=+1 ---

    # Skip taps 9-11 (kr=-1 for ch1), advance to tap 12
    add                 lr3 lr3 lr15;;

    # kr=0: load row 0 left, channel 1
    ldr_cyclic_mult_reg lr5 cr1 lr0;;

    # Tap 12 (kr=0, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: load row 1 left, channel 1
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    # Tap 15 (kr=+1, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # --- LEFT HALF: channel 2, kr=0 and kr=+1 ---

    # Skip taps 18-20, advance to tap 21
    add                 lr3 lr3 lr15;;

    # kr=0: load row 0 left, channel 2
    ldr_cyclic_mult_reg lr5 cr2 lr0;;

    # Tap 21 (kr=0, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: load row 1 left, channel 2
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    # Tap 24 (kr=+1, kc=-1)
    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # Quantize left half, store to temp[0]
    aaq;;
    xmem.store_aaq_result lr0 cr6;;

    # --- RIGHT HALF: channel 0, kr=0 and kr=+1 ---
    add                 lr3 lr0 3;
    reset_acc;;

    # kr=0: load row 0 right, channel 0
    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: load row 1 right, channel 0
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # --- RIGHT HALF: channel 1, kr=0 and kr=+1 ---
    add                 lr3 lr3 lr15;;

    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # --- RIGHT HALF: channel 2, kr=0 and kr=+1 ---
    add                 lr3 lr3 lr15;;

    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # Quantize right half, store to temp[128]
    aaq;;
    xmem.store_aaq_result lr4 cr6;;

    # --- STRIDE STEP ---
    ldr_cyclic_mult_reg lr0 cr6 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    reset_acc;;

    acc.stride          64 on off lr0;;

    ldr_cyclic_mult_reg lr4 cr6 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;;

    acc.stride          64 on off lr2;;

    # Store output
    aaq;;
    xmem.store_aaq_result lr7 cr5;;

    add                 lr7 lr7 cr12;
    add                 lr8 lr8 cr12;;

    blt                 lr8 lr11 row0_filter_loop;;

# ===========================================================================
# Rows 1-127 (main loop): all 3 kernel rows, 3 channels, 16 filters
# ===========================================================================

    add                 lr5 lr12 lr0;
    SET                 lr9 cr9;;

row_loop:
    SET                 lr8 cr0;;

filter_loop:
    ldr_mult_reg        r0 lr8 cr3;
    SET                 lr3 cr0;;

    reset_acc;;

    # === LEFT HALF: channel 0 (taps 0..8) ===

    # kr=-1: load row(2r-1) left, ch0
    sub                 lr14 lr5 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=0: load row(2r) left, ch0
    add                 lr3 lr3 1;
    ldr_cyclic_mult_reg lr5 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: load row(2r+1) left, ch0
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # === LEFT HALF: channel 1 (taps 9..17) ===

    add                 lr3 lr3 1;
    sub                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    ldr_cyclic_mult_reg lr5 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # === LEFT HALF: channel 2 (taps 18..26) ===

    add                 lr3 lr3 1;
    sub                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    ldr_cyclic_mult_reg lr5 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # Quantize left half, store to temp[0]
    aaq;;
    xmem.store_aaq_result lr0 cr6;;

    # === RIGHT HALF: channel 0 (taps 0..8) ===
    SET                 lr3 cr0;
    reset_acc;;

    # kr=-1: row(2r-1) right, ch0
    sub                 lr14 lr5 lr6;;
    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=0: row(2r) right, ch0
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # kr=+1: row(2r+1) right, ch0
    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # === RIGHT HALF: channel 1 (taps 9..17) ===

    add                 lr3 lr3 1;
    sub                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr1 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # === RIGHT HALF: channel 2 (taps 18..26) ===

    add                 lr3 lr3 1;
    sub                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr4;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr14 lr5 lr6;;

    add                 lr14 lr14 lr4;;

    ldr_cyclic_mult_reg lr14 cr2 lr0;;

    mult.ve.cyclic      lr10 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr0 0 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    mult.ve.cyclic      lr1 2 lr0 lr3;
    acc;;

    # Quantize right half, store to temp[128]
    aaq;;
    xmem.store_aaq_result lr4 cr6;;

    # --- STRIDE STEP ---
    ldr_cyclic_mult_reg lr0 cr6 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    reset_acc;;

    acc.stride          64 on off lr0;;

    ldr_cyclic_mult_reg lr4 cr6 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;;

    acc.stride          64 on off lr2;;

    # Store output
    aaq;;
    xmem.store_aaq_result lr7 cr5;;

    add                 lr7 lr7 cr12;
    add                 lr8 lr8 cr12;;

    blt                 lr8 lr11 filter_loop;;

    # Advance to next output row
    add                 lr5 lr5 lr12;
    add                 lr9 lr9 1;;

    blt                 lr9 lr13 row_loop;;

end:
    bkpt;;
