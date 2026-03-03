# Depthwise convolution: 1 channel, 64x64, 3x3 kernel, same-size output.
#
# Each 128-byte SIMD chunk packs 2 spatial rows (64 cols x 2 rows).
# 32 chunks total.  The cyclic register holds 3 neighboring chunks
# (S0=chunk_{i-1}, S1=chunk_i, S2=chunk_{i+1}) so that vertical
# neighbor access is a simple offset into the cyclic register:
#
#   kr=-1 (row above): cyclic offset 64  -> [S0[64:128], S1[0:64]]
#   kr= 0 (same row):  cyclic offset 128 -> S1
#   kr=+1 (row below): cyclic offset 192 -> [S1[64:128], S2[0:64]]
#
# Horizontal shifts add +/-1 to the base cyclic offset.
#
# Border handling:
#   - Column 0 / column 63 edges are masked via r_mask.
#   - Chunk 0 (top border): cyclic register initialises to zeros,
#     so S0 = zeros without loading.
#   - Chunk 31 (bottom border): unrolled with special mask slots
#     that zero positions 64-127 (row 63's "below" contribution).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (32 chunks x 128 bytes = 4096 bytes)
#   cr1 = kernel base  (9 bytes in positions 0-8, padded to 128 bytes)
#   cr2 = output base  (32 chunks x 512 bytes = 16384 bytes, 32-bit acc)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Kernel layout in r0[0..8]:
#   [0]=k(-1,-1)  [1]=k(-1, 0)  [2]=k(-1,+1)
#   [3]=k( 0,-1)  [4]=k( 0, 0)  [5]=k( 0,+1)
#   [6]=k(+1,-1)  [7]=k(+1, 0)  [8]=k(+1,+1)
#
# Mask slots in r_mask:
#   slot 0: all zeros          -> no masking               (kc = 0)
#   slot 1: bits {0, 64} set   -> zero col 0 of each row   (kc = -1)
#   slot 2: bits {63, 127} set -> zero col 63 of each row  (kc = +1)
#   slot 3: bits {64..127}     -> zero bottom row           (last chunk, kr=+1, kc=0)
#   slot 4: bits {0, 64..127}  -> left border + bottom row  (last chunk, kr=+1, kc=-1)
#   slot 5: bits {63..127}     -> right border + bottom row (last chunk, kr=+1, kc=+1)
#
# Register allocation:
#   lr0  = 0    (mask_shift, mask_offset=0, cyclic index S0, kernel idx 0)
#   lr1  = S0 input offset (loop variable, = (chunk-1)*128)
#   lr2  = output write offset (loop variable)
#   lr3  = temp (kernel index / cyclic offset scratch)
#   lr4  = temp (cyclic offset scratch)
#   lr5  = 128  (chunk stride, cyclic index S1)
#   lr6  = 256  (cyclic index S2)
#   lr7  = 512  (output stride)
#   lr8  = 1    (mask_offset for kc=-1, kernel idx 1)
#   lr9  = 2    (mask_offset for kc=+1, kernel idx 2)
#   lr10 = 3840 (main loop end: 30 * 128)
#   lr11 = 3    (mask_offset for last-chunk slot 3, kernel idx 3)
#   lr12 = 4    (mask_offset for last-chunk slot 4, kernel idx 4)
#   lr13 = 5    (mask_offset for last-chunk slot 5, kernel idx 5)
#   lr14 = unused
#   lr15 = unused

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_reg        r0 lr0 cr1;
    set                 lr5 128;
    set                 lr7 512;;

    ldr_mult_mask_reg   lr0 cr3;
    set                 lr6 256;
    set                 lr8 1;;

    set                 lr9 2;
    set                 lr10 3840;;

    set                 lr11 3;
    set                 lr12 4;;

    set                 lr13 5;;

# ===========================================================================
# Chunk 0 (rows 0-1) -- top border
# S0 = zeros (cyclic reg initialised to 0), only load S1 and S2.
# ===========================================================================

    ldr_cyclic_mult_reg lr0 cr0 lr5;;

    ldr_cyclic_mult_reg lr5 cr0 lr6;
    reset_acc;;

    # --- kr=-1 (row above): base cyclic offset 64 ---
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr0;
    acc;;

    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr8;
    acc;;

    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr9;
    acc;;

    # --- kr=0 (same row): base cyclic offset 128 ---
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr11;
    acc;;

    mult.ve             r0 lr5 lr0 lr0 lr12;
    acc;;

    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr13;
    acc;;

    # --- kr=+1 (row below): base cyclic offset 192 ---
    set                 lr3 6; set lr4 191;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    set                 lr3 7; set lr4 192;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    set                 lr3 8; set lr4 193;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Store chunk 0 output
    str_acc_reg         lr0 cr2;;

# ===========================================================================
# Main loop: chunks 1 .. 30
# ===========================================================================

    set                 lr1 0;
    set                 lr2 512;;

row_loop:
    # Load three neighbouring chunks into cyclic register
    add                 lr3 lr1 lr5; add lr4 lr1 lr6;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    ldr_cyclic_mult_reg lr3 cr0 lr5;;

    ldr_cyclic_mult_reg lr4 cr0 lr6;
    reset_acc;;

    # --- kr=-1 (row above): base cyclic offset 64 ---
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr0;
    acc;;

    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr8;
    acc;;

    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr9;
    acc;;

    # --- kr=0 (same row): base cyclic offset 128 ---
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr11;
    acc;;

    mult.ve             r0 lr5 lr0 lr0 lr12;
    acc;;

    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr13;
    acc;;

    # --- kr=+1 (row below): base cyclic offset 192 ---
    set                 lr3 6; set lr4 191;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    set                 lr3 7; set lr4 192;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    set                 lr3 8; set lr4 193;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Store and advance
    str_acc_reg         lr2 cr2;;

    incr                lr1 128;
    incr                lr2 512;;

    blt                 lr1 lr10 row_loop;;

# ===========================================================================
# Chunk 31 (rows 62-63) -- bottom border
# S2 data is stale but kr=+1 uses special masks (slots 3/4/5) to zero
# positions 64-127, so stale S2 content is harmless.
# ===========================================================================

    # Load S0 = chunk_30, S1 = chunk_31
    add                 lr3 lr1 lr5;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    ldr_cyclic_mult_reg lr3 cr0 lr5;
    reset_acc;;

    # --- kr=-1 (row above): normal masks ---
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr0;
    acc;;

    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr8;
    acc;;

    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr9;
    acc;;

    # --- kr=0 (same row): normal masks ---
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr11;
    acc;;

    mult.ve             r0 lr5 lr0 lr0 lr12;
    acc;;

    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr13;
    acc;;

    # --- kr=+1 (row below): special masks (slots 4/3/5) to zero bottom row ---
    set                 lr3 6; set lr4 191;
    mult.ve             r0 lr4 lr12 lr0 lr3;
    acc;;

    set                 lr3 7; set lr4 192;
    mult.ve             r0 lr4 lr11 lr0 lr3;
    acc;;

    set                 lr3 8; set lr4 193;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # Store chunk 31 output
    str_acc_reg         lr2 cr2;;

end:
    bkpt;;
