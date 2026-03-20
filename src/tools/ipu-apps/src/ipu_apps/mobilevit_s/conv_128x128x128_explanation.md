# Regular Convolution 128×128×128 → 128×128×128 on the IPU

This document explains, step by step, how a standard 2D convolution with 128 input
channels and 128 output channels on a 128×128 spatial image runs on the IPU. It assumes
a 3×3 kernel with zero-padding and stride 1, so the output has the same spatial size as
the input. The output is **quantized back to 8 bits** — the accumulator works in 32-bit
internally, but a quantization step converts each output element to a single byte before
storing, so the output has the same format as the input.

---

## 1. What the Convolution Computes

For every output pixel `(row, col)` and every output channel `f` (0–127):

```
output[row][col][f] = sum over ic=0..127, kr=-1..+1, kc=-1..+1 of:
    input[row+kr][col+kc][ic] * kernel[f][ic][kr][kc]
```

That is 128 input channels × 9 kernel positions = **1,152 multiply-accumulate operations
per output pixel per output channel**. The entire output is 128×128×128 pixels, so the
total work is 128×128×128×1,152 ≈ 2.4 billion MACs.

---

## 2. The IPU's Core Processing Model

The IPU processes data in **128-byte vectors**. Its multiply unit (`mult.ve`) multiplies
two 128-element vectors element-wise in a single cycle, producing 128 partial products.
The accumulator (`acc`) adds those 128 products into a running 128-element sum
(`r_acc`, stored as 32-bit integers — 512 bytes total).

A single VLIW (Very Long Instruction Word) cycle can execute one memory operation, one
multiply, one accumulate, and several register updates **all in parallel**. The algorithm
is designed to keep all these slots busy every cycle.

---

## 3. Memory Layout — How the Data Is Organized

### 3.1 Input (128×128 spatial, 128 channels)

Since each spatial row is exactly 128 bytes wide (128 columns × 1 byte per element),
each **chunk** holds exactly **one spatial row of one channel**.

The chunks are interleaved by channel within each row:

```
Address 0:        row 0, channel 0   (128 bytes)
Address 128:      row 0, channel 1   (128 bytes)
Address 256:      row 0, channel 2   (128 bytes)
...
Address 128×127:  row 0, channel 127 (128 bytes)
Address 128×128:  row 1, channel 0   (128 bytes)
Address 128×129:  row 1, channel 1   (128 bytes)
...
```

The **row group stride** (distance from one row to the same channel in the next row) is
`128 channels × 128 bytes = 16,384 bytes`.

**Total input size**: 128 rows × 128 channels × 128 bytes = **2 MB**.

### 3.2 Kernel Weights (128 output filters × 128 input channels × 3×3)

The kernel is stored filter by filter. Each filter has 128 input channels, but the IPU
can only load 128 bytes at a time into its multiply register `r0`. Since each input
channel contributes 9 kernel weights (3×3), we can fit **8 input channels** into one
128-byte block: 8 × 9 = 72 bytes of data + 56 bytes of padding.

Each filter therefore needs `128 / 8 = 16 blocks` of 128 bytes:

```
Block 0:  ic 0–7   → 72 bytes of weights, 56 bytes padding
Block 1:  ic 8–15  → 72 bytes of weights, 56 bytes padding
...
Block 15: ic 120–127 → 72 bytes of weights, 56 bytes padding
```

Within each 72-byte region, the 9 weights for each input channel are stored consecutively:

```
Byte offset within block for input channel ic (0–7):
  ic*9 + 0: kernel weight at (kr=-1, kc=-1)  — top-left neighbor
  ic*9 + 1: kernel weight at (kr=-1, kc= 0)  — top-center
  ic*9 + 2: kernel weight at (kr=-1, kc=+1)  — top-right
  ic*9 + 3: kernel weight at (kr= 0, kc=-1)  — left
  ic*9 + 4: kernel weight at (kr= 0, kc= 0)  — center
  ic*9 + 5: kernel weight at (kr= 0, kc=+1)  — right
  ic*9 + 6: kernel weight at (kr=+1, kc=-1)  — bottom-left
  ic*9 + 7: kernel weight at (kr=+1, kc= 0)  — bottom-center
  ic*9 + 8: kernel weight at (kr=+1, kc=+1)  — bottom-right
```

**Total kernel size**: 128 filters × 16 blocks × 128 bytes = **256 KB**.

### 3.3 Output (128×128 spatial, 128 channels)

Although the accumulator works internally in 32 bits (128 elements × 4 bytes = 512 bytes
in `r_acc`), the final result is **quantized to 8 bits** before storing. Each output
element is therefore 1 byte, and one output chunk is 128 elements × 1 byte = **128 bytes**
— the same size as an input chunk. The layout mirrors the input exactly:

```
Address 0:        row 0, filter 0   (128 bytes)
Address 128:      row 0, filter 1   (128 bytes)
...
Address 128×127:  row 0, filter 127 (128 bytes)
Address 128×128:  row 1, filter 0   (128 bytes)
...
```

**Total output size**: 128 rows × 128 filters × 128 bytes = **2 MB** (same as the input).

This is a major advantage of quantization: the output can be fed directly as input to
the next convolution layer without any format conversion.

---

## 4. The Cyclic Register — Accessing Spatial Neighbors

The convolution needs to access 3 neighboring rows for each output row (the row above,
the current row, and the row below). The IPU has a special 512-byte **cyclic register**
(`r_cyclic`) designed for exactly this.

Think of `r_cyclic` as a circular buffer of 512 bytes. We load three 128-byte chunks
into it at fixed positions:

```
Index   0–127:   S0 = previous row (row - 1) of the current input channel
Index 128–255:   S1 = current row  (row)     of the current input channel
Index 256–383:   S2 = next row     (row + 1) of the current input channel
Index 384–511:   (unused, wraps around to index 0)
```

The `mult.ve` instruction reads from `r_cyclic` using a **cyclic offset** — an index
that wraps around at 512. By choosing the right offset, we select which of the 3 rows
to read and whether to shift horizontally:

| Kernel position | Row  | Shift | Cyclic offset |
|-----------------|------|-------|---------------|
| top-left        | S0   | left  | 0 - 1 = 511 (wraps) |
| top-center      | S0   | none  | 0             |
| top-right       | S0   | right | 1             |
| left            | S1   | left  | 128 - 1 = 127 |
| center          | S1   | none  | 128           |
| right           | S1   | right | 129           |
| bottom-left     | S2   | left  | 256 - 1 = 255 |
| bottom-center   | S2   | none  | 256           |
| bottom-right    | S2   | right | 257           |

**Horizontal shift** works because the cyclic register is circular: reading from offset
127 gives you element 127 of S0 followed by element 0 of S1 — which is exactly the
"shift left by 1" of S1 (each lane reads its left neighbor). Similarly, offset 129 is
"shift right by 1."

---

## 5. Border Handling with Masks

Zero-padding means that out-of-bounds pixels are treated as zero. Instead of branching
(which is expensive), the IPU uses a **mask register**. The mask is a 128-bit vector:
**a set bit zeroes out the corresponding element** in the multiply result.

Six mask slots handle all border cases:

| Slot | Name         | What it zeroes         | Used when              |
|------|--------------|------------------------|------------------------|
| 0    | No mask      | Nothing                | Interior taps          |
| 1    | Left border  | Element 0 (leftmost)   | Any tap with kc = -1   |
| 2    | Right border | Element 127 (rightmost) | Any tap with kc = +1  |
| 3    | Bottom row   | (see below)            | kr = +1 on last row    |
| 4    | Left+bottom  | Union of 1 and 3       | kc=-1, kr=+1 on last  |
| 5    | Right+bottom | Union of 2 and 3       | kc=+1, kr=+1 on last  |

For the 128×128 case, since each chunk is exactly one row, the "bottom row" mask zeros
the entire chunk — effectively making the non-existent row 128 all zeros.

The **top border** is handled differently: the cyclic register initializes to all zeros,
so for the very first row (row 0), we simply skip loading S0 — it's already zero. No
mask needed.

---

## 6. The Algorithm — Loop Structure

The computation is organized as three nested loops:

```
FOR each spatial row (0 to 127):            ← outer loop, "chunk loop"
  FOR each output filter f (0 to 127):      ← middle loop, "filter loop"
    reset accumulator
    FOR each input channel group g (0 to 15, 8 channels per group): ← inner loop
      load kernel block [f][g] into r0      (128 bytes)
      FOR each input channel ic in group (8 channels):
        load S0, S1, S2 into r_cyclic       (3 memory loads)
        apply 9 taps: mult.ve + acc × 9     (9 multiply-accumulate pairs)
      END
    END
    quantize r_acc from 32-bit to 8-bit
    store quantized output to output[row][f] (1 memory store, 128 bytes)
  END
END
```

### 6.1 Processing One Input Channel (the Innermost Kernel)

For a single input channel, the 9 multiply-accumulate operations look like this:

```asm
; Load 3 spatial neighbors into r_cyclic
ldr_cyclic_mult_reg <S0_addr> cr0 lr0 ;;   ; S0 at cyclic index 0
ldr_cyclic_mult_reg <S1_addr> cr0 lr4 ;;   ; S1 at cyclic index 128
ldr_cyclic_mult_reg <S2_addr> cr0 lr8 ;;   ; S2 at cyclic index 256

; 9 taps: for each (kr, kc), do mult.ve with appropriate cyclic_offset
;         and mask_slot, then accumulate
mult.ve r0 <cyclic_off> <mask_slot> lr0 lr6 ; acc ;;
; ... 8 more mult.ve + acc pairs, incrementing lr6 (kernel index) each time
```

Each `mult.ve` broadcasts one kernel weight (selected by `lr6`, the byte index into r0)
across all 128 lanes and multiplies it by the 128 spatial elements read from `r_cyclic`
at the given offset. The `acc` instruction adds those 128 products into `r_acc`.

After all 9 taps, `lr6` has advanced by 9, pointing to the next input channel's weights.

### 6.2 The Input Channel Group Loop

Eight channels share one kernel block in r0. After processing all 8 channels (72 kernel
bytes, 9 taps each = 72 multiply-accumulate pairs), we load the next kernel block and
continue accumulating into the same `r_acc`. This repeats 16 times to cover all 128
input channels.

### 6.3 The Filter Loop

After all 128 input channels are accumulated, `r_acc` holds the final 128 output values
for this row and this filter as 32-bit integers. The values are then **quantized to
8 bits** (scaled and clamped to fit in one byte) and stored to memory as 128 bytes. The
accumulator is reset, the first kernel block of the next filter is loaded, and the
process repeats.

### 6.4 The Spatial (Row) Loop — Three Sections

The row loop is split into three sections to handle borders:

1. **Row 0 (top border)**: Skip loading S0 — `r_cyclic` starts as all zeros, giving
   automatic zero-padding for the top edge. Load S1 (row 0) and S2 (row 1).

2. **Rows 1–126 (interior)**: Load all three neighbors normally. S0 = row-1, S1 = row,
   S2 = row+1.

3. **Row 127 (bottom border)**: Load S0 (row 126) and S1 (row 127). Skip loading S2.
   Use bottom-border masks (slots 3–5) for the kr=+1 taps to zero them out.

---

## 7. Data Reading Order — A Concrete Trace

Here is the exact sequence of memory reads for processing **row 5, filter 3**:

```
1. Load kernel block 0 for filter 3  (128 bytes from kernel memory)
   This contains weights for input channels 0–7.

2. Input channel 0:
   Read row 4, ch 0 → r_cyclic[0]     (input address = 4×16384 + 0×128)
   Read row 5, ch 0 → r_cyclic[128]   (input address = 5×16384 + 0×128)
   Read row 6, ch 0 → r_cyclic[256]   (input address = 6×16384 + 0×128)
   → 9 multiply-accumulates using kernel bytes 0–8

3. Input channel 1:
   Read row 4, ch 1 → r_cyclic[0]     (input address = 4×16384 + 1×128)
   Read row 5, ch 1 → r_cyclic[128]   (input address = 5×16384 + 1×128)
   Read row 6, ch 1 → r_cyclic[256]   (input address = 6×16384 + 1×128)
   → 9 multiply-accumulates using kernel bytes 9–17

4–9. Input channels 2–7: same pattern.

10. Load kernel block 1 for filter 3  (next 128 bytes from kernel memory)
    This contains weights for input channels 8–15.

11–18. Input channels 8–15: same pattern as above.

... (repeat for blocks 2–15, covering channels 16–127)

Final: Quantize r_acc to 8-bit, store → output row 5, filter 3  (128 bytes)
```

**Total memory traffic per (row, filter):**
- 16 kernel block loads × 128 bytes = 2,048 bytes
- 128 input channels × 3 neighbor loads × 128 bytes = 49,152 bytes
- 1 output store × 128 bytes = 128 bytes (quantized to 8-bit)

**Total for the entire convolution:**
- 128 rows × 128 filters × (2,048 + 49,152 + 128) = **128 × 128 × 51,328 ≈ 841 MB**
  of memory traffic.

Note: compared to a 32-bit output (512 bytes per store), quantization reduces output
memory traffic by 4× and total output storage from 8 MB to 2 MB.

---

## 8. Summary Diagram

```
Input (2 MB)              Kernel (256 KB)           Output (2 MB)
128×128 spatial           128 filters × 128ch       128×128 spatial
128 channels              × 3×3 weights             128 channels
1 byte/element            1 byte/weight             1 byte/element
                 ┌──────────────────┐
  row r, ch ic ──┤                  │
  row r±1, ch ic─┤  r_cyclic (512B) ├──► mult.ve ──► acc ──► r_acc
                 │  3 neighbor rows │    ×9 taps     (128    (128 elements
                 └──────────────────┘    per ch     elements) ×32-bit
                                                      │       internal)
                 ┌──────────────────┐                  │
  filter f ──────┤  r0 (128B)       ├──► broadcast ────┘
                 │  8ch × 9 weights │    1 weight          ┌──────────┐
                 └──────────────────┘    to 128 lanes      │ quantize │
                                                           │ 32b → 8b │
                                                           └────┬─────┘
                                                                ▼
                                                       store to output
                                                       128 bytes (8-bit)
                                                       after all 128 ch
```
