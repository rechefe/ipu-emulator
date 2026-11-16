# The memory models, one by one

For the thesis and the defense: what each model computes, in plain language,
with its equation, a worked example and its assumptions. The code is in
`src/ipu-memsim/src/ipu_memsim/`.

## The setting, and the one equation everything shares

The emulator runs a kernel and gives its exact cycle count with a perfect
memory (`ideal`). The tool records every XMEM access the kernel makes
(cycle, read or write, row) and then asks a memory model, access by access,
"how many cycles does the IPU wait for this one?". Then

```
estimated runtime = ideal cycles + sum of all waits (stalls)
```

This is legitimate because, in the specification, a stall freezes the whole
pipeline and the stalled word has changed nothing (IPU-ARCH-14, IPU-ARCH-17):
a stall makes everything later happen later, but cannot change what happens.
An access recorded at cycle `c` therefore reaches the model at time
`c + (stalls so far)`.

**The example kernel used below:** `residual_add_256x144` (y = a + b for 256
tokens of 144 channels). Its trace: **1,163 ideal cycles, 865 accesses**;
three tables of 288 rows each (`a` at rows 0 to 287, `b` at 288 to 575, `y`
at 576 to 863), each row touched once. All numbers below were produced by
the tool on this trace.

## Toy models (`models.py`)

They do not follow the hardware. They are simple shapes that bracket the
answer and test the replay code.

### Ideal

Every wait is 0. Runtime = 1,163. Its only use: check that the replay
reproduces the emulator.

### AllMiss

Every access waits `penalty` cycles: `stall = accesses x penalty`.
With penalty 10: 865 x 10 = **8,650**, runtime 9,813. An upper bound: nothing
is ever on chip.

### PageFirstTouch

The first access to each page (1024 consecutive rows) waits `penalty`; later
accesses to that page are free: `stall = pages touched x penalty`.
The example's 864 rows all lie in rows 0 to 863, one page, so with penalty
2,111 the stall is **2,111**, runtime 3,274. A memory that is large enough
and starts empty.

### Banked

`num_banks` banks; after an access a bank is busy for `busy_cycles`. An access
to a busy bank waits until it is free:

```
wait = max(0, free_at[bank] - t);   free_at[bank] = t + wait + busy_cycles
```

The row-to-bank mapping is the knob. With 16 banks and `busy_cycles = 2`:
"interleave" (`bank = row mod 16`) gives a stall of **1** cycle, because
neighbouring rows are in different banks; "blocked" (`bank = row div 1024`)
gives **577**, because `a`, `b` and `y` are all in bank 0 and the kernel
touches two or three rows of it per output row.

### Cache

A textbook cache: `size_rows / line_rows` lines in sets of `assoc`, least
recently used line evicted, a miss waits `miss_penalty`.
At 2 MB (16,384 rows), 1-row lines, penalty 10: 864 misses and 1 hit, stall
**8,640**. Almost every access misses because the kernel touches each row
once: a cache keeps what was used before, and nothing is used twice. This
is why a plain cache model is the wrong picture for these kernels, and why
the next model matters.

## The `cache_unit` model (`cache_unit.py`): XMEM as the specification describes it

Source: `docs/content/specs/cache-unit.md` and
`docs/content/requirements/memory-access.md` (line references in
[memory_sketch.md](memory_sketch.md)).

**What it models.** XMEM is `num_banks` banks of `bank_rows` rows. Each table
(one array of the kernel) owns some banks. Page `p` of a table (its rows
`p x bank_rows` onward) lives in the table's bank number `p mod (banks of the
table)`. For each access:

1. If that bank holds page `p`: a hit. Wait only if its data is still on the
   way: `wait = max(0, ready - t)`.
2. Otherwise a miss. If the page's data is in DRAM (an input, or a page that
   was drained), the DMA fetches the whole page; the wait lasts until it has
   arrived. A new output page needs no fetch: the bank is simply claimed.
3. A write marks the bank dirty. When the kernel is `jump_back` rows into the
   next page, the previous bank is given back: free at once if clean, drained
   to DRAM by the DMA first if dirty.
4. The DMA works in the background: it prefetches the pages ahead into free
   banks, and drains dirty ones, while the IPU computes. There is one DMA;
   transfers queue.

**Transfer time** (`dram.py`). With `fill_cycles_per_bank = F` (the swept
parameter), moving `r` rows takes `ceil(F x r / 1024)` cycles.

**Worked example**, F = 2,111, 16 banks. The three tables get one bank each.

- Cycle 0: the kernel reads `a` row 0. Miss; the DMA fetches `a`'s 288 rows:
  `ceil(2111 x 288 / 1024) = 594` cycles. The IPU waits 594.
- It then reads `b` row 0. Miss; 594 more cycles (the DMA does one thing at a
  time).
- It writes `y` row 0. A new page: the bank is claimed, no wait.
- Every other access is a hit: 862 hits.

Stall = 594 + 594 = **1,188**, runtime **2,351**, which is 102% more than
ideal: this kernel computes for 1,163 cycles and waits 1,188 for its data.
After the kernel ends the DMA still drains `y` (594 cycles); the kernel does
not wait for that (IPU-EXEC-18), so it is reported separately.

With `residency = "stays_in_xmem"` (the previous kernel left `a` and `b` in
XMEM, and `y` stays for the next one) there is nothing to fetch: stall **0**.
The truth for a real network is between 0 and 1,188, and question Q2 decides
where.

**Assumptions of this model** (each is a parameter; the sensitivity ranking
in [questions.md](questions.md) says which ones matter):

| Assumption | Parameter |
|---|---|
| One `kernel.toml` region is one table | (fixed; Q7) |
| Banks are shared between tables by size | `bank_allocation` |
| `jump_back` is the largest look-back the kernel makes | `jump_back` |
| A page whose bank holds another page of the same table evicts it; the spec has no eviction | (fixed; Q5) |
| One DMA, first come first served | `dma_channels` |
| The DMA prefetches one page ahead | `prefetch_distance` |
| A partial bank takes proportionally less time, with no fixed start-up cost | `fill_whole_bank`, Q9 |
| Each kernel is replayed alone | `residency` |
| One emulator row is one hardware row, also for FP32 kernels | (fixed; Q13) |

## The working-set formulas (`formulas.py`)

For each table of a kernel, two closed forms of the layer shape: its size in
rows, and its look-back (how far behind its furthest row the kernel goes).
From them, the banks the table needs so that no bank is loaded twice:

```
banks = min( ceil(rows / R),  ceil(look_back / R) + 1 )      R = rows per bank
```

Example: the 3x3 convolution reads three spatial rows of all `C_in` channels,
so its input look-back is `3 x C_in - 1` (the project book's INT8 result; it
also holds on the FP32 kernel). For `C_in = 32`: look-back 95, so
`ceil(95 / 1024) + 1 = 2` banks at most, however large the image.

Checked against the replay for all 37 recorded kernels at three bank sizes:
rows and look-back exact in all 426 table checks; banks equal in 416, and in
10 the formula asks for one bank more than needed (its window bound is not
tight when the window happens to line up with bank boundaries). It never
asks for too few.

## What this tool cannot say

- **No absolute chip numbers.** The bank fill time, the clock and the bank
  sharing rule are not known; results are curves over them, with the chip as
  a point to be marked later.
- **No real DMA arbitration.** One DMA serving requests in order is an
  assumption; the real controller's priorities between fills and drains,
  and between two contexts, are not modelled.
- **No inter-kernel data flow.** Each kernel is replayed on its own; the
  residency switch gives two bounds, not the real hand-off.
- **No pipeline effects.** The extra cycle of a late hit decision (OQ-16) and
  load-after-store stalls (OQ-17) are not modelled.
- **No energy, no area, no bandwidth cost of DRAM traffic.**
- **Nothing replayed outside MobileViT-S.** The tables for DeiT-Tiny, Swin-T
  and YOLOv8n count parameters and operations exactly, but their knees are
  extrapolated from formulas validated on MobileViT-S shapes only.
- **Not validated against hardware.** The FPGA comparison code has only seen
  synthetic data.
- **The repository's golden tests were not run on the development machine**
  (torch does not install there), so kernel correctness rests on the
  project's CI.
