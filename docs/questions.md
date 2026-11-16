# Open questions for the memory what-if tool

Two kinds of question. **Structural** questions change how the model works,
so they need an answer. **Swept parameters** are only numbers: the tool
sweeps them, the thesis reports the curve, and the real chip is one point on
it; for these the architects are asked for a realistic range only.

Each entry has the default the tool uses today (ASSUMED) so work can
continue. "OQ-n" refers to `docs/content/requirements/open-questions.md`.
Every parameter named here is in the parameter registry
(`ipu_memsim/params.py`; `memsim params` prints it).

## Ask about these first (Stage 10 sensitivity ranking)

From `memsim sensitivity --spec src/ipu-memsim/experiments/sensitivity_2mb.toml`
(MobileViT-S, 2 MB / 16 banks, bank fill 2,111 ns; one parameter varied at a
time). "Spread" is the largest change in stall, in percentage points of ideal
cycles, over the parameter's low / default / high settings, for the layer
type it moves most. TOY MODEL numbers: they rank the questions.

| Rank | Parameter (question) | Settings tried | Largest spread | Layer types moved by more than 1 point |
|---|---|---|---|---|
| 1 | `bank_allocation` (Q4) | one bank each / by size / equal | 20,102 (depthwise conv) | 6 of 8 |
| 2 | `jump_back` (Q8, Q5) | 0 / from the trace / never | 10,605 (depthwise conv) | 5 of 8 |
| 3 | `fill_ns_per_bank` (Q1) | 250 / 2,111 / 8,000 ns | 374 (elementwise) | 8 of 8 |
| 4 | `fill_whole_bank` (Q9) | rows the table has / always a full bank | 353 (elementwise) | 7 of 8 |
| 5 | `clock_ns` (Q1b) | 0.5 / 1 / 2 ns | 153 (elementwise) | 8 of 8 |
| 6 | `residency` (Q2) | stays in XMEM / DRAM round trip | 102 (elementwise) | 6 of 8 |
| 7 | `prefetch_before_start` (Q3) | no / yes | 102 (elementwise) | 7 of 8 |
| 8 | `prefetch_distance` (Q20) | 0 / 1 / 16 pages | 19 (depthwise conv) | 2 of 8 |
| 9 | `dma_channels` (Q10) | 1 / 4 | 0.4 (pointwise conv) | 0 of 8 |

How to read it:

- **Ranks 1 and 2 are cliffs, not slopes.** With one bank per table, or with
  a bank released too early, a kernel reloads a bank on almost every access.
  By-size and equal sharing give identical results, and `jump_back = never`
  differs from the trace-derived value only for depthwise (3.7% against
  9.1%). So the question for the architects is not a number but a rule: who
  guarantees that each table has enough banks and a safe `jump_back`, and
  what happens when it has not (Q4, Q5, Q8)?
- **Ranks 3 to 7 scale the cost of moving data.** They matter for every
  layer type and most for the light ones (residual add, LayerNorm). Fill
  time and clock are one question (time per bank in cycles). `fill_whole_bank`
  ranks high because most MobileViT-S tables are much smaller than a bank.
- **Ranks 8 and 9 can wait.** How far the DMA prefetches and whether there
  are several DMA channels barely change the results at this operating
  point.

## A. Structural: ask these first

| # | Question | Who | Tool default today (ASSUMED) |
|---|---|---|---|
| Q2 | **Between two layers, does the output go to DRAM and come back, or stay in XMEM for the next kernel?** OQ-11 (unassigned) asks how one kernel hands a table to the next. | Cache-unit owner | `residency = "dram_roundtrip"` (pessimistic). The switch `stays_in_xmem` shows the other case; both are in the per-layer-type table. |
| Q4 | **Who decides how many banks each table gets, and by what rule?** `xmem_bank_list` is per table (cache-unit.md section 5). | Team lead / SW | `bank_allocation = "proportional"`: by table size, at least 1 each; or an explicit list. |
| Q5 | **What happens when a kernel returns to a page whose bank was released** (weights re-read for every output row, `repetitions`)? Reload from DRAM, or must the table keep enough banks? | Cache-unit owner | Reload from DRAM, counted as `conflicts` / `refetches`. The spec has no eviction (CU-20). |
| Q7 | **What is a "table"?** Is one `kernel.toml` region one table (OQ-8, unassigned)? LayerNorm declares 11 regions and softmax 6, so they cannot run in 4 banks and LayerNorm uses 11 of 16. | Architects | One region = one table; offset = row minus the region's base row. |
| Q15 | **What does the FPGA report?** Stall counter, per-bank request counters, the fill time of a bank, a cycle counter? Which settings change without a rebuild? | FPGA owner | Not needed by the model; needed to check it (test plan in `memsim_design_notes.md`). |
| Q19 | **Which input tables are weights and which are activations?** Needed only for `stays_in_xmem`: weights always come from DRAM. | Kernel authors | By region name (`weight_tables`), overridable per layer in the network file (`weights = [...]`). For the single-stream matmul kernels `a` is the weights (CONFIRMED, `kernels/matmul/matmul_x128_c2/kernel.toml:37`); in the attention kernels both inputs are activations. |
| Q3 | **Is the input prefetched before the kernel starts?** CU-16 allows it. | Architects | `prefetch_before_start = false` (cold start). |
| Q8 | **Who sets `jump_back` per kernel**, and to what? | SW / architects | `jump_back = "auto"`: the largest look-back the kernel actually makes. |

## B. Swept parameters: ask for a realistic range only

| # | Parameter | Registry name | Swept over | Anchor point |
|---|---|---|---|---|
| Q1 | Cycles to fill one 1024-row bank from HBM | `fill_cycles_per_bank` (or `fill_ns_per_bank` + `clock_ns`) | 0, 250, 500, 1000, 2111, 4000, 8000 | 2111 cycles: DRAMsim3 `HBM2_8Gb_x128`, one channel, 1 ns clock. A simulator's answer for an assumed access pattern, not a measurement. |
| Q1b | Chip clock period | `clock_ns` | not swept; converts ns to cycles | 1 ns, ASSUMED (the emulator defines no clock; OQ-16 speaks of 1 GHz) |
| Q6 | XMEM size and layout: is 16 x 1024 rows final? | `xmem_mb`, `num_banks`, `bank_rows` | 0.5, 1, 2, 4, 8 MB | 16 banks x 1024 rows = 2 MB (cache-unit.md section 4) |
| Q20 | How far ahead the DMA prefetches | `prefetch_distance` | 0, 1, all banks | The spec's DMA (section 8.1) fills every free bank of the table; the tool's default of 1 is the cautious choice. |
| Q10 | DMA transfers at the same time | `dma_channels` | 1, 2, 4 | 1, ASSUMED |

What to ask for Q1 and Q6: "what is the slowest and the fastest bank fill you
consider realistic, and which bank layouts are still on the table?" The
answer picks the part of the curve that matters.

## C. Smaller modelling questions

| # | Question | Who | Tool default today |
|---|---|---|---|
| Q9 | **Partial banks.** A table of 64 rows occupies a whole bank. Does the DMA move 64 rows or 1024, and does a transfer have a fixed start-up cost? | Cache-unit owner | Only the table's rows, in proportion (`fill_whole_bank = false`). |
| Q11 | **Load-after-store in the pipeline** (OQ-17): extra stall cycles. | Architects | Not modelled. |
| Q12 | **Stall decision timing** (OQ-16, IPU-MEM-12): one extra cycle per miss? | Architects | Not modelled. |
| Q13 | **FP32 kernels.** Most kernels run only in `FP32`, an emulator-only format with 512-byte rows. What are their hardware row counts? | Kernel authors | One emulator row = one hardware row (128 elements). |
| Q14 | **`ST.AAQ` spill** (OQ-35): 4 rows per store in 8-bit formats. | AaQ owner | The trace records what the emulator did. |
| Q16 | The repo has IPU-MEM-1 to 12 and 20 to 23; is 13 to 19 missing or just unused? | Requirements owner | - |
| Q17 | The project book's profiling emulator (peak lookahead per region) is not on `master`. | Ze'evi | The tool computes its own working set from the trace. |
| Q18 | **Second chip.** Shared DRAM/HBM? How do chips exchange tensors? | Team lead | Design note only. |
