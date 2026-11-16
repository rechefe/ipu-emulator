# Open questions for the memory what-if tool

Ranked by how much the answer changes the tool's numbers. Each has the default
the tool uses today (ASSUMED) so work can continue. "OQ-n" refers to
`docs/content/requirements/open-questions.md`.

## A. Change the results by a large factor

| # | Question | Who | Tool default today (ASSUMED) |
|---|---|---|---|
| Q1 | **DRAM/HBM speed as the DMA sees it.** How many cycles does it take to fill one 1024-row bank (latency + bandwidth)? Which HBM generation, how many channels, what clock (OQ-16 mentions 1 GHz)? | Architects | `[dram] latency_cycles = 100`, `rows_per_cycle = 1.0` (one 128-byte row per cycle = 128 GB/s at 1 GHz). Invented placeholder, not a measurement. |
| Q2 | **Does data go back to DRAM between kernels?** OQ-11 (unassigned) asks how one kernel hands a table to the next inside XMEM. If outputs stay in XMEM, the next layer's input costs no DRAM read. | Cache-unit owner | Every kernel is replayed alone: inputs come from DRAM, outputs go to DRAM. |
| Q3 | **Is the input prefetched before the kernel starts?** CU-16 allows the DMA to fill banks before the context is active. How much of that is realistic depends on free banks during the previous kernel. | Architects | `prefetch_before_start = false` (cold start); a switch turns it on. |
| Q4 | **How many banks does each table get, and who decides?** `xmem_bank_list` is per table (cache-unit.md §5). 16 banks shared by, say, input + weights + output. | Team lead / SW | Proportional to table size, at least 1 each (`bank_allocation = "proportional"`), or an explicit list. |
| Q5 | **What happens when a kernel returns to a page whose bank was released** (weights re-read for every output row, `repetitions`)? Reload from DRAM each time, or must the table be given enough banks to stay resident? | Cache-unit owner | Reload from DRAM, counted as a miss ("re-fetch"). The report shows the count so the cost is visible. |
| Q6 | **Is 16 x 128 KB the layout being frozen**, or are other splits on the table (more, smaller banks; bigger banks)? Which sizes should the thesis sweep? | Team lead | Sweeps cover 4 to 64 banks and 256 to 4096 rows per bank. |

## B. Change the results moderately

| # | Question | Who | Tool default today |
|---|---|---|---|
| Q7 | **Row address to table mapping** (OQ-8, unassigned). Is a `kernel.toml` region the right "table"? Kernels pin regions at fixed rows (for example `residual_add` at rows 0, 1024, 2048). | Architects | One region = one table; offset = row minus the region's base row. |
| Q8 | **Who sets `jump_back` per kernel**, and to what? | SW / architects | Derived from the trace: the largest look-back the kernel actually makes (or a fixed value from the config). |
| Q9 | **Partial banks.** A table of 64 rows occupies a whole 1024-row bank. Does the DMA move 64 rows or 1024? | Cache-unit owner | Only the table's rows are moved. |
| Q10 | **One DMA or several?** Can a fill and a drain run at the same time? | Cache-unit owner | One DMA, first come first served (`dma_channels = 1`). |
| Q11 | **Load-after-store in the pipeline** (OQ-17): extra stall cycles when a word loads a row an older word still has to store. | Architects | Not modelled (0 cycles). |
| Q12 | **Stall decision timing** (OQ-16, IPU-MEM-12): one extra cycle per miss? | Architects | Not modelled. |
| Q13 | **FP32 kernels.** Most kernels in the repo run only in `FP32`, an emulator-only format with 512-byte rows. What will their hardware (8-bit) row counts be? | Kernel authors | One emulator row = one hardware row (same 128 elements). |
| Q14 | **`ST.AAQ` spill** (OQ-35): in 8-bit formats the emulator writes 4 rows per store. | AaQ owner | The trace records what the emulator did; in FP32 it is one row. |

## C. Needed for validation, not for the model

| # | Question | Who |
|---|---|---|
| Q15 | **What does the FPGA build expose?** Stall counters, per-bank hit/miss counters, a cycle counter, a trace port? Which memory settings can be changed without a rebuild (bank count, DMA rate, `jump_back`)? | FPGA owner |
| Q16 | **Which requirement numbers exist?** The repo has IPU-MEM-1 to 12 and 20 to 23; there is no IPU-MEM-13 to 19 (the brief expected 1 to 20). Is something missing or is the numbering just sparse? | Requirements owner |
| Q17 | **The profiling emulator of the project book** (per-region address sets, peak lookahead in lines) is not on `master`. Is it on another branch? The tool computes its own working-set number from the trace instead. | Ze'evi |
| Q18 | **Second chip.** Do two chips share one DRAM/HBM, and how do they exchange tensors? | Team lead |
