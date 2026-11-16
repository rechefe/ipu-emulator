# STATUS: memory what-if tool (`src/ipu-memsim`)

Last updated after Stage 6. Branch `performance-modeling` (local, never
pushed), branched from `master` at `c5c04f7c` ("Merge pull request #251").
`master` already has both the convolution and the transformer kernels, so no
integration branch was needed.

## 1. What exists

| Stage | State | What |
|---|---|---|
| 0 Orientation | done | [docs/memory_sketch.md](docs/memory_sketch.md), [docs/questions.md](docs/questions.md), test baseline (section 6) |
| 1 Trace recorder | done | `ipu_memsim.record`, `ipu_memsim.trace` |
| 2 Replay + models | done | `ipu_memsim.replay`, `ipu_memsim.models` (Ideal, AllMiss, PageFirstTouch, Banked, Cache), `ipu_memsim.cache_unit` (follows the cache-unit spec), `ipu_memsim.config` |
| 3 Sweeps + plot | done | `ipu_memsim.sweep`, `ipu_memsim.svgplot` |
| 4 Network table | done | `ipu_memsim.network`, `networks/mobilevit_s.toml`, `ipu_memsim.workset` |
| 5 DRAM backend | done | `ipu_memsim.dram` (interface, fixed model, table model), `ipu_memsim.dram_trace` (DRAMsim3 adapter; DRAMsim3 built and ran on this machine) |
| 6 Design notes | done | [docs/memsim_design_notes.md](docs/memsim_design_notes.md) |

The emulator and the ISA are untouched. The recorder uses only the reference
model's public API (`observe_xmem_reads`, `step`). Files changed outside the
new package, all additive: `.gitignore`, `ruff.toml`, `tools/BUILD.bazel`,
`tools/import_graph.toml`, `docs/BUILD.bazel` (registering the new package
with the repo's lint, import-graph and architecture-page checks).

## 2. How to run it

Everything runs through Bazel. Outputs go to `memsim-out/` (in `.gitignore`).

```bash
# 1. Record a trace (once per kernel and shape).
bazel run //src/ipu-memsim:memsim -- record residual_add
bazel run //src/ipu-memsim:memsim -- record pointwise_conv_unified \
    --set height=128 --set width=128 --set in_channels=64 --set out_channels=32 \
    --out memsim-out/pointwise_128x128_64to32.csv.gz

# 2. Replay it through a memory model.
bazel run //src/ipu-memsim:memsim -- replay memsim-out/pointwise_128x128_64to32.csv.gz \
    --config src/ipu-memsim/configs/cache_unit.toml

# 3. Sweeps: a CSV and four SVG charts.
bazel run //src/ipu-memsim:memsim -- sweep memsim-out/pointwise_128x128_64to32.csv.gz \
    --out memsim-out/sweep_pointwise

# 4. The per-layer-type table of a network.
C=src/ipu-memsim/configs
bazel run //src/ipu-memsim:memsim -- network src/ipu-memsim/networks/mobilevit_s.toml \
    --out memsim-out/mobilevit_s \
    --configs $C/study_05mb_slow.toml $C/study_2mb_slow.toml $C/study_8mb_slow.toml $C/study_2mb_fast.toml

# 5. Measure bank-fill times with DRAMsim3 (built separately, see ipu_memsim/dram_trace.py).
bazel run //src/ipu-memsim:memsim -- dram-measure --dramsim3 /path/to/dramsim3main.out \
    --ini /path/to/DRAMsim3/configs/HBM2_8Gb_x128.ini --out my_hbm.toml

# Tests.
bazel test //src/ipu-memsim/...
```

Configs are in `src/ipu-memsim/configs/`, one per model, each commented. A
new network is a new file like `networks/mobilevit_s.toml`.

## 3. The one timing assumption

**Stalls serialize: estimated runtime = ideal cycles + sum of stall cycles.**
It is stated in `ipu_memsim/replay.py` and rests on two requirements: the
cache agent is the only source of backpressure (IPU-ARCH-14) and a stalled
word has changed nothing (IPU-ARCH-17), so a stall delays later words but
cannot change which rows the program touches. Example: 1000 cycles with three
50-cycle stalls is 1150 cycles; an access recorded at cycle 700 after two
stalls reaches the model at time 800.

## 4. CONFIRMED, ASSUMED, UNKNOWN

Full list with file and line references: [docs/memory_sketch.md](docs/memory_sketch.md).
Questions, ranked: [docs/questions.md](docs/questions.md).

**CONFIRMED (from the repo's specs)**

- Row = 1024 bits; bank = page = 1024 rows; 16 banks = 2 MB; up to 16 tables.
- Page of an access = `offset[19:10]`, its bank = the table's bank list at
  `page % banks`. Hit = tag match; a miss stalls CTRL.
- A read bank is published only when the whole bank has arrived.
- A write miss claims a free bank with no DRAM read.
- `jump_back`: at row `>= jump_back` of page `p`, the bank of page `p-1` is
  released (read) or flushed to DRAM (written).
- The DMA runs in the background, decoupled from the pipeline. So XMEM is a
  cache in front of DRAM, and a `scratch_pad` table is a scratchpad.
- The requirements are IPU-MEM-1 to 12 and 20 to 23 (no 13 to 19 exist).
- The emulator has no stall model; its XMEM is a flat 512 MiB space.

**ASSUMED (each is a config key or constructor argument)**

| Assumption | Where to change it |
|---|---|
| One `kernel.toml` region = one table | the region list in `ipu_memsim/record.py` |
| One emulator row = one hardware row, also in FP32 | reported sizes use 128 bytes per row (`cache_unit.ROW_BYTES`) |
| DMA time = latency + rows / bandwidth; values are placeholders | `[dram]` in the config |
| One DMA, first come first served | `dma_channels` |
| Banks shared between tables by size, at least 1 each | `bank_allocation` |
| `jump_back` = the largest look-back the kernel makes | `jump_back` |
| The DMA prefetches the next page of an input when a bank is free | `prefetch` |
| Cold start: nothing is in XMEM at cycle 0 | `prefetch_before_start` |
| A page whose bank holds another page of the same table evicts it (the spec has no eviction; counted as `conflicts`) | not switchable; see Q5 |
| The DMA moves only the rows a table has | `fill_whole_bank` |
| Every kernel runs alone: inputs from DRAM, outputs to DRAM | not modelled otherwise; see Q2 |

**UNKNOWN**: Q1 to Q18 in [docs/questions.md](docs/questions.md). The five
that move the numbers most are repeated in section 9.

## 5. Decisions log

| Decision | Reason |
|---|---|
| New package `src/ipu-memsim`, registered like the others | The repo checks imports, lint and the architecture page per package; a loose script folder would fail them. |
| Record through `ReferenceModel.step()` + `observe_xmem_reads()` | Public API only, so no emulator change and no reach into private state. |
| No golden check while recording | Goldens need torch, which Bazel cannot fetch on this Mac (section 6). Correctness of the kernels is the kernel tests' job. |
| One trace entry per row written, per load read | Matches the emulator's `xmem_reads` / `xmem_writes` exactly in FP32; the header stores `stats_match`. |
| Trace = gzip CSV with a JSON header line | Readable in any editor, about 1 MB for 270,000 accesses, loads in under a second. |
| Added a spec-following `cache_unit` model beyond the brief's list | Stage 0 found a real cache-unit spec in the repo; the toy models cannot express banks per table, `jump_back` or the background DMA. |
| `access(t, op, row, region)` instead of `access(t, op, row)` | The region is already in the trace; looking it up again per access would only slow the loop. |
| Charts as hand-written SVG | matplotlib is not in the repo's locked dependencies; a 150-line SVG writer keeps the dependency list at numpy + tomli. |
| DRAMsim3 (not Ramulator 2) for the DRAM side | It builds here with a plain `make` (no cmake on this machine) and ships HBM/HBM2 configs. |
| `ipu-memsim` not added to the dependency lock inputs | Re-locking needs network access and regenerates `requirements_lock.txt`; it uses only numpy and tomli, both already pinned. To do before a PR. |
| Multi-stage kernels (3 of 67 folders) are not recorded | They run several programs; the recorder handles one. Reported as "not measured" in the network table. |

## 6. Test results against the baseline

Machine: macOS arm64, Bazel 8.3.0. On this machine plain `bazel test //...`
runs **0 tests**, before and after my changes: Bazel cannot fetch
`torch==2.14.1+cpu` (no macOS wheel), which stops the analysis of every
target that needs torch. With `-k` (keep going) the torch-free targets run:

| | Test targets run | Passing | Test cases | Failing |
|---|---|---|---|---|
| Baseline, `master` `c5c04f7c`, `bazel test -k //...` | 271 | 271 | 7199 passed, 265 skipped | 0 |
| This branch, `bazel test -k //...` | 277 | 277 | 7268 passed, 265 skipped | 0 |

The difference is the 6 new targets of `//src/ipu-memsim` (69 test cases).
No existing test changed result. The torch-dependent targets (among them the kernel
golden tests) could not be built here in either run, so they are
**not verified on this machine**; run `bazel test //...` on a machine where
torch installs (CI, Linux) before merging. The book's "996 tests, 987
passing" is from another tree and another runner and was not reproduced.

Mandatory sanity tests (`test/test_models.py`): an infinite preloaded cache
replays to exactly the ideal cycle count; a cache that never hits stalls
accesses x penalty; the same trace and config give the same result. The
`cache_unit` model is checked against hand-computed schedules
(`test/test_cache_unit.py`). The recorder is checked on `residual_add`,
`pointwise_conv_unified` and `matmul_small`: entries equal the emulator's
`xmem_reads`/`xmem_writes`, and two runs give byte-identical files.

## 7. Measured so far

**Cycle counts against the project book** (the four single-kernel numbers match exactly):

| Kernel and shape | Book | Recorded here |
|---|---|---|
| `residual_add`, 64 + 96 + 128 + 160 channels | 2,260 | 325 at 64 channels and 485 at 96, that is 5 x channels + 5; by that formula the four sum to 2,260 (128 and 160 were not recorded) |
| `pointwise_conv_unified` 128x128, 64 to 32 | 316,553 | 316,553 |
| `depthwise_conv_universal` 64x64x64 | 19,686 | 19,686 |
| `conv_universal` 64x64, 32 to 32 | 318,564 | 318,564 |
| `matmul_144x144_x128` (OutProj L3) | 43,921 | 43,921 |

**Speed.** Recording 316,553 cycles takes 43 s. A replay of its 270,849
accesses takes about 0.17 s (75 replays in 13.0 s for the whole sweep), so
no vectorising was needed. The 37-layer MobileViT-S study took about 20
minutes, nearly all of it recording; traces are kept and reused.

**Sweep, pointwise 128x128 64 to 32** (`memsim-out/sweep_pointwise/`; TOY
MODEL, placeholder DRAM). Estimated runtime in cycles, ideal 316,553:

| XMEM | DRAM 1 row/cycle | DRAM 0.1 row/cycle | DRAM 0.02 row/cycle |
|---|---|---|---|
| 4 banks (0.5 MB) | 329,049 | 430,713 | 882,553 |
| 8 banks (1 MB) | 318,926 | 337,646 | 593,838 |
| 16 banks (2 MB) | 318,926 | 337,646 | 593,838 |
| 64 banks (8 MB) | 318,926 | 337,646 | 593,838 |

With 16 banks and a different bank size (DRAM 0.1): 256-row banks (0.5 MB)
322,286; 1024-row banks (2 MB) 337,646; 4096-row banks (8 MB) 399,086.
Bigger banks are slower here because a bank is usable only when all of it
has arrived.

**MobileViT-S per-layer-type table** (`memsim-out/mobilevit_s/`; TOY MODEL;
each kernel replayed alone and cold; counts and convolution shapes as in
`networks/mobilevit_s.toml`, which is a subset for the convolutions). Stall
cycles, and percent of the type's ideal cycles; "slow" DRAM = 0.1 row/cycle,
"fast" = 1 row/cycle; 8 MB gave the same numbers as 2 MB for every type:

| Layer type | Ideal cycles | Largest footprint (rows) | Largest peak live (rows) | 0.5 MB, slow | 2 MB, slow | 2 MB, fast |
|---|---|---|---|---|---|---|
| standard conv | 1,486,064 | 2,721 | 2,402 | 10,433,640 (702%) | 57,750 (3.9%) | 6,486 (0.4%) |
| depthwise conv | 175,026 | 16,660 | 535 | 172,460,520 (98,534%) | 162,086 (92.6%) | 3,718 (2.1%) |
| pointwise conv | 540,551 | 12,321 | 322 | 145,100 (26.8%) | 41,693 (7.7%) | 4,973 (0.9%) |
| attention | 906,484 | 2,816 | 519 | softmax needs 6 banks | 192,540 (21.2%) | 22,584 (2.5%) |
| MLP | 13,423,440 | 2,400 | 582 | 1,538,716 (11.5%) | 1,538,716 (11.5%) | 182,236 (1.4%) |
| normalization | 63,360 | 730 | 489 | needs 11 banks | 47,400 (74.8%) | 9,600 (15.2%) |
| elementwise | 16,710 | 864 | 4 | 86,160 (516%) | 86,160 (516%) | 11,856 (71%) |
| reshape | 65,704 | 2,305 | 578 | 1,520,868 (2,315%) | 23,468 (35.7%) | 2,640 (4.0%) |

No kernel exists for: global average pooling, the 640 to 1000 classifier, a
general stride-2 standard convolution, SiLU after a convolution. Batch
normalization is folded into the convolution kernels.

What the table shows, with the caveat that the DRAM speed is a placeholder:

- **Above the point where each table has enough banks, more XMEM changes
  nothing** (2 MB = 8 MB for every type). The largest footprint of any layer
  here is 16,660 rows (2.03 MB), and it is streamed, with at most 535 rows
  live at once.
- **Below it there is a cliff, not a slope.** At 0.5 MB (4 banks) the
  depthwise and standard convolutions get one bank per table, alternate
  between two pages of the same table and reload a bank on almost every
  access (the model's `conflicts`). Whether the real hardware would thrash
  like this or simply cannot be configured this way is question Q5.
- **The stall that remains is the time to move the data once.** Layers that
  compute little per row (residual add, LayerNorm, unfold) are dominated by
  it; the matmuls and convolutions are not. This is why Q1 (DRAM speed) and
  Q2 (does data return to DRAM between layers) decide the real numbers.
- LayerNorm declares 11 regions and softmax 6. With one region = one table
  (Q7) they cannot run in 4 banks at all.

**DRAMsim3, HBM2 (`HBM2_8Gb_x128.ini`, one channel, consecutive addresses,
IPU clock 1 ns):** a 1024-row bank fills in 2,111 cycles and drains in
2,050; 64 rows fill in 191. This is in
`configs/cache_unit_hbm2_dramsim3.toml` and corresponds to about 0.49 rows
per cycle. It is DRAMsim3's answer for an assumed access pattern, not a
measurement of the team's HBM. On macOS DRAMsim3 aborts while exiting after
writing its statistics; the adapter reads the statistics and ignores the
exit status.

## 8. Next three steps

1. **Get Q1 to Q5 answered** (section 9) and put the answers in the configs.
   Until then every stall number is a toy-model number.
2. **Model the hand-off between layers** (Q2): replay a whole chain of
   kernels on one `cache_unit`, so a layer's output banks can be the next
   layer's input banks. This is what turns the per-kernel table into a
   per-network number. Design sketch in `docs/memsim_design_notes.md`.
3. **Set the real MobileViT-S shapes and counts** in
   `networks/mobilevit_s.toml` from the team's model export (the convolution
   shapes there are from the public architecture and only a subset), then
   rerun the network table and the sweeps per layer type.

## 9. For Ze'evi

**In plain words.** The tool works end to end. You record a kernel once on
the emulator (which is unchanged), and then replay that recording through
different memory setups in a fraction of a second each. The cycle counts it
records match the four numbers in your project book exactly.

The most useful thing I found is that the repo already contains a real
specification of the memory system (`docs/content/specs/cache-unit.md` and
the IPU-MEM requirements). It answers several of your "unknown" items: XMEM
is a cache of 16 banks of 1024 rows in front of DRAM, each array ("table")
owns a list of banks, a miss stalls the processor until a whole bank has
been copied in, and `jump_back` is the kernel's promise that it is finished
with the previous bank. I built a model that follows that specification
(`cache_unit`), next to the simple toy models you asked for.

What you can say today: for the MobileViT-S kernels in the repo, once every
array has enough banks, making XMEM bigger than 2 MB does not help, and
making it much smaller makes some convolutions collapse. What you cannot
say yet is any absolute number of stall cycles, because the speed of the
DRAM link in the model is a placeholder I chose, and because each kernel is
replayed on its own, as if every layer's data went to DRAM and back.

One thing to do on another machine: the repo's torch-based tests cannot be
built on this Mac (before or after my changes), so run `bazel test //...`
where torch installs before merging.

**Questions to take to the team lead and the architects** (full list in
`docs/questions.md`):

1. How fast is the DMA: how many cycles to fill one 1024-row bank from HBM,
   and at what clock? (Q1)
2. Between two layers, does the output go to DRAM and come back, or can it
   stay in XMEM for the next kernel? (Q2, the repo's own OQ-11)
3. Who decides how many banks each array gets, and by what rule? (Q4)
4. When a kernel goes back to data whose bank was already released (weights
   re-read per output row), is it reloaded from DRAM, or must the array be
   given enough banks to stay? (Q5)
5. Is 16 banks of 1024 rows the layout being frozen, or are other splits
   still open? Which sizes should the study sweep? (Q6)
6. What will the FPGA let us read out: a stall counter, per-bank request
   counters, the fill time of a bank? (Q15)
7. Is one `kernel.toml` region the right "table"? LayerNorm has 11 regions,
   which cannot fit 4 banks and uses 11 of 16. (Q7)

