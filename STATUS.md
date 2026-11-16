# STATUS: memory what-if tool (`src/ipu-memsim`)

Last updated after Stage 9 (the follow-up brief: fill-time sweeps, the
menu-driven front end, the residency switch). Branch `performance-modeling` (local, never
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
| 7 Fill-time sweeps | done | `ipu_memsim.dram.BankFillDram` (fill time per bank, chip clock), `prefetch_distance`, `experiments/fill_time_2mb.toml`, `experiments/size_x_fill_time.toml`, break-even, heatmaps |
| 8 Menu front end | done | `ipu_memsim.params` (the parameter registry), `ipu_memsim.experiment` (`run_sweep(spec)`), `memsim sweep --spec`, `ipu_memsim.ui` (Streamlit app, optional extra) |
| 9 Residency switch | done | `residency = "dram_roundtrip"` / `"stays_in_xmem"` in `cache_unit`, `experiments/residency_x_fill_time.toml` |

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

# 6. Run an experiment file: one or two swept parameters over a kernel or a network.
#    Writes results.csv, break_even.csv, the charts and spec.toml (the exact
#    experiment, so the chart can be made again) to memsim-out/<name>/.
bazel run //src/ipu-memsim:memsim -- sweep --spec src/ipu-memsim/experiments/fill_time_2mb.toml
bazel run //src/ipu-memsim:memsim -- sweep --spec src/ipu-memsim/experiments/size_x_fill_time.toml
bazel run //src/ipu-memsim:memsim -- sweep --spec src/ipu-memsim/experiments/residency_x_fill_time.toml

# 7. List every tunable parameter with its default, range and status label.
bazel run //src/ipu-memsim:memsim -- params cache_unit

# Tests.
bazel test //src/ipu-memsim/...
```

### The menu (browser app)

```bash
cd src/ipu-memsim
uv run --extra ui streamlit run src/ipu_memsim/ui/app.py
```

`uv` installs Streamlit into `src/ipu-memsim/.venv` on the first run (about
40 packages; nothing is added to the repo's Bazel lock) and the app opens at
`http://localhost:8501`. In the left column: pick the workload (a network and
its layer types, one kernel, or a saved trace), the memory model, the
parameter to sweep and its values (a list or min/max/steps, linear or log),
optionally a second parameter for a heatmap, and edit any other parameter.
Every parameter shows its status label, and its tooltip gives the
description and the source. A yellow banner lists the placeholder parameters
in use. Press **Run**; the page then shows the chart (its "..." menu saves a
PNG), the table, the break-even table, and buttons to download the CSV and
the spec, or to save everything to `memsim-out/<name>/`. "Save this
experiment" and "Load an experiment" write and read the same TOML file that
`memsim sweep --spec` takes.

To check the app without a browser (starts it headless and runs one sweep):

```bash
cd src/ipu-memsim && uv run --extra ui python test/ui_smoke.py
```

Replaying needs only numpy and tomli. Recording a missing trace (done once,
on demand, then kept in `memsim-out/traces/`) needs the kernel framework but
not torch.

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
| Bank fill time: 1024 cycles per 1024-row bank by default, a placeholder; a partial bank takes proportionally less | `fill_cycles_per_bank` (or `fill_ns_per_bank`); swept in the experiments |
| Chip clock period 1 ns (only used to convert a fill time in ns) | `clock_ns` |
| The DMA prefetches at most 1 page ahead (the spec allows as many as the table has free banks) | `prefetch_distance` |
| Activations go to DRAM and back between layers | `residency` |
| Which input tables are weights: by name (`w`, `filters`, `gamma`, ...), or per layer in the network file | `weight_tables`, `weights = [...]` |
| One DMA, first come first served | `dma_channels` |
| Banks shared between tables by size, at least 1 each | `bank_allocation` |
| `jump_back` = the largest look-back the kernel makes | `jump_back` |
| The DMA prefetches the next page of an input when a bank is free | `prefetch` |
| Cold start: nothing is in XMEM at cycle 0 | `prefetch_before_start` |
| A page whose bank holds another page of the same table evicts it (the spec has no eviction; counted as `conflicts`) | not switchable; see Q5 |
| The DMA moves only the rows a table has | `fill_whole_bank` |

**UNKNOWN**: [docs/questions.md](docs/questions.md), now in two groups:
structural questions (they change how the model works) and swept parameters
(only numbers: ask for a realistic range). Section 9 repeats the short list.

The status label of every parameter is in the registry
(`ipu_memsim/params.py`); `memsim params` prints it.

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
| Fill time is its own DRAM backend (`bank_fill`), and the default when a config names no kind | It is the number the architects can give directly. The earlier latency + bandwidth model stays as `kind = "fixed"`; the four study configs name it explicitly, so their results are unchanged. |
| The default fill time is 1024 cycles, not the DRAMsim3 2,111 | The brief: the HBM2 result is an anchor point, not the truth. 1024 (one row per cycle) is round enough not to be mistaken for a measurement. Charts of a fill-time sweep mark 2,111 with a dashed line. |
| `prefetch_distance` is an option of `cache_unit`, not a separate model class | The spec supports background prefetch (cache-unit.md:115 and section 8.1: the DMA fills any free bank of the table). A second class would copy the whole model for one loop bound. Default 1 keeps the earlier results; the spec's behaviour is a distance of `num_banks`. |
| One parameter registry (`params.py`); config loading, sweeps, the CLI's `params` and the menu are all driven by it | The brief's design rule. A test checks the registry against every model's constructor, so a new argument without a registry entry fails. |
| `xmem_mb` is a derived parameter | "XMEM size" is not a key of any model: it sets `num_banks` (whole banks of `bank_rows`) for `cache_unit` and `size_rows` for the toy cache. |
| Streamlit, run through `uv`, as the optional extra `ipu-memsim[ui]` | It installs cleanly on this Mac with one command and needs no layout code. It is not in the repo's Bazel lock: re-locking needs network access and would pull about 40 packages into every build. |
| PNG download through the chart's own menu | Streamlit's charts (Vega) save PNG themselves; converting the core's SVG to PNG would need a system library. The core still writes SVG files. |
| Residency is a per-kernel switch, not a chained replay | Chaining needs to know which output feeds which input across kernels with different layouts. The switch gives the two bounds now: every activation from DRAM, or every activation page that fits its banks already present and no output drained. |
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

| This branch after Stages 7 to 9, `bazel test -k //...` | 278 | 277 | 7295 passed, 265 skipped | 1 (the package's own lint target; fixed afterwards, see below) |

The difference to the baseline is the 7 new targets of `//src/ipu-memsim`
(97 test cases). The last full run had one failure, `//src/ipu-memsim:ruff`:
it was linting the `.venv` folder that `uv` creates for the browser app. The
lint glob now excludes it and that target passes; the other 277 targets were
not rerun after that one-line change.
No existing test changed result. The torch-dependent targets (among them the kernel
golden tests) could not be built here in either run, so they are
**not verified on this machine**; run `bazel test //...` on a machine where
torch installs (CI, Linux) before merging. The book's "996 tests, 987
passing" is from another tree and another runner and was not reproduced.

Stage 7 acceptance (`test/test_experiment.py`): a fill time of 0 reproduces
the ideal cycle count; stall never falls when the fill time grows. Both also
hold on all 37 recorded MobileViT-S traces at 1, 2 and 8 MB and prefetch
distances 1 and 16 (checked with a script, not a Bazel test, because traces
are not in git): no violation, so there is nothing to explain.
Stage 8 acceptance: `test_run_sweep_gives_one_row_per_group_and_value`
(shape of the table), `test_the_registry_covers_every_key_a_config_accepts`,
and `test/ui_smoke.py` (the app starts headless and runs one sweep; it is run
with `uv`, not Bazel, because Streamlit is not in the lock). The real server
was also started once and answered on its health URL.

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

**Stage 7: stall against bank fill time, 2 MB / 16 banks**
(`memsim-out/fill_time_2mb/`; TOY MODEL: the fill time is swept, the other
placeholders are listed in `spec.toml`; each kernel alone, `dram_roundtrip`).
Stall as a percentage of the type's ideal cycles:

| Layer type | 250 | 500 | 1,000 | 2,111 (HBM2 anchor) | 4,000 | 8,000 | Break-even (stall > 10%) |
|---|---|---|---|---|---|---|---|
| standard conv | 0.09 | 0.19 | 0.37 | 0.79 | 1.50 | 2.99 | above 8,000 |
| depthwise conv | 0.44 | 0.87 | 1.74 | 3.67 | 6.96 | 51.9 | first at 8,000 (line estimate 4,271) |
| pointwise conv | 0.18 | 0.37 | 0.74 | 1.56 | 2.95 | 5.90 | above 8,000 |
| attention | 0.51 | 1.02 | 2.04 | 4.30 | 8.14 | 16.3 | first at 8,000 (4,915) |
| MLP | 0.27 | 0.55 | 1.10 | 2.31 | 4.39 | 8.77 | above 8,000 |
| normalization | 1.66 | 3.25 | 6.49 | 13.7 | 25.9 | 51.8 | first at 2,111 (1,539) |
| elementwise | 12.1 | 24.2 | 48.4 | 102 | 193 | 386 | first at 250 (206) |
| reshape | 0.83 | 1.67 | 3.33 | 7.03 | 13.3 | 27.1 | first at 4,000 (3,004) |

A fill time of 0 gives 0 stall for every type. "Line estimate" is where a
straight line between the two neighbouring swept values crosses 10%; it is
not a replay. The depthwise jump between 4,000 and 8,000 is where the DMA
stops keeping up with the IPU, so the estimate there is rough: sweep that
interval more finely before quoting it.

**Grid: XMEM size against fill time** (`memsim-out/size_x_fill_time/`, one
heatmap per layer type). For every type, 1, 2, 4 and 8 MB give the same
numbers as the 2 MB row above (normalization and attention need at least 11
and 6 banks, so they start at 2 MB and 1 MB). At 0.5 MB the convolutions
thrash at every fill time (depthwise: 2,381% at 250 cycles), reshape reaches
1,777%, and pointwise crosses 10% near 3,900 cycles. So for these kernels
the fill time decides the stall, and XMEM size matters only below the point
where each table has enough banks.

**Stage 9: residency** (`memsim-out/residency_x_fill_time/`), at 2 MB and
the 2,111-cycle anchor, stall in percent of ideal cycles:

| Layer type | `dram_roundtrip` | `stays_in_xmem` |
|---|---|---|
| standard conv | 0.79 | 0.39 |
| depthwise conv | 3.67 | 2.47 |
| pointwise conv | 1.56 | 0.32 |
| attention | 4.30 | 0.00 |
| MLP | 2.31 | 1.57 |
| normalization | 13.7 | 0.28 |
| elementwise | 102 | 0.00 |
| reshape | 7.03 | 0.00 |

`stays_in_xmem` is the optimistic bound: every activation page that fits the
table's banks is already there, and no output is drained. What remains is
the weights (always from DRAM) and inputs larger than their banks. The truth
for a real network lies between the two columns.

**Replay time.** 259 replays (fill-time sweep, 37 traces) take 12 s; the
1,110 replays of the grid take 51 s, about 0.05 s each. The command line
prints progress and refuses a grid above `max_replays` (default 20,000)
without `--yes`; the app shows a progress bar and asks for confirmation
above its limit (default 2,000).

**DRAMsim3, HBM2 (`HBM2_8Gb_x128.ini`, one channel, consecutive addresses,
IPU clock 1 ns):** a 1024-row bank fills in 2,111 cycles and drains in
2,050; 64 rows fill in 191. This is in
`configs/cache_unit_hbm2_dramsim3.toml` and corresponds to about 0.49 rows
per cycle. It is DRAMsim3's answer for an assumed access pattern, not a
measurement of the team's HBM. On macOS DRAMsim3 aborts while exiting after
writing its statistics; the adapter reads the statistics and ignores the
exit status.

## 8. Next three steps

1. **Ask the structural questions** (section 9). They decide which column
   and which model variant is the real one; the swept parameters only need a
   range.
2. **Chain the kernels of a network on one `cache_unit`**, so residency is
   modelled layer to layer instead of as two bounds. This needs, per pair of
   layers, which output table feeds which input table.
3. **Set the real MobileViT-S shapes and counts** in
   `networks/mobilevit_s.toml` from the team's model export (the convolution
   list is a subset from the public architecture), and sweep the interval
   where a layer type crosses the threshold more finely.

Before a PR: add `ipu-memsim` to the dependency lock inputs
(`tools/update_requirements_lock.py`), and run `bazel test //...` on a
machine where torch installs.

## 9. For Ze'evi

**What you can do from the menu now.** Start the app (two lines in section
2). Pick MobileViT-S or a single kernel, pick the memory model, choose from
a dropdown which parameter to sweep and over which values, optionally a
second one for a heatmap, and press Run. You get the chart, the table, and
for each layer type the fill time at which stall passes 10% (you can change
the 10%). Every other parameter is on the page with its default, and a label
telling you whether it is CONFIRMED by a document, ASSUMED by the tool, or
UNKNOWN. You can save the experiment as a file and load it again, and the
same file runs from the command line, so every chart in the thesis can be
reproduced from the `spec.toml` saved next to it.

**What the first curves say.** At today's 2 MB, the heavy layers (the
convolutions and the matmuls) lose under 10% even at 8,000 cycles per bank,
except depthwise, which collapses somewhere between 4,000 and 8,000. The
light layers (residual add, LayerNorm, reshape) are dominated by moving
their data: residual add loses 12% already at 250 cycles. Making XMEM larger
than 2 MB changes nothing for these kernels; 0.5 MB breaks the convolutions
at any fill time. If activations can stay in XMEM between layers, most of
the stall of the light layers disappears.

**Which numbers are still placeholders.**

- The fill time itself is no longer a placeholder: it is the swept axis.
  The 2,111-cycle point is DRAMsim3's HBM2 answer, marked on the charts as
  an anchor, not as the chip.
- Still ASSUMED and able to move the curves: how banks are shared between
  tables, `jump_back`, how far ahead the DMA prefetches (the tool uses 1
  page; the spec allows more), one DMA at a time, the 1 ns clock, and that
  each kernel is replayed on its own.
- The convolution shapes and the layer counts in the MobileViT-S file are a
  subset from the public architecture, so totals across the network are not
  quotable yet.
- The torch-based tests of the repo still cannot run on this Mac.

**Questions for the team lead and the architects, structural ones first.**

1. Between two layers, does the output go to DRAM and come back, or stay in
   XMEM for the next kernel? (Q2)
2. Who decides how many banks each table gets, and by what rule? (Q4)
3. When a kernel returns to a page whose bank was released, is it reloaded
   from DRAM, or must the table keep enough banks? (Q5)
4. What is a "table": is one `kernel.toml` region one table? LayerNorm has
   11. (Q7)
5. What does the FPGA report: a stall counter, per-bank request counters,
   the fill time of a bank? (Q15)

And two requests for a range, not a decision:

6. What is the slowest and the fastest bank fill time you consider
   realistic, and at what clock? (Q1)
7. Which XMEM sizes and bank layouts are still open besides 16 x 1024 rows?
   (Q6)
