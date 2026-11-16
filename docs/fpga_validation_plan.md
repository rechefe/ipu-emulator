# FPGA validation plan

How to use the first FPGA measurements to check and calibrate the memory
what-if tool. The comparison code exists and is tested on synthetic data
(`ipu_memsim/fpga.py`, `memsim fpga-compare`); nothing here has been run
against real hardware yet. Which counters the FPGA exposes is still unknown
(question Q15 in [questions.md](questions.md)).

## What to bring back from the FPGA

One CSV file, one row per run (format in `ipu_memsim/fpga.py`):

```
kernel,config,measured_cycles,measured_stall_cycles,measured_fetches,measured_bank_fill_cycles,note
residual_add_256x144_default,fpga_16banks,2375,1212,3,,
```

| Column | Needed? | What it is | Why |
|---|---|---|---|
| `kernel` | yes | the name of the trace recorded for that kernel and shape | pairs the run with a replay |
| `config` | yes | the name of a model config describing the FPGA setup (`configs/fpga/*.toml`) | bank count, bank size, banks per table |
| `measured_cycles` | yes | cycles from kernel start to kernel done | the number the tool predicts |
| `measured_stall_cycles` | ask for it | cycles CTRL was stalled | separates memory time from compute time |
| `measured_fetches` | ask for it | bank fills from DRAM during the run (`bank_pub` events) | deterministic: should match the model exactly |
| `measured_bank_fill_cycles` | ask for it | time from `bank_req` to `bank_pub` for a full bank | calibrates the fill time directly |

Also ask, once: the table setup used (banks per table, `jump_back`), the
clock frequency, and which of bank count, DMA rate and `jump_back` can be
changed without a rebuild.

## Which kernels to run first

Small ones first, so a wrong assumption shows up cheaply. Cycle counts are the
emulator's (ideal, no memory stalls).

| Order | Kernel (trace) | Ideal cycles | What it tests |
|---|---|---|---|
| 1 | `residual_add_256x144_default` | 1,163 | Almost pure data movement: three small tables, each read or written once. Total minus ideal is the fill time of three partial banks. |
| 2 | `layernorm_64x192_default` | 2,944 | 11 tables: does one region = one table hold (Q7)? |
| 3 | `matmul_144x144_x128_default` | 43,921 | Weights re-read: every table must stay resident (Q5). |
| 4 | `qk_scores_256x36_default` | 39,426 | Attention scores. |
| 5 | `pointwise_conv_unified`, 32x32, 96 to 144 | 125,649 | Medium streaming kernel, tables around one bank. |
| 6 | `pointwise_conv_unified`, 128x128, 64 to 32 | 316,553 | 8 input pages, 4 output pages: prefetch, `jump_back` release, background drain. |
| 7 | `depthwise_conv_universal`, 64x64x64 | 19,686 | The kernel most sensitive to bank sharing and to prefetch (Stage 10). |

## Which parameters to change on the FPGA

In the order of the Stage 10 sensitivity ranking (most informative first):

1. **Banks per table** (if it can be set): give the depthwise input 1, 2 and
   3 banks. The model predicts a cliff between too few and enough.
2. **`jump_back`**: the trace-derived value against a larger one.
3. **Total banks enabled**: 16 against 8.
4. **DMA rate**, if it can be throttled: this is the fill-time axis.

## How to calibrate, then check

1. **Compute-only check.** Kernel 1 with its data preloaded (or subtract the
   stall counter). Measured cycles should equal the ideal count plus a
   constant (pipeline depth, start-up). That constant is not a memory effect;
   note it and subtract it from every later run.
2. **Fit the fill time.** `memsim fpga-compare --fit` fits one number, the
   cycles to fill a 1024-row bank, on half of the runs, by minimising the
   squared relative error of total cycles. If `measured_bank_fill_cycles` is
   available, compare the fitted value with it: they should agree.
3. **Convert to the clock.** Fitted cycles times the clock period is the fill
   time in ns; put both in a config (`fill_ns_per_bank`, `clock_ns`).
4. **Check on the other half.** The report gives, on runs *not* used for the
   fit, the relative error per run, the bias, and the rank correlation.

```bash
bazel run //src/ipu-memsim:memsim -- fpga-compare --measured fpga_runs.csv \
    --configs src/ipu-memsim/configs/fpga --traces memsim-out/traces --fit
```

## What agreement means

- **Fetch counts** are deterministic in the model. A mismatch is a wrong
  assumption (bank sharing, `jump_back`, prefetch, what a table is), not a
  parameter to tune. Fix the assumption before fitting anything.
- **Rank correlation near 1** with a large bias: the structure is right and a
  time constant is wrong (fill time, clock, a fixed start-up cost per
  transfer, question Q9).
- **Good fit on the calibration half, poor on the other half**: the model is
  missing a mechanism; look at which kernels disagree most (the report sorts
  them).

## Proof that the code works (synthetic data only)

`memsim fpga-compare --synthetic 2111 --noise 0.02 --fit` on eight recorded
kernels and two configs: "measurements" generated from the model at 2,111
cycles per bank with 2% random noise; the fit returned 2,138 cycles; on the 7
held-out runs the mean absolute error was 0.54%, the largest 1.01%, the rank
correlation 1.000, and every fetch count matched. This shows the harness
recovers a known fill time; it says nothing about the real chip.
