# Results log

Every result the tool has produced, with the command that reproduces it, its
label, and the question to the architects that would sharpen it. All
commands run from the repository root; `M` stands for
`bazel run //src/ipu-memsim:memsim --`, and `E` for
`src/ipu-memsim/experiments`. Outputs go to `memsim-out/` (not in git); each
experiment folder holds the `spec.toml` that produced it.

Labels: **replayed** = recorded on the emulator and replayed (the cycle
counts are exact, the stalls depend on the model's parameters); **toy** = a
replay that rests on placeholder or assumed parameters; **extrapolated** = no
replay, a formula applied to a shape; **counted** = arithmetic on layer
shapes; **synthetic** = generated from the model itself to test code.

Question numbers refer to [questions.md](questions.md).

## 1. Emulator cycle counts (replayed, exact)

| Kernel and shape | Ideal cycles | Command |
|---|---|---|
| `residual_add`, 64 channels | 325 | `M record residual_add` |
| `pointwise_conv_unified` 128x128, 64 to 32 | 316,553 | `M record pointwise_conv_unified --set height=128 --set width=128 --set in_channels=64 --set out_channels=32` |
| `depthwise_conv_universal` 64x64x64 | 19,686 | recorded by `M network src/ipu-memsim/networks/mobilevit_s.toml ...` |
| `conv_universal` 64x64, 32 to 32 | 318,564 | same |
| `matmul_144x144_x128` | 43,921 | same |

They equal the project book's numbers. Nothing sharpens them: they are the
emulator's.

## 2. Stage 3 sweeps on the pointwise convolution (toy)

`M sweep memsim-out/pointwise_128x128_64to32.csv.gz --out memsim-out/sweep_pointwise`

Runtime 318,926 to 882,553 cycles over 4 to 64 banks and three DRAM speeds
(table in STATUS.md, section 7). Uses the older latency-plus-bandwidth DRAM
model with invented speeds. Sharpened by Q1; superseded by sections 4 and 5.

## 3. Stage 4 per-layer-type table (toy)

`M network src/ipu-memsim/networks/mobilevit_s.toml --out memsim-out/mobilevit_s --configs src/ipu-memsim/configs/study_05mb_slow.toml src/ipu-memsim/configs/study_2mb_slow.toml src/ipu-memsim/configs/study_8mb_slow.toml src/ipu-memsim/configs/study_2mb_fast.toml`

Per layer type: ideal cycles, footprint, peak live rows, stall under four
configurations (STATUS.md, section 7). Layer types with no kernel are listed.
Sharpened by Q1, Q2; the layer list by the team's model export.

## 4. Stall against bank fill time, 2 MB (toy; the fill time is the swept axis)

`M sweep --spec E/fill_time_2mb.toml` and, finer, `M sweep --spec E/fill_time_fine.toml`

Stall in percent of ideal cycles at 250 / 2,111 / 8,000 cycles per bank:
standard conv 0.09 / 0.79 / 2.99; depthwise 0.44 / 3.67 / 51.9; pointwise
0.18 / 1.56 / 5.90; attention 0.51 / 4.30 / 16.3; MLP 0.27 / 2.31 / 8.77;
normalization 1.66 / 13.7 / 51.8; elementwise 12.1 / 102 / 386; reshape 0.83 /
7.03 / 27.1.

Break-even (stall first above 10%), from the finer sweep: normalization at
1,600 cycles (line estimate 1,540); reshape at 3,200 (3,003); attention at
5,000 (4,915); depthwise at 5,500 (5,280); elementwise at or below 250;
standard conv, pointwise and MLP not within 8,000.

Sharpened by: Q1 (where on the axis the chip is), then Q2 and Q4.

## 5. XMEM size against fill time (toy)

`M sweep --spec E/size_x_fill_time.toml`

One heatmap per layer type. 1, 2, 4 and 8 MB give the same stall as 2 MB
(LayerNorm needs 11 banks, softmax 6, so they have no value below that); at
0.5 MB the convolutions thrash at every fill time. Sharpened by Q4, Q6, Q7.

## 6. Residency: DRAM round trip against stays in XMEM (toy; two bounds)

`M sweep --spec E/residency_x_fill_time.toml`

At 2 MB and 2,111 cycles, stall percent round trip / stays: standard conv
0.79 / 0.39; depthwise 3.67 / 2.47; pointwise 1.56 / 0.32; attention 4.30 /
0.00; MLP 2.31 / 1.57; normalization 13.7 / 0.28; elementwise 102 / 0.00;
reshape 7.03 / 0.00. Sharpened by Q2 (which column is real) and Q19.

## 7. Sensitivity of the assumptions (toy; ranks the questions)

`M sensitivity --spec E/sensitivity_2mb.toml`

Largest spread in stall (percentage points) per parameter: `bank_allocation`
20,102; `jump_back` 10,605; `fill_ns_per_bank` 374; `fill_whole_bank` 353;
`clock_ns` 153; `residency` 102; `prefetch_before_start` 102;
`prefetch_distance` 19; `dma_channels` 0.4. The baseline column equals the
2,111 column of section 4 exactly. Full table:
`memsim-out/sensitivity_2mb/sensitivity.csv`; pair heatmaps in `pair/`.
Sharpened by Q4, Q8, Q5 first.

## 8. Knees (toy)

`M knee --spec E/knee_2mb.toml`

At 2,111 cycles per bank, banks of 1024 rows. Plateau knee (more banks stop
helping) / formula's minimum: standard conv 5 / 5; depthwise 6 / 5; pointwise
7 / 5; attention 6 / 6; MLP 4 / 4; normalization 11 / 11; elementwise 3 / 3;
reshape 5 / no formula. In MB: 0.375 (elementwise) to 1.375 (normalization).
Threshold knee (stall at most 10%): 4 to 6 banks, and none for normalization
and elementwise, whose stall is above 10% with any number of banks. Where the
knee is above the formula, the extra banks are for prefetching a streamed
table (pointwise input and output) or for the depthwise start-up effect
(`formulas.measured_banks` docstring). Sharpened by Q4, Q7.

## 9. Working-set formulas (replayed validation)

`M formulas`

426 table checks (37 traces, bank sizes 1024, 256, 64): rows and look-back
exact in 426; banks equal to the replay's minimum in 416, one more than
needed in 10, never fewer. Kernels without a formula: concat, fold, unfold.
The formulas hold for the MobileViT-S shapes they were checked on; a wider
range of shapes needs new traces.

## 10. Network-only tables for other families (counted; knees extrapolated)

`M netstats src/ipu-memsim/networks/deit_tiny.toml src/ipu-memsim/networks/swin_t.toml src/ipu-memsim/networks/yolov8n.toml`

Parameters counted against published: DeiT-Tiny 5,717,416 (exact), Swin-T
28,288,354 (exact), YOLOv8n 3,157,200 (exact). Operations per byte: about 83
to 85 for the MLP layers of both transformers, 0.3 to 0.5 for LayerNorm and
residual adds, 162 for YOLOv8n's 3x3 convolutions. Extrapolated knees
(banks): DeiT-Tiny MLP 5, attention 4, normalization 11; Swin-T MLP 22,
normalization 15, attention 4. Not covered: patch embeddings, stride-2
convolutions, convolutions wider than 128, pooling, embeddings. The shapes
are from the public architectures, written by hand; confirm with
`scripts/export_layers.py`. Sharpened by: kernels for these shapes, or at
least traces of the existing kernels at these shapes.

## 11. DRAMsim3 HBM2 anchor (simulator output, not a measurement)

`M dram-measure --dramsim3 /path/to/dramsim3main.out --ini /path/to/DRAMsim3/configs/HBM2_8Gb_x128.ini`

A 1024-row bank fills in 2,111 cycles and drains in 2,050 at a 1 ns clock,
one channel, consecutive addresses. Sharpened by Q1 and by the FPGA.

## 12. FPGA harness self-test (synthetic)

`M fpga-compare --measured memsim-out/fpga/synthetic_measurements.csv --configs src/ipu-memsim/configs/fpga --traces memsim-out/fpga_demo_traces --synthetic 2111 --noise 0.02 --fit`

Fitted fill time 2,138 against the true 2,111; held-out mean error 0.54%,
rank correlation 1.000. Proves the code, says nothing about the chip.
Replaced by real data when the FPGA delivers it (Q15).
