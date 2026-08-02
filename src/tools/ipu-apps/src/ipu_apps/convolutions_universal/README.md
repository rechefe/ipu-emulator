# convolutions_universal — App Catalog

This package holds every convolution-family app on the row-addressed ISA
(`mb/195`). All apps share:

- **Row-addressed XMEM**: `LDR_MULT_REG`/`LDR_CYCLIC_MULT_REG`'s `offset`/`base`,
  `STR_POST_AAQ_REG`, and `LDR_MULT_MASK_REG` operands are XMEM **row numbers**
  (`byte_addr // 128`), not byte addresses. `R_CYCLIC` operands
  (`LDR_CYCLIC_MULT_REG`'s `index`, `MULT.RC.*`'s `rc_idx`) stay
  **element-addressed** (a 512-element ring) — never rescale those.
- **INT8**, zero-padded ("same") convolution, unless noted otherwise.
- A `benchmark/benchmark.py` + `benchmark/results.md` reporting real
  (emulator-measured) cycle counts and MULT-slot utilization — not analytical
  estimates.
- A `test/test_<app>.py` with a bit-exact NumPy/`ipu_math` reference.

For a line-by-line walkthrough of how one of these kernels is actually built,
see [`depthwise/depthwise_conv_universal/GUIDE.md`](depthwise/depthwise_conv_universal/GUIDE.md) —
it's the most mature app here (9 cyc/channel) and a good template for reading
the others.

## Catalog

| App | Shape | Stride | Special | cyc/unit* | Status |
|---|---|---|---|---|---|
| [`conv/conv_universal`](conv/conv_universal/README.md) | 3×3, cols∈{16,32,64} | 1 | walking-pointer rotating-slot | ~11 cyc/ch | Mature |
| [`conv/conv_universal_bn_activation`](conv/conv_universal_bn_activation) | 3×3, cols∈{16,32,64} | 1 | + per-channel bias, ReLU | ~9 cyc/ch | Mature |
| [`conv/conv_first_layer`](conv/conv_first_layer) | 3×3, 256×256×3→128×128×16 | 2 | fixed-shape first-layer conv, INT8+BN+ReLU | — | Mature, not generalized |
| [`conv/conv_universal_wide384`](conv/conv_universal_wide384) | 3×3, width≥384 (mult. of 128), even out_channels | 1 | 3-slot R_CYCLIC strip generalizing `conv_first_layer`'s trick to arbitrary `cpr=width/128` | ~24% MULT util | **Correctness-first only** — no rotating-slot pipelining yet; 384 is just the minimum supported width, not a tuned target. See its own docstring. |
| [`depthwise/depthwise_conv_universal`](depthwise/depthwise_conv_universal/GUIDE.md) | 3×3 depthwise, cols∈{16,32,64} | 1 | no bias, no activation | **9 cyc/ch** | Mature — see the guide |
| [`depthwise/depthwise_conv_universal_bn_activation`](depthwise/depthwise_conv_universal_bn_activation) | 3×3 depthwise, cols∈{16,32,64} | 1 | + per-channel bias, ReLU | 10 cyc/ch | Mature |
| [`depthwise/depthwise_conv_stride2_128`](depthwise/depthwise_conv_stride2_128) | 3×3 depthwise, cols=128 | 2 | two-stage: unmodified `depthwise_conv_universal` + ACC.STRIDE decimate pass | ~79% MULT util (stage 1 dominated) | Complete, benchmarked |
| [`depthwise/depthwise_conv_stride2_narrow`](depthwise/depthwise_conv_stride2_narrow) | 3×3 depthwise, cols∈{16,32,64} | 2 | same two-stage pattern; ACC.STRIDE's `elements_in_row=cols` does row-splitting for free | ~103% (stats-composition quirk, see below) | Complete, benchmarked |
| [`pointwise/pointwise_conv_unified`](pointwise/pointwise_conv_unified) | 1×1 | 1 | multi-pass, `oc_per_reg=1` padded kernel; wins on awkward `in_ch` | G+2 cyc/OC | Mature |
| [`pointwise/pointwise_conv_unified_bn_activation`](pointwise/pointwise_conv_unified_bn_activation) | 1×1 | 1 | + per-channel bias (bias region mirrors kernel layout), ReLU | — | Mature |
| [`residual_add`](residual_add) | elementwise add | — | residual/skip-connection add for conv towers | — | Mature |

*cyc/unit = cycles per channel (depthwise) or per output-channel (others), where documented — see each app's own README/docstring for the exact definition.

## Two-stage apps (stride-2 family)

`depthwise_conv_stride2_128` and `depthwise_conv_stride2_narrow` don't
implement stride-2 directly in one pass. Instead:

1. **Stage 1** runs the existing, proven `depthwise_conv_universal` completely
   unmodified (or via a thin cols=128 subclass) at full spatial resolution.
2. **Stage 2** is a short, separate program — sharing the same `IpuState`/XMEM —
   that decimates stage 1's full-resolution output down to the stride-2 result,
   using `ACC.STRIDE` to pick out kept rows/columns.

This pattern is reusable: it lets you get a new stride/shape variant working
by composing an already-correct kernel with a small reshuffle pass, rather
than hand-deriving a new fused pipeline. See either app's `__init__.py` for
the mechanics (`tempfile`-based binary assembly, resetting
`state.program_counter = 0` between stages).

**Known quirk:** `RunStats.total_cycles` is *overwritten*, not accumulated,
across sequential `run_test()` calls sharing one `IpuState` — so
`state.stats.mult_utilization` read directly after stage 2 divides combined
MULT-active cycles by stage-2-only cycles, occasionally producing >100%
figures (see `depthwise_conv_stride2_narrow/benchmark/results.md`). Both
benchmark scripts work around this by recomputing
`mult_active_cycles / (cycles1 + cycles2)` manually; the underlying
`emulator.py`/`stats.py` bug has not been fixed at the source.

## Adding a new app

The two-stage pattern above and the wide384/first_layer R_CYCLIC-strip
technique are the two proven ways to add spatial coverage without a full
redesign:

- **New stride/decimation variant of an existing kernel** → two-stage pattern
  (run the base kernel unmodified, add a decimate pass).
- **New spatial width/shape not covered by chunk-interleaved packing** →
  R_CYCLIC-strip pattern (see `conv_universal_wide384`'s module docstring for
  the full technique and its generalization from `conv_first_layer`).

Whichever you pick, follow the project's standard checklist in the top-level
`CLAUDE.md`/`SKILL.md`: never hand-assign opcodes, keep `execute_*` operand
names matching `instruction_spec.py`, and add the app to this table plus its
own `benchmark/` and `test/`.
