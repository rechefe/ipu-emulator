# Bootcamp Drill 3/3: `softmax`

**Difficulty: *** (adds AAQ + multi-pass control flow + cyclic/AGG registers)**

## Learning objective

Drills 1-2 were single-pass loops with one CR-driven address walk. This
drill introduces the pattern almost every non-trivial IPU kernel uses:
**multiple passes over the same data**, each pass a complete loop, with
intermediate results staged through XMEM between passes. By the end you
should be able to explain:

- Why softmax needs 4 separate passes (max, exp, sum, normalize) instead of
  one, and why the max/sum passes are *reductions* while the exp/normalize
  passes are *maps*.
- The difference between `AGG.MAX`/`AGG.SUM` (many lanes -> ONE `r_acc` slot,
  chosen by an LR) and `ACC.ADD`/`ACC.ADD.FIRST` from drills 1-2 (one
  multiply result -> ALL 128 `r_acc` lanes independently).
- Why `ACTIVATE.QUANTIZE` + `STR_POST_AAQ_REG` is a different (and more
  "real") output path than `STR_ACC_REG` (which drills 1-2 used, and which
  is explicitly simulation-only).
- Why softmax is reformulated in terms of `exp2` (base 2) instead of the
  mathematically natural `exp` (base e), and what role the resident `C_VEC`
  constant plays.

## ISA concepts introduced

- `AGG.MAX.FIRST` / `AGG.SUM.FIRST` (ACC slot, reduction start)
- `ACTIVATE.QUANTIZE` (AAQ slot) with `identity`, `exp2`, and `reciprocal`
- `STR_POST_AAQ_REG` (STORE slot -- the real hardware output path, as
  opposed to drills 1-2's simulation-only `STR_ACC_REG`)
- `MULT.EE` (broadcast a single Ra element, scaled by a CR, to all 128 lanes)
- A genuine 4-pass kernel structure, with three passes reading back data a
  previous pass staged in XMEM

## Why this drill runs in FP32 "wide-vector debug mode"

Softmax needs a real exponential and a real division. Doing those precisely
in INT8 requires a fixed-point requantization scheme that's a topic of its
own -- see project memory on quantized softmax if you're curious. This drill
sidesteps that entirely by running in the emulator's FP32 debug mode (every
lane is 4 bytes, holding an IEEE float) so you can focus on the multi-pass
*structure* without also learning INT8 quantization. The harness builds this
mode for you automatically (`SoftmaxApp.make_state()`).

## Why this drill stops at 128 rows

Passes 1 and 3 park one scalar (row max, row sum) per row into `r_acc`, which
only has 128 slots. A production kernel loops over *groups* of <=128 rows to
handle arbitrarily many rows (see project memory: `softmax_rows`). This
drill fixes `rows <= 128` so there's exactly one group and no group loop --
the 4-pass structure is the point, not the group-chunking logic stacked on
top of it in a real kernel.

## What it computes

Numerically-stable softmax, row-wise, over `rows` (<=128) rows of 128 FP32
logits each:

```
softmax(x)_j = exp(x_j - max(x)) / sum_k exp(x_k - max(x))
```

## Try it yourself

1. **Run the demo**:
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m ipu_apps.bootcamp.softmax --rows 16
   ```
2. **Re-add the numerical-stability trick -- a real stub to fill in.**
   [`exercise/softmax_unstable_stub.asm`](exercise/softmax_unstable_stub.asm)
   ships with Pass 1 (the max reduction) and the max-subtract half of Pass 2
   already stubbed OUT, so it computes `2^(c*x[r,j])` directly on raw logits
   instead of `2^(c*x[r,j] - maxvec[r])`. Your task is to re-add both pieces
   (find the `# TODO` markers) so the program matches `../softmax.asm`
   again. Self-check with:
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m pytest src/tools/ipu-apps/test/test_bootcamp_softmax_exercise.py -v
   ```
   `test_softmax_solution_matches_numpy` runs the answer key
   (`exercise/softmax_solution.asm`) against a numpy softmax reference,
   including a large-magnitude (`scale=50`) case. `test_stub_diverges_on_large_logits`
   shows what the unfixed stub does instead at that same scale (in this
   emulator: an `OverflowError` out of `ACTIVATE.QUANTIZE`'s FP32 pack step,
   since `2^(c*x_j)` overflows FP32 for large logits) -- that's the failure
   subtracting the row max before exponentiating is meant to prevent.
3. **Add a fifth pass**: compute the row-wise entropy `-sum_j p_j * log2(p_j)`
   of the softmax output as a bonus reduction pass, staged to its own XMEM
   region. You'll need another `AGG.SUM`-style reduction and the `identity`
   or a suitable activation to get `log2(p_j)` (hint: check what activation
   functions are available in `instruction_spec.py`'s `ActivationFn`
   description before assuming one exists).
4. **Compare against drill 2's single-pass style.** Try (on paper, or for
   real) writing softmax as ONE pass instead of four. You'll find you can't:
   the max and the sum are both needed before you can compute any output
   element, and both require seeing the whole row first -- this is why
   reductions force a multi-pass structure whenever a later step depends on
   a global property of a still-earlier step's output.

## Progression recap

| Drill | ISA surface added | Control flow |
|---|---|---|
| 1. `residual_add` | LR, LOAD/XMEM | one counted loop |
| 2. `conv` | + MULT (real), ACC (real) | one counted loop |
| 3. `softmax` (this one) | + AAQ/`ACTIVATE`, AGG reductions | four passes, staged through XMEM |

You've now touched every major slot in the VLIW word except masking
(`R_MASK`/`mask_shift`) and branching beyond a simple counted loop (`BEQ`,
`BGE`, `BR`) -- those show up in the project's full-scale apps
(`conv_first_layer`, `softmax_rows_partial`, etc.), which this bootcamp was
deliberately scoped to stay short of.
