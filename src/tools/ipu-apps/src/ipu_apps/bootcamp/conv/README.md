# Bootcamp Drill 2/3: `conv`

**Difficulty: ** (adds MULT + ACC "for real")**

## Learning objective

Drill 1 used MULT/ACC only as plumbing (multiply-by-one). This drill uses
them for their actual purpose: a genuine multiply-accumulate sliding window.
By the end you should be able to explain:

- Why `rc_idx` on `MULT.RC.VE` is not just "which slot to read" but a **free
  whole-vector shift** -- and how that turns a sliding-window convolution
  into a handful of MULT+ACC cycles instead of a per-output-element loop.
- Why this drill produces no output for the two edge lanes (0 and 127), and
  what the real masking mechanism (`R_MASK` + `mask_shift`) would need to do
  instead if you wanted them.
- How `ACC.ADD.FIRST` vs `ACC.ADD` divide a multi-tap accumulation into "start
  the sum" and "add to the sum" steps.

## ISA concepts introduced

- `MULT.RC.VE` with a **non-zero `rc_idx`** (whole-vector shift) and a **CR
  register scalar source** (tap weight read directly from a CR's low byte,
  no LR indirection)
- `SUB`/`ADD` used to synthesize small signed constants (-1, +1) into LRs
- The `ACC.ADD.FIRST` -> `ACC.ADD` -> `ACC.ADD` three-tap accumulation
  pattern

Still **not** used: masking (`mask_offset`/`mask_shift` stay at their
pass-through defaults), AAQ/`ACTIVATE`, multi-pass control flow. Those are
drill 3.

## What it computes

A single-channel "valid" 1D convolution, kernel size 3, stride 1, no padding:

```
out[r][p] = w[-1]*in[r][p-1] + w[0]*in[r][p] + w[1]*in[r][p+1]
```

for `num_rows` independent 128-element INT8 rows, at output positions
`p = 1 .. 126`. (Positions 0 and 127 would need a neighbour outside the
loaded row; see `conv.asm`'s "WHY NO MASKING" section for exactly what
happens there and why it's safe to just not read those two lanes.)

## Try it yourself

1. **Run the demo**:
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m ipu_apps.bootcamp.conv --rows 4
   ```
2. **Inspect the edge lanes.** Modify `__main__.py` to print `out[:, 0]` and
   `out[:, 127]` (currently discarded) alongside the valid range. Work out by
   hand what value each edge lane actually holds, given the wraparound
   behaviour described in `conv.asm`, and confirm your prediction against the
   emulator's output.
3. **Widen the kernel to 5 taps -- with a real stub to fill in.** Instead of
   editing this file directly, go work through
   [`exercise/`](exercise/conv5_stub.asm): open `exercise/conv5_stub.asm`,
   find the `# TODO` markers, and add two more MULT.RC.VE/ACC.ADD pairs with
   `rc_idx = -2` and `rc_idx = +2` (plus the two new rc_idx-constant LRs feeding
   them). The valid output range shrinks to `p = 2 .. 125` -- the exercise
   harness (`exercise/__init__.py`) already has `VALID_LO`/`VALID_HI` and the
   two extra tap-weight CRs (`cr11`/`cr12`) set up for you. Self-check with:
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m pytest src/tools/ipu-apps/test/test_bootcamp_conv_exercise.py -v
   ```
   `test_conv5_solution_matches_numpy` runs the answer key
   (`exercise/conv5_solution.asm`) against a numpy 5-tap valid-conv
   reference (`p = 2..125`) -- compare your fixed-up stub against it (don't
   peek at the solution file until you've tried).
5. **Turn on masking.** As a bridge to drill 3's ISA surface: try setting up
   `R_MASK` so lanes 0 and 127 are correctly zeroed by hardware (`pad_mode`
   ZERO) instead of silently holding wrapped garbage/wrong-neighbour values.
   You'll need `LDR_MULT_MASK_REG` and a real `mask_shift` value instead of
   the always-zero `lr0` used here -- see `conv_first_layer` (referenced in
   project memory) for the pattern, though that app is far more elaborate
   than what you need here.

## Next step

Once this passes and you can explain the "shift is free via rc_idx" trick,
move on to [`../softmax/`](../softmax/README.md), which adds the AAQ
(`ACTIVATE`) stage and a real multi-pass (max / exp / sum / normalize)
control-flow structure.
