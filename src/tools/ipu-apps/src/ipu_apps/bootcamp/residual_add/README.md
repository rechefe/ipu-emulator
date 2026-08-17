# Bootcamp Drill 1/3: `residual_add`

**Difficulty: * (easiest)**

## Learning objective

Get comfortable with the two most basic ISA surfaces before touching real
compute: the **LR (loop register)** slot and the **LOAD/XMEM** slots. By the
end of this drill you should be able to explain, without looking anything up:

- Why a VLIW word's slots see a register **snapshot** taken at the start of
  the cycle, and how that forces you to pipeline a load one cycle ahead of
  the instruction that consumes it.
- Why `.asm` XMEM operands are **row numbers**, not byte addresses, and why
  `STR_ACC_REG`'s output stride differs from the input stride here.
- Why two operations that logically belong "together" (like bumping a
  counter and branching on it) sometimes have to live in **separate VLIW
  words** to see each other's effects.

## ISA concepts introduced

- `SET`, `ADD` (LR slot)
- `LDR_CYCLIC_MULT_REG` (LOAD slot, cyclic addressing into `R_CYCLIC`)
- `MULT.RC.VE` used as an identity copy (multiply by the CR1-is-always-1
  trick) -- introduced here only as plumbing; drill 2 is where MULT starts
  doing real work
- `ACC.ADD.FIRST` / `ACC.ADD` (ACC slot) -- again, only to combine two loads,
  not yet a "real" multiply-accumulate
- `STR_ACC_REG` (simulation-only accumulator store)
- `BLT` (COND slot) and the counted-loop idiom

Explicitly **not** used: masking, AAQ/`ACTIVATE`, multi-pass control flow.
Those show up in drills 2 and 3.

## What it computes

`out[v] = A[v] + B[v]` for `num_vectors` pairs of 128-element INT8 vectors,
producing INT32 sums (the accumulator is always 32-bit regardless of the
INT8 input width).

## Try it yourself

1. **Run the demo**: from the repo root,
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m ipu_apps.bootcamp.residual_add --vectors 8
   ```
2. **Break the pipelining on purpose.** In `residual_add.asm`, move the `LDR_CYCLIC_MULT_REG lr1 cr3 lr0;` load into the *same* word as the `MULT.RC.VE` that's supposed to consume the value loaded the cycle *before* it (i.e. try issuing the load and its consuming MULT in the same word instead of one cycle apart). Re-run the test and see what breaks -- then explain why, in terms of the snapshot-read rule.
3. **Extend it to subtraction -- with a real stub to fill in.** There's no
   `ACC.SUB` used in this drill, but it exists (see `instruction_spec.py`).
   Instead of just editing this file, go work through
   [`exercise/`](exercise/residual_sub_stub.asm): open
   `exercise/residual_sub_stub.asm`, find the `# TODO` marker, and swap the
   one `ACC.ADD` for `ACC.SUB` so it computes `A[v] - B[v]` instead of the
   sum. Self-check with:
   ```bash
   PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
     python -m pytest src/tools/ipu-apps/test/test_bootcamp_residual_add_exercise.py -v
   ```
   `test_residual_sub_solution_matches_numpy` runs the answer key
   (`exercise/residual_sub_solution.asm`) against
   `a.astype(np.int32) - b.astype(np.int32)` -- compare your fixed-up stub
   against it (don't peek at the solution file until you've tried).
4. **Try three inputs.** Extend the harness and assembly to add three tensors `A + B + C` instead of two. You'll need a third `LDR_CYCLIC_MULT_REG`/`MULT.RC.VE`/`ACC.ADD` triple per vector, and a third input base CR.

## Next step

Once this passes and you can explain the snapshot-timing and row-addressing
rules above, move on to [`../conv/`](../conv/README.md), which swaps the
"multiply by one" placeholder for a real multiply-accumulate sliding window.
