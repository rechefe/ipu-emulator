"""Exercise harness for bootcamp drill 3 (softmax): re-add numerical stability.

Reuses SoftmaxApp verbatim (same CR/XMEM setup, same FP32 wide-vector
state) -- the exercise only changes which .asm file is assembled, not the
memory map or register conventions.

Usage::

    from ipu_apps.bootcamp.softmax.exercise import SoftmaxExerciseApp, SOLUTION_ASM_PATH, STUB_ASM_PATH

    app = SoftmaxExerciseApp(
        inst_path="softmax.bin",
        input_path="logits.bin",
        output_path="probs.bin",
        rows=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path

from ipu_apps.bootcamp.softmax import SoftmaxApp

_EXERCISE_DIR = Path(__file__).resolve().parent

STUB_ASM_PATH = _EXERCISE_DIR / "softmax_unstable_stub.asm"
SOLUTION_ASM_PATH = _EXERCISE_DIR / "softmax_solution.asm"


class SoftmaxExerciseApp(SoftmaxApp):
    """Same CR/XMEM setup and FP32 wide-vector state as SoftmaxApp.

    Defaults ASM_PATH to the answer key (softmax_solution.asm); assemble
    STUB_ASM_PATH directly (as the test file does) to run the learner's
    as-shipped stub instead.
    """

    ASM_PATH = SOLUTION_ASM_PATH
