"""Exercise harness for bootcamp drill 1 (residual_add): add -> subtract.

Reuses ResidualAddApp's CR/XMEM setup verbatim (the exercise doesn't change
the memory map or register conventions, only the ACC op inside the loop
body), just pointing ASM_PATH at one of the two exercise .asm files instead
of ../residual_add.asm.

Usage::

    from ipu_apps.bootcamp.residual_add.exercise import ResidualSubApp, SOLUTION_ASM_PATH, STUB_ASM_PATH

    app = ResidualSubApp(
        inst_path="residual_sub.bin",
        input_a_path="tensor_a.bin",
        input_b_path="tensor_b.bin",
        output_path="output.bin",
        num_vectors=4,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path

from ipu_apps.bootcamp.residual_add import ResidualAddApp

_EXERCISE_DIR = Path(__file__).resolve().parent

STUB_ASM_PATH = _EXERCISE_DIR / "residual_sub_stub.asm"
SOLUTION_ASM_PATH = _EXERCISE_DIR / "residual_sub_solution.asm"


class ResidualSubApp(ResidualAddApp):
    """Same CR/XMEM setup as ResidualAddApp; ASM_PATH points at the solution.

    Pass ``asm_path=STUB_ASM_PATH`` to the constructor (or set
    ``ASM_PATH`` on a subclass) to run the learner's stub instead of the
    answer key -- see ``base.IpuApp`` for how ``ASM_PATH``/``inst_path`` are
    consumed; this class's default matches the answer key so importing it
    with no further configuration exercises the solution.
    """

    ASM_PATH = SOLUTION_ASM_PATH
