"""Regression test for the BLT signed-comparison sign-extension bug.

LR/CR registers are ``LR_CR_SCALAR_BITS`` (20) bits wide. A counter initialized
to ``-1`` is stored as ``0xFFFFF`` (1048575). ``execute_blt`` must interpret that
as the signed value ``-1`` when comparing against a positive bound — otherwise a
loop with a negative-initialized counter (the standard ``first_index - 1`` startup
used by the cyclic-multiply matmul kernels) never branches and runs its body once.

The bug: ``execute_blt`` calls ``_to_signed_32`` (sign bit at bit 31), so
``0xFFFFF`` reads back as ``+1048575``, and ``BLT (-1) (127)`` does NOT branch.

Run just this file::

    cd src/tools/ipu-emu-py
    uv run pytest test/test_blt_sign_extend.py -v
"""

from __future__ import annotations

from ipu_emu.execute import decode_instruction_word
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.ipu_state import IpuState

from ipu_as.lark_tree import assemble


def _make_state(asm_code: str, *, cr: dict[int, int] | None = None) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    load_program(state, decoded)
    return state


def test_negative_one_stores_as_20bit_mask():
    """Sanity check: setting a CR to -1 reads back as the 20-bit all-ones value."""
    state = IpuState()
    state.regfile.set_cr(9, -1)
    assert state.regfile.get_cr(9) == 0xFFFFF  # 1048575


def test_blt_branches_on_negative_lr():
    """BLT(-1, 127) must branch: -1 < 127.

    Mirrors the matmul k-loop startup ``SET lr5 cr9`` where cr9 = -1.
    With the sign-extension bug, lr5 reads as +1048575 and BLT falls through,
    leaving lr2 == 0 instead of 1.
    """
    state = _make_state(
        """\
SET lr0 cr9;;
SET lr1 cr10;;
BLT lr0 lr1 less_branch;;
SET lr2 cr11;;
BKPT;;
less_branch:
SET lr2 cr12;;
BKPT;;
""",
        cr={9: -1, 10: 127, 11: 0, 12: 1},
    )
    run_until_complete(state)
    assert state.regfile.get_lr(2) == 1, (
        "BLT did not branch on a negative counter — "
        "execute_blt sign-extends at bit 31 instead of LR_CR_SCALAR_BITS (20)"
    )


def test_blt_does_not_branch_when_not_less():
    """BLT(-1, -5) must NOT branch: -1 is greater than -5.

    A decisive negative-vs-negative case that pins down sign handling without
    relying on loop timing.
    """
    state = _make_state(
        """\
SET lr0 cr9;;
SET lr1 cr10;;
BLT lr0 lr1 neg_branch;;
SET lr2 cr11;;
BKPT;;
neg_branch:
SET lr2 cr12;;
BKPT;;
""",
        cr={9: -1, 10: -5, 11: 7, 12: 9},
    )
    run_until_complete(state)
    # -1 < -5 is false → fall through → lr2 == 7.
    assert state.regfile.get_lr(2) == 7, (
        "BLT branched when it should not have for negative operands"
    )
