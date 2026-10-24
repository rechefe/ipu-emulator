"""The instruction handlers, one module per slot (E04-14).

Each module holds one slot's ``execute_<name>(ipu, *, <operands>)`` handlers,
whose keyword-only parameters are exactly the instruction's operand names in
``instruction_spec.py``, and the private helpers only they use. A slot module
never imports another one: what two slots share lives in
:mod:`ipu_emu.geometry` or on :class:`~ipu_emu.cycle.CycleView`.

:data:`HANDLER_MODULES` names the module that holds each slot's handlers;
``ipu_emu.ipu`` binds every spec instruction's ``execute_fn`` from it.
"""

from __future__ import annotations

from types import MappingProxyType, ModuleType

from ipu_emu.slots import aaq, aaq_exact, acc, brk, cond, load, lr, mult
from ipu_isa.slots import Slot

__all__ = ["HANDLER_MODULES"]

#: The module holding each slot's handlers. BREAK's is ``brk`` (``break`` is
#: a keyword).
HANDLER_MODULES: MappingProxyType[Slot, ModuleType] = MappingProxyType(
    {
        Slot.LR: lr,
        Slot.LOAD: load,
        Slot.MULT: mult,
        Slot.ACC: acc,
        Slot.AAQ: aaq,
        Slot.AAQ_EXACT: aaq_exact,
        Slot.COND: cond,
        Slot.BREAK: brk,
    }
)
