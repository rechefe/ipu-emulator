"""Declarative register descriptors for the IPU register file.

Each RegDescriptor defines a single register (or array of registers) in the
IPU.  The RegFile class uses REGFILE_SCHEMA to build the full register state
and the debug CLI auto-generates read/write commands from the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class RegDtype(Enum):
    """Data type for register storage and debug display."""
    UINT8 = "uint8"
    INT8 = "int8"
    UINT32 = "uint32"
    INT32 = "int32"
    UINT128 = "uint128"  # mask register


class RegKind(Enum):
    """Which pipeline stage / functional group the register belongs to."""
    MULT = "mult"
    ACC = "acc"
    LR = "lr"
    CR = "cr"
    MISC = "misc"


@dataclass(frozen=True)
class RegDescriptor:
    """Declarative description of a single register (or register array).

    Attributes:
        name:       Human-readable name used in debug CLI (e.g. "r0", "lr").
        kind:       Pipeline stage this register belongs to.
        size_bytes: Size of *one* register element in bytes.
        count:      Number of elements (e.g. 2 for r0/r1, 16 for LR array).
        dtype:      Element data type (for display / numpy view).
        cyclic:     If True, index wraps around (like r_cyclic).
        word_view:  If True, also expose a uint32 word view (like r_acc).
        debug_aliases: Extra names accepted by the debug CLI.
    """
    name: str
    kind: RegKind
    size_bytes: int
    count: int = 1
    dtype: RegDtype = RegDtype.UINT8
    cyclic: bool = False
    word_view: bool = False
    debug_aliases: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The single source of truth for the IPU register file layout.
# Adding a register here automatically creates storage, debug commands, and
# JSON serialisation support.
# ---------------------------------------------------------------------------

REGFILE_SCHEMA: list[RegDescriptor] = [
    # Mult-stage registers
    RegDescriptor(
        name="r",
        kind=RegKind.MULT,
        size_bytes=128,
        count=2,
        dtype=RegDtype.UINT8,
        debug_aliases=("r0", "r1"),
    ),
    RegDescriptor(
        name="r_cyclic",
        kind=RegKind.MULT,
        size_bytes=512,
        count=1,
        dtype=RegDtype.UINT8,
        cyclic=True,
        debug_aliases=("rcyclic",),
    ),
    RegDescriptor(
        name="r_mask",
        kind=RegKind.MULT,
        size_bytes=128,
        count=1,
        dtype=RegDtype.UINT128,
        debug_aliases=("rmask",),
    ),
    # Accumulator stage
    RegDescriptor(
        name="r_acc",
        kind=RegKind.ACC,
        size_bytes=512,
        count=1,
        dtype=RegDtype.UINT8,
        word_view=True,
        debug_aliases=("acc",),
    ),
    # LR / CR scalar register files
    RegDescriptor(
        name="lr",
        kind=RegKind.LR,
        size_bytes=4,
        count=16,
        dtype=RegDtype.UINT32,
    ),
    RegDescriptor(
        name="cr",
        kind=RegKind.CR,
        size_bytes=4,
        count=16,
        dtype=RegDtype.UINT32,
    ),
    # Misc / forwarding registers
    RegDescriptor(
        name="mult_res",
        kind=RegKind.MISC,
        size_bytes=512,
        count=1,
        dtype=RegDtype.UINT8,
        word_view=True,
    ),
    RegDescriptor(
        name="mem_bypass",
        kind=RegKind.MISC,
        size_bytes=128,
        count=1,
        dtype=RegDtype.UINT8,
    ),
]
