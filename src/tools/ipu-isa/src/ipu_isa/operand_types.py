"""The operand-type registry: every fact about each operand type, once (E03-05).

Each operand type of ``instruction_spec`` (``LrIdx``, ``ActivationFn``, ...) has
one entry here giving its kind, width, value domain, whether it can be read as a
register, the names it carries on the wire and its documentation. Every other
table (the valid-type set, the widths the union layout packs, the emulator's
decoded-field suffixes, the SV typedefs, the operand docs) is derived from it.

The entries are authored as dicts (Q30), checked against a fixed key set, and
loaded into frozen :class:`OperandType` records in :data:`OPERAND_TYPES`.

``token_class``, ``field_suffix`` and ``sv_typedef`` are wire format: the
emulator keys decoded instruction words by ``field_suffix``, and the downstream
RTL repo names its types after ``sv_typedef``. They are written out, not
computed from Python class names, so a refactor cannot change them by accident.
Changing one is a breaking change that a PR must call out, like a ``*_WIDTH``
change (CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from ipu_isa import schema
from ipu_isa.acc_stride_enums import (
    ELEMENTS_IN_ROW_NAMES,
    HORIZONTAL_STRIDE_NAMES,
    VERTICAL_STRIDE_NAMES,
)
from ipu_isa.activations import ACTIVATION_FN_NAMES
from ipu_isa.errors import IsaLookupError
from ipu_isa.geometry import INST_MEM_WORDS
from ipu_isa.incr_mod_pow2_k import (
    LR_MOD_POW2_K_FIELD_BITS,
    LR_MOD_POW2_K_MAX,
    LR_MOD_POW2_K_MIN,
)
from ipu_isa.mult_mask_offset import MULT_MASK_OFFSET_FIELD_BITS, MULT_MASK_SLOT_COUNT
from ipu_isa.registers import REGISTER_DEFINITIONS


class OperandKind(Enum):
    """What an operand's value names: a register, an enum member, an immediate or a label."""

    REGISTER = "register"
    ENUM = "enum"
    IMMEDIATE = "immediate"
    LABEL = "label"


@dataclass(frozen=True, slots=True)
class Domain:
    """The values an operand accepts: assembly names in encoding order
    (registers and enums), or an inclusive integer range (immediates and
    labels). An immediate whose width the union packing decides has neither."""

    members: tuple[str, ...] = ()
    value_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class OperandType:
    """One operand type of the registry: its width, value domain and wire names."""

    name: str  # "LrIdx": the spec key; appears as an SV comment
    kind: OperandKind
    bits: int | None  # None: decided by union packing
    domain: Domain
    readable: bool  # a "read" flag is legal on this operand
    token_class: str  # the assembler token class (wire: names decoded fields today)
    field_suffix: str  # decoded-word field-name suffix (wire format)
    sv_typedef: str | None  # SV enum typedef, for REGISTER and ENUM kinds
    summary: str  # one line
    doc: str  # long form, for the generated operand-types page and editor hover

    def field_codes(self, bits: int | None) -> FieldCodes:
        """How a ``bits``-wide field holds this type's values; the one rule
        disassembly (``codec.disasm.type_text``) and the reserved-code fault
        (``faults.reserved_operand_codes``) read.

        * REGISTER and ENUM: code ``i`` is ``domain.members[i]``.
        * An immediate or label whose value range fits the field: the range is
          stored from code 0 upward (``INC.MODP2`` stores ``k - 1``), so a code
          plus the range's low end is its value, and only the range's size of
          codes is defined.
        * Any other type, or ``bits`` None: every code is its own value.
        """
        if self.kind in (OperandKind.REGISTER, OperandKind.ENUM):
            members = self.domain.members
            return FieldCodes(members=members, offset=0, defined=len(members))
        if self.domain.value_range is not None and bits is not None:
            low, high = self.domain.value_range
            size = high - low + 1
            if size <= 1 << bits:
                return FieldCodes(members=None, offset=low, defined=size)
        return FieldCodes(members=None, offset=0, defined=None)


@dataclass(frozen=True, slots=True)
class FieldCodes:
    """How the codes of one field map to an operand type's values (``OperandType.field_codes``)."""

    members: tuple[str, ...] | None  # REGISTER and ENUM: code i names members[i]
    offset: int  # a code's value is code + offset
    defined: int | None  # codes 0..defined-1 mean a value; None: every code does


_LR_COUNT = REGISTER_DEFINITIONS["lr"]["count"]
_CR_COUNT = REGISTER_DEFINITIONS["cr"]["count"]


def code_bits(count: int) -> int:
    """Bits to encode one of ``count`` values (codes ``0..count-1``).

    The width of an index or enum field, for the registry here and for an
    assembler's enum tokens (E05-02: bit layouts are computed in ``ipu_isa``).
    """
    return (count - 1).bit_length()


def _members(bank: str) -> tuple[str, ...]:
    return tuple(REGISTER_DEFINITIONS[bank]["assembler_values"])


#: The keys each authored entry holds (all required).
ENTRY_KEYS = schema.Keys(
    frozenset(
        {
            "kind",
            "bits",
            "domain",
            "readable",
            "token_class",
            "field_suffix",
            "sv_typedef",
            "summary",
            "doc",
        }
    )
)

_SPEC: dict[str, dict[str, Any]] = {
    "MultStageReg": {
        "kind": OperandKind.REGISTER,
        "bits": 2,
        "domain": Domain(members=_members("r")),
        "readable": True,
        "token_class": "MultStageRegField",
        "field_suffix": "mult_stage_reg_field",
        "sv_typedef": "mult_stage_reg_field_t",
        "summary": "Multiply-stage register R0 or R1 (2-bit field).",
        "doc": "Multiply-stage field in the VLIW encoding. Assembly accepts **`r0`** and **`r1`** only; "
        "the field is **two bits** wide (encoding `2` is reserved). Used as the destination of "
        "`LD.MULT` and as the **`ra`** operand of `MULT.RC.VV`.",
    },
    "LrIdx": {
        "kind": OperandKind.REGISTER,
        "bits": code_bits(_LR_COUNT),
        "domain": Domain(members=_members("lr")),
        "readable": True,
        "token_class": "LrRegField",
        "field_suffix": "lr_reg_field",
        "sv_typedef": "lr_reg_field_t",
        "summary": "Loop register LR0–LR15.",
        "doc": "Loop register index: resolves to **`lr0`** … **`lr15`**. Often used for addresses, strides, "
        "and control values. When marked `read: live` in the spec, the emulator reads the **current** "
        "LR value after earlier slots in the same cycle.",
    },
    "CrIdx": {
        "kind": OperandKind.REGISTER,
        "bits": code_bits(_CR_COUNT),
        "domain": Domain(members=_members("cr")),
        "readable": True,
        "token_class": "CrRegField",
        "field_suffix": "cr_reg_field",
        "sv_typedef": "cr_reg_field_t",
        "summary": "Configuration register CR0–CR15.",
        "doc": "Constant-register index: **`cr0`** … **`cr15`**. CRs are **read-only** in assembly; the "
        "harness initializes them (e.g. base pointers, strides). **`cr15`** conventionally holds "
        "dstructure configuration but is a valid `CrIdx` operand like any other CR. **`SET`** in "
        "the LR slot copies the full **32-bit** CR value into an LR.",
    },
    "LcrIdx": {
        "kind": OperandKind.REGISTER,
        "bits": code_bits(_LR_COUNT + _CR_COUNT),
        "domain": Domain(members=_members("lr") + _members("cr")),
        "readable": True,
        "token_class": "LcrRegField",
        "field_suffix": "lcr_reg_field",
        "sv_typedef": "lcr_reg_field_t",
        "summary": "A loop register or a configuration register in one field.",
        "doc": "LR **or** CR index in one field: lower indices map to **`lr0`–`lr15`**, higher indices to "
        "**`cr0`–`cr15`** in the usual combined ordering used by the assembler. Used as **`src_b`** "
        "on **`ADD`**/**`SUB`**, as **`step`** on **`INC.MODP2`**, and as **`src`** on "
        "**`MULT.RC.VE`** (an LR's stored value selects an element from `R0`/`R1`; a CR's low byte "
        "is the scalar directly).",
    },
    "LrdIdx": {
        "kind": OperandKind.REGISTER,
        "bits": code_bits(_LR_COUNT // 2),
        "domain": Domain(members=tuple(f"lrd{2 * i}" for i in range(_LR_COUNT // 2))),
        "readable": True,
        "token_class": "LrdRegField",
        "field_suffix": "lrd_reg_field",
        "sv_typedef": "lrd_reg_field_t",
        "summary": "Register pair LRDn = LR(n+1):LR(n), n even.",
        "doc": "Register-pair index: **`lrd0`**, **`lrd2`**, **`lrd4`**, … **`lrd14`** — named after "
        "the lower register in the pair — each aliasing two adjacent **`LR`** registers as one "
        "8-byte-element value: **`LRDn`** = **`LR(n+1)`** (high elements) concatenated with **`LR(n)`** "
        "(low elements). Purely an assembler/emulator view; there is no separate physical storage. "
        "Used as **`dest`** on **`ADDB`** / **`ADDBI`**.",
    },
    "DstructureCrIdx": {
        "kind": OperandKind.REGISTER,
        "bits": code_bits(_CR_COUNT),
        "domain": Domain(members=_members("cr")),
        "readable": False,
        "token_class": "DstructureCrRegField",
        "field_suffix": "dstructure_cr_reg_field",
        "sv_typedef": "dstructure_cr_reg_field_t",
        "summary": "Configuration register holding the dstructure (valid elements, partition, pad mode).",
        "doc": "Constant-register index: **`cr0`** … **`cr15`**, selecting which CR supplies the "
        "**valid element mask** / dstructure configuration (`valid_elements`, `partition`) for "
        "`AGG.SUM`, `AGG.SUM.FIRST`, `AGG.MAX`, `AGG.MAX.FIRST`, `ACTIVATE.QUANTIZE`, and the "
        "masking multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, "
        "`MULT.EE`). Unlike **`CrIdx`**, **`cr15`** is allowed here — it's the conventional "
        "dstructure register — but the operand is always mandatory; there is no implicit "
        "fallback to `cr15` when it is omitted.",
    },
    "ElementsInRow": {
        "kind": OperandKind.ENUM,
        "bits": code_bits(len(ELEMENTS_IN_ROW_NAMES)),
        "domain": Domain(members=tuple(ELEMENTS_IN_ROW_NAMES)),
        "readable": False,
        "token_class": "ElementsInRowField",
        "field_suffix": "elements_in_row_field",
        "sv_typedef": "elements_in_row_field_t",
        "summary": "ACC.STRIDE elements per row: 16, 32 or 64.",
        "doc": "ACC-slot immediate: elements per row for **`ACC.STRIDE`**. Valid values: **`16`**, **`32`**, "
        "**`64`** (minimum is 16; encoded 0→16, 1→32, 2→64). See `acc_stride_enums` in `ipu_isa`.",
    },
    "HorizontalStride": {
        "kind": OperandKind.ENUM,
        "bits": code_bits(len(HORIZONTAL_STRIDE_NAMES)),
        "domain": Domain(members=tuple(HORIZONTAL_STRIDE_NAMES)),
        "readable": False,
        "token_class": "HorizontalStrideField",
        "field_suffix": "horizontal_stride_field",
        "sv_typedef": "horizontal_stride_field_t",
        "summary": "ACC.STRIDE horizontal stride mode.",
        "doc": "ACC-slot immediate: horizontal stride mode for **`ACC.STRIDE`**. Valid values: **`off`**, "
        "**`on`**, **`on_inv`** (2-bit encoded enum; **`reserved3`** is reserved). Expand padding "
        "is fixed hardware behaviour and is not programmable. See `acc_stride_enums` in `ipu_isa`.",
    },
    "VerticalStride": {
        "kind": OperandKind.ENUM,
        "bits": code_bits(len(VERTICAL_STRIDE_NAMES)),
        "domain": Domain(members=tuple(VERTICAL_STRIDE_NAMES)),
        "readable": False,
        "token_class": "VerticalStrideField",
        "field_suffix": "vertical_stride_field",
        "sv_typedef": "vertical_stride_field_t",
        "summary": "ACC.STRIDE vertical stride mode.",
        "doc": "ACC-slot immediate: **vertical stride** bit pattern for `ACC.STRIDE` (see "
        "`acc_stride_enums`).",
    },
    "ActivationFn": {
        "kind": OperandKind.ENUM,
        "bits": code_bits(len(ACTIVATION_FN_NAMES)),
        "domain": Domain(members=tuple(ACTIVATION_FN_NAMES)),
        "readable": False,
        "token_class": "ActivationFnField",
        "field_suffix": "activation_fn_field",
        "sv_typedef": "activation_fn_field_t",
        "summary": "ACTIVATE.QUANTIZE activation function keyword.",
        "doc": "AAQ-slot keyword on **`ACTIVATE.QUANTIZE`**: one of **identity**, **relu**, **relu6**, "
        "**sigmoid**, **tanh**, **gelu**, **softplus**, **elu**, **exp2**, **reciprocal**, **rsqrt**, **silu**, **window** "
        "(see ``ACTIVATION_FN_NAMES`` in ``ipu_isa.activations``). Emulator-only calibration (including α and the "
        "**window** bounds a, b) "
        "is described in **Building Applications** "
        "(`docs/content/building-applications.md`).",
    },
    "LrIncDecImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": None,
        "domain": Domain(),
        "readable": False,
        "token_class": "LrIncDecImmediate",
        "field_suffix": "lr_inc_dec_immediate",
        "sv_typedef": None,
        "summary": "INC/DEC immediate; width decided by the LR slot's union packing.",
        "doc": "Unsigned immediate for **`INC`** / **`DEC`** in the LR slot. The bit width **W** is not "
        "hardcoded — it is derived from the LR slot union layout so the total slot width stays "
        "constant. Valid range: **`0`** to **`2^W − 1`**.",
    },
    "AddbiImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": 8,
        "domain": Domain(value_range=(-128, 255)),
        "readable": False,
        "token_class": "AddbiImmediate",
        "field_suffix": "addbi_immediate",
        "sv_typedef": None,
        "summary": "ADDBI byte immediate, read as signed int8.",
        "doc": "8-bit byte immediate for **`ADDBI`**: written as an unsigned value **`0`**–**`255`** "
        "or an equivalent signed **`-128`**–**`127`** literal (both spellings encode the same "
        "bit pattern). The encoded byte is reinterpreted as a signed two's-complement `int8` "
        "when broadcast-added to **`ADDB`**/**`ADDBI`**'s 8 destination byte elements.",
    },
    "LrModPow2KImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": LR_MOD_POW2_K_FIELD_BITS,
        "domain": Domain(value_range=(LR_MOD_POW2_K_MIN, LR_MOD_POW2_K_MAX)),
        "readable": False,
        "token_class": "LrModPow2KImmediate",
        "field_suffix": "lr_mod_pow2_k_immediate",
        "sv_typedef": None,
        "summary": "INC.MODP2 exponent k, encoded as k - 1.",
        "doc": "Four-bit immediate for **`INC.MODP2`**: encodes exponent **k** with semantic "
        "**k ∈ [1, 9]** as **(k − 1)** in the word.",
    },
    "MultMaskOffsetImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": MULT_MASK_OFFSET_FIELD_BITS,
        "domain": Domain(value_range=(0, MULT_MASK_SLOT_COUNT - 1)),
        "readable": False,
        "token_class": "MultMaskOffsetImmediate",
        "field_suffix": "mult_mask_offset_immediate",
        "sv_typedef": None,
        "summary": "R_MASK slot index for a masked multiply.",
        "doc": "Unsigned **3-bit** immediate on multiply instructions: **`mask_offset`** selects slot "
        "**`0`**–**`7`**, each a **128-bit** region of **`R_MASK`** (eight mask slots total). "
        "**`mask_shift`** remains an **`LrIdx`**.",
    },
    "LrOrReshapeMaskImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": None,
        "domain": Domain(),
        "readable": False,
        "token_class": "LrOrReshapeMaskImmediate",
        "field_suffix": "lr_or_reshape_mask_immediate",
        "sv_typedef": None,
        "summary": "ACC.RESHAPE first element: an immediate or an LR; width decided by union packing.",
        "doc": "Operand on **`ACC.RESHAPE`** (bit width auto-derived from union layout): selects the "
        "first participating element slot (elements **`reshape_mask`**…**`7`** of the "
        "source/dest LRD pair participate). Accepts either an unsigned immediate **`0`**–**`7`** "
        "or an LR register (its value is used directly as the mask; a resolved value "
        "greater than **`8`** raises an error rather than being clamped). "
        "**`0`** uses all 8 slots; **`7`** uses only slot 7; **`8`** uses none.",
    },
    "BreakImmediate": {
        "kind": OperandKind.IMMEDIATE,
        "bits": 16,
        "domain": Domain(value_range=(-(1 << 15), (1 << 16) - 1)),
        "readable": False,
        "token_class": "BreakImmediateType",
        "field_suffix": "break_immediate_type",
        "sv_typedef": None,
        "summary": "BREAK.IFEQ comparison value.",
        "doc": "16-bit value for **`BREAK`** / breakpoint slot conditions.",
    },
    "Label": {
        "kind": OperandKind.LABEL,
        "bits": code_bits(INST_MEM_WORDS),
        "domain": Domain(value_range=(0, INST_MEM_WORDS - 1)),
        "readable": False,
        "token_class": "LabelToken",
        "field_suffix": "label_token",
        "sv_typedef": None,
        "summary": "Branch target: a label or a relative offset.",
        "doc": "Branch target: a symbolic **`label`** or a relative offset accepted by the cond slot "
        "(e.g. `loop`, `+3`).",
    },
}


def _load(spec: Mapping[str, dict[str, Any]]) -> Mapping[str, OperandType]:
    """Check the authored entries and freeze them into records."""
    records = {}
    for name, entry in spec.items():
        path = f"OPERAND_TYPES.{name}"
        schema.check_keys(path, entry, ENTRY_KEYS)
        kind = entry["kind"]
        has_typedef = kind in (OperandKind.REGISTER, OperandKind.ENUM)
        if (entry["sv_typedef"] is not None) != has_typedef:
            raise schema.SpecError(
                f"{path}.sv_typedef: set exactly for REGISTER and ENUM operand types"
            )
        if entry["readable"] and kind is not OperandKind.REGISTER:
            raise schema.SpecError(f"{path}.readable: only a REGISTER operand can be read")
        if entry["bits"] is None and kind is not OperandKind.IMMEDIATE:
            raise schema.SpecError(f"{path}.bits: only an IMMEDIATE width may be derived")
        records[name] = OperandType(name=name, **entry)
    schema.check_unique("OPERAND_TYPES", "field_suffix", (t.field_suffix for t in records.values()))
    schema.check_unique("OPERAND_TYPES", "token_class", (t.token_class for t in records.values()))
    return MappingProxyType(records)


#: Every operand type, in authoring order.
OPERAND_TYPES: Mapping[str, OperandType] = _load(_SPEC)


def operand_type(name: str) -> OperandType:
    """The record of operand type ``name``."""
    try:
        return OPERAND_TYPES[name]
    except KeyError:
        raise IsaLookupError(
            f"unknown operand type {name!r}; known: {sorted(OPERAND_TYPES)}"
        ) from None
