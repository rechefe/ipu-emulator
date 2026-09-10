"""Generate the ETISS ``IPU`` architecture plugin sources from the instruction spec.

Sibling of :mod:`ipu_as.gen_codegen` (which emits SystemVerilog): both read
``ipu_common.instruction_spec`` / ``ipu_common.registers`` so the ETISS backend
never contains a hand-typed opcode, bit offset, or register size.

Emitted artefacts
-----------------

``IPU_gen.h``
    ``struct IPU`` (an ``ETISS_CPU`` followed by the register file, the
    read-before-write shadow copies, config and run statistics), the address
    map, word geometry, and one opcode enum per slot.

``IPUFuncs_gen.h``
    One prototype per instruction, named ``ipu_<execute_fn without the
    ``execute_`` prefix>``, with the operands of ``INSTRUCTION_SPEC`` in spec
    order.  ``IPUFuncs.c`` implements them; a missing handler is a link error
    (the C analogue of the ``AttributeError`` ``ipu.py`` raises).

``IPUDecode_gen.cpp``
    The single catch-all ``InstructionDefinition`` covering the whole VLIW
    word: it slices each slot's opcode and operand fields with
    ``BitArrayRange`` and appends the C calls for that word to the ETISS
    ``CodePart``, plus the ASM printer used by tracing and GDB.

``ipu_etiss_layout.py``
    The same geometry and address map for the Python-side runner, so the two
    can never disagree.

Bit ranges come from ``CompoundInst.get_fields()`` — the assembler's own
LSB-first field walk, which is also what ``ipu_emu.execute.decode_instruction_word``
uses — so the generated extraction is the Python decoder by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_as import inst as _inst
from ipu_as import utils
from ipu_as.compound_inst import CompoundInst
from ipu_common.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    INSTRUCTION_SPEC,
    SLOT_COUNT,
    SLOT_UNIONS,
)
from ipu_common.acc_stride_enums import (
    ELEMENTS_IN_ROW_VALUES,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_common.incr_mod_pow2_k import (
    LR_MOD_POW2_K_ENCODED_MAX,
    LR_MOD_POW2_K_MIN,
)
from ipu_common.reshape_mask import RESHAPE_ELEMENT_COUNT, RESHAPE_MASK_LR_OFFSET
from ipu_common.registers import REGISTER_DEFINITIONS

# ---------------------------------------------------------------------------
# Address map / geometry (single definition, exported to C and Python)
# ---------------------------------------------------------------------------

IMEM_BASE = 0x00000000
XMEM_BASE = 0x10000000
XMEM_SIZE = 1 << 23          # 8 MB, matches ipu_emu.xmem.XMEM_SIZE_BYTES
XMEM_WIDTH = 128             # one addressable row, matches XMEM_WIDTH_BYTES
INST_MEM_SIZE = 1024         # matches ipu_emu.ipu_state.INST_MEM_SIZE
LANES = 128

# Registers that the shadow (read-before-write) copy has to carry.  Derived
# below from the spec: a register is shadowed when some operand reads it from
# the snapshot, or when a handler reads it from the snapshot internally.
# ``lr``/``cr`` are always shadowed (INC/DEC/ADDB/INCR_MOD_POW2/RESHAPE read
# them from the snapshot inside the handler, not through an operand).
_SHADOW_REGS = ("lr", "cr", "r", "r_cyclic", "r_acc", "mult_res")

# Which RunStats counter a non-NOP instruction in each slot bumps.
# Mirrors ``ipu_emu.ipu._SLOT_STAT``.
_SLOT_STAT = {
    "mult": "mult_active_cycles",
    "acc": "acc_active_cycles",
    "load": "xmem_reads",
    "store": "xmem_writes",
    "acc_store": "xmem_writes",
}

# Slot -> the Inst subclass name prefix used in CompoundInst.get_fields().
_SLOT_CLASS = {
    "cond": "CondInst",
    "lr": "LrInst",
    "load": "LoadInst",
    "store": "StoreInst",
    "acc_store": "AccStoreInst",
    "mult": "MultInst",
    "acc": "AccInst",
    "aaq": "AaqInst",
    "break": "BreakInst",
}

# C scalar type for register storage, by RegDtype value.
_REG_C_TYPE = {
    "uint8": "etiss_uint8",
    "uint32": "etiss_uint32",
    "uint128": "etiss_uint8",   # r_mask is a byte blob in the emulator too
}


def _snake(name: str) -> str:
    return utils.camel_case_to_snake_case(name)


def _slot_prefixes(slot: str) -> list[str]:
    """Field-name prefixes for a slot: one per sub-slot instance."""
    base = _snake(_SLOT_CLASS[slot])
    if SLOT_COUNT[slot] > 1:
        return [f"{base}_{i}" for i in range(SLOT_COUNT[slot])]
    return [base]


def _field_ranges() -> dict[str, tuple[int, int]]:
    """name -> (lsb, msb) for every field of the compound word."""
    ranges: dict[str, tuple[int, int]] = {}
    shift = 0
    for name, bits in CompoundInst.get_fields():
        ranges[name] = (shift, shift + bits - 1)
        shift += bits
    return ranges


def _opcode_field_name(slot: str, prefix: str) -> str:
    return f"{prefix}_token_0_{slot}_inst_opcode"


def _operand_field_name(prefix: str, field_idx: int, canonical_type: str) -> str:
    token_cls = _inst.OPERAND_TYPE_MAP[canonical_type]
    return f"{prefix}_token_{field_idx + 1}_{_snake(token_cls.__name__)}"


def _c_ident(instruction_name: str) -> str:
    """`ACC.ADD.FIRST` -> `ACC_ADD_FIRST` (opcode enum member)."""
    return instruction_name.upper().replace(".", "_").replace("-", "_")


def _handler_name(execute_fn: str) -> str:
    """`execute_ldr_mult_reg` -> `ipu_ldr_mult_reg`."""
    assert execute_fn.startswith("execute_"), execute_fn
    return "ipu_" + execute_fn[len("execute_"):]


def _is_nop(instruction_name: str) -> bool:
    return instruction_name == "NOP"


# ---------------------------------------------------------------------------
# Operand model
# ---------------------------------------------------------------------------

class Operand:
    """One handler argument: how to extract it and how to pass it to C."""

    def __init__(self, name: str, op_type: str, read: str | None,
                 field_idx: int | None, canonical_type: str | None):
        self.name = name
        self.op_type = op_type
        self.read = read
        self.field_idx = field_idx
        self.canonical_type = canonical_type

    @property
    def c_type(self) -> str:
        if self.op_type == "MultStageReg" and self.read:
            return "const etiss_uint8 *"
        return "etiss_uint32"

    def c_arg_template(self) -> str:
        """The generated-C argument expression, with ``{V}`` where the raw value goes.

        Raw operand values are translate-time constants, so every index
        computation below is folded into a literal in the emitted C; only the
        register *reads* happen at run time.
        """
        live = self.read == "live"
        snap = self.read == "snapshot"
        if not (live or snap):
            return "{V}ULL"
        pfx = "" if live else "s_"
        if self.op_type == "LrIdx":
            return f"((IPU*)cpu)->{pfx}LR[{{V}}]"
        if self.op_type in ("CrIdx", "DstructureCrIdx"):
            return f"((IPU*)cpu)->{pfx}CR[{{V}}]"
        if self.op_type == "LcrIdx":
            # LR0-15 then CR0-15 in one index space (ipu.py _resolve_reads).
            # Which side it lands on is known here, so no runtime branch.
            lr_count = REGISTER_DEFINITIONS["lr"]["count"]
            return _LcrTemplate(pfx, lr_count)
        if self.op_type == "MultStageReg":
            # Ra data comes from the cycle-start shadow copy (issue #157).
            return f"(&((IPU*)cpu)->{pfx}R[({{V}}) * IPU_R_SIZE])"
        # LrdIdx and every immediate/label type: the raw index is passed
        # through and the handler resolves it (ipu.py _resolve_operand's
        # fallback branch does exactly this).
        return "{V}ULL"


class _LcrTemplate(str):
    """An LcrIdx argument, resolved to LR or CR at translate time.

    Subclasses ``str`` so it flows through the same emission path; the
    ``{V}`` substitution is done by :func:`_cxx_arg` which special-cases it.
    """

    def __new__(cls, prefix: str, lr_count: int):
        obj = super().__new__(cls, "<lcr>")
        obj.prefix = prefix
        obj.lr_count = lr_count
        return obj


def _cxx_arg(template, var: str) -> str:
    """Turn an argument template into a C++ expression that builds the C text.

    ``var`` is the name of the C++ variable holding the operand's raw value.
    """
    if isinstance(template, _LcrTemplate):
        pfx, n = template.prefix, template.lr_count
        return (
            f'(({var}) < {n}'
            f' ? std::string("((IPU*)cpu)->{pfx}LR[") + std::to_string({var}) + "]"'
            f' : std::string("((IPU*)cpu)->{pfx}CR[") + std::to_string(({var}) - {n}) + "]")'
        )
    parts = template.split("{V}")
    out = f'std::string("{parts[0]}")'
    for tail in parts[1:]:
        out += f' + std::to_string({var}) + "{tail}"'
    return out


def _operands_for(slot: str, instruction_name: str) -> list[Operand]:
    spec = INSTRUCTION_SPEC[slot][instruction_name]
    bindings = dict(
        (operand_name, field_idx)
        for field_idx, operand_name in SLOT_UNIONS[slot].opcode_bindings.get(
            instruction_name, []
        )
    )
    union_fields = SLOT_UNIONS[slot].fields
    operands: list[Operand] = []
    for op in spec["operands"]:
        field_idx = bindings.get(op["name"])
        if field_idx is None:
            raise RuntimeError(
                f"{slot}/{instruction_name}: operand {op['name']!r} has no union "
                "field binding — instruction_spec and the union layout disagree"
            )
        operands.append(
            Operand(
                name=op["name"],
                op_type=op["type"],
                read=op.get("read"),
                field_idx=field_idx,
                canonical_type=union_fields[field_idx].canonical_type,
            )
        )
    return operands


# ---------------------------------------------------------------------------
# IPU_gen.h
# ---------------------------------------------------------------------------


def _state_field_list() -> list[tuple[str, str, int]]:
    """Field order of the runner's state blob: (name, kind, count).

    Kinds: ``U32``/``U64``/``F64`` are scalars, ``U32ARR`` is *count* 32-bit
    words, ``BLOB`` is *count* raw bytes.  Emitted into both ``IPU_gen.h``
    (as an X-macro) and ``ipu_etiss_layout.py`` so the runner and the Python
    driver can never disagree about the wire format.
    """
    fields: list[tuple[str, str, int]] = [
        ("dtype", "U32", 0),
        ("elu_alpha", "F64", 0),
        ("break_mode", "U32", 0),
        ("max_cycles", "U64", 0),
    ]
    for name, meta in REGISTER_DEFINITIONS.items():
        if name.endswith("_wide_debug"):
            continue
        if meta["vector"]:
            fields.append((name.upper(), "BLOB", meta["size_bytes"] * meta["count"]))
        else:
            fields.append((name.upper(), "U32ARR", meta["count"]))
    fields += [
        ("cycles", "U64", 0),
        ("mult_active_cycles", "U64", 0),
        ("acc_active_cycles", "U64", 0),
        ("xmem_reads", "U64", 0),
        ("xmem_writes", "U64", 0),
        ("error_code", "U32", 0),
        ("error_detail", "U64", 0),
    ]
    return fields


def _register_fields() -> list[tuple[str, str, int]]:
    """(C name, C element type, total byte/element count) for struct IPU.

    Wide-vector debug registers are skipped: that mode is emulator-only and
    the ETISS backend does not implement it (see the ETISS integration spec).
    """
    out: list[tuple[str, str, int]] = []
    for name, meta in REGISTER_DEFINITIONS.items():
        if name.endswith("_wide_debug"):
            continue
        ctype = _REG_C_TYPE[meta["dtype"].value]
        if meta["vector"]:
            out.append((name.upper(), ctype, meta["size_bytes"] * meta["count"]))
        else:
            out.append((name.upper(), ctype, meta["count"]))
    return out


def gen_ipu_header() -> str:
    regs = _register_fields()
    lines: list[str] = []
    a = lines.append
    a("/* GENERATED by ipu_as.gen_etiss — do not edit. */")
    a("#ifndef IPU_GEN_H")
    a("#define IPU_GEN_H")
    a("")
    a('#include "etiss/jit/CPU.h"')
    a('#include "etiss/jit/System.h"')
    a('#include "etiss/jit/ReturnCode.h"')
    a('#include "etiss/jit/types.h"')
    a("")
    a("#ifdef __cplusplus")
    a('extern "C" {')
    a("#endif")
    a("")
    a("/* ---- word geometry -------------------------------------------------- */")
    a(f"#define IPU_WORD_BITS   {CompoundInst.bits()}")
    a(f"#define IPU_WORD_BYTES  {_instruction_aligned_bytes()}")
    a(f"#define IPU_WORD_FETCH_BITS {_instruction_aligned_bytes() * 8}")
    a(f"#define IPU_INST_MEM_SIZE {INST_MEM_SIZE}")
    a("")
    a("/* ---- address map ----------------------------------------------------- */")
    a(f"#define IPU_IMEM_BASE   0x{IMEM_BASE:08x}ULL")
    a("#define IPU_IMEM_END    (IPU_IMEM_BASE + (etiss_uint64)IPU_INST_MEM_SIZE * IPU_WORD_BYTES)")
    a(f"#define IPU_XMEM_BASE   0x{XMEM_BASE:08x}ULL")
    a(f"#define IPU_XMEM_SIZE   0x{XMEM_SIZE:08x}ULL")
    a(f"#define IPU_XMEM_WIDTH  {XMEM_WIDTH}")
    a(f"#define IPU_LANES       {LANES}")
    a("")
    a("/* ---- register geometry ----------------------------------------------- */")
    for name, meta in REGISTER_DEFINITIONS.items():
        if name.endswith("_wide_debug"):
            continue
        a(f"#define IPU_{name.upper()}_SIZE  {meta['size_bytes']}")
        a(f"#define IPU_{name.upper()}_COUNT {meta['count']}")
    a("")
    a("/* ---- operand enums and immediates (from ipu_common) ------------------ */")
    a(f"#define IPU_LR_MOD_POW2_K_MIN {LR_MOD_POW2_K_MIN}")
    a(f"#define IPU_LR_MOD_POW2_K_ENCODED_MAX {LR_MOD_POW2_K_ENCODED_MAX}")
    a(f"#define IPU_RESHAPE_ELEMENT_COUNT {RESHAPE_ELEMENT_COUNT}")
    a(f"#define IPU_RESHAPE_MASK_LR_OFFSET {RESHAPE_MASK_LR_OFFSET}")
    a(f"#define IPU_ELEMENTS_IN_ROW_MAX {len(ELEMENTS_IN_ROW_VALUES) - 1}")
    a("#define ipu_elements_per_row(enc) ("
      + " : ".join(f"(enc) == {i} ? {v}u" for i, v in enumerate(ELEMENTS_IN_ROW_VALUES))
      + " : 0u)")
    _h = [get_horizontal_stride_bits(i) for i in range(3)]
    _v = [get_vertical_stride_bits(i) for i in range(3)]
    if _h != _v:
        raise RuntimeError("horizontal and vertical stride decode tables diverged")
    a(f"#define IPU_STRIDE_MAX {len(_h) - 1}")
    a("#define ipu_stride_enabled(enc) ("
      + " : ".join(f"(enc) == {i} ? {int(en)}" for i, (en, _inv) in enumerate(_h))
      + " : 0)")
    a("#define ipu_stride_inverted(enc) ("
      + " : ".join(f"(enc) == {i} ? {int(inv)}" for i, (_en, inv) in enumerate(_h))
      + " : 0)")
    a("")
    a("/* ---- error codes (mirror ipu_emu.errors.EmulatorError conditions) ----- */")
    for i, name in enumerate(_ERROR_CODES, start=1):
        a(f"#define IPU_ERR_{name} {i}")
    a(f"#define IPU_ERR_COUNT {len(_ERROR_CODES)}")
    a("")
    a("/* ---- break handling --------------------------------------------------- */")
    a("#define IPU_BREAK_IGNORE 0   /* run_until_complete: breaks do not stop */")
    a("#define IPU_BREAK_STOP   1   /* run_with_debug / GDB: stop at the break */")
    a("")
    a("/* ---- opcode enums (position in INSTRUCTION_SPEC == opcode) ------------ */")
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        for opcode, instruction_name in enumerate(INSTRUCTION_SPEC[slot]):
            a(f"#define IPU_{slot.upper()}_OP_{_c_ident(instruction_name)} {opcode}")
        a(f"#define IPU_NUM_{slot.upper()}_OP {len(INSTRUCTION_SPEC[slot])}")
    a("")
    a("/* ---- runner state blob (mirrored by ipu_etiss_layout.STATE_FIELDS) ---- */")
    a('#define IPU_STATE_MAGIC "IPUSTATE"')
    a("#define IPU_STATE_VERSION 1")
    a("/* X(member, kind, count): kind is U32 | U64 | F64 | U32ARR | BLOB. */")
    a("#define IPU_STATE_FIELDS(X) \\")
    for _name, _kind, _count in _state_field_list():
        a(f"    X({_name}, {_kind}, {_count}) \\")
    a("    /* end of list */")
    a("")
    a("/* ---- CPU state -------------------------------------------------------- */")
    a("struct IPU")
    a("{")
    a("    ETISS_CPU cpu; /* must stay first: ETISS casts IPU* <-> ETISS_CPU* */")
    a("")
    a("    /* live register file */")
    for cname, ctype, count in regs:
        a(f"    {ctype} {cname}[{count}];")
    a("")
    a("    /* cycle-start shadow copy (VLIW read-before-write semantics) */")
    for reg in _SHADOW_REGS:
        cname, ctype, count = next(r for r in regs if r[0] == reg.upper())
        a(f"    {ctype} s_{cname}[{count}];")
    a("")
    a("    /* configuration (emulator-only knobs, not part of the ISA) */")
    a("    etiss_uint32 dtype;       /* DType: 0 = INT8, 1..7 = FP8 exponent bits */")
    a("    double       elu_alpha;")
    a("    etiss_uint32 break_mode;")
    a("    etiss_uint64 max_cycles;  /* 0 = unlimited */")
    a("")
    a("    /* run statistics (mirror ipu_emu.stats.RunStats) */")
    a("    etiss_uint64 cycles;")
    a("    etiss_uint64 mult_active_cycles;")
    a("    etiss_uint64 acc_active_cycles;")
    a("    etiss_uint64 xmem_reads;")
    a("    etiss_uint64 xmem_writes;")
    a("")
    a("    /* first error raised during the run (0 = none) */")
    a("    etiss_uint32 error_code;")
    a("    etiss_uint64 error_detail;")
    a("};")
    a("typedef struct IPU IPU;")
    a("")
    a("#ifdef __cplusplus")
    a("}")
    a("#endif")
    a("#endif /* IPU_GEN_H */")
    return "\n".join(lines) + "\n"


# Error conditions the C handlers can raise, mirroring the EmulatorError call
# sites in ipu.py.  The runner maps these back to Python exceptions.
_ERROR_CODES = (
    "XMEM_ROW_NEGATIVE",
    "XMEM_ROW_RANGE",
    "XMEM_ACCESS",
    "MULT_STAGE_OPERAND",
    "LDR_MULT_REG_DEST",
    "CYCLIC_INDEX",
    "PAD_MODE_NEEDS_FLOAT",
    "INCR_MOD_POW2_K",
    "RESHAPE_MASK_RANGE",
    "RESHAPE_INDEX_RANGE",
    "ACTIVATE_REQUIRES_INT8",
    "DSTRUCTURE_FIELD",
    "STRIDE_OPERAND",
    "REGISTER_INDEX_RANGE",
    "ACC_STRIDE_RANGE",
    "ACTIVATION_OVERFLOW",
    "LR_CONFLICT",
    "MAX_CYCLES",
)


def _instruction_aligned_bytes() -> int:
    """Bytes per instruction in the assembler's ``--format bin`` stream."""
    bits = CompoundInst.bits()
    word = 32
    if bits % word:
        bits += word - (bits % word)
    return (bits // word) * 4


# ---------------------------------------------------------------------------
# IPUFuncs_gen.h
# ---------------------------------------------------------------------------

def gen_funcs_header() -> str:
    lines: list[str] = []
    a = lines.append
    a("/* GENERATED by ipu_as.gen_etiss — do not edit. */")
    a("/*")
    a(" * One prototype per instruction in INSTRUCTION_SPEC.  IPUFuncs.c must")
    a(" * define every one of them: a missing or misnamed handler is a link")
    a(" * error, which is the C analogue of the AttributeError ipu.py raises")
    a(" * when execute_fn names a method that does not exist.")
    a(" *")
    a(" * Argument names and order match the instruction's `operands` list.")
    a(" */")
    a("#ifndef IPU_FUNCS_GEN_H")
    a("#define IPU_FUNCS_GEN_H")
    a("")
    a('#include "IPU/IPU_gen.h"')
    a("")
    a("#ifdef __cplusplus")
    a('extern "C" {')
    a("#endif")
    a("")
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        a(f"/* ---- {slot} slot ---- */")
        for instruction_name, spec in INSTRUCTION_SPEC[slot].items():
            if _is_nop(instruction_name):
                continue
            ops = _operands_for(slot, instruction_name)
            args = ["IPU *cpu", "ETISS_System *system"]
            args += [f"{op.c_type}{op.name}" if op.c_type.endswith("*")
                     else f"{op.c_type} {op.name}" for op in ops]
            ret = "int" if slot == "break" else "void"
            a(f"{ret} {_handler_name(spec['execute_fn'])}({', '.join(args)});")
        a("")
    a("/* Shadow-copy refresh, emitted at the start of every VLIW word. */")
    a("void ipu_snapshot(IPU *cpu);")
    a("/* Raise an emulator error (first one wins), mirroring EmulatorError. */")
    a("void ipu_raise(IPU *cpu, etiss_uint32 code, etiss_uint64 detail);")
    a("")
    a("#ifdef __cplusplus")
    a("}")
    a("#endif")
    a("#endif /* IPU_FUNCS_GEN_H */")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# IPUDecode_gen.cpp
# ---------------------------------------------------------------------------

def _emit_extract(prefix: str, ops, ranges, indent: str) -> list[str]:
    """C++ lines that pull each operand's raw field value out of the word."""
    out: list[str] = []
    for op in ops:
        field = _operand_field_name(prefix, op.field_idx, op.canonical_type)
        lsb, msb = ranges[field]
        var = f"{prefix}_{op.name}"
        out.append(f"{indent}static etiss::instr::BitArrayRange R_{var}({msb}, {lsb});")
        out.append(f"{indent}const etiss_uint32 {var} = (etiss_uint32)R_{var}.read(ba);")
    return out


def _emit_call(handler: str, prefix: str, ops, indent: str,
               statement: bool = True, sink: str = "cp.code()") -> list[str]:
    """C++ lines appending the handler call (or, if not *statement*, the bare call)."""
    expr = f'std::string("{handler}((IPU*)cpu, system")'
    for op in ops:
        expr += ' + ", " + ' + _cxx_arg(op.c_arg_template(), f"{prefix}_{op.name}")
    expr += ' + ")"'
    if statement:
        expr += ' + ";\\n"'
        return [f"{indent}{sink} += {expr};"]
    return [f"{indent}const std::string call = {expr};"]


def _register_index_limit(op: "Operand") -> str | None:
    """The C expression bounding a register-index operand, or None.

    Union fields are as wide as the widest operand that shares them, so a
    hand-crafted binary can carry an index far outside the register file --
    ``SET``'s ``CrIdx`` rides an 8-bit field, for instance.  The Python
    emulator trips an assertion there; without an explicit check the generated
    C would index past the end of the register array.
    """
    lr = "IPU_LR_COUNT"
    cr = "IPU_CR_COUNT"
    return {
        "LrIdx": lr,
        "CrIdx": cr,
        "DstructureCrIdx": cr,
        "LcrIdx": f"({lr} + {cr})",
        "LrdIdx": f"({lr} / 2)",
        "MultStageReg": "2",
    }.get(op.op_type)


def _emit_validation(prefix: str, ops, indent: str, flag: str = "operands_ok",
                     sink: str = "cp.code()") -> list[str]:
    """Operand range checks ipu.py performs while resolving, before the handler.

    Emits an ``ipu_raise`` into the generated block and clears *flag* so the
    caller can suppress the instruction, which is what the Python dispatcher
    does by raising before it calls the handler.
    """
    out: list[str] = []
    for op in ops:
        limit = _register_index_limit(op)
        if limit is None:
            continue
        var = f"{prefix}_{op.name}"
        code = ("IPU_ERR_MULT_STAGE_OPERAND" if op.op_type == "MultStageReg"
                else "IPU_ERR_REGISTER_INDEX_RANGE")
        out.append(f"{indent}if ({var} >= {limit}) {{")
        out.append(
            f'{indent}    {sink} += std::string("ipu_raise((IPU*)cpu, {code}, ")'
            f' + std::to_string({var}) + "ULL);\\n";'
        )
        out.append(f"{indent}    {flag} = false;")
        out.append(f"{indent}}}")
    return out


def _emit_instruction(slot: str, prefix: str, instruction_name: str, spec: dict,
                      ranges: dict, indent: str) -> list[str]:
    """Everything one non-NOP instruction contributes to the generated block."""
    ops = _operands_for(slot, instruction_name)
    out = _emit_extract(prefix, ops, ranges, indent)
    checks = _emit_validation(prefix, ops, indent + "    ")
    if checks:
        out.append(f"{indent}bool operands_ok = true;")
        out += checks
        out.append(f"{indent}if (operands_ok) {{")
        body_indent = indent + "    "
    else:
        body_indent = indent

    stat = _SLOT_STAT.get(slot)
    if stat:
        # The Python dispatcher bumps the counter only once it actually reaches
        # this slot, so an error raised by an earlier slot must suppress it.
        out.append(f'{body_indent}cp.code() += "if (((IPU*)cpu)->error_code == 0)'
                   f' {{ ((IPU*)cpu)->{stat} += 1; }}\\n";')
    out += _emit_call(_handler_name(spec["execute_fn"]), prefix, ops, body_indent)
    if checks:
        out.append(f"{indent}}}")
    return out


# ---------------------------------------------------------------------------
# IPU_gen.h
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IPUDecode_gen.cpp
# ---------------------------------------------------------------------------

def gen_decode_source() -> str:
    ranges = _field_ranges()
    lines: list[str] = []
    a = lines.append
    a("/* GENERATED by ipu_as.gen_etiss - do not edit. */")
    a("/*")
    a(" * One ETISS instruction == one IPU VLIW word.  The opcode tree is not")
    a(" * used (code and mask are both all-zero, so this definition matches every")
    a(" * word); the slots are decoded here and turned into calls into IPUFuncs.c.")
    a(" *")
    a(" * Slot execution order is the order of Ipu.execute_vliw_cycle in ipu.py:")
    a(" * break, LR x3, load, mult, acc, aaq, store, acc_store, cond.")
    a(" */")
    a('#include "IPUArch.h"')
    a('#include "IPU/IPU_gen.h"')
    a("")
    a("#include <sstream>")
    a("#include <string>")
    a("")
    a("static std::string ipu_vliw_asm(etiss::instr::BitArray &ba, etiss::instr::Instruction &);")
    a("")
    a("static etiss::instr::InstructionDefinition ipu_vliw(")
    a('    ISA_IPU, "IPU_VLIW", (uint64_t)0, (uint64_t)0,')
    a("    [](etiss::instr::BitArray &ba, etiss::CodeSet &cs,")
    a("       etiss::instr::InstructionContext &ic) -> bool {")
    a("        etiss::CodePart &cp = cs.append(etiss::CodePart::INITIALREQUIRED);")
    a('        cp.code() = std::string("/* IPU VLIW */\\n");')
    a("")

    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        for prefix in _slot_prefixes(slot):
            field = _opcode_field_name(slot, prefix)
            lsb, msb = ranges[field]
            a(f"        static etiss::instr::BitArrayRange R_{prefix}_op({msb}, {lsb});")
            a(f"        const etiss_uint32 {prefix}_op = (etiss_uint32)R_{prefix}_op.read(ba);")
    a("")
    a('        cp.code() += "ipu_snapshot((IPU*)cpu);\\n";')
    a("")

    prefix = _slot_prefixes("break")[0]
    a("        /* break slot runs first and may stop before any side effect */")
    a(f"        switch ({prefix}_op) {{")
    for instruction_name, spec in INSTRUCTION_SPEC["break"].items():
        a(f"        case IPU_BREAK_OP_{_c_ident(instruction_name)}: {{")
        if _is_nop(instruction_name):
            a("            break;")
            a("        }")
            continue
        ops = _operands_for("break", instruction_name)
        for line in _emit_extract(prefix, ops, ranges, " " * 12):
            a(line)
        for line in _emit_call(_handler_name(spec["execute_fn"]), prefix, ops,
                               " " * 12, statement=False):
            a(line)
        a('            cp.code() += std::string("if (((IPU*)cpu)->break_mode && ") + call')
        a('                + ") { cpu->instructionPointer = " + std::to_string(ic.current_address_)')
        a('                + "ULL; return ETISS_RETURNCODE_BREAKPOINT; }\\n";')
        a("            break;")
        a("        }")
    a("        }")
    a("")

    a("        /* LR sub-slots: three independent instructions per word.")
    a("         * ipu.py resolves all three and checks for a write conflict before")
    a("         * executing any of them, so a fault in the third sub-slot has to")
    a("         * suppress the first two. */")
    a("        {")
    a("            int lr_targets[8];")
    a("            int lr_target_count = 0;")
    a("            bool operands_ok = true;")
    a("            std::string lr_code;")
    a("            std::string lr_error;")
    for prefix in _slot_prefixes("lr"):
        a(f"            switch ({prefix}_op) {{")
        for instruction_name, spec in INSTRUCTION_SPEC["lr"].items():
            a(f"            case IPU_LR_OP_{_c_ident(instruction_name)}: {{")
            if _is_nop(instruction_name):
                a("                break;")
                a("            }")
                continue
            ops = _operands_for("lr", instruction_name)
            for line in _emit_extract(prefix, ops, ranges, " " * 16):
                a(line)
            for line in _emit_validation(prefix, ops, " " * 16, sink="lr_error"):
                a(line)
            write_op = _lr_write_operand(spec)
            if write_op is not None:
                var = f"{prefix}_{write_op.name}"
                if write_op.op_type == "LrdIdx":
                    a(f"                lr_targets[lr_target_count++] = 2 * (int){var};")
                    a(f"                lr_targets[lr_target_count++] = 2 * (int){var} + 1;")
                else:
                    a(f"                lr_targets[lr_target_count++] = (int){var};")
            for line in _emit_call(_handler_name(spec["execute_fn"]), prefix, ops,
                                   " " * 16, sink="lr_code"):
                a(line)
            a("                break;")
            a("            }")
        a("            }")
    a("            for (int i = 0; i < lr_target_count; ++i)")
    a("                for (int j = i + 1; j < lr_target_count; ++j)")
    a("                    if (lr_targets[i] == lr_targets[j]) {")
    a('                        lr_error += std::string("ipu_raise((IPU*)cpu, IPU_ERR_LR_CONFLICT, ")')
    a('                            + std::to_string(lr_targets[i]) + "ULL);\\n";')
    a("                        operands_ok = false;")
    a("                    }")
    a("            cp.code() += operands_ok ? lr_code : lr_error;")
    a("        }")
    a("")

    for slot in ("load", "mult", "acc", "aaq", "store", "acc_store"):
        prefix = _slot_prefixes(slot)[0]
        a(f"        /* {slot} slot */")
        a(f"        switch ({prefix}_op) {{")
        for instruction_name, spec in INSTRUCTION_SPEC[slot].items():
            a(f"        case IPU_{slot.upper()}_OP_{_c_ident(instruction_name)}: {{")
            if _is_nop(instruction_name):
                a("            break;")
                a("        }")
                continue
            for line in _emit_instruction(slot, prefix, instruction_name, spec,
                                          ranges, " " * 12):
                a(line)
            a("            break;")
            a("        }")
        a("        }")
        a("")

    a("        /* default next PC is the following word; a taken branch overrides it */")
    a('        cp.code() += std::string("cpu->nextPc = ")')
    a('            + std::to_string(ic.current_address_ + IPU_WORD_BYTES) + "ULL;\\n";')
    prefix = _slot_prefixes("cond")[0]
    a(f"        switch ({prefix}_op) {{")
    for instruction_name, spec in INSTRUCTION_SPEC["cond"].items():
        a(f"        case IPU_COND_OP_{_c_ident(instruction_name)}: {{")
        if _is_nop(instruction_name):
            a("            break;")
            a("        }")
            continue
        for line in _emit_instruction("cond", prefix, instruction_name, spec,
                                      ranges, " " * 12):
            a(line)
        a("            ic.force_block_end_ = true;")
        a("            break;")
        a("        }")
    a("        }")
    a("")
    a("        /* A fault aborts the cycle: ipu.py propagates the exception")
    a("         * before the cond slot runs and before run_until_complete counts")
    a("         * the cycle, so neither the PC nor the cycle counter moves. */")
    a('        cp.code() += "if (((IPU*)cpu)->error_code != 0)"')
    a('                     " { return ETISS_RETURNCODE_GENERALERROR; }\\n";')
    a('        cp.code() += "cpu->instructionPointer = cpu->nextPc;\\n";')
    a('        cp.code() += "((IPU*)cpu)->cycles += 1;\\n";')
    a('        cp.code() += "if (((IPU*)cpu)->max_cycles != 0 && ((IPU*)cpu)->cycles >= ((IPU*)cpu)->max_cycles)"')
    a('                     " { ipu_raise((IPU*)cpu, IPU_ERR_MAX_CYCLES, ((IPU*)cpu)->cycles); }\\n";')
    a('        cp.code() += "if (((IPU*)cpu)->error_code != 0) { return ETISS_RETURNCODE_GENERALERROR; }\\n";')
    a('        cp.code() += "if (cpu->instructionPointer >= IPU_IMEM_END)"')
    a('                     " { return ETISS_RETURNCODE_CPUFINISHED; }\\n";')
    a('        cp.getAffectedRegisters().add("instructionPointer", 64);')
    a("        return true;")
    a("    },")
    a("    0, ipu_vliw_asm);")
    a("")
    a(_gen_asm_printer(ranges))
    return "\n".join(lines) + "\n"


def _lr_write_operand(spec: dict) -> Operand | None:
    """The LR-slot operand naming the written register (ipu.py _build_plan)."""
    for op in spec["operands"]:
        if op["name"] in ("dest", "reg") and "read" not in op:
            return Operand(op["name"], op["type"], None, None, None)
    return None


def _gen_asm_printer(ranges: dict[str, tuple[int, int]]) -> str:
    """A compact per-slot disassembly for tracing and GDB."""
    lines: list[str] = []
    a = lines.append
    a("static std::string ipu_vliw_asm(etiss::instr::BitArray &ba, etiss::instr::Instruction &)")
    a("{")
    a("    std::stringstream ss;")
    a("    bool first = true;")
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        names = list(INSTRUCTION_SPEC[slot])
        for prefix in _slot_prefixes(slot):
            field = _opcode_field_name(slot, prefix)
            lsb, msb = ranges[field]
            a("    {")
            a(f"        static etiss::instr::BitArrayRange R({msb}, {lsb});")
            a("        const unsigned op = (unsigned)R.read(ba);")
            a("        static const char *names[] = {"
              + ", ".join(f'"{n}"' for n in names) + "};")
            a(f"        if (op < {len(names)} && std::string(names[op]) != \"NOP\") {{")
            a('            if (!first) ss << "; ";')
            a("            ss << names[op];")
            a("            first = false;")
            a("        }")
            a("    }")
    a('    if (first) ss << "NOP";')
    a('    ss << ";;";')
    a("    return ss.str();")
    a("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ipu_etiss_layout.py
# ---------------------------------------------------------------------------

def gen_python_layout() -> str:
    return f'''"""GENERATED by ipu_as.gen_etiss — do not edit.

Address map and word geometry shared by the ETISS plugin and the Python
runner, so the two can never disagree about where memory lives.
"""

WORD_BITS = {CompoundInst.bits()}
WORD_BYTES = {_instruction_aligned_bytes()}
INST_MEM_SIZE = {INST_MEM_SIZE}

IMEM_BASE = 0x{IMEM_BASE:08x}
IMEM_BYTES = INST_MEM_SIZE * WORD_BYTES
XMEM_BASE = 0x{XMEM_BASE:08x}
XMEM_SIZE = 0x{XMEM_SIZE:08x}
XMEM_WIDTH = {XMEM_WIDTH}
LANES = {LANES}

#: Error code -> symbolic name, mirroring the IPU_ERR_* constants in IPU_gen.h.
ERROR_NAMES = {{
{chr(10).join(f"    {i}: {name!r}," for i, name in enumerate(_ERROR_CODES, start=1))}
}}

STATE_MAGIC = b"IPUSTATE"
STATE_VERSION = 1

#: Wire format of the runner's state blob, in order, as (name, kind, count).
#: Kinds: U32/U64/F64 scalars, U32ARR = count 32-bit words, BLOB = count bytes.
#: The blob starts with STATE_MAGIC, a uint32 version and a uint64 PC (in
#: words), then these fields, all little-endian.
STATE_FIELDS = [
{chr(10).join(f"    ({n!r}, {k!r}, {c})," for n, k, c in _state_field_list())}
]
'''


# ---------------------------------------------------------------------------
# Public accessors (the Python-side runner reads these directly, so the address
# map and wire format have exactly one definition)
# ---------------------------------------------------------------------------

#: Emulator error conditions, in IPU_ERR_* order (code == index + 1).
ERROR_CODES = _ERROR_CODES

#: code -> symbolic name for the IPU_ERR_* constants.
ERROR_NAMES = {i: name for i, name in enumerate(_ERROR_CODES, start=1)}

STATE_MAGIC = b"IPUSTATE"
STATE_VERSION = 1


def instruction_aligned_bytes() -> int:
    """Bytes per instruction in the assembler's ``--format bin`` stream."""
    return _instruction_aligned_bytes()


def state_field_list() -> list[tuple[str, str, int]]:
    """Wire format of the runner's state blob; see :func:`_state_field_list`."""
    return _state_field_list()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OUTPUTS = {
    "IPU_gen.h": gen_ipu_header,
    "IPUFuncs_gen.h": gen_funcs_header,
    "IPUDecode_gen.cpp": gen_decode_source,
    "ipu_etiss_layout.py": gen_python_layout,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the ETISS IPU architecture sources from instruction_spec."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--only",
        choices=sorted(_OUTPUTS),
        help="generate a single artefact instead of all of them",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = [args.only] if args.only else sorted(_OUTPUTS)
    for name in wanted:
        (args.output_dir / name).write_text(_OUTPUTS[name]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
