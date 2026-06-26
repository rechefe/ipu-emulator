from dataclasses import dataclass

import lark
import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate
from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    COMPOUND_LAYOUT_SLOT_ORDER,
    InstructionDoc,
    SLOT_UNIONS,
    is_hardware_slot,
)


def _operand_type_md_link(typ: str) -> str:
    slug = typ.lower().replace("_", "-")
    return f"[`{typ}`](operand-types.md#{slug})"


def _md_table_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


# ===========================================================================
# Operand Type Resolution
# ===========================================================================
# Maps string type names from instruction_spec → actual token classes.
# This is the ONLY place where type names are resolved to Python classes.
# ===========================================================================

OPERAND_TYPE_MAP: dict[str, type[ipu_token.IpuToken]] = {
    "MultStageReg": reg.MultStageRegField,
    "LrIdx": reg.LrRegField,
    "CrIdx": reg.CrRegField,
    "DstructureCrIdx": reg.DstructureCrRegField,
    "LcrIdx": reg.LcrRegField,
    "LrIncDecImmediate": immediate.LrIncDecImmediate,
    "ElementsInRow": immediate.ElementsInRowField,
    "HorizontalStride": immediate.HorizontalStrideField,
    "VerticalStride": immediate.VerticalStrideField,
    "LrModPow2KImmediate": immediate.LrModPow2KImmediate,
    "MultMaskOffsetImmediate": immediate.MultMaskOffsetImmediate,
    "ActivationFn": immediate.ActivationFnField,
    "BreakImmediate": immediate.BreakImmediateType,
    "Label": ipu_token.LabelToken,
}


def _build_struct_table(slot_type: str) -> dict[str, "InstructionFormat"]:
    """Build struct_by_opcode_table from INSTRUCTION_SPEC for a given slot.

    Resolves string operand type names to actual token classes using
    OPERAND_TYPE_MAP, and wraps each instruction in an InstructionFormat.
    """
    slot_spec = INSTRUCTION_SPEC[slot_type]
    result = {}
    for inst_name, inst_def in slot_spec.items():
        operand_classes = [
            OPERAND_TYPE_MAP[op["type"]] for op in inst_def["operands"]
        ]
        result[inst_name] = InstructionFormat(
            operands=operand_classes, doc=inst_def["doc"]
        )
    return result


@dataclass
class InstructionFormat:
    operands: list[type[ipu_token.IpuToken]]
    doc: InstructionDoc | None = None


def validate_inst_structure(cls: type) -> type:
    """Class decorator to validate instruction structure."""
    cls._validate_instr_structure()
    return cls


class Inst:
    def __init__(self, inst: dict[str, any]):
        self.opcode = self.opcode_type()(inst["opcode"])
        opcode_idx = self.opcode.encode()
        struct_table = self.struct_by_opcode_table()
        struct_names = list(struct_table.keys())
        struct_entry = struct_table[struct_names[opcode_idx]]
        operand_types = self._operand_types_from_struct(struct_entry)

        if len(inst["operands"]) != len(operand_types):
            raise ValueError(
                f"Instruction {inst['opcode'].token.value} expects {len(operand_types)} operands, "
                f"got {len(inst['operands'])}, in Line {self.opcode.token.line}, Column {self.opcode.token.column}."
            )

        self.operands = [
            op_type(op) for op_type, op in zip(operand_types, inst["operands"])
        ]
        self.specific_operand_types = operand_types

    def _get_full_token_list(self) -> list[ipu_token.IpuToken]:
        full_token_list = [None for _ in range(1 + len(self.operand_types()))]
        full_token_list[0] = self.opcode
        for i, operand in enumerate(self.operands):
            mapped_index = self._inst_mapping_table[self.opcode.encode()][i]
            full_token_list[mapped_index + 1] = operand
        for i in range(len(full_token_list)):
            if full_token_list[i] is None:
                full_token_list[i] = self.operand_types()[i - 1].default()
        return full_token_list

    @classmethod
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        raise NotImplementedError(
            "struct_by_opcode_table property must be implemented by subclasses"
        )

    @classmethod
    def _slot_type_name(cls) -> str:
        """Return the slot name string (e.g. 'lr', 'mult') for this Inst subclass."""
        raise NotImplementedError(
            "_slot_type_name must be implemented by subclasses"
        )

    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        """Return canonical token types for each union field (derived from SLOT_UNIONS)."""
        slot_union = SLOT_UNIONS[cls._slot_type_name()]
        return [OPERAND_TYPE_MAP[f.canonical_type] for f in slot_union.fields]

    @classmethod
    def _validate_instr_structure(cls) -> None:
        cls._inst_mapping_table = {}
        cls._slot_union = SLOT_UNIONS[cls._slot_type_name()]
        names = cls.opcode_type().enum_array()
        for opcode_idx, (opcode_name, _) in enumerate(
            cls.struct_by_opcode_table().items()
        ):
            assert opcode_name == names[opcode_idx], (
                f"Configuration of {cls.__name__} is invalid: opcode table order "
                f"does not match {cls.opcode_type().__name__}.enum_array()"
            )
            bindings = cls._slot_union.opcode_bindings.get(opcode_name, [])
            cls._inst_mapping_table[opcode_idx] = [fi for fi, _ in bindings]

    @classmethod
    def nop_inst(cls) -> str:
        raise NotImplementedError("nop_inst method must be implemented by subclasses")

    @staticmethod
    def _reversed_inst_mapping_table(mapping: list[int]) -> dict[int, int]:
        return {j: i for i, j in enumerate(mapping)}

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        raise NotImplementedError(
            "opcode_type method must be implemented by subclasses"
        )

    def encode(self) -> int:
        encoded_inst = 0
        shift_amount = 0
        canonical_types = self.all_tokens()
        full_token_list = self._get_full_token_list()
        for i in range(len(full_token_list) - 1, -1, -1):
            token = full_token_list[i]
            canonical_width = canonical_types[i].bits()
            encoded_inst |= token.encode() << shift_amount
            shift_amount += canonical_width
        return encoded_inst

    @classmethod
    def bits(cls) -> int:
        return sum(token_type.bits() for token_type in cls.all_tokens())

    @classmethod
    def find_inst_type_by_opcode(cls, opcode: str) -> type["Inst"]:
        opl = opcode.lower()
        for subclass in cls.__subclasses__():
            if any(opl == name.lower() for name in subclass.struct_by_opcode_table()):
                return subclass
        raise ValueError(f"Opcode '{opcode}' not found in any Inst subclass.")

    @classmethod
    def all_tokens(cls) -> list[type[ipu_token.IpuToken]]:
        return [cls.opcode_type()] + cls.operand_types()

    @classmethod
    def decode(cls, value: int) -> str:
        """Decode an instruction word to assembly text.

        Uses the opcode to select the correct semantic token types for each
        union field, so cross-type-sharing fields disassemble correctly.
        """
        all_tok = cls.all_tokens()
        n = len(all_tok)
        total_bits = cls.bits()
        opcode_bits = cls.opcode_type().bits()

        # Extract opcode (top bits) to pick semantic types.
        opcode_value = (value >> (total_bits - opcode_bits)) & ((1 << opcode_bits) - 1)
        try:
            opcode_name = cls.opcode_type().enum_array()[opcode_value]
        except (IndexError, AttributeError):
            opcode_name = None

        # Map all_tokens index → semantic token class for this opcode.
        semantic: dict[int, type[ipu_token.IpuToken]] = {}
        if opcode_name is not None and hasattr(cls, "_slot_union"):
            bindings = cls._slot_union.opcode_bindings.get(opcode_name, [])
            struct_entry = cls._struct_entry(opcode_name)
            for (field_idx, _), op_class in zip(bindings, struct_entry.operands):
                semantic[field_idx + 1] = op_class  # +1: index 0 is the opcode token

        decoded_tokens = []
        shift_amount = 0
        for tok_idx in range(n - 1, -1, -1):  # LSB → MSB
            tok_type = all_tok[tok_idx]
            token_bits = tok_type.bits()
            token_value = (value >> shift_amount) & ((1 << token_bits) - 1)
            actual_type = semantic.get(tok_idx, tok_type)
            decoded_tokens.append(actual_type.decode(token_value))
            shift_amount += token_bits
        return " ".join(reversed(decoded_tokens))

    @classmethod
    def desc(cls) -> list[str]:
        res = []
        res.append(f"{cls.__name__} - {cls.bits()} bits:")
        for token_type in cls.all_tokens():
            res.append(f"\t{token_type.__name__} - {token_type.bits()} bits")
        return res

    @classmethod
    def description(cls) -> str:
        """Return human-readable description of this instruction type."""
        raise NotImplementedError(
            "description method must be implemented by subclasses"
        )

    @classmethod
    def get_all_instruction_classes(cls) -> list[type["Inst"]]:
        """Return instruction subclasses in compound-layout slot order."""
        slot_to_class = {
            subclass._slot_type_name(): subclass for subclass in cls.__subclasses__()
        }
        seen: list[type["Inst"]] = []
        for slot in COMPOUND_LAYOUT_SLOT_ORDER:
            inst_class = slot_to_class.get(slot)
            if inst_class is None:
                continue
            seen.append(inst_class)
        # Any subclasses not in COMPOUND_LAYOUT_SLOT_ORDER (should not happen).
        for subclass in cls.__subclasses__():
            if subclass not in seen:
                seen.append(subclass)
        return seen

    @classmethod
    def _operand_types_from_struct(
        cls, struct_entry: InstructionFormat | list[type[ipu_token.IpuToken]]
    ) -> list[type[ipu_token.IpuToken]]:
        return (
            struct_entry.operands
            if isinstance(struct_entry, InstructionFormat)
            else struct_entry
        )

    @classmethod
    def _struct_entry(cls, opcode: str) -> InstructionFormat:
        struct_entry = cls.struct_by_opcode_table()[opcode]
        return (
            struct_entry
            if isinstance(struct_entry, InstructionFormat)
            else InstructionFormat(struct_entry)
        )

    @classmethod
    def _render_instruction_docs(cls, heading: str, intro: str, slot_type: str) -> str:
        lines = [f"## {heading}", ""]
        if not is_hardware_slot(slot_type):
            lines.append(
                "> **Simulation-only slot** — not implemented in real IPU hardware. "
                "Excluded from hardware codegen."
            )
            lines.append("")
        if intro.strip():
            lines.append(intro.strip())
            lines.append("")

        for opcode, struct_entry in cls.struct_by_opcode_table().items():
            if opcode == "NOP":
                continue  # documented once globally in the Slots section
            instruction_format = cls._struct_entry(opcode)
            if instruction_format.doc is None:
                continue
            spec_ops = INSTRUCTION_SPEC[slot_type][opcode]["operands"]
            lines.extend(
                cls._render_opcode_doc(opcode, instruction_format.doc, spec_ops)
            )
            lines.append("")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _render_opcode_doc(
        opcode: str,
        doc: InstructionDoc,
        spec_operands: list[dict],
    ) -> list[str]:
        lines = [f"### `{opcode}` — {doc.title}", ""]
        lines.append(f"**Syntax:** `{doc.syntax}`")
        lines.append("")

        if spec_operands:
            lines.append(
                "**Operands:** *(the **Type** column links to the "
                "[operand type reference](operand-types.md))*"
            )
            lines.append("")
            lines.append("| Name | Type | Details |")
            lines.append("|------|------|---------|")
            for i, sop in enumerate(spec_operands):
                name = sop["name"]
                typ = sop["type"]
                link = _operand_type_md_link(typ)
                detail = doc.operands[i] if i < len(doc.operands) else "—"
                lines.append(
                    f"| `{name}` | {link} | {_md_table_cell(detail)} |"
                )
            lines.append("")
        elif doc.operands:
            lines.append("**Operands:**")
            lines.extend([f"- {operand}" for operand in doc.operands])
            lines.append("")

        lines.append("**General description:**")
        lines.append(doc.summary)
        lines.append("")

        if doc.notes:
            lines.append(f"> {doc.notes}")
            lines.append("")

        if doc.operation:
            lines.append("**Pseudo code:**")
            lines.append(f"`{doc.operation}`")
            lines.append("")

        if doc.example:
            lines.append("**Example of usage:**")
            lines.append("```asm")
            lines.append(doc.example)
            lines.append("```")

        return lines


@validate_inst_structure
class LoadInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "load"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.LoadInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("load")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return LoadInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="LOAD Instructions",
            intro=(
                "First-stage memory loads that feed the multiply unit "
                "(mult-stage registers, cyclic register, and mask register)."
            ),
            slot_type="load",
        )


@validate_inst_structure
class StoreInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "store"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.StoreInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("store")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return StoreInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="STORE Instructions",
            intro=(
                "Last-stage memory stores that drain **POST_AAQ_REG** to external "
                "memory after activation and quantization."
            ),
            slot_type="store",
        )


@validate_inst_structure
class AccStoreInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "acc_store"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.AccStoreInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("acc_store")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return AccStoreInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="ACC_STORE Instructions",
            intro=(
                "Simulation-only slot for storing **R_ACC** to external memory "
                "(`STR_ACC_REG`). Not implemented in real IPU hardware."
            ),
            slot_type="acc_store",
        )


@validate_inst_structure
class MultInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "mult"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.MultInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("mult")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return MultInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="MULT Instructions",
            intro="""Multiplication instructions for element-wise and element-vector operations.
The multiplication result (`mult_result`) is forwarded to the ACC stage in the CPU and not stored in any register in the way.
""",
            slot_type="mult",
        )


@validate_inst_structure
class AccInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "acc"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.AccInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("acc")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return AccInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="ACC Instructions",
            intro="Accumulation instructions for combining values with optional masking and shifting.",
            slot_type="acc",
        )


@validate_inst_structure
class AaqInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "aaq"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.AaqInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("aaq")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return AaqInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="AAQ Instructions",
            intro="Activation and quantization: aggregate r_acc into AAQ registers; ACTIVATE writes activated lanes from r_acc into POST_AAQ_REG; AAQ quantizes POST_AAQ_REG.",
            slot_type="aaq",
        )


@validate_inst_structure
class LrInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "lr"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.LrInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("lr")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return LrInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="LR Instructions",
            intro="Loop register manipulation instructions for controlling loop counters and addresses.",
            slot_type="lr",
        )


@validate_inst_structure
class CondInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "cond"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.CondInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("cond")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return CondInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0), instr_id=addr
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="Conditional Branch Instructions",
            intro="Control flow instructions for branching based on conditions or unconditionally.",
            slot_type="cond",
        )


@validate_inst_structure
class BreakInst(Inst):
    @classmethod
    def _slot_type_name(cls) -> str:
        return "break"

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.BreakInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("break")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return BreakInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "NOP", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="Break Instructions",
            intro="Debug break instructions for halting execution and entering debug mode.",
            slot_type="break",
        )
