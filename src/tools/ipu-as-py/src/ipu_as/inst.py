from dataclasses import dataclass

import lark
import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate
from ipu_common.instruction_spec import INSTRUCTION_SPEC, InstructionDoc


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
    "LcrIdx": reg.LcrRegField,
    "AaqRegIdx": reg.AaqRegField,
    "ElementsInRow": immediate.ElementsInRowField,
    "HorizontalStride": immediate.HorizontalStrideField,
    "VerticalStride": immediate.VerticalStrideField,
    "AggMode": immediate.AggModeField,
    "PostFn": immediate.PostFnField,
    "Immediate": immediate.LrImmediateType,
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
        struct_entry = self.struct_by_opcode_table()[inst["opcode"].token.value]
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
            mapped_index = self._inst_mapping_table[self.opcode.token.value][i]
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
    def _validate_instr_structure(cls) -> None:
        cls._inst_mapping_table = dict()
        for opcode, struct_entry in cls.struct_by_opcode_table().items():
            assert (
                opcode in cls.opcode_type().enum_array()
            ), f"Configuration of {cls.__name__} is invalid, opcode key must be of type {cls.opcode_type().__name__}"

            cls._inst_mapping_table[opcode] = cls._find_instruction_inst_mapping(
                cls._operand_types_from_struct(struct_entry)
            )

    @classmethod
    def _find_instruction_inst_mapping(
        cls, operand_types: list[type[ipu_token.IpuToken]]
    ):
        inst_mapping = [None for _ in operand_types]
        full_token_list = [False for _ in cls.operand_types()]
        for i, token in enumerate(operand_types):
            for j, token_type in enumerate(cls.operand_types()):
                if token == token_type and not full_token_list[j]:
                    full_token_list[j] = True
                    inst_mapping[i] = j
                    break
        assert None not in inst_mapping, f"Configuration of {cls.__name__} is invalid"
        return inst_mapping

    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        raise NotImplementedError(
            "operand_types property must be implemented by subclasses"
        )

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
        for token in reversed(self._get_full_token_list()):
            encoded_inst |= token.encode() << shift_amount
            shift_amount += token.bits()
        return encoded_inst

    @classmethod
    def bits(cls) -> int:
        return sum(token_type.bits() for token_type in cls.all_tokens())

    @classmethod
    def find_inst_type_by_opcode(cls, opcode: str) -> type["Inst"]:
        for subclass in cls.__subclasses__():
            if opcode in subclass.struct_by_opcode_table().keys():
                return subclass
        raise ValueError(f"Opcode '{opcode}' not found in any Inst subclass.")

    @classmethod
    def all_tokens(cls) -> list[type[ipu_token.IpuToken]]:
        return [cls.opcode_type()] + cls.operand_types()

    @classmethod
    def decode(cls, value: int) -> str:
        decoded_tokens = []
        shift_amount = 0
        for token_type in reversed(cls.all_tokens()):
            token_bits = token_type.bits()
            token_value = (value >> shift_amount) & ((1 << token_bits) - 1)
            decoded_tokens.append(token_type.decode(token_value))
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
        """Return all instruction subclasses."""
        return cls.__subclasses__()

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
    def _render_instruction_docs(cls, heading: str, intro: str) -> str:
        lines = [f"## {heading}", ""]
        if intro.strip():
            lines.append(intro.strip())
            lines.append("")

        for opcode, struct_entry in cls.struct_by_opcode_table().items():
            instruction_format = cls._struct_entry(opcode)
            if instruction_format.doc is None:
                continue
            lines.extend(cls._render_opcode_doc(opcode, instruction_format.doc))
            lines.append("")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _render_opcode_doc(opcode: str, doc: InstructionDoc) -> list[str]:
        lines = [f"### {opcode} - {doc.title}", doc.summary, ""]
        lines.append(f"**Syntax:** `{doc.syntax}`")
        lines.append("")

        if doc.operands:
            lines.append("**Operands:**")
            lines.extend([f"- {operand}" for operand in doc.operands])
            lines.append("")

        if doc.operation:
            lines.append("**Operation:**")
            lines.append("```")
            lines.append(f"{doc.operation}")
            lines.append("```")

        if doc.example:
            lines.append("**Example:**")
            lines.append("```asm")
            lines.append(doc.example)
            lines.append("```")

        return lines


@validate_inst_structure
class XmemInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [
            reg.MultStageRegField,
            reg.LrRegField,
            reg.LrRegField,
            reg.CrRegField,
        ]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.XmemInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, InstructionFormat]:
        return _build_struct_table("xmem")

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return XmemInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "xmem_nop", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="XMEM Instructions",
            intro="Memory access instructions for loading and storing data between registers and memory.",
        )


@validate_inst_structure
class MultInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [
            reg.MultStageRegField,
            reg.LrRegField,
            reg.LrRegField,
            reg.LrRegField,
            reg.LrRegField,
        ]

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
                    token=lark.Token("TOKEN", "mult_nop", line=0, column=0),
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
        )


@validate_inst_structure
class AccInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [
            reg.AaqRegField,
            immediate.ElementsInRowField,
            immediate.HorizontalStrideField,
            immediate.VerticalStrideField,
            reg.LrRegField,
        ]

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
                    token=lark.Token("TOKEN", "acc_nop", line=0, column=0),
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
        )


@validate_inst_structure
class AaqInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [
            immediate.AggModeField,
            immediate.PostFnField,
            reg.CrRegField,
            reg.AaqRegField,
        ]

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
                    token=lark.Token("TOKEN", "aaq_nop", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="AAQ Instructions",
            intro="Activation and quantization: aggregate r_acc into AAQ registers.",
        )


@validate_inst_structure
class LrInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, reg.LcrRegField, reg.LcrRegField, immediate.LrImmediateType]

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
                    token=lark.Token("TOKEN", "incr", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "lr0", line=0, column=0),
                        instr_id=addr,
                    ),
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "0", line=0, column=0),
                        instr_id=addr,
                    ),
                ],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="LR Instructions",
            intro="Loop register manipulation instructions for controlling loop counters and addresses.",
        )


@validate_inst_structure
class CondInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken]

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
                    token=lark.Token("TOKEN", "b", line=0, column=0), instr_id=addr
                ),
                "operands": [
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "+1", line=0, column=0), instr_id=addr
                    )
                ],
            }
        )

    @classmethod
    def description(cls) -> str:
        return cls._render_instruction_docs(
            heading="Conditional Branch Instructions",
            intro="Control flow instructions for branching based on conditions or unconditionally.",
        )


@validate_inst_structure
class BreakInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, immediate.BreakImmediateType]

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
                    token=lark.Token("TOKEN", "break_nop", line=0, column=0),
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
        )
