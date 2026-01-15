from dataclasses import dataclass

import lark
import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate


@dataclass
class InstructionDoc:
    title: str
    summary: str
    syntax: str
    operands: list[str]
    operation: str | None = None
    example: str | None = None


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
            lines.append(f"**Operation:** `{doc.operation}`")
            lines.append("")

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
        return [reg.RxRegField, reg.LrRegField, reg.CrRegField]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.XmemInstOpcode

    @classmethod
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        return {
            "ldr": InstructionFormat(
                operands=[reg.RxRegField, reg.LrRegField, reg.CrRegField],
                doc=InstructionDoc(
                    title="Load Register",
                    summary="Load data from memory into a register.",
                    syntax="ldr Rx Lr Cr",
                    operands=[
                        "Rx: Destination data register (R register, 128-byte)",
                        "Lr: Base address register (holds memory address)",
                        "Cr: Offset register added to the base address",
                    ],
                    operation="Rx = Memory[Lr + Cr]",
                    example="set lr0 0x1000;;\nldr r0 lr0 cr0;;",
                ),
            ),
            "str": InstructionFormat(
                operands=[reg.RxRegField, reg.LrRegField, reg.CrRegField],
                doc=InstructionDoc(
                    title="Store Register",
                    summary="Store data from a register into memory.",
                    syntax="str Rx Lr Cr",
                    operands=[
                        "Rx: Source data register (R register, 128-byte)",
                        "Lr: Base address register (holds memory address)",
                        "Cr: Offset register added to the base address",
                    ],
                    operation="Memory[Lr + Cr] = Rx",
                    example="set lr1 0x2000;;\nstr r1 lr1 cr1;;",
                ),
            ),
            "xmem_nop": InstructionFormat(
                operands=[],
                doc=InstructionDoc(
                    title="No Operation",
                    summary="No operation for the XMEM pipeline.",
                    syntax="xmem_nop",
                    operands=[],
                    example="xmem_nop;;",
                ),
            ),
        }

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
        return [reg.RxRegField, reg.RxRegField, reg.LrRegField, reg.LrRegField]
    
    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.MultInstOpcode
    
    @classmethod
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        return {
            "mult.ee": InstructionFormat(
                operands=[reg.RxRegField, reg.LrRegField],
                doc=InstructionDoc(
                    title="Element-wise Multiply",
                    summary="Multiply elements of two registers element by element.",
                    syntax="mult.ee Ra  Lr1",
                    operands=[
                        "Ra: First source register (R register)",
                        "Lr1: offset for Ra from RC ",
                    ],
                    operation="for i in [0, 127]: result[i] = Ra[i] * RC[i+Lr1]",
                    example="# Element-wise multiplication\nmult.ee r4 lr0;;",
                ),
            ),
            "mult.ev": InstructionFormat(
                operands=[reg.RxRegField, reg.LrRegField, reg.LrRegField],
                doc=InstructionDoc(
                    title="Element-Vector Multiply",
                    summary="Multiply a vector by a loop-indexed element.",
                    syntax="mult.ev Ra Lr1 Lr2",
                    operands=[
                        "Ra: First source register (R register) Ra inside(R4,R5)",
                        "Lr1: offset for Ra from RC ",
                        "Lr2: element index from Ra",
                    ],
                    operation="for i in [0, 127]: result[i] = Ra[i] * RC[Lr2 + i + Lr1]",
                    example="set lr0 0;;\nmult.ev r7 lr0 lr1;;",
                ),
            ),
            "mult_nop": InstructionFormat(
                operands=[],
                doc=InstructionDoc(
                    title="No Operation",
                    summary="No operation for the MULT pipeline.",
                    syntax="mult_nop",
                    operands=[],
                    example="mult_nop;;",
                ),
            ),
        }
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
            intro="Multiplication instructions for element-wise and element-vector operations.",
            
        )

@validate_inst_structure
class AccInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.RxRegField, reg.LrRegField]
    
    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.AccInstOpcode
    
    @classmethod
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        return {
            "acc": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, reg.LrRegField,reg.LrRegField],
                doc=InstructionDoc(
                    title="Accumulate",
                    summary="Accumulate values from a register into an accumulator.",
                    syntax="acc Lr1 Lr2 Lr3 Lr4",
                    operands=[
                        "Lr1: Index to select one of 8 masks from RM (mask register)",
                        "Lr2: Offset for the accumulator (valid range: -3 to 3)",
                        "Lr3: Shift amount applied within each partition without crossing partition boundaries. Example: with 2 partitions and shift=1, data [a0,a1,...,a63,b0,b1,...,b63] becomes [0,a0,a1,...,a62,0,b0,b1,...,b62]",
                        "Lr4: When set to 1, shifts the upper half of RT (RT[128:255]) to the lower half (RT[0:127]) before accumulation",
                    ],
                    operation="for i in [0, 127]: RT[Lr2 + i] += RP_shifted[i] if (RM[Lr1*128 + i] == 1 AND RM_shifted[i] == 1) else 0  (where RP is data from previous pipeline stage, RP_shifted applies Lr3 shift within partitions, RM_shifted applies Lr3 shift to the mask)",                    
                    example="# Accumulate with mask and shift\nacc lr0 lr1 lr2 lr3;;",
                ),
            ),
            "acc_nop": InstructionFormat(
                operands=[],
                doc=InstructionDoc(
                    title="No Operation",
                    summary="No operation for the ACC pipeline.",
                    syntax="acc_nop",
                    operands=[],
                    example="acc_nop;;",
                ),
            ),
        }
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
class LrInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, immediate.LrImmediateType]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.LrInstOpcode

    @classmethod
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        return {
            "incr": InstructionFormat(
                operands=[reg.LrRegField, immediate.LrImmediateType],
                doc=InstructionDoc(
                    title="Increment Register",
                    summary="Increment a loop register by an immediate value.",
                    syntax="incr Lr imm",
                    operands=[
                        "Lr: Loop/address register to increment",
                        "imm: Immediate value to add",
                    ],
                    operation="Lr = Lr + imm",
                    example="set lr0 10;;\nincr lr0 1;;\nincr lr0 5;;",
                ),
            ),
            "set": InstructionFormat(
                operands=[reg.LrRegField, immediate.LrImmediateType],
                doc=InstructionDoc(
                    title="Set Register",
                    summary="Assign an immediate value to a loop register.",
                    syntax="set Lr imm",
                    operands=[
                        "Lr: Loop/address register to set",
                        "imm: Immediate value to assign",
                    ],
                    operation="Lr = imm",
                    example="set lr0 0x1000;;\nset lr1 100;;\nset lr2 -5;;",
                ),
            ),
        }

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
    def struct_by_opcode_table(
        cls,
    ) -> dict[str, InstructionFormat | list[type[ipu_token.IpuToken]]]:
        return {
            "beq": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Branch if Equal",
                    summary="Branch when two registers are equal.",
                    syntax="beq Lr1 Lr2 label",
                    operands=[
                        "Lr1: First register to compare",
                        "Lr2: Second register to compare",
                        "label: Branch target",
                    ],
                    operation="if (Lr1 == Lr2) goto label",
                    example="beq lr0 lr1 end;;",
                ),
            ),
            "bne": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Branch if Not Equal",
                    summary="Branch when two registers differ.",
                    syntax="bne Lr1 Lr2 label",
                    operands=[
                        "Lr1: First register to compare",
                        "Lr2: Second register to compare",
                        "label: Branch target",
                    ],
                    operation="if (Lr1 != Lr2) goto label",
                    example="bne lr0 lr1 different;;",
                ),
            ),
            "blt": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Branch if Less Than",
                    summary="Branch when the first register is less than the second.",
                    syntax="blt Lr1 Lr2 label",
                    operands=[
                        "Lr1: First register (lhs)",
                        "Lr2: Second register (rhs)",
                        "label: Branch target",
                    ],
                    operation="if (Lr1 < Lr2) goto label",
                    example="blt lr0 lr1 smaller;;",
                ),
            ),
            "bnz": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Branch if Not Zero",
                    summary="Branch when the test register is not equal to the baseline (often zero).",
                    syntax="bnz LrTest LrBase label",
                    operands=[
                        "LrTest: Register to test",
                        "LrBase: Baseline register (commonly a zero register)",
                        "label: Branch target",
                    ],
                    operation="if (LrTest != LrBase) goto label",
                    example="bnz lr3 lr0 main_loop;;",
                ),
            ),
            "bz": InstructionFormat(
                operands=[reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Branch if Zero",
                    summary="Branch when the test register equals the baseline (often zero).",
                    syntax="bz LrTest LrBase label",
                    operands=[
                        "LrTest: Register to test",
                        "LrBase: Baseline register (commonly a zero register)",
                        "label: Branch target",
                    ],
                    operation="if (LrTest == LrBase) goto label",
                    example="bz lr0 lr1 zero;;",
                ),
            ),
            "b": InstructionFormat(
                operands=[ipu_token.LabelToken],
                doc=InstructionDoc(
                    title="Unconditional Branch",
                    summary="Always branch to a label (relative or absolute).",
                    syntax="b label",
                    operands=["label: Branch target"],
                    operation="goto label",
                    example="b start;;",
                ),
            ),
            "br": InstructionFormat(
                operands=[reg.LrRegField],
                doc=InstructionDoc(
                    title="Branch to Register",
                    summary="Branch to the address held in a loop register.",
                    syntax="br Lr",
                    operands=["Lr: Register containing target address"],
                    operation="goto address_in(Lr)",
                    example="br lr0;;",
                ),
            ),
            "bkpt": InstructionFormat(
                operands=[],
                doc=InstructionDoc(
                    title="Breakpoint",
                    summary="Halt execution (debug breakpoint).",
                    syntax="bkpt",
                    operands=[],
                    operation="halt",
                    example="bkpt",
                ),
            ),
        }

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
