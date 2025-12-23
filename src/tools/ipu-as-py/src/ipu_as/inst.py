import lark
import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate


def validate_inst_structure(cls: type) -> type:
    """Class decorator to validate instruction structure."""
    cls._validate_instr_structure()
    return cls


class Inst:
    def __init__(self, inst: dict[str, any]):
        self.opcode = self.opcode_type()(inst["opcode"])
        if len(inst["operands"]) != len(
            self.struct_by_opcode_table()[inst["opcode"].token.value]
        ):
            raise ValueError(
                f"Instruction {inst['opcode'].token.value} expects {len(self.struct_by_opcode_table()[inst['opcode'].token.value])} operands, "
                f"got {len(inst['operands'])}, in Line {self.opcode.token.line}, Column {self.opcode.token.column}."
            )

        self.operands = [
            op_type(op)
            for op_type, op in zip(
                self.struct_by_opcode_table()[inst["opcode"].token.value],
                inst["operands"],
            )
        ]
        self.specific_operand_types = self.struct_by_opcode_table()[
            inst["opcode"].token.value
        ]

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
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        raise NotImplementedError(
            "struct_by_opcode_table property must be implemented by subclasses"
        )

    @classmethod
    def _validate_instr_structure(cls) -> None:
        cls._inst_mapping_table = dict()
        for opcode, token_list in cls.struct_by_opcode_table().items():
            assert (
                opcode in cls.opcode_type().enum_array()
            ), f"Configuration of {cls.__name__} is invalid, opcode key must be of type {cls.opcode_type().__name__}"

            cls._inst_mapping_table[opcode] = cls._find_instruction_inst_mapping(
                token_list
            )

    @classmethod
    def _find_instruction_inst_mapping(cls, token_list: list[type[ipu_token.IpuToken]]):
        inst_mapping = [None for _ in token_list]
        full_token_list = [False for _ in cls.operand_types()]
        for i, token in enumerate(token_list):
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


@validate_inst_structure
class XmemInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.RxRegField, reg.LrRegField, reg.CrRegField]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.XmemInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "ldr": [reg.RxRegField, reg.LrRegField, reg.CrRegField],
            "str": [reg.RxRegField, reg.LrRegField, reg.CrRegField],
            "xmem_nop": [],
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
        return """
## XMEM Instructions

Memory access instructions for loading and storing data between registers and memory.

### ldr - Load Register
Loads data from memory into a register.

**Syntax:** `ldr Rx Lr Cr`

**Operands:**

- `Rx`: Destination data register (where loaded value will be stored) - Must be an R register (128-byte)
- `Lr`: Base address register (contains memory address)
- `Cr`: Offset register (added to base address)

**Operation:** `Rx = Memory[Lr + Cr]`

**Example:**
```
set lr0 0x1000    # Set base address
ldr r0 lr0 cr0  # Load from address 0x1000 + cr0 into r0
```

### str - Store Register
Stores data from a register to memory.

**Syntax:** `str Rx Lr Cr`

**Operands:**

- `Rx`: Source data register (value to store) - Must be an R register (128-byte)
- `Lr`: Base address register (contains memory address)
- `Cr`: Offset register (added to base address)

**Operation:** `Memory[Lr + Cr] = Rx`

**Example:**
```
set lr1 0x2000    # Set base address
str r1 lr1 cr1  # Store r1 to address (0x2000 + cr1)
```

### xmem_nop - No Operation
No operation for the XMEM pipeline.

**Syntax:** `xmem_nop`

**Operands:** None

**Example:**
```
xmem_nop          # Pipeline stall or placeholder
```
"""


@validate_inst_structure
class MacInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.MacInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "mac.ee": [reg.RxRegField, reg.RxRegField, reg.RxRegField],
            "mac.ev": [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField],
            "mac.agg": [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField],
            "mac_nop": [],
        }

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return MacInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "mac_nop", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )

    @classmethod
    def description(cls) -> str:
        return """
## MAC Instructions

Multiply-accumulate instructions for vector and scalar operations.

### mac.ee - Element-wise Multiply-Accumulate
Performs element-wise multiplication and accumulation.

**Syntax:** `mac.ee Rd Ra Rb`

**Operands:**

- `Rd`: Destination register (accumulator) - must be an RQ register
- `Ra`: First source register (multiplicand) - must be an R register
- `Rb`: Second source register (multiplier) - must be an R register

**Operation:** for each index i `Rd[i] = Rd[i] + (Ra[i] * Rb[i])`, i runs from 0 to 127

**Example:**
```
# Vector dot product step
mac.ee rq0 r4 r5
```

### mac.ev - Element-Vector Multiply-Accumulate
Performs element-vector multiplication with accumulation using loop register.

**Syntax:** `mac.ev Rd, Ra, Rb, Lr`

**Operands:**

- `Rd`: Destination register (accumulator) - must be an RQ register
- `Ra`: First source register (multiplicand) - must be an R register
- `Rb`: Second source register (multiplier) - must be an R register
- `Lr`: index register (controls iteration)

**Operation:** for each index i `Rd[i] = Rd[i] + (Ra[i] * Rb[Lr])` - i runs from 0 to 127

**Example:**
```
set lr0 0
mac.ev rq0 r7, r9 lr0
```

### mac.agg - Aggregate Multiply-Accumulate
Performs aggregated multiplication and accumulation across elements.

**Syntax:** `mac.agg Rd Ra Rb Lr`

**Operands:**
- `Rd`: Destination register (accumulator) - must be an RQ register
- `Ra`: First source register (multiplicand) - must be an R register
- `Rb`: Second source register (multiplier) - must be an R register
- `Lr`: index register (controls iteration)

**Operation:** `Rd[Lr] = sum(Ra[i] * Rb[i])` - i runs from 0 to 127

**Example:**
```
# Reduction operation
mac.agg rq4 r0 r1 lr0  # Aggregate multiply-accumulate
```

### mac_nop - No Operation
No operation for the MAC pipeline.

**Syntax:** `mac_nop`

**Operands:** None

**Example:**
```
mac_nop  # Pipeline placeholder
```
"""


@validate_inst_structure
class LrInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, immediate.LrImmediateType]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.LrInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "incr": [reg.LrRegField, immediate.LrImmediateType],
            "set": [reg.LrRegField, immediate.LrImmediateType],
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
        return """
## LR Instructions

Loop register manipulation instructions for controlling loop counters and addresses.

### incr - Increment Register
Increments a loop register by an immediate value.

**Syntax:** `incr Lr imm`

**Operands:**

- `Lr`: Loop/address register to increment
- `imm`: Immediate value (constant to add)

**Operation:** `Lr = Lr + imm`

**Example:**
```
set lr0 10       # Initialize lr0 to 10
incr lr0 1       # lr0 = 11
incr lr0 5       # lr0 = 16
```

### set - Set Register
Sets a loop register to an immediate value.

**Syntax:** `set Lr imm`

**Operands:**

- `Lr`: Loop/address register to set
- `imm`: Immediate value (constant to assign)

**Operation:** `Lr = imm`

**Example:**
```
set lr0 0x1000   # lr0 = 0x1000 (4096)
set lr1 100      # lr1 = 100
set lr2 -5       # lr2 = -5
```
"""


@validate_inst_structure
class CondInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.CondInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "beq": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bne": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "blt": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bnz": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bz": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "b": [ipu_token.LabelToken],
            "br": [reg.LrRegField],
            "bkpt": [],
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
        return """
## Conditional Branch Instructions

Control flow instructions for branching based on conditions or unconditionally.

### beq - Branch if Equal
Branches to a label if two registers are equal.

**Syntax:** `beq Lr1 Lr2 label`

**Operands:**

- `Lr1`: First register to compare
- `Lr2`: Second register to compare
- `label`: Branch target label

**Operation:** `if (Lr1 == Lr2) goto label`

**Example:**
```
loop:
    incr lr0 1
    beq lr0 lr1 end  # Branch to 'end' if lr0 == lr1
    b loop
end:
```

### bne - Branch if Not Equal
Branches to a label if two registers are not equal.

**Syntax:** `bne Lr1 Lr2 label`

**Operation:** `if (Lr1 != Lr2) goto label`

**Example:**
```
bne lr0 lr1 different  # Branch if lr0 != lr1
```

### blt - Branch if Less Than
Branches to a label if first register is less than second.

**Syntax:** `blt Lr1 Lr2 label`

**Operation:** `if (Lr1 < Lr2) goto label`

**Example:**
```
blt lr0 lr1 smaller  # Branch if lr0 < lr1
```

### bnz - Branch if Not Zero
Branches to a label if comparison result is not zero.

**Syntax:** `bnz Lr1 label`

**Operation:** `if (Lr1 != 0) goto label`

**Example:**
```
bnz lr0 nonzero  # Branch if lr0 != 0
```

### bz - Branch if Zero
Branches to a label if comparison result is zero.

**Syntax:** `bz Lr1 label`

**Operation:** `if (Lr1 == 0) goto label`
**Example:**
```
bz lr0 zero  # Branch if lr0 == 0
```

### b - Unconditional Branch
Always branches to the specified label.

**Syntax:** `b label`

**Operands:**

- `label`: Branch target label

**Operation:** `goto label`

**Example:**
```
b start        # Jump to 'start' label
b +5           # Jump forward 5 instructions
b -3           # Jump backward 3 instructions
```

### br - Branch to Register
Branches to the address stored in a register.

**Syntax:** `br Lr`

**Operands:**

- `Lr`: Register containing branch target address

**Operation:** `goto address_in(Lr)`

**Example:**
```
set lr0, 0x100
br lr0         # Jump to address 0x100
```

### bkpt - Breakpoint
Halts execution (breakpoint for debugging).

**Syntax:** `bkpt`

**Operands:** None

**Operation:** Halt execution

**Example:**
```
bkpt           # Stop here for debugging
```
"""
