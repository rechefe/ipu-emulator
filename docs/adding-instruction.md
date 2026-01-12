# Adding Instructions

This is a guide on how to add a new instruction to the IPU emulator.

## Assembler - Adding Instruction
First we'll have to add the instruction in the assembler.

The assembler source code can be found under `src/tools/ipu-as-py`

There you can find two files:

- `src/ipu_as/inst.py`
- `src/ipu_as/opcodes.py`

You'll first have to add the opcode in the corresponding opcode class in `opcodes.py`, for example:

```py
class MacInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "mac.ee",
            "mac.ev",
            "mac.agg",
            "zero_rq",
            "new_instruction" # <--------- Here is my new instruction opcode :)
            "mac_nop",
        ]
```

Now we can continue to add the instruction description in `inst.py`, You'll have to find the corresponding instruction class, in this case it will be `MacInst`.

We'll find a method under this class called `struct_by_opcode_table`, such method should be implemented under each class which implements an instruction.

The `struct_by_opcode_table` returns a dictionary with the instruction opcode (added before) as the key and an `InstructionFormat` class instance as the value.

For example:

```py
"mac.ee": InstructionFormat(
                operands=[reg.RxRegField, reg.RxRegField, reg.RxRegField],
                doc=InstructionDoc(
                    title="Element-wise Multiply-Accumulate",
                    summary="Multiply and accumulate element by element.",
                    syntax="mac.ee Rd Ra Rb",
                    operands=[
                        "Rd: Destination accumulator (RQ register)",
                        "Ra: First source register (R register)",
                        "Rb: Second source register (R register)",
                    ],
                    operation="for i in [0, 127]: Rd[i] = Rd[i] + (Ra[i] * Rb[i])",
                    example="# Vector dot product step\nmac.ee rq0 r4 r5;;",
                ),
            ),
```

Notice - 
- `operands` - should include the list of operands that the instruction will be composed of (registers, immediates, etc.).

- `doc` - an `InstructionDoc` class instance with the following information:
    - `title` - meaningful name for the instruction.
    - `summary` - more elaborate word explanation of the instruction.
    - `syntax` - syntax of the instruction.
    - `operands` - small description of each operand meaning (state its type RQ/R/LR/Whatever).
    - `operation` - pseudo-code for what does this instruction do.
    - `example` - simple example for usage of this instruction.

!!! note "Instruction documentation" 
    The `InstructionDoc` field is later used to generate documentation for the IPU assembler and emulator operation for everybody to use. Please think and review your explanations so that everybody could easily use it.

## Emulator - Adding Instruction

TODO - @eyal