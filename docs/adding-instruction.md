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

Once you've added the instruction to the assembler, you need to implement the actual instruction execution logic in the emulator.

The emulator source code can be found under `src/lib/ipu`. Depending on the instruction type, you'll need to modify the appropriate file:

- `ipu_acc_inst.c/h` - for accumulator-related instructions
- `ipu_cond_inst.c/h` - for conditional instructions
- `ipu_lr_inst.c/h` - for load/register instructions
- `ipu_mult_inst.c/h` - for multiplication-related instructions
- `ipu_xmem_inst.c/h` - for external memory instructions

### Step 1: Create Instruction Handler Functions

Add handler functions in the appropriate instruction file. Each specific instruction gets its own handler that performs the operation:

```c
void ipu__execute_your_instruction_name(ipu__obj_t *ipu,
                                        inst_parser__inst_t inst,
                                        const ipu__regfile_t *regfile_snapshot)
{
    // Extract operand fields from inst struct
    inst_parser__mult_stage_reg_field_t ra_idx = 
        (inst_parser__mult_stage_reg_field_t)inst.mult_inst_token_1_mult_stage_reg_field;
    
    // Access register data
    ipu__r_reg_t *ra_reg_ptr = ipu__get_mult_stage_r_reg(ipu, ra_idx);
    
    // Perform operation - use ipu_math functions for arithmetic
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++) 
    {
        ipu_math__mult(&ra_reg_ptr->bytes[i],
                       &rb_reg.bytes[i],
                       &ipu->misc.mult_res.words[i],
                       ipu__get_cr_dtype(&ipu->regfile));
    }
}
```

### Step 2: Register the Instruction in the Dispatcher

Add a dispatch function (e.g., `ipu__execute_mult_instruction`) that routes to your handler based on opcode:

```c
void ipu__execute_mult_instruction(ipu__obj_t *ipu,
                                   inst_parser__inst_t inst,
                                   const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.mult_inst_token_0_mult_inst_opcode)
    {
    case INST_PARSER__MULT_INST_OPCODE_YOUR_INSTRUCTION:
        ipu__execute_your_instruction_name(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__MULT_INST_OPCODE_ANOTHER_INSTRUCTION:
        ipu__execute_another_instruction(ipu, inst, regfile_snapshot);
        break;
    default:
        assert(false && "Invalid MULT instruction type");
        break;
    }
}
```

### Step 3: Key Data Types and Helpers

Use these key types and helper functions:

**Instruction representation:**
- `inst_parser__inst_t` - the parsed instruction from the assembler

**Register types:**
- `ipu__r_reg_t` - standard R register (128 bytes)
- `ipu__r_acc_reg_t` - accumulator register (512 bytes)
- `ipu__regfile_t` - full register file with all stage registers

**State:**
- `ipu__obj_t` - the IPU state object containing regfile, memory, PC, etc.
- `ipu__regfile_t *regfile_snapshot` - snapshot of registers at instruction fetch time

**Helper functions:**
- `ipu__get_mult_stage_r_reg()` - get pointer to mult stage R register
- `ipu__get_cr_dtype()` - get data type from CR register
- `ipu_math__add()`, `ipu_math__mult()` - arithmetic operations on data elements

### Step 4: Add Tests

Add unit tests in `test/` directory to verify your instruction works correctly. Run tests using:

```bash
bazel test //test:all -c dbg
```

!!! important "Consistency"
    Ensure your emulator implementation matches the pseudo-code in the instruction's `operation` field defined in the assembler. Operands must be extracted from the `inst_parser__inst_t` struct fields matching your instruction format.