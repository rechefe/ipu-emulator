# Assembly Language Syntax

The IPU assembler is a Python-based tool that converts assembly language into binary instructions for the IPU emulator.

## Features

- **Jinja2 Preprocessing** - Use templates and macros to generate assembly
- **Label Resolution** - Forward and backward label references
- **Multiple Instruction Types** - XMEM, MAC, LR, and Conditional instructions
- **C Code Generation** - Generate C header files from assembly
- **Syntax Validation** - Lark-based parser with detailed error messages

## Basic Syntax

Assembly files use a simple, line-based syntax:

```asm
# Comments start with #
label:                      # Labels end with colon
    ldr r0 lr0 cr0;       # Load from memory
    mac.ee rq4 r2 r3;     # Multiply-accumulate
    beq lr0 lr1 label;    # Branch if equal
```

## Instruction Separators

The IPU supports two types of instruction separators for different execution models:

### Compound Instruction 

An IPU instruction is a compound instruction, meaning its composed of few instructions running in parallel.

**Syntax:**
```asm
xmem_inst; mac_inst; lr_inst; cond_inst;;
```

**Execution:** All instructions in the compound execute simultaneously.

**Instruction Types:**

1. **XMEM type** - Memory operations
2. **MAC type** - Multiply-accumulate
3. **LR type** - Loop register operations (2 of this type can be executed in parallel)
4. **COND type** - Control flow 

**Example:**
```asm
# All execute in parallel in one cycle
ldr r0 lr0 cr0; mac.ee rq4 r2 r3; incr lr0 1; b next
```

**Important Rules:**

- Each pipeline can only execute one instruction per compound
- Missing pipelines use NOP (no operation)
- Order doesn't matter within a compound instruction
- Pipelines are independent and don't interfere

**Complete Example:**
```asm
# Sequential execution (4 cycles)
ldr r0 lr0 cr0;;
mac.ee rq8 r2 r3;;
incr lr0 1;;
b next;;

# Parallel execution (1 cycle)
ldr rq0 lr0 cr0; mac.ee rq4 r2 r3; incr lr0 1; b next;;
```

## Instruction Format

Each instruction follows the pattern:

```
opcode operand1 operand2 operand3
```

**Register Types:**

- `r0-r11` - Data registers (128-byte vectors)
- `mem_bypass` - Equivalent to R register - used if loading R register and applying MAC using it is needed in the same instruction.
- `rq0, rq4, rq8` - Aliases of `r0-r11` registers (512-byte vectors) - e.g. `rq = {r0, r1, r2, r3}`
- `lr0-lr15` - General registers - IPU can read and write LRs
- `cr0-cr15` - Configuration registers - IPU can only read CRs

**Immediate Values:**

- Decimal: `42`
- Hexadecimal: `0x2A`
- Binary: `0b00101010`
- Negative: `-10`

## Labels

Labels mark positions in code for jumps and branches:

```asm
start:              # Label definition
    set lr0 10
loop:
    incr lr0 1
    bne lr0 lr1 loop    # Branch to label
    b start             # Jump to label
```

**Label References:**

- **Backward**: Reference to a label already defined
- **Forward**: Reference to a label defined later
- **Relative**: `+N` or `-N` (jump N instructions forward/backward)

**Examples:**
```asm
b +5        # Jump 5 instructions forward
b -2        # Jump 2 instructions backward
b loop      # Jump to label 'loop'
```

## Jinja2 Preprocessing

The assembler supports [Jinja2 templates](https://jinja.palletsprojects.com/) for powerful code generation:

### Variables and Expressions

```jinja2
{% set loop_count = 10 %}
{% set base_addr = 0x1000 %}

set lr0 {{ base_addr }};
set lr1 {{ loop_count }};
```

### Loops

```jinja2
{% for i in range(8) %}
ldr r{{ i }} lr0 cr{{ i }};;
{% endfor %}
```

### Macros

```jinja2
{% macro load_vector(reg_num, addr) %}
set lr0 {{ addr }};
ldr r{{ reg_num }} lr0 cr0;;
{% endmacro %}

{{ load_vector(0, 0x1000) }}
{{ load_vector(1, 0x2000) }}
```

### Conditionals

```jinja2
{% set use_optimization = true %}

{% if use_optimization %}
    # Optimized code path - parallel execution
    ldr r0 lr0 cr0; mac.agg rq0 r2 r3; incr lr0 128; b next;;
{% else %}
    # Standard code path - sequential
    ldr rx0 lr0 cr0;;
    mac.ee rx1 rx2 rx3;;
    incr lr0 128;;
    b next;;
{% endif %}
```

### Include Files

```jinja2
{% include 'common_macros.j2' %}
{% include 'constants.j2' %}
```

## Advanced Features

### Loop Unrolling

```jinja2
{% macro unroll_loop(count) %}
{% for i in range(count) %}
    ldr r{{ i % 8 }} lr0 cr0; mac.ee rx0 rx{{ i % 8 }} rx7; incr lr0 128;;
{% endfor %}
{% endmacro %}

{{ unroll_loop(16) }}
```

### Parametric Code Generation

```jinja2
{% set VECTOR_SIZE = 128 %}
{% set NUM_LAYERS = 4 %}
{% set ACTIVATION = 'relu' %}

{% for layer in range(NUM_LAYERS) %}
layer_{{ layer }}:
    # Load weights for layer {{ layer }}
    set lr0 {{ 0x1000 + layer * VECTOR_SIZE }};;
    ldr rx1 lr0 cr0;;
    
    # Matrix multiply
    mac.ev rx0 rx1 rx2 lr0;;
    
    {% if ACTIVATION == 'relu' %}
    # ReLU activation
    blt rx0 lr0 skip_relu_{{ layer }};;
    set rx0 0;;
skip_relu_{{ layer }}:
    {% endif %}
{% endfor %}
```

### Compound Instruction Generation

```jinja2
{% macro parallel_load_mac(rx_out, rx1, rx2, addr, offset) %}
ldr {{ rx1 }} lr0 cr0; mac.ee {{ rx_out }} {{ rx1 }} {{ rx2 }}; incr lr0 {{ offset }};;
{% endmacro %}

# Generates efficient parallel instructions
{{ parallel_load_mac('rx0', 'rx1', 'rx2', 0x1000, 128) }}
```

## Using the Assembler

### Command Line

```bash
# Assemble to binary
ipu-as input.asm -o output.bin

# Generate C header
ipu-as input.asm --c-gen --out-dir ./generated

# With preprocessing
ipu-as template.asm.j2 -o output.bin

# Define variables
ipu-as template.asm.j2 --define OPTIMIZE=true --define SIZE=256
```

### In Bazel

```starlark
load("//asm_rules:defs.bzl", "ipu_asm")

ipu_asm(
    name = "my_program",
    src = "program.asm",
)
```

### Preprocessing Context

The assembler automatically provides these variables in templates:

- `__file__` - Current file path
- `__line__` - Current line number
- Custom variables via command line: `--define VAR=value`

## Complete Example

```asm
# Matrix-vector multiplication with loop unrolling
{% set ROWS = 8 %}
{% set COLS = 128 %}

# Initialize base addresses
set lr0 0x1000      ;; # Matrix base address
set lr1 0x2000      ;; # Vector base address
set lr2 0x3000      ;; # Result base address
set lr3 {{ ROWS }}  ;; # Loop counter

main_loop:
    # Parallel load and MAC operations
    {% for i in range(COLS // 16) %}
    ldr r{{ i % 4 }} lr0 cr{{ i }}; mac.ev rq4 r{{ i % 4 }} r6 lr1; incr lr0, 16;;
    {% endfor %}
    
    # Store result and update
    str rx7 lr2 cr0; incr lr2, 128; incr lr3, 1;;
    
    # Loop condition
    bnz lr3 lr0 main_loop;;

end:
    bkpt
```

## Additional Resources

- [Jinja2 Template Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [Instruction Reference](instructions.md) - Complete instruction set documentation
