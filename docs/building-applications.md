# Building IPU Applications

This guide shows how to build a complete IPU application using the fully connected neural network layer as an example. The complete code is in [src/apps/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/apps/fully_connected).

## Application Structure

An IPU application typically consists of:

1. **Assembly code** (`.asm`) - IPU program with compute operations (see [Assembly Syntax Guide](assembly-syntax.md))
2. **Python host code** - Sets up the IPU state, loads data, and runs the emulator
3. **BUILD.bazel** - Build configuration for assembling `.asm` files
4. **Test data** - Input/output binary files

## Step 1: Write the Assembly Program

Create your IPU assembly program in `fully_connected.asm`. The assembly program contains the compute logic that runs on the IPU. See the [Assembly Syntax Guide](assembly-syntax.md) for details on writing IPU assembly code with Jinja2 preprocessing.

## Step 2: Assemble the Program

Create a `BUILD.bazel` to assemble the `.asm` file:

```starlark
load("//:asm_rules.bzl", "assemble_asm")

assemble_asm(
    name = "assemble_fully_connected",
    src = "fully_connected.asm",
)
```

Then build:

```bash
bazel build //src/apps/fully_connected:assemble_fully_connected
```

## Step 3: Write the Python Host Code

Use the `ipu_emu` package to load data, run the emulator, and check results:

```python
from ipu_emu.ipu_state import IpuState
from ipu_emu.emulator import run

# Create IPU state and load the assembled program
state = IpuState()
state.load_program("assemble_fully_connected.bin")

# Load data into external memory
state.xmem.load_binary("inputs.bin", base_addr=0x0000, width=128, height=10)
state.xmem.load_binary("weights.bin", base_addr=0x20000, width=128, height=64)

# Set up control registers with base addresses
state.regfile.set_scalar("cr", 0, 0x0000)   # input base
state.regfile.set_scalar("cr", 1, 0x20000)  # weight base
state.regfile.set_scalar("cr", 2, 0x40000)  # output base

# Run the emulator
run(state, max_cycles=1_000_000)

# Save output
state.xmem.dump_binary("output.bin", base_addr=0x40000, width=64, height=10)
```

See the fully_connected app module at `src/tools/ipu-emu-py/src/ipu_emu/apps/fully_connected.py` for the complete implementation.

## Step 4: Run

```bash
cd src/tools/ipu-emu-py

# Run the fully_connected app
uv run python -m ipu_emu.apps.fully_connected \
    path/to/instructions.bin \
    path/to/inputs.bin \
    path/to/weights.bin \
    output.bin INT8
```

## Key Concepts

- **Memory Layout**: Define base addresses for inputs, weights, and outputs in external memory
- **Register Setup**: Initialize LR/CR registers before execution
- **Emulator Run**: The emulator executes instructions until the program counter exceeds instruction memory
- **Bazel Integration**: The `assemble_asm` rule compiles `.asm` files to `.bin` executables

See the [Assembly Syntax Guide](assembly-syntax.md) for more details on writing IPU programs.
