# Building IPU Applications

This guide shows how to build a complete IPU application using the fully connected neural network layer as an example. The complete code is in [src/apps/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/apps/fully_connected).

## Application Structure

An IPU application typically consists of:

1. **Assembly code** (`.asm`) - IPU program with compute operations (see [Assembly Syntax Guide](assembly-syntax.md))
2. **C host code** (`.c`) - Sets up the IPU, loads data, and verifies results
3. **BUILD.bazel** - Build configuration
4. **Test data** - Input/output binary files

## Step 1: Write the Assembly Program

Create your IPU assembly program in `fully_connected.asm`. The assembly program contains the compute logic that runs on the IPU. See the [Assembly Syntax Guide](assembly-syntax.md) for details on writing IPU assembly code with Jinja2 preprocessing.

## Step 2: Write the Host Code

Create `fully_connected.c` with the main function and IPU lifecycle callbacks:

```c
#include "ipu/ipu.h"
#include "emulator/emulator.h"
#include "logging/logger.h"

#define INPUT_BASE_ADDR 0x0000
#define WEIGHTS_BASE_ADDR 0x20000
#define OUTPUT_BASE_ADDR 0x40000

/**
 * @brief Setup function to initialize IPU state before execution
 */
void ipu_setup(ipu__obj_t *ipu, int argc, char **argv) {
    LOG_INFO("Setting up IPU...");
    
    // Clear accumulator registers
    for (int i = 0; i < IPU__RQ_REGS_NUM; i++) {
        ipu__clear_rq_reg(ipu, i);
    }
    
    // Load data into external memory
    emulator__load_binary_to_xmem(
        ipu->xmem, argv[2], INPUT_BASE_ADDR, 128, 10);
    emulator__load_binary_to_xmem(
        ipu->xmem, argv[3], WEIGHTS_BASE_ADDR, 128, 64);
    
    // Initialize control registers with base addresses
    ipu->regfile.cr_regfile.cr[0] = INPUT_BASE_ADDR;
    ipu->regfile.cr_regfile.cr[1] = WEIGHTS_BASE_ADDR;
    ipu->regfile.cr_regfile.cr[2] = OUTPUT_BASE_ADDR;
    
    LOG_INFO("IPU setup complete.");
}

/**
 * @brief Teardown function to check results and cleanup after execution
 */
void ipu_teardown(ipu__obj_t *ipu, int argc, char **argv) {
    LOG_INFO("IPU Teardown - Final State:");
    
    // Check final program counter
    LOG_INFO("Final Program Counter: %u", ipu->program_counter);
    
    // Check loop register values
    LOG_INFO("Loop Register lr0: %u", ipu->regfile.lr_regfile.lr[0]);
    LOG_INFO("Loop Register lr1: %u", ipu->regfile.lr_regfile.lr[1]);
    
    // Save results from memory
    emulator__dump_xmem_to_binary(
        ipu->xmem, argv[4], OUTPUT_BASE_ADDR, 64, 10);
    
    // Cleanup resources
    free(ipu->xmem);
    free(ipu);
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv) {
    emulator__test_config_t config = {
        .test_name = "Fully Connected Layer",
        .max_cycles = 1000000,
        .progress_interval = 100,
        .setup = ipu_setup,
        .teardown = ipu_teardown
    };
    
    return emulator__run_test(argc, argv, &config);
}
```

### Choosing Data Type

To choose the data type you can use the following function:

```c
ipu__set_cr_dtype(ipu, dtype);
```

Where `dtype` is an enum from an `ipu_math__dtype_t` type.
`ipu_math` module must be included:

```c
#include "ipu_math/ipu_math.h"
```

Valid options are:

- `IPU_MATH__DTYPE_INT8`
- `IPU_MATH__DTYPE_FP8_E4M3`
- `IPU_MATH__DTYPE_FP8_E5M2`

> [!WARNING] 
> Choosing a data type is done via writing to the `CR15` register - notice not to override it with anything else and only access it with the data type function above.

## Step 3: Create the BUILD Configuration

Create `BUILD.bazel`:

```starlark
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//:asm_rules.bzl", "assemble_asm")

# Assemble the IPU program
assemble_asm(
    name = "assemble_fully_connected",
    src = "fully_connected.asm",
)

# Build the host binary
cc_binary(
    name = "fully_connected",
    srcs = ["fully_connected.c"],
    deps = [
        "//:emulator",
        "//:ipu",
        "//:logger",
        "//:xmem",
    ],
)
```

## Step 4: Build and Run

```bash
# Build the application
bazel build //src/apps/fully_connected:fully_connected

# Run with the assembled program
bazel run //src/apps/fully_connected:fully_connected -- \
    $(bazel info bazel-bin)/src/apps/fully_connected/assemble_fully_connected.bin \
    inputs.bin \
    weights.bin \
    output.bin
```

You could always add `sh` script to run automatically with predefined files.

## Key Concepts

- **Memory Layout**: Define base addresses for inputs, weights, and outputs in external memory
- **Setup Function**: Initialize registers and load data before IPU execution
- **Teardown Function**: Check register values and save results after IPU execution
- **Main Function**: Configure the emulator test with lifecycle callbacks
- **Bazel Integration**: The `assemble_asm` rule compiles `.asm` files to `.bin` executables

See the [Assembly Syntax Guide](assembly-syntax.md) for more details on writing IPU programs.
