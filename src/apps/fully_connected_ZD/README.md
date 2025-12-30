# Fully Connected Layer - Parallel Neuron Computation (ZD)

This implementation computes a fully connected neural network layer using the `mac.ev` instruction to process **all neurons in parallel**.

## Overview

Unlike the sequential approach, this implementation calculates partial results for all 128 neurons simultaneously in each iteration:

- **Input**: 10 samples × 128 elements (int8)
- **Weights**: 128×128 matrix in **transposed layout** (int8)
- **Output**: 10 samples × 128 neurons (int32)

## Key Innovation: Parallel Computation

### Traditional Approach (Sequential)
```
For each neuron (64 iterations):
    Compute full dot product for one neuron
    Result: one neuron complete
```

### This Approach (Parallel)
```
For each input element (128 iterations):
    Multiply input[element] with weights[element][:] (all 128 neurons)
    Result: partial products for ALL 128 neurons
```

After 128 iterations, all neurons are complete!

## Weight Matrix Layout - CRITICAL

The weight matrix MUST be in **transposed layout**:

```
weights[input_element_idx][neuron_idx]
```

- Each **row** contains all neuron weights for one input element position
- This allows `mac.ev` to compute partial products for all neurons simultaneously

### Example:
```
Row 0: [w_neuron0_elem0, w_neuron1_elem0, ..., w_neuron127_elem0]
Row 1: [w_neuron0_elem1, w_neuron1_elem1, ..., w_neuron127_elem1]
...
Row 127: [w_neuron0_elem127, w_neuron1_elem127, ..., w_neuron127_elem127]
```

## How mac.ev Enables Parallel Computation

The `mac.ev` (element-vector) instruction:
```assembly
mac.ev rq8 mem_bypass r0 element_idx
```

- Takes **ONE element** from `r0[element_idx]` (the input)
- Multiplies it with **ALL 128 elements** from `mem_bypass` (the weight row)
- Accumulates to **128 positions** in `rq8` (partial products for all neurons)

This is executed 128 times with different weight rows, building up the complete results.

## Files

### Source Code
- `fully_connected_ZD.asm` - Assembly implementation with parallel computation
- `fully_connected_ZD.c` - C setup/teardown code
- `test_fully_connected_ZD.sh` - Test runner script

### Data Generation
- `generate_test_data.py` - Generates test data with correct transposed layout
  - Creates: `inputs_10x128_int8.bin` (1,280 bytes)
  - Creates: `weights_128x128_int8.bin` (16,384 bytes, **transposed**)
  - Creates: `output_10x128_int32.bin` (5,120 bytes, expected results)

### Binary Data
- `inputs_10x128_int8.bin` - Input activations
- `weights_128x128_int8.bin` - Weight matrix (transposed layout)
- `output_10x128_int32.bin` - Expected output for verification

## Memory Layout

```
0x00000 - 0x004FF:  Input activations (10 × 128 bytes = 1,280 bytes)
0x20000 - 0x23FFF:  Weights (128 × 128 bytes = 16,384 bytes)
0x40000 - 0x413FF:  Output activations (10 × 128 × 4 bytes = 5,120 bytes)
```

## Assembly Algorithm

```assembly
start:
    set input_idx 0
    set input_limit 1280  # 10 samples × 128 bytes

input_loop:
    ldr r0 input_idx input_base              # Load input vector (128 bytes)

    set weight_byte_offset 0                 # Start at first weight row
    set element_idx 0                        # Start at first element
    set last_element_idx 127                 # 128 elements total

element_loop:
    ldr mem_bypass weight_byte_offset weight_base  # Load weight row for element[i]
    mac.ev rq8 mem_bypass r0 element_idx           # All neurons += weights × input[i]
    incr weight_byte_offset 128                    # Next weight row
    incr element_idx 1                             # Next element
    blt element_idx last_element_idx element_loop  # Loop 128 times

    # Store results (4 R registers = 128 neurons × 4 bytes)
    str r8 output_offset output_base
    incr output_offset 128
    str r9 output_offset output_base
    incr output_offset 128
    str r10 output_offset output_base
    incr output_offset 128
    str r11 output_offset output_base
    incr output_offset 128

    incr input_idx 128
    blt input_idx input_limit input_loop
```

## Performance

- **Iterations per input sample**: 128 (once per input element)
- **Neurons computed**: 128 (all in parallel)
- **Total cycles**: ~128-140 per input sample
- **Efficiency**: Same cycles per neuron as sequential, but conceptually parallel

## Usage

### Generate Test Data
```bash
cd src/apps/fully_connected_ZD
python3 generate_test_data.py
```

### Build and Run
```bash
# Build the project
make

# Run the test
./test_fully_connected_ZD.sh <binary> <inst_file> \
    inputs_10x128_int8.bin \
    weights_128x128_int8.bin \
    output_actual.bin
```

### Verify Results
Compare `output_actual.bin` with `output_10x128_int32.bin`

## Key Differences from Standard Implementation

| Aspect | Standard (fully_connected) | This (fully_connected_ZD) |
|--------|---------------------------|---------------------------|
| MAC Operation | `mac.agg` (full dot product) | `mac.ev` (element × vector) |
| Loop Structure | 64 iterations (per neuron) | 128 iterations (per element) |
| Neurons | 64 | 128 |
| Weight Layout | Row per neuron | Row per input element |
| Computation | Sequential neurons | Parallel neurons |
| Weight Size | 8,192 bytes | 16,384 bytes |
| Output Size | 2,560 bytes | 5,120 bytes |

## Why This Approach?

This implementation is designed for hardware that:
- Has `mac.ev` but NOT `mac.agg`
- Can benefit from parallel accumulation across multiple neurons
- Has sufficient memory for transposed weight storage

The transposed weight layout enables computing all neurons simultaneously by broadcasting each input element to all neuron accumulators.
