# Fully Connected Layer: 128×256 (Batch Size = 1)

This example demonstrates a fully connected neural network layer with:
- **Input size**: 128 elements
- **Output neurons**: 256 neurons
- **Batch size**: 1 sample
- **Data type**: int8 inputs/weights, int32 outputs

## Architecture

### Memory Layout

```
INPUT_BASE_ADDR   = 0x0000   : Input activations (128 bytes)
WEIGHTS_BASE_ADDR = 0x20000  : Weight matrix (32768 bytes, transposed)
ZEROS_BASE_ADDR   = 0x30000  : Zero buffer for register clearing (1024 bytes)
OUTPUT_BASE_ADDR  = 0x40000  : Output activations (1024 bytes)
```

### Weight Matrix Layout (CRITICAL)

The weight matrix is stored in **transposed layout** to enable parallel computation:

```
weights[input_element_idx][neuron_idx]
```

- **Shape**: 128 elements × 256 neurons = 32768 bytes
- **Each row**: Contains all 256 neuron weights for one input element position
- **Size per row**: 256 bytes (2 R registers)
- **This layout enables**: `mac.ev` instruction to compute all neurons in parallel

### Register Usage

The implementation uses 8 R registers (r8-r15) to hold 256 neurons:
- Each R register holds 32 int32 values (128 bytes)
- 8 registers × 32 values = 256 neurons total
- Total accumulator space: 1024 bytes

## Algorithm

```
For the single input sample:
    1. Clear 8 R registers (r8-r15) by loading zeros
    2. Load input vector (128 bytes)
    3. For each of 128 input elements:
        a. Load weight row (256 neuron weights for this element)
        b. mac.ev: Multiply input[element] with all neuron weights, accumulate
    4. Store results (8 R registers = 256 neurons = 1024 bytes)
```

## Files

- `fully_connected_128x256.c` - C implementation with IPU setup/teardown
- `fully_connected_128x256.asm` - Assembly implementation
- `generate_test_data.py` - Script to generate test data
- `BUILD.bazel` - Bazel build configuration
- `test_fully_connected_128x256.sh` - Test runner script

### Generated Data Files

- `inputs_1x128_int8.bin` - Input activations (128 bytes)
- `weights_128x256_int8.bin` - Weight matrix, transposed (32768 bytes)
- `output_1x256_int32.bin` - Expected output (1024 bytes)
- `zeros_1024_bytes.bin` - Zero buffer (1024 bytes)

## Building and Running

### Build the binary
```bash
bazel build //src/apps/fully_connected_ZD/128x256:fully_connected_128x256
bazel build //src/apps/fully_connected_ZD/128x256:assemble_fully_connected_128x256
```

### Run the test
```bash
bazel test //src/apps/fully_connected_ZD/128x256:fully_connected_128x256_test --test_output=all
```

### Generate new test data
```bash
cd src/apps/fully_connected_ZD/128x256
python3 generate_test_data.py
```

## Key Differences from 128×128 Example

1. **More output neurons**: 256 vs 128
   - Requires 8 R registers instead of 4
   - Output size: 1024 bytes instead of 512 bytes

2. **Batch size**: 1 sample instead of 10
   - Simplified loop structure
   - No batching overhead

3. **Weight matrix size**: 32768 bytes vs 16384 bytes
   - Each weight row is 256 bytes instead of 128 bytes
   - More memory bandwidth required

## Performance Characteristics

- **Compute**: 128 input elements × 256 neurons = 32,768 MAC operations per sample
- **Memory accesses**:
  - Input read: 128 bytes (1 R register)
  - Weight reads: 128 rows × 256 bytes = 32768 bytes
  - Output write: 1024 bytes (8 R registers)
- **Parallelism**: All 256 neurons computed in parallel per input element
