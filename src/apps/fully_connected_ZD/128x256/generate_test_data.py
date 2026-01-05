#!/usr/bin/env python3
"""
Generate test data for fully connected layer: 128 inputs × 256 output neurons, batch size = 1.

This script creates:
1. Input matrix: 1 sample × 128 elements (int8)
2. Weight matrix: 128 input elements × 256 neurons (int8) - TRANSPOSED layout
3. Expected output: 1 sample × 256 neurons (int32)

Weight matrix layout is CRITICAL:
- weights[input_element_idx][neuron_idx]
- Each row contains all neuron weights for one input element position
- This enables parallel computation of all neurons using mac.ev
"""

import struct
import random

# Configuration
NUM_SAMPLES = 1
INPUT_SIZE = 128
NUM_NEURONS = 256

# Random seed for reproducibility
random.seed(42)

def clamp_int8(value):
    """Clamp value to int8 range [-128, 127]."""
    if value > 127:
        return 127
    elif value < -128:
        return -128
    return value

def save_int8_binary(data, filename):
    """Save int8 data to binary file."""
    with open(filename, 'wb') as f:
        for value in data:
            # Pack as signed byte
            f.write(struct.pack('b', clamp_int8(value)))
    print(f"Saved {filename}: {len(data)} bytes")

def save_int32_binary(data, filename):
    """Save int32 data to binary file."""
    with open(filename, 'wb') as f:
        for value in data:
            # Pack as signed 32-bit integer (little-endian)
            f.write(struct.pack('<i', value))
    print(f"Saved {filename}: {len(data) * 4} bytes")

def generate_inputs():
    """
    Generate input activations.
    Shape: (NUM_SAMPLES, INPUT_SIZE)
    Values: Random int8 in range [-10, 10]
    Returns: Flat list of NUM_SAMPLES * INPUT_SIZE values
    """
    inputs = []
    for sample in range(NUM_SAMPLES):
        for element in range(INPUT_SIZE):
            inputs.append(random.randint(-10, 10))

    print(f"\nGenerated inputs: {NUM_SAMPLES} sample × {INPUT_SIZE} elements")
    print(f"Sample input[0][:10] = {inputs[:10]}")
    return inputs

def generate_weights_transposed():
    """
    Generate weight matrix in TRANSPOSED layout for parallel computation.

    Shape: (INPUT_SIZE, NUM_NEURONS)
    Layout: weights[input_element_idx][neuron_idx]

    Each ROW contains all neuron weights for one input element position.
    This allows mac.ev to compute partial products for all neurons in parallel.

    Values: Random int8 in range [-5, 5]
    Returns: Flat list of INPUT_SIZE * NUM_NEURONS values
    """
    weights = []
    for element in range(INPUT_SIZE):
        for neuron in range(NUM_NEURONS):
            weights.append(random.randint(-5, 5))

    print(f"\nGenerated weights (transposed layout): {INPUT_SIZE} elements × {NUM_NEURONS} neurons")
    print(f"weights[0] = all neurons' weights for input element 0")
    print(f"Sample weights[0][:10] = {weights[:10]}")
    return weights

def compute_expected_output(inputs, weights_transposed):
    """
    Compute expected output using matrix multiplication.

    Args:
        inputs: Flat list of NUM_SAMPLES * INPUT_SIZE values
        weights_transposed: Flat list of INPUT_SIZE * NUM_NEURONS values

    Returns:
        output: Flat list of NUM_SAMPLES * NUM_NEURONS values (int32)

    The computation matches the assembly implementation:
    For each sample:
        For each neuron:
            output[sample][neuron] = sum(input[sample][i] × weight[i][neuron] for i in range(INPUT_SIZE))
    """
    output = []

    for sample in range(NUM_SAMPLES):
        for neuron in range(NUM_NEURONS):
            acc = 0
            for element in range(INPUT_SIZE):
                input_idx = sample * INPUT_SIZE + element
                weight_idx = element * NUM_NEURONS + neuron

                input_val = inputs[input_idx]
                weight_val = weights_transposed[weight_idx]

                acc += input_val * weight_val

            output.append(acc)

    print(f"\nComputed expected output: {NUM_SAMPLES} sample × {NUM_NEURONS} neurons")
    print(f"Sample output[0][:10] = {output[:10]}")

    return output

def main():
    print("=" * 70)
    print("Generating test data for 128×256 fully connected layer (batch=1)")
    print("=" * 70)

    # Generate inputs
    inputs = generate_inputs()
    save_int8_binary(inputs, "inputs_1x128_int8.bin")

    # Generate weights in transposed layout
    weights_transposed = generate_weights_transposed()
    save_int8_binary(weights_transposed, "weights_128x256_int8.bin")

    # Compute expected output
    expected_output = compute_expected_output(inputs, weights_transposed)
    save_int32_binary(expected_output, "output_1x256_int32.bin")

    print("\n" + "=" * 70)
    print("Test data generation complete!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  1. inputs_1x128_int8.bin       - {NUM_SAMPLES} sample × {INPUT_SIZE} elements = {NUM_SAMPLES * INPUT_SIZE} bytes")
    print(f"  2. weights_128x256_int8.bin    - {INPUT_SIZE} elements × {NUM_NEURONS} neurons = {INPUT_SIZE * NUM_NEURONS} bytes (TRANSPOSED)")
    print(f"  3. output_1x256_int32.bin      - {NUM_SAMPLES} sample × {NUM_NEURONS} neurons = {NUM_SAMPLES * NUM_NEURONS * 4} bytes")
    print("\nWeight matrix layout:")
    print(f"  Row i = weights for all {NUM_NEURONS} neurons at input element i")
    print(f"  This enables parallel computation of all neurons per iteration")
    print()

if __name__ == "__main__":
    main()
