#!/bin/bash
# Test script for IPU Fully Connected Layer example

set -e

# Get the paths to the binary and data files
IPU_BINARY="$1"
INST_FILE="$2"
INPUTS_FILE="$3"
WEIGHTS_FILE="$4"
OUTPUT_BASENAME="$5"
DTYPE="${6:-FP8_E4M3}"  # Default to FP8_E4M3 if not specified
EXPECTED_OUTPUT="$7"  # Expected output file for comparison

# Write output to test undeclared outputs directory so it's preserved with test logs
if [ -n "$TEST_UNDECLARED_OUTPUTS_DIR" ]; then
    OUTPUTS_FILE="$TEST_UNDECLARED_OUTPUTS_DIR/$OUTPUT_BASENAME"
else
    OUTPUTS_FILE="$OUTPUT_BASENAME"
fi

echo "Running IPU Fully Connected Layer Test..."
echo "Binary: $IPU_BINARY"
echo "Instruction file: $INST_FILE"
echo "Inputs file: $INPUTS_FILE"
echo "Weights file: $WEIGHTS_FILE"
echo "Data type: $DTYPE"
echo "Output will be saved to: $OUTPUTS_FILE"
if [ -n "$EXPECTED_OUTPUT" ]; then
    echo "Expected output: $EXPECTED_OUTPUT"
fi

# Set log level to INFO
export LOG_LEVEL=INFO

# Run the IPU fully connected program
"$IPU_BINARY" "$INST_FILE" "$INPUTS_FILE" "$WEIGHTS_FILE" "$OUTPUTS_FILE" "$DTYPE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Test FAILED: IPU Fully Connected Layer returned exit code $EXIT_CODE"
    exit 1
fi

echo "IPU execution completed successfully"

# Compare output with expected output if provided
if [ -n "$EXPECTED_OUTPUT" ]; then
    echo "Comparing output with expected result..."
    if cmp -s "$OUTPUTS_FILE" "$EXPECTED_OUTPUT"; then
        echo "Test PASSED: Output matches expected result"
        exit 0
    else
        echo "Test FAILED: Output does not match expected result"
        echo "Expected: $EXPECTED_OUTPUT"
        echo "Actual:   $OUTPUTS_FILE"
        # Show byte-by-byte diff for debugging
        echo "Byte difference:"
        cmp "$OUTPUTS_FILE" "$EXPECTED_OUTPUT" || true
        exit 1
    fi
else
    echo "Test PASSED: IPU Fully Connected Layer executed successfully (no expected output to compare)"
    echo "Output saved to: $OUTPUTS_FILE"
    exit 0
fi
