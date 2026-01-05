#!/bin/bash
# Test script for IPU Fully Connected Layer 128×256 (batch=1)

set -e

# Get the paths to the binary and data files
IPU_BINARY="$1"
INST_FILE="$2"
INPUTS_FILE="$3"
WEIGHTS_FILE="$4"
OUTPUT_BASENAME="$5"

# Write output to test undeclared outputs directory so it's preserved with test logs
if [ -n "$TEST_UNDECLARED_OUTPUTS_DIR" ]; then
    OUTPUTS_FILE="$TEST_UNDECLARED_OUTPUTS_DIR/$OUTPUT_BASENAME"
else
    OUTPUTS_FILE="$OUTPUT_BASENAME"
fi

echo "Running IPU Fully Connected Layer 128×256 Test (batch=1)..."
echo "Binary: $IPU_BINARY"
echo "Instruction file: $INST_FILE"
echo "Inputs file: $INPUTS_FILE"
echo "Weights file: $WEIGHTS_FILE"
echo "Output will be saved to: $OUTPUTS_FILE"

# Set log level to INFO
export LOG_LEVEL=INFO

# Run the IPU fully connected program
"$IPU_BINARY" "$INST_FILE" "$INPUTS_FILE" "$WEIGHTS_FILE" "$OUTPUTS_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test PASSED: IPU Fully Connected Layer 128×256 executed successfully"
    echo "Output saved to: $OUTPUTS_FILE"
    exit 0
else
    echo "Test FAILED: IPU Fully Connected Layer 128×256 returned exit code $EXIT_CODE"
    exit 1
fi
