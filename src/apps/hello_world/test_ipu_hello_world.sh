#!/bin/bash
# Test script for IPU Hello World example

set -e

# Get the paths to the binary and instruction file
IPU_BINARY="$1"
INST_FILE="$2"

echo "Running IPU Hello World Test..."
echo "Binary: $IPU_BINARY"
echo "Instruction file: $INST_FILE"

# Set log level to INFO
export LOG_LEVEL=INFO

# Run the IPU hello world program
"$IPU_BINARY" "$INST_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test PASSED: IPU Hello World executed successfully"
    exit 0
else
    echo "Test FAILED: IPU Hello World returned exit code $EXIT_CODE"
    exit 1
fi
