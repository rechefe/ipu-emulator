#!/usr/bin/env bash
# Run a single app's test without Bazel.
# Usage: ./run_test.sh <app_name>
# Example: ./run_test.sh fully_connected
#
# Run all emulator/assembler unit tests (no app name needed):
#   ./run_test.sh --unit

set -euo pipefail
cd "$(dirname "$0")"

PYPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"
export PYTHONPATH="$PYPATH"

APPS_SRC="src/tools/ipu-apps/src/ipu_apps"
APPS_TEST="src/tools/ipu-apps/test"

# Map app name -> env-var prefix
declare -A PREFIX_MAP=(
    [fully_connected]=FC
    [depthwise_conv]=DCONV
    [depthwise_conv_64x64x1]=DCONV64
    [depthwise_conv_128x128x8]=DCONV8
    [depthwise_conv_128x128x64]=DCONV64
    [depthwise_conv_64x64x256]=DCONV256
    [conv2ch]=CONV2CH
    [conv2ch2ch]=CONV2CH2CH
    [pointwise_conv]=PCONV
    [pointwise_conv_32x32x16]=PCONV16
    [pointwise_conv_32x32x32]=PCONV32
    [pointwise_conv_128x128_16to64]=PCONV128_16TO64
    [pointwise_conv_128x128_64to16]=PCONV128_64TO16
    [conv_128x128_1to8]=CONV1TO8
    [conv_128x128_4to8]=CONV4TO8
    [conv_128x128_8to16]=CONV8TO16
    [conv_64x64_8to16]=CONV64_8TO16
    [conv_32x32x16]=CONV32X32X16
)

if [[ "${1:-}" == "--unit" ]]; then
    echo "=== Running emulator unit tests ==="
    python -m pytest src/tools/ipu-emu-py/test/ -v
    echo ""
    echo "=== Running assembler tests ==="
    python -m pytest src/tools/ipu-as-py/test/ -v
    exit 0
fi

if [[ -z "${1:-}" ]]; then
    echo "Usage: ./run_test.sh <app_name>"
    echo "       ./run_test.sh --unit"
    echo ""
    echo "Available apps:"
    for app in $(echo "${!PREFIX_MAP[@]}" | tr ' ' '\n' | sort); do
        echo "  $app"
    done
    exit 1
fi

APP="$1"
PREFIX="${PREFIX_MAP[$APP]:-}"

if [[ -z "$PREFIX" ]]; then
    echo "Error: unknown app '$APP'"
    echo "Available apps:"
    for app in $(echo "${!PREFIX_MAP[@]}" | tr ' ' '\n' | sort); do
        echo "  $app"
    done
    exit 1
fi

ASM_FILE="$APPS_SRC/$APP/$APP.asm"
if [[ ! -f "$ASM_FILE" ]]; then
    echo "Error: assembly file not found: $ASM_FILE"
    exit 1
fi

BIN_FILE="/tmp/${APP}.bin"
DATA_DIR="$APPS_SRC/$APP/test_data_format"
TEST_FILE="$APPS_TEST/test_${APP}.py"

if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: test file not found: $TEST_FILE"
    exit 1
fi

echo "=== Assembling $ASM_FILE ==="
python -m ipu_as.cli assemble --input "$ASM_FILE" --output "$BIN_FILE" --format bin

echo "=== Running $TEST_FILE ==="
export "${PREFIX}_INST_BIN=$BIN_FILE"
export "${PREFIX}_DATA_DIR=$DATA_DIR"
python -m pytest "$TEST_FILE" -v
