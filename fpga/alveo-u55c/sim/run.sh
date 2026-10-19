#!/usr/bin/env bash
# Simulate the prototype kernel with the stub IPU (Verilator 5). Not part of
# `bazel test`: CI has no Verilator. Run from anywhere in the checkout:
#
#   fpga/alveo-u55c/sim/run.sh
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root"
bazel build //src/ipu-host:register_blocks
gen="$(bazel info bazel-bin)/src/ipu-host"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
mapfile -t rtl < <(sed -e "s#\$GEN#$gen#" -e '/^#/d' -e '/^$/d' fpga/alveo-u55c/rtl/filelist_stub.f)
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --top-module tb_ipu_proto_kernel \
  -Mdir "$out" "${rtl[@]}" fpga/alveo-u55c/sim/tb_ipu_proto_kernel.sv >"$out/build.log" 2>&1 \
  || { cat "$out/build.log"; exit 1; }
"$out/Vtb_ipu_proto_kernel" | tee "$out/log"
grep -q '^PASS$' "$out/log"
