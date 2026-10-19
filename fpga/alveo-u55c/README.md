# IPU prototype on the Alveo U55C

The platform around the IPU for the FPGA prototype. The specification is [FPGA prototype](../../docs/content/specs/fpga-prototype.md); the decisions are [ADR 0007](../../docs/adr/0007-fpga-prototype-on-alveo-u55c.md).

## Files

| Path | What it is |
|---|---|
| `rtl/ipu_proto_kernel.sv` | The Vitis RTL kernel: control routing, the platform registers, the socket exerciser, the IPU and the cache unit |
| `rtl/ipu_socket_pkg.sv` | The socket: sizes and the `tbl_view` type the IPU and the platform share |
| `rtl/ipu_top_stub.sv` | A stand-in `ipu_top` for bring-up before the IPU RTL exists |
| `rtl/proto_cache_unit.sv`, `rtl/proto_pkg.sv` | The prototype cache unit: XMEM, bank states, DMA to HBM |
| `rtl/axil_split2.sv` | Splits the control port between the IPU and the platform |
| `rtl/filelist_stub.f` | Compile order; `$GEN` is `bazel-bin/src/ipu-host` |
| `vitis/` | `kernel.xml`, the packaging script and the `v++` link configuration |
| `sim/` | A Verilator simulation of the kernel with the stub |
| `host/` | The card as a `Device` (XRT) and `run_flat`, which runs a program and compares it with the reference model |
| `Makefile` | `rtl`, `xo`, `xclbin`, `sim` |

The register blocks come from `src/ipu-host/src/ipu_host/rdl/` (`bazel build //src/ipu-host:register_blocks`). Never edit the generated SystemVerilog.

## Checks

| Command | Needs | Checks |
|---|---|---|
| `bazel test //fpga/alveo-u55c:all` | Bazel | slang elaborates the kernel with the stub and the generated blocks, with no diagnostics in our files |
| `bazel test //src/ipu-host:all` | Bazel | the register maps, the generated blocks, and the host driver on the modelled card |
| `fpga/alveo-u55c/sim/run.sh` | Verilator 5 | the host path, commit, fill, load, store, done and drain, and the stub's empty kernel |
| `make xclbin` | Vitis 2023.2, the U55C platform | synthesis and implementation. Not yet run. |

## Run a program

On the model (any machine):

```bash
bazel run //src/ipu-as:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin"
bazel run //fpga/alveo-u55c:run_flat -- "$PWD/prog.bin" --format int8 --cr 2=3 --cr 3=40
```

On the card, after `make xclbin` and `source /opt/xilinx/xrt/setup.sh`:

```bash
bazel run //fpga/alveo-u55c:run_flat -- "$PWD/prog.bin" --format int8 --cr 2=3 \
    --xclbin "$PWD/fpga/alveo-u55c/build/hw/ipu_proto.xclbin"
```

## Drop in the IPU

The RTL repository provides a file list whose files define `module ipu_top` with the ports of `rtl/ipu_top_stub.sv` ([External interfaces](../../docs/content/requirements/interfaces.md)), import `ipu_socket_pkg`, and may use `ipu_host_regs`, `ipu_host_regs_pkg`, `ipu_host_addr_pkg` and `ipu_instr_pkg`. Then:

```bash
make -C fpga/alveo-u55c xclbin IPU_RTL=/path/to/ipu_rtl.f
```

The Makefile puts `ipu_instr_pkg.sv` and that list where the stub was: after the packages and the generated register blocks, before the platform modules.
