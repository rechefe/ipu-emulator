# The prototype kernel with the stub IPU, in compile order. $GEN is the
# directory of //src/ipu-host:register_blocks (bazel-bin/src/ipu-host).
fpga/alveo-u55c/rtl/ipu_socket_pkg.sv
fpga/alveo-u55c/rtl/proto_pkg.sv
$GEN/ipu_host_addr_pkg.sv
$GEN/proto_platform_addr_pkg.sv
$GEN/ipu_host_regs_pkg.sv
$GEN/proto_platform_regs_pkg.sv
$GEN/ipu_host_regs.sv
$GEN/proto_platform_regs.sv
fpga/alveo-u55c/rtl/axil_split2.sv
fpga/alveo-u55c/rtl/ipu_top_stub.sv
fpga/alveo-u55c/rtl/proto_cache_unit.sv
fpga/alveo-u55c/rtl/ipu_proto_kernel.sv
