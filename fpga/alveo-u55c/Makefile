# Build the IPU prototype for the Alveo U55C (fpga-prototype.md, "Build").
#
#   make xclbin                       # hardware build with the stub IPU
#   make xclbin IPU_RTL=/path/ipu.f   # with the RTL repository's IPU
#   make xclbin TARGET=hw_emu         # hardware emulation
#   make sim                          # Verilator simulation with the stub
#
# Needs Vitis 2023.2 and XRT, with the U55C platform installed. `make rtl`
# needs only Bazel.

PLATFORM  ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET    ?= hw
FREQ_MHZ  ?= 100
IPU_RTL   ?= stub
BUILD     ?= build/$(TARGET)

ROOT      := $(shell git rev-parse --show-toplevel)
HERE      := $(ROOT)/fpga/alveo-u55c
GEN        = $(shell cd $(ROOT) && bazel info bazel-bin)/src/ipu-host

XO        := $(BUILD)/ipu_proto_kernel.xo
XCLBIN    := $(BUILD)/ipu_proto.xclbin

.PHONY: rtl xo xclbin sim clean

# The file list: the platform, the generated register blocks, and the IPU
# (the stub, or the RTL repository's list, which replaces ipu_top_stub.sv).
rtl: $(BUILD)/rtl.f
$(BUILD)/rtl.f: FORCE
	@mkdir -p $(BUILD)
	cd $(ROOT) && bazel build //src/ipu-host:register_blocks //src/ipu-gen:ipu-gen
	cd $(ROOT) && bazel run //src/ipu-gen:ipu-gen -- sv-package --output $(abspath $(BUILD))/ipu_instr_pkg.sv
	sed -e '/^#/d' -e '/^$$/d' -e 's#$$GEN#$(GEN)#' -e 's#^fpga/#$(ROOT)/fpga/#' \
	    $(HERE)/rtl/filelist_stub.f > $@.tmp
	if [ "$(IPU_RTL)" != stub ]; then \
	    awk -v instr=$(abspath $(BUILD))/ipu_instr_pkg.sv -v list=$(IPU_RTL) \
	        '/ipu_top_stub.sv/ { print instr; while ((getline l < list) > 0) print l; next } { print }' \
	        $@.tmp > $@.tmp2 && mv $@.tmp2 $@.tmp; \
	fi
	mv $@.tmp $@

xo: $(XO)
$(XO): $(BUILD)/rtl.f $(HERE)/vitis/kernel.xml $(HERE)/vitis/package_kernel.tcl
	vivado -mode batch -nojournal -nolog -source $(HERE)/vitis/package_kernel.tcl \
	    -tclargs $(abspath $(BUILD))/rtl.f $(HERE)/vitis/kernel.xml $(abspath $@) $(abspath $(BUILD))

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) $(HERE)/vitis/connectivity.cfg
	v++ --link --target $(TARGET) --platform $(PLATFORM) --kernel_frequency $(FREQ_MHZ) \
	    --config $(HERE)/vitis/connectivity.cfg --temp_dir $(BUILD)/link -o $@ $<

sim:
	$(HERE)/sim/run.sh

clean:
	rm -rf build

FORCE:
