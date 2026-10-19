# Package the prototype kernel as a Vitis .xo (Vivado batch mode).
#
#   vivado -mode batch -source package_kernel.tcl -tclargs <filelist> <kernel.xml> <out.xo> <build_dir>
#
# <filelist> holds one SystemVerilog file per line, in compile order
# (build/rtl.f, written by the Makefile).
set filelist   [lindex $argv 0]
set kernel_xml [lindex $argv 1]
set xo_path    [lindex $argv 2]
set build_dir  [lindex $argv 3]

set part xcu55c-fsvh2892-2L-e
create_project -force kernel_pack $build_dir/kernel_pack -part $part

set fp [open $filelist r]
set files [split [string trim [read $fp]] "\n"]
close $fp
add_files -norecurse $files
set_property file_type SystemVerilog [get_files *.sv]
set_property top ipu_proto_kernel [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $build_dir/ip -vendor ipu.proto -library kernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $build_dir/ip/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $build_dir/ip $build_dir/ip/component.xml

set core [ipx::current_core]
set_property core_revision 1 $core
foreach up [ipx::get_user_parameters] { ipx::remove_user_parameter [get_property NAME $up] $core }
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
ipx::associate_bus_interfaces -busif m_axi_hbm -clock ap_clk $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property vitis_drc {ctrl_protocol user_managed} $core
set_property ipi_drc {ignore_freq_hz true} $core

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete

package_xo -force -xo_path $xo_path -kernel_name ipu_proto_kernel -ip_directory $build_dir/ip -kernel_xml $kernel_xml
