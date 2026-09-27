"""The generated SystemVerilog package elaborates cleanly under slang."""

from pyslang import DiagnosticEngine
from pyslang.ast import Compilation
from pyslang.syntax import SyntaxTree

from ipu_as import gen_codegen

# Declaring the compound type elaborates every slot struct and union member;
# slang rejects a packed union whose members differ in width.
_ELAB_TOP = """
module ipu_instr_pkg_elab_top;
  import ipu_instr_pkg::*;
  ipu_compound_inst_t inst;
  if ($bits(ipu_compound_inst_t) != int'(IPU_COMPOUND_INST_WIDTH)) begin : g_width
    $error("ipu_compound_inst_t is not IPU_COMPOUND_INST_WIDTH bits");
  end
endmodule
"""


def test_sv_package_compiles():
    compilation = Compilation()
    for text, name in (
        (gen_codegen.render_template("ipu_instr_pkg.sv.j2"), "ipu_instr_pkg.sv"),
        (_ELAB_TOP, "ipu_instr_pkg_elab_top.sv"),
    ):
        compilation.addSyntaxTree(SyntaxTree.fromText(text, name=name))
    diagnostics = compilation.getAllDiagnostics()
    report = DiagnosticEngine.reportAll(compilation.sourceManager, diagnostics)
    assert not diagnostics, report
