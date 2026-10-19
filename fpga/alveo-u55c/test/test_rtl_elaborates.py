"""slang elaborates the prototype kernel with the stub IPU.

Our RTL must be free of diagnostics. The generated register blocks may carry
PeakRDL's own style warnings (see //src/ipu-host:test_register_blocks_compile),
and nothing else.
"""

from pathlib import Path

from pyslang import DiagnosticEngine
from pyslang.ast import Compilation
from pyslang.syntax import SyntaxTree

FILELIST = Path("fpga/alveo-u55c/rtl/filelist_stub.f")
GENERATED = "src/ipu-host"
GENERATOR_WARNINGS = {"sign-conversion", "unnamed-generate"}


def sources():
    lines = FILELIST.read_text().splitlines()
    paths = [line.replace("$GEN", GENERATED) for line in lines if line and not line.startswith("#")]
    return [Path(p) for p in paths]


def test_kernel_elaborates_with_the_stub():
    compilation = Compilation()
    for path in sources():
        compilation.addSyntaxTree(SyntaxTree.fromText(path.read_text(), name=str(path)))
    engine = DiagnosticEngine(compilation.sourceManager)
    diagnostics = [
        d
        for d in compilation.getAllDiagnostics()
        if engine.getOptionName(d.code) not in GENERATOR_WARNINGS
    ]
    report = DiagnosticEngine.reportAll(compilation.sourceManager, diagnostics)
    assert not diagnostics, report
    top = [inst.name for inst in compilation.getRoot().topInstances]
    assert top == ["ipu_proto_kernel"]
