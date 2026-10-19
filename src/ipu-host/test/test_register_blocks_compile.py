"""slang elaborates the generated register blocks.

PeakRDL's output draws two slang warnings: sign-conversion (a signed loop
index in an unsigned address compare) and unnamed-generate. They are the
generator's style, not ours, and are the only diagnostics allowed.
"""

from pathlib import Path

import pytest
from pyslang import DiagnosticEngine
from pyslang.ast import Compilation
from pyslang.syntax import SyntaxTree

from ipu_host.regmap import MAP_NAMES

GENERATED = Path("src/ipu-host")
GENERATOR_WARNINGS = {"sign-conversion", "unnamed-generate"}


@pytest.mark.parametrize("name", MAP_NAMES)
def test_register_block_elaborates(name):
    compilation = Compilation()
    for path in (
        GENERATED / f"{name}_regs_pkg.sv",
        GENERATED / f"{name}_regs.sv",
        GENERATED / f"{name}_addr_pkg.sv",
    ):
        compilation.addSyntaxTree(SyntaxTree.fromText(path.read_text(), name=path.name))
    engine = DiagnosticEngine(compilation.sourceManager)
    diagnostics = [
        d
        for d in compilation.getAllDiagnostics()
        if engine.getOptionName(d.code) not in GENERATOR_WARNINGS
    ]
    report = DiagnosticEngine.reportAll(compilation.sourceManager, diagnostics)
    assert not diagnostics, report
