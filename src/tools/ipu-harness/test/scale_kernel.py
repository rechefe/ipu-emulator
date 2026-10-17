"""The epic's worked example, the ``scale`` kernel (y = x * s), written to a folder.

The runner's tests run it end to end (E06-11). ``with_reference=False`` leaves out
``reference.py`` and gives every case an expected file, so nothing needs torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_harness.manifest import KernelManifest, load_manifest

#: The epic's program stores y through ACTIVATE.QUANTIZE and ST.AAQ, which the
#: emulator runs in INT8, FP32 and INT32 only (the AaQ stage is a placeholder,
#: OQ-20). This one stores the accumulator (ST.ACC), so it runs in E4 too: y is
#: an ``accumulator`` output, one accumulator vector per row of x.
ASM = """\
{#- scale.asm -- y = x * s, one XMEM row (128 elements) per iteration. -#}
{%- set acc_rows = 1 if format in ("fp32", "int32") else 4 %}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ST.ACC LR2 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 {{ acc_rows }} ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

TOML = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = {{ type = "int", min = 1, max = 256 }}
s    = {{ type = "float" }}

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]
store = "accumulator"

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
{reference}tolerance = {{ fp32 = "EXACT", int8 = "EXACT", e4 = "FP8_ULP" }}

[cases.default]
rows = 4
s = 2.0
{default_files}
[cases.one_row]
rows = 1
s = -3.0
{one_row_files}
[cases.mine]
rows = 2
s = 2.0
inputs = {{ x = "data/x.npy" }}
expected = {{ y = "data/y.npy" }}
"""

REFERENCE = '''\
import torch


def reference(x: torch.Tensor, *, rows: int, s: float) -> dict[str, torch.Tensor]:
    """y = x * s."""
    return {"y": x * s}
'''


def _files(name: str, rows: int, s: float, directory: Path) -> str:
    """An inputs/expected pair for a case, small integers exact in every format."""
    x = np.arange(rows * 128, dtype=np.float64).reshape(rows, 128) % 17 - 8
    np.save(directory / "data" / f"{name}_x.npy", x)
    np.save(directory / "data" / f"{name}_y.npy", x * s)
    return f'inputs = {{ x = "data/{name}_x.npy" }}\nexpected = {{ y = "data/{name}_y.npy" }}\n'


def write_scale(root: Path, *, with_reference: bool = True, asm: str = ASM) -> KernelManifest:
    """Write the kernel under ``root/elementwise/scale`` and load it."""
    directory = root / "elementwise" / "scale"
    (directory / "data").mkdir(parents=True)
    x = np.arange(2 * 128, dtype=np.float64).reshape(2, 128) % 17 - 8
    np.save(directory / "data" / "x.npy", x)
    np.save(directory / "data" / "y.npy", 2 * x)
    (directory / "scale.asm").write_text(asm)
    if with_reference:
        (directory / "reference.py").write_text(REFERENCE)
        toml = TOML.format(
            reference='reference = "reference.py:reference"\n', default_files="", one_row_files=""
        )
    else:
        toml = TOML.format(
            reference="",
            default_files=_files("default", 4, 2.0, directory),
            one_row_files=_files("one_row", 1, -3.0, directory),
        )
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)
