"""One CR names one table (:mod:`ipu_harness.table_bases`).

A CR an XMEM instruction uses as its base will hold a bare table ID, so a kernel
reaches each tensor through exactly one CR and keeps any offset in an LR. The
rule is held to the scale kernel with each way of breaking it, and to every
kernel in the tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scale_kernel import ASM, write_scale

from ipu_emu.errors import ConfigurationError
from ipu_harness._toml_source import ManifestError
from ipu_harness.cases import expand
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, run_kernel, stage
from ipu_harness.variants import variant

FP32 = variant("fp32")


def _scale(tmp_path: Path, *replacements: tuple[str, str], asm: str = ASM) -> KernelManifest:
    """The scale kernel, with text replacements in its kernel.toml."""
    manifest = write_scale(tmp_path, with_reference=False, asm=asm)
    path = manifest.directory / "kernel.toml"
    text = path.read_text()
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    path.write_text(text)
    return load_manifest(manifest.directory)


def _stage(manifest: KernelManifest):
    case = next(c for c in expand(manifest) if c.name == "default")
    return stage(manifest, case, FP32)


def test_the_scale_kernel_reaches_each_tensor_through_its_one_cr(tmp_path):
    manifest = _scale(tmp_path)
    case = next(c for c in expand(manifest) if c.name == "default")
    assert run_kernel(manifest, case, FP32, RunOptions()).passed


def test_an_abi_offset_is_refused_at_load(tmp_path):
    with pytest.raises(ManifestError, match=r"x: a CR holds a table ID, never an offset"):
        _scale(tmp_path, ('x    = "CR2"', 'x    = { register = "CR2", offset = 1 }'))


def test_a_cr_holding_a_value_over_a_base_is_refused_at_load(tmp_path):
    with pytest.raises(ManifestError, match=r"binds xb, a value over base\(x\), to CR7"):
        _scale(
            tmp_path,
            ('qf = "case_qf"', 'qf = "case_qf"\nxb = "base(x) + 1"'),
            ('qf   = "CR6"', 'qf   = "CR6"\nxb   = "CR7"'),
        )


def test_a_base_cr_bound_to_no_tensor_is_refused(tmp_path):
    asm = ASM.replace("LD.MULT R0 LR1 {{ cr.x }}", "LD.MULT R0 LR1 {{ cr.rows }}")
    with pytest.raises(ConfigurationError, match=r"LD\.MULT addresses a table through CR4"):
        _stage(_scale(tmp_path, asm=asm))


def test_cr0_is_a_base_only_for_a_tensor_bound_to_it(tmp_path):
    asm = ASM.replace("LD.MULT R0 LR1 {{ cr.x }}", "LD.MULT R0 LR1 CR0")
    with pytest.raises(ConfigurationError, match=r"through CR0.*bind a tensor to it"):
        _stage(_scale(tmp_path, asm=asm))


def test_a_free_tables_cr_is_not_read_as_a_number(tmp_path):
    """Its ID is the binder's to choose, so a program may not depend on it."""
    asm = ASM.replace("SET LR2 CR0", "SET LR2 {{ cr.x }}")
    assert asm != ASM
    with pytest.raises(ConfigurationError, match=r"SET reads CR2, which holds a table ID"):
        _stage(_scale(tmp_path, asm=asm))


def _staged_kernels() -> list[KernelManifest]:
    return [m for m in default_registry().kernels() if m.stages is None]


@pytest.mark.parametrize("manifest", _staged_kernels(), ids=lambda m: m.name)
def test_every_kernel_reaches_each_tensor_through_one_cr(manifest):
    """The default case stages (the check runs while it does)."""
    case = next(c for c in expand(manifest) if c.name == "default")
    stage(manifest, case, (case.formats or manifest.formats)[0])
