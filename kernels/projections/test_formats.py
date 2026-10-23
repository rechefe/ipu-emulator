"""The projections run in FP32 only (E07-03, ADR 0007).

Every template stores its output rows with ``ACTIVATE.EXACT``, the
emulator-only FP32 store: NX8's ``ACTIVATE.QUANTIZE`` has no SiLU, and the
plan keeps the identity entries FP32 too (plan item 25). These tests pin that
an NX8 variant of any template is refused when its program is loaded, before
anything runs, rather than answering in a mode the kernel was not written for.

When a template moves to ``ACTIVATE.QUANTIZE`` (or NX8 gains SiLU), its test
fails: add the NX8 variants to its ``formats`` and drop it here.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from ipu_emu.errors import EmulatorError
from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.runner import run_kernel
from ipu_harness.variants import variant

FAMILY = Path(__file__).parent

TEMPLATES = [
    "proj_p4",
    "proj_p4_2tg",
    "proj_p4_2tg_silu",
    "packed_output_linear",
    "packed_output_linear_p4",
]


def _template_copy(tmp_path: Path, template: str, formats: str) -> Path:
    """``template`` in a copy of the family folder, its ``[kernel] formats``
    replaced, so a test can run it in a variant it does not list."""
    copy = tmp_path / "projections"
    shutil.copytree(FAMILY, copy, ignore=shutil.ignore_patterns("test_*.py", "BUILD.bazel"))
    toml = copy / template / "kernel.toml"
    text, n = re.subn(
        r"^formats = .*$", f"formats = {formats}", toml.read_text(), count=1, flags=re.M
    )
    assert n == 1
    toml.write_text(text)
    return copy / template


def _small_case(manifest):
    """The template's default case: its own small shape."""
    return next(c for c in expand(manifest) if c.name == "default")


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_is_fp32_only(template):
    assert load_manifest(FAMILY / template).formats == (variant("fp32"),)


@pytest.mark.parametrize("template", TEMPLATES)
def test_nx8_is_refused_at_load_by_the_fp32_store(tmp_path, template):
    manifest = load_manifest(_template_copy(tmp_path, template, '["fp32", "e4"]'))
    with pytest.raises(EmulatorError, match="ACTIVATE.EXACT, which is valid in FP32 mode only"):
        run_kernel(manifest, _small_case(manifest), variant("e4"))
