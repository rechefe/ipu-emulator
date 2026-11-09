"""Kernels set every loop register they read (the kernel-start contract).

An LR holds what an instruction wrote, so a kernel that reads an LR no instruction of
its program set reads whatever the last program left. ``[abi]`` used to preset the
loop registers of 27 kernel manifests; the programs now set them.

Each test runs the default case of a kernel with all sixteen LRs filled with
seeded garbage right before the ``[abi]`` writes and holds the outputs to the golden
as usual. A program that reads an LR it never set fails here, the same way it
fails in a chain after a kernel that left other LRs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from zlib import crc32

import numpy as np
import pytest

from ipu_harness import abi, runner, stages
from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, run_kernel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

LR_COUNT = 16

#: The kernel folders (registry entries over a template count with it) whose
#: manifests used to bind LRs in ``[abi]``, plus the two kernels that read an LR
#: no one set.
KERNEL_DIRS = frozenset(
    {
        "attn_scores_km_16x60",
        "attn_scores_km_256x36",
        "attn_scores_km_64x48",
        "attn_v_bcast_36",
        "attn_v_bcast_48",
        "attn_v_bcast_60",
        "qk_scores_16x60",
        "qk_scores_256x36",
        "qk_scores_64x48",
        "conv_universal_wide384",
        "pointwise_conv_tpl",
        "fully_connected",
        "matmul_small",
        "matmul_x128_2tg_c2",
        "matmul_x128_2tg_c3",
        "matmul_x128_c2",
        "matmul_x128_c2_qkv",
        "matmul_x128_c3",
        "matmul_x128_c4",
        "proj_p4",
        "proj_p4_2tg",
        "proj_p4_2tg_silu",
        "fold_16x16x192",
        "fold_32x32x144",
        "fold_8x8x240",
        "unfold_16x16x192",
        "unfold_32x32x144",
        "unfold_8x8x240",
    }
)

KERNELS_ROOT = Path(__file__).resolve().parents[3] / "kernels"

_REGISTRY = default_registry()
NAMES = sorted(k.name for k in _REGISTRY.kernels() if k.directory.name in KERNEL_DIRS)


def _with_garbage_lrs(monkeypatch: pytest.MonkeyPatch, seed: str) -> None:
    """Fill all LRs with seeded garbage just before the harness writes ``[abi]``."""
    rng = np.random.default_rng(crc32(seed.encode()))
    real = abi.apply

    def apply(model, writes):
        for i in range(LR_COUNT):
            model.write_register("lr", int(rng.integers(0, 2**32)), i)
        real(model, writes)

    monkeypatch.setattr(runner, "apply", apply)
    monkeypatch.setattr(stages, "apply", apply)


def test_the_list_names_kernels_the_registry_has():
    assert {k.directory.name for k in _REGISTRY.kernels() if k.name in NAMES} == KERNEL_DIRS


@pytest.mark.parametrize("name", NAMES)
def test_the_default_case_passes_with_garbage_lrs(name, monkeypatch):
    manifest = _REGISTRY.kernel(name)
    case = next(c for c in expand(manifest) if c.name == "default")
    _with_garbage_lrs(monkeypatch, name)
    result = run_kernel(manifest, case, case.formats[0], RunOptions(), cache=None)
    assert result.passed, result.describe()


def _lr_bindings(value: object, path: str) -> list[str]:
    if isinstance(value, str):
        return [f"{path} = {value}"] if re.fullmatch(r"LR\d+", value.strip()) else []
    if isinstance(value, dict):
        return [b for k, v in value.items() for b in _lr_bindings(v, f"{path}.{k}")]
    return []


def test_no_kernel_toml_binds_an_lr_in_abi():
    found = []
    for path in sorted(KERNELS_ROOT.glob("**/kernel.toml")):
        abi_table = tomllib.loads(path.read_text(encoding="utf-8")).get("abi", {})
        found += _lr_bindings(abi_table, str(path.relative_to(KERNELS_ROOT)))
    assert found == []
