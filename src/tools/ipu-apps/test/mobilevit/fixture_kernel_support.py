"""Shared plumbing for the MobileViT cross-family tests in this directory.

* :func:`kernel_inst` -- a registered kernel's assembled binary, assembled once
  per process through the registry (:func:`~ipu_apps.kernel_registry.cases.assemble_kernel`,
  which finds the ``.asm`` via ``importlib.resources`` -- no source-tree path).
* :func:`kernel_program` -- a //kernels kernel's program words for the given
  parameters (the packed chains load the registered packed kernels, E07-23).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ipu_apps.kernel_registry.cases import assemble_kernel
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat

_INST: dict[str, Path] = {}
_INST_DIR: Path | None = None


def kernel_inst(name: str) -> Path:
    """Assemble (once per process) and return registered kernel ``name``'s binary."""
    global _INST_DIR
    if name not in _INST:
        if _INST_DIR is None:
            _INST_DIR = Path(tempfile.mkdtemp(prefix="mobilevit_inst_"))
        _INST[name] = assemble_kernel(name, _INST_DIR)
    return _INST[name]


def kernel_program(name: str, **params: object) -> bytes:
    """The //kernels kernel ``name``'s program in FP32, as a ``.bin`` file holds
    it: its default case's parameters, ``params`` replacing them."""
    manifest = default_registry().kernel(name)
    (case,) = (c for c in expand(manifest) if c.name == "default")
    plan = plan_case(manifest, {**case.values, **params}, ElementFormat.FP32)
    return assemble(plan).to_bin()
