"""Shared plumbing for the MobileViT cross-family tests in this directory.

* :func:`kernel_inst` -- a registered kernel's assembled binary, assembled once
  per process through the registry (:func:`~ipu_apps.kernel_registry.cases.assemble_kernel`,
  which finds the ``.asm`` via ``importlib.resources`` -- no source-tree path).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ipu_apps.kernel_registry.cases import assemble_kernel

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
