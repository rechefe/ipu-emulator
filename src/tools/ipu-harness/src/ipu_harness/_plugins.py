"""Kernel-side Python loaded as plugins (Q62): a kernel folder's file by path, an
``ipu_kernels`` module by name. The harness never imports either statically.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType


@functools.lru_cache(maxsize=64)
def _executed(path: Path, digest: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"_ipu_kernel_{path.parent.name}_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_file(path: Path) -> ModuleType:
    """A kernel folder's ``.py`` file (no ``__init__.py`` needed), executed once
    per content: an edit is seen, an unchanged file is not run again.

    Raises:
        Exception: Whatever executing the file raises.
    """
    return _executed(path, hashlib.sha256(path.read_bytes()).hexdigest())


def import_named(module: str) -> ModuleType | None:
    """``module`` imported by name, or None if it (or a package above it) does not exist.

    Raises:
        Exception: Whatever importing an existing module raises.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name is not None and (module + ".").startswith(error.name + "."):
            return None
        raise
