"""Cross-family use is declared in FAMILY_DEPS only (E06-18, Q62).

Reads every family's sources (``kernels/<family>/`` and
``ipu_kernels/<family>/``) and fails on a use of another family that
``kernels/families.bzl`` does not list: a Python import of
``ipu_kernels.<other>``, or a path into ``kernels/<other>/`` (a stage program,
an ``{% include %}`` of another family's macro) in any source file. Also checks
that KERNELS names every kernel folder.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Bazel runs the test from the runfiles root; the sources are symlinks into
#: the checkout, which is where a folder missing from KERNELS would be.
ROOT = Path.cwd()
KERNELS_DIR = (ROOT / "kernels" / "families.bzl").resolve().parent
IPU_KERNELS = KERNELS_DIR.parent / "src" / "tools" / "ipu-kernels" / "src" / "ipu_kernels"

#: The files a family's use of another can be in.
_SOURCES = ("*.py", "*.toml", "*.asm", "*.asm.j2")


def load_families(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """KERNELS and FAMILY_DEPS from ``families.bzl`` (two dict literals)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["KERNELS"], found["FAMILY_DEPS"]


def _imports(path: Path) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append((node.lineno, node.module))
    return found


def _import_uses(path: Path, where: Path, others: Sequence[str]) -> list[str]:
    found = []
    for line, module in _imports(path):
        parts = module.split(".")
        if parts[0] == "ipu_kernels" and len(parts) > 1 and parts[1] in others:
            found.append(f"{where}:{line}: imports {module}")
    return found


def _path_uses(path: Path, where: Path, others: Sequence[str]) -> list[str]:
    if not others:
        return []
    use = re.compile(r"(?:kernels/|\.\./)(" + "|".join(map(re.escape, others)) + r")/")
    return [
        f"{where}:{number}: uses kernels/{match[1]}/"
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if (match := use.search(text))
    ]


def violations(
    kernels_dir: Path,
    ipu_kernels: Path,
    families: Sequence[str],
    family_deps: Mapping[str, Sequence[str]],
) -> list[str]:
    """Every use of one family by another that ``family_deps`` does not list."""
    problems = []
    for family in families:
        allowed = {family, *family_deps.get(family, ())}
        others = [f for f in families if f not in allowed]
        for root in (kernels_dir / family, ipu_kernels / family):
            paths = [p for pattern in _SOURCES for p in sorted(root.rglob(pattern))]
            for path in paths if root.is_dir() else []:
                where = path.relative_to(root.parent)
                if path.suffix == ".py":
                    problems += _import_uses(path, where, others)
                problems += _path_uses(path, where, others)
    return [f"{p} (not in FAMILY_DEPS)" for p in problems]


def test_kernels_names_every_kernel_folder():
    kernels, _ = load_families(KERNELS_DIR / "families.bzl")
    folders = {(p.parent.parent.name, p.parent.name) for p in KERNELS_DIR.glob("*/*/kernel.toml")}
    listed = {(family, name) for family, names in kernels.items() for name in names}
    assert listed == folders, "add the kernel to KERNELS in kernels/families.bzl"


def test_every_cross_family_use_is_declared():
    kernels, family_deps = load_families(KERNELS_DIR / "families.bzl")
    assert set(family_deps) <= set(kernels), "FAMILY_DEPS names a family with no kernels"
    assert violations(KERNELS_DIR, IPU_KERNELS, list(kernels), family_deps) == []


def _plant(tmp_path: Path, files: Mapping[str, str]) -> tuple[Path, Path]:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path / "kernels", tmp_path / "ipu_kernels"


def test_an_undeclared_import_is_reported(tmp_path):
    kernels, ipu_kernels = _plant(
        tmp_path,
        {
            "kernels/attention/pack.py": "from ipu_kernels.softmax import rows\n",
            "ipu_kernels/attention/layers.py": "import ipu_kernels.softmax.pack\n",
            "kernels/softmax/s/kernel.toml": "",
        },
    )
    families = ["attention", "softmax"]
    assert violations(kernels, ipu_kernels, families, {}) == [
        "attention/pack.py:1: imports ipu_kernels.softmax (not in FAMILY_DEPS)",
        "attention/layers.py:1: imports ipu_kernels.softmax.pack (not in FAMILY_DEPS)",
    ]
    assert violations(kernels, ipu_kernels, families, {"attention": ["softmax"]}) == []


def test_an_undeclared_stage_or_include_is_reported(tmp_path):
    kernels, ipu_kernels = _plant(
        tmp_path,
        {
            "kernels/attention/qk/kernel.toml": 'stages = "../../softmax/rows/rows.asm"\n',
            "kernels/attention/qk/qk.asm": '{% include "../softmax/macros.asm.j2" %}\n',
            "kernels/attention/qk/stages.py": 'PROGRAM = "kernels/softmax/rows/rows.asm"\n',
        },
    )
    found = violations(kernels, ipu_kernels, ["attention", "softmax"], {})
    assert found == [
        "attention/qk/stages.py:1: uses kernels/softmax/ (not in FAMILY_DEPS)",
        "attention/qk/kernel.toml:1: uses kernels/softmax/ (not in FAMILY_DEPS)",
        "attention/qk/qk.asm:1: uses kernels/softmax/ (not in FAMILY_DEPS)",
    ]
