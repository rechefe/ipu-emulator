"""Cross-family use is declared in FAMILY_DEPS only (E06-18, Q62).

Reads every family's sources (``kernels/<family>/`` and
``ipu_kernels/<family>/``) and fails on a use of another family that
``kernels/families.bzl`` does not list: a Python import of
``ipu_kernels.<other>``, or a path into ``kernels/<other>/`` (a stage program,
an ``{% include %}`` of another family's macro) in any source file. Also checks
that KERNELS names every kernel folder.

A pack name cannot resolve to another family (a pack is looked up in the
kernel folder, its family folder's pack.py, its own ipu_kernels.<family> and
the shared ipu_kernels.pack),
so such an edge can only be an import or a re-export, which this catches.
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
IPU_KERNELS = KERNELS_DIR.parent / "src" / "ipu-kernels" / "src" / "ipu_kernels"

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


def _module_of(path: Path, root: Path) -> list[str]:
    """The package parts of a file under ``ipu_kernels/`` (``[ipu_kernels, softmax, sub]``)."""
    return ["ipu_kernels", *path.relative_to(root).parent.parts]


def _imported(path: Path, package: list[str] | None) -> list[tuple[int, str]]:
    """Every module a Python file imports, relative imports resolved against
    ``package`` (``None``: the file is in no package, relative imports are skipped)."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None or node.level > len(package):
                    continue
                base = package[: len(package) - node.level + 1]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            # `from ipu_kernels import softmax`: the names may be modules.
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
    return found


def _import_uses(path: Path, where: Path, others: Sequence[str], package) -> list[str]:
    found = []
    for line, module in _imported(path, package):
        parts = module.split(".")
        if parts[0] == "ipu_kernels" and len(parts) > 1 and parts[1] in others:
            found.append(f"{where}:{line}: imports {'.'.join(parts[:2])}")
    return sorted(set(found), key=found.index)


def _text_uses(path: Path, where: Path, others: Sequence[str]) -> list[str]:
    """Paths into another family (``kernels/<other>/``, or ``../../<other>/`` from a
    kernel folder; ``../<x>/`` is a sibling kernel of the same family) and module
    names in strings (``import_module("ipu_kernels.<other>.pack")``)."""
    if not others:
        return []
    names = "|".join(map(re.escape, others))
    use = re.compile(rf"(?<![\w/])(?:kernels/|\.\./\.\./)({names})/|\bipu_kernels\.({names})\b")
    found = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for match in use.finditer(text):
            other = match[1] or match[2]
            found.append(
                f"{where}:{number}: uses {'kernels/' + other + '/' if match[1] else 'ipu_kernels.' + other}"
            )
    return found


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
                text = _text_uses(path, where, others)
                if path.suffix == ".py":
                    package = _module_of(path, ipu_kernels) if root.parent == ipu_kernels else None
                    imports = _import_uses(path, where, others, package)
                    # An import line also matches the text check: report it once, as an import.
                    lines = {p.split(": ")[0] for p in imports}
                    text = [p for p in text if p.split(": ")[0] not in lines]
                    problems += imports
                problems += text
    return [f"{p} (not in FAMILY_DEPS)" for p in problems]


def test_kernels_names_every_kernel_folder():
    kernels, _ = load_families(KERNELS_DIR / "families.bzl")
    folders = {(p.parent.parent.name, p.parent.name) for p in KERNELS_DIR.glob("*/*/kernel.toml")}
    listed = {(family, name) for family, names in kernels.items() for name in names}
    assert listed == folders, "add the kernel to KERNELS in kernels/families.bzl"


def test_family_deps_name_other_families():
    kernels, family_deps = load_families(KERNELS_DIR / "families.bzl")
    for family, uses in family_deps.items():
        assert family in kernels, f"FAMILY_DEPS: {family} is no family"
        for other in uses:
            assert other in kernels and other != family, f"FAMILY_DEPS[{family}]: {other}"


def test_every_cross_family_use_is_declared():
    kernels, family_deps = load_families(KERNELS_DIR / "families.bzl")
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
            "kernels/attention/qk/test_qk.py": "from ipu_kernels.softmax import rows\n",
            "ipu_kernels/attention/layers.py": "import ipu_kernels.softmax.pack\n",
            "ipu_kernels/attention/pack.py": "from ipu_kernels import softmax\n",
            "ipu_kernels/attention/sub/stages.py": "from ...softmax import pack\n",
            "ipu_kernels/attention/gen.py": 'M = importlib.import_module("ipu_kernels.softmax.pack")\n',
        },
    )
    families = ["attention", "softmax"]
    assert violations(kernels, ipu_kernels, families, {}) == [
        "attention/qk/test_qk.py:1: imports ipu_kernels.softmax (not in FAMILY_DEPS)",
        "attention/gen.py:1: uses ipu_kernels.softmax (not in FAMILY_DEPS)",
        "attention/layers.py:1: imports ipu_kernels.softmax (not in FAMILY_DEPS)",
        "attention/pack.py:1: imports ipu_kernels.softmax (not in FAMILY_DEPS)",
        "attention/sub/stages.py:1: imports ipu_kernels.softmax (not in FAMILY_DEPS)",
    ]
    assert violations(kernels, ipu_kernels, families, {"attention": ["softmax"]}) == []


def test_an_undeclared_file_use_is_reported(tmp_path):
    kernels, ipu_kernels = _plant(
        tmp_path,
        {
            # A stage program of another family, from a kernel folder (E06-14).
            "kernels/attention/qk/stages.py": 'ctx.run_stage("../../softmax/rows/rows.asm")\n',
            "kernels/attention/qk/notes.toml": 'source = "kernels/softmax/rows"\n',
            # Not uses: a sibling kernel of the same family, and a provenance note.
            "kernels/attention/qk/qk.asm": "# from an old harness folder, softmax/rows/rows.asm\n",
            "kernels/attention/qk/more.py": 'ctx.run_stage("../softmax/x.asm")\n',
        },
    )
    assert violations(kernels, ipu_kernels, ["attention", "softmax"], {}) == [
        "attention/qk/stages.py:1: uses kernels/softmax/ (not in FAMILY_DEPS)",
        "attention/qk/notes.toml:1: uses kernels/softmax/ (not in FAMILY_DEPS)",
    ]
