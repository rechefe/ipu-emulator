"""Declared Python dependencies agree with the lock and with the Bazel libraries.

Checks, over the inputs listed in the header of ``requirements_lock.txt``:

- every requirement in every ``pyproject.toml`` (runtime and extras) and in
  ``tools/requirements-tools.in`` is pinned in the lock, so "added a dependency,
  forgot to relock" fails here (refresh with
  ``bazel run //tools:update_requirements_lock``);
- each package's ``RUNTIME_DEPS`` (its ``deps.bzl``, which its ``py_library``
  maps through ``requirement()``) equals its ``[project].dependencies`` minus the
  workspace packages (``ipu-*``).

Names compare after PEP 503 normalisation.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

LOCK = Path("requirements_lock.txt")

# Paths of the runtime_deps.json manifests in runfiles, one per package.
MANIFESTS = (
    Path("src/tools/ipu-as/runtime_deps.json"),
    Path("src/tools/ipu-emu/runtime_deps.json"),
    Path("src/tools/ipu-isa/runtime_deps.json"),
    Path("src/tools/ipu-debug/runtime_deps.json"),
    Path("src/tools/ipu-gen/runtime_deps.json"),
    Path("src/tools/ipu-kernels/runtime_deps.json"),
    Path("src/tools/ipu-harness/runtime_deps.json"),
)

# The repository's own distributions: Bazel wires them as py_library targets.
WORKSPACE_PREFIX = "ipu-"

_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", re.MULTILINE)


def normalize(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    """Distribution name of a PEP 508 requirement string.

    Handles extras, every version operator, URLs and environment markers:
    the name is always the leading identifier.
    """
    match = _NAME.match(spec)
    if not match:
        raise ValueError(f"not a requirement: {spec!r}")
    return normalize(match.group(1))


def is_workspace(name: str) -> bool:
    return normalize(name).startswith(WORKSPACE_PREFIX)


def lock_pins(lock_text: str) -> set[str]:
    """Normalised names pinned (``name==version``) in a uv/pip lock."""
    return {normalize(m.group(1)) for m in _PIN.finditer(lock_text)}


def lock_inputs(lock_text: str) -> list[str]:
    """The input files named in the ``uv pip compile`` command of the lock header."""
    for line in lock_text.splitlines():
        if not line.startswith("#"):
            break
        words = shlex.split(line.lstrip("#"))
        if words[:3] == ["uv", "pip", "compile"]:
            inputs = []
            for word in words[3:]:
                if word.startswith("-"):
                    break
                inputs.append(word)
            return inputs
    raise ValueError("lock header has no `uv pip compile` command")


def pyproject_requirements(pyproject: dict) -> tuple[list[str], list[str]]:
    """(runtime, extras) requirement names of a parsed ``pyproject.toml``."""
    project = pyproject.get("project", {})
    runtime = [requirement_name(s) for s in project.get("dependencies", [])]
    extras = [
        requirement_name(s)
        for group in project.get("optional-dependencies", {}).values()
        for s in group
    ]
    return runtime, extras


def requirements_in(text: str) -> list[str]:
    """Requirement names in a pip ``.in`` file (comments, blanks and options skipped)."""
    names = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.append(requirement_name(line))
    return names


def unlocked(declared: dict[str, list[str]], pins: set[str]) -> list[str]:
    """``source: name`` for every declared non-workspace name missing from the lock."""
    return sorted(
        f"{source}: {name}"
        for source, names in declared.items()
        for name in names
        if not is_workspace(name) and name not in pins
    )


def runtime_mismatch(runtime: list[str], runtime_deps: list[str]) -> tuple[set[str], set[str]]:
    """(missing from RUNTIME_DEPS, not in pyproject) for one package."""
    want = {normalize(n) for n in runtime if not is_workspace(n)}
    have = {normalize(n) for n in runtime_deps}
    return want - have, have - want


# ---------------------------------------------------------------------------
# The repository.
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _manifests() -> dict[str, list[str]]:
    """package path -> RUNTIME_DEPS."""
    out = {}
    for path in MANIFESTS:
        data = json.loads(path.read_text())
        out[data["package"]] = data["runtime_deps"]
    return out


def _declared_by_source() -> dict[str, list[str]]:
    lock_text = LOCK.read_text()
    declared = {}
    for name in lock_inputs(lock_text):
        path = Path(name)
        if path.name == "pyproject.toml":
            runtime, extras = pyproject_requirements(_load_toml(path))
            declared[name] = runtime + extras
        else:
            declared[name] = requirements_in(path.read_text())
    return declared


def test_every_lock_input_is_available():
    """Each file the lock is compiled from is in this test's data."""
    missing = [p for p in lock_inputs(LOCK.read_text()) if not Path(p).is_file()]
    assert not missing, f"add to //tools:test_deps_consistency data: {missing}"


def test_every_declared_requirement_is_locked():
    missing = unlocked(_declared_by_source(), lock_pins(LOCK.read_text()))
    assert not missing, (
        "declared but not pinned in requirements_lock.txt; run "
        f"`bazel run //tools:update_requirements_lock`: {missing}"
    )


def test_every_package_has_a_manifest():
    packages = {
        str(Path(p).parent)
        for p in lock_inputs(LOCK.read_text())
        if Path(p).name == "pyproject.toml"
    }
    assert packages == set(_manifests()), (
        "every pyproject.toml in the lock needs a deps.bzl RUNTIME_DEPS and a "
        "runtime_deps_manifest listed in MANIFESTS"
    )


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_runtime_deps_match_pyproject(manifest):
    data = json.loads(manifest.read_text())
    runtime, _ = pyproject_requirements(_load_toml(Path(data["package"]) / "pyproject.toml"))
    missing, extra = runtime_mismatch(runtime, data["runtime_deps"])
    assert not missing and not extra, (
        f"{data['package']}/deps.bzl RUNTIME_DEPS must equal [project].dependencies "
        f"minus ipu-* packages; missing {sorted(missing)}, not in pyproject {sorted(extra)}"
    )


# ---------------------------------------------------------------------------
# The checks themselves, on scratch input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ("click>=8.3.1", "click"),
        ("ml_dtypes~=0.5", "ml-dtypes"),
        ("Jinja2 != 3.0.0", "jinja2"),
        ("pkg[extra,other]>=1; python_version < '3.11'", "pkg"),
        ("zope.interface", "zope-interface"),
        ("ipu-as", "ipu-as"),
    ],
)
def test_requirement_name(spec, name):
    assert requirement_name(spec) == name


def test_unpinned_requirement_is_reported():
    pyproject = tomllib.loads(
        '[project]\ndependencies = ["numpy>=1", "requests>=2", "ipu-isa"]\n'
        '[project.optional-dependencies]\ndev = ["pytest"]\n'
    )
    runtime, extras = pyproject_requirements(pyproject)
    pins = lock_pins("numpy==2.2.6 \\\n    --hash=sha256:00\npytest==9.1.1\n")
    assert unlocked({"p/pyproject.toml": runtime + extras}, pins) == ["p/pyproject.toml: requests"]


def test_unpinned_tool_is_reported():
    names = requirements_in("# tools\nuv\n-c constraints.txt\nruff  # lint\n")
    assert unlocked({"tools/requirements-tools.in": names}, {"uv"}) == [
        "tools/requirements-tools.in: ruff"
    ]


def test_runtime_dep_missing_from_runtime_deps_is_reported():
    assert runtime_mismatch(["click", "lark", "ipu-isa"], ["click"]) == ({"lark"}, set())
    assert runtime_mismatch(["click"], ["click", "pytest"]) == (set(), {"pytest"})
    assert runtime_mismatch(["ml_dtypes"], ["ml-dtypes"]) == (set(), set())


def test_lock_inputs_parses_header():
    header = (
        "# This file was autogenerated by uv via the following command:\n"
        "#    uv pip compile a/pyproject.toml tools/x.in --extra dev -o lock.txt\n"
        "click==8.5.0\n"
    )
    assert lock_inputs(header) == ["a/pyproject.toml", "tools/x.in"]
