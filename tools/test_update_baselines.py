"""``tools/update_baselines.sh`` runs every baseline update target.

Parses the script's target lists and the BUILD files of the packages that hold
baselines, and checks that the two agree: every ``update_*`` binary in those
packages is run by the script, and every target the script runs exists. A new
baseline with its own update target therefore fails here until the script runs
it.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path("tools/update_baselines.sh")

#: The packages with baseline files (docs/content/code-conventions.md, "Baselines").
BASELINE_PACKAGES = ("src/tools/ipu-apps", "src/tools/ipu-as-py", "src/tools/ipu-emu-py")

_LABEL = re.compile(r"//(?P<package>[\w./-]+):(?P<name>\w+)")
_UPDATE_RULE = re.compile(r"\bname\s*=\s*\"(?P<name>update_\w+)\"")


def _script_array(text: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\((?P<body>.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert match, f"{SCRIPT} has no {name}=( ... ) array"
    return match.group("body").split()


def _script_variable(text: str, name: str) -> str:
    match = re.search(rf"^{name}=(?P<value>\S+)$", text, re.MULTILINE)
    assert match, f"{SCRIPT} has no {name}= line"
    return match.group("value")


def script_targets() -> list[str]:
    text = SCRIPT.read_text(encoding="utf-8")
    return [*_script_array(text, "UPDATE_TARGETS"), _script_variable(text, "KERNEL_TARGET")]


def build_targets() -> set[str]:
    targets = set()
    for package in BASELINE_PACKAGES:
        build = Path(package, "BUILD.bazel").read_text(encoding="utf-8")
        targets |= {f"//{package}:{m['name']}" for m in _UPDATE_RULE.finditer(build)}
    return targets


def test_script_targets_are_labels_in_baseline_packages() -> None:
    for target in script_targets():
        match = _LABEL.fullmatch(target)
        assert match, f"{target!r} in {SCRIPT} is not a //package:name label"
        assert match["package"] in BASELINE_PACKAGES, target


def test_script_runs_each_target_once() -> None:
    targets = script_targets()
    assert len(targets) == len(set(targets)), targets


def test_script_runs_every_update_target() -> None:
    in_script = set(script_targets())
    in_build = build_targets()
    assert in_build, "no update_* targets found in the baseline packages"
    assert not in_build - in_script, f"update targets missing from {SCRIPT}"
    assert not in_script - in_build, f"targets in {SCRIPT} with no BUILD rule"
