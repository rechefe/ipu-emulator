"""Tests import what Bazel declares: no test edits `sys.path` to find a module.

A shared fixture is a `py_library` (the MobileViT ones are
`:mobilevit_fixtures`), so Bazel knows it is a dependency.
"""

from pathlib import Path

TESTS = Path(__file__).parent


def test_no_test_inserts_into_sys_path():
    scanned = sorted(TESTS.rglob("*.py"))
    assert len(scanned) > 10, "the tests are this target's data"
    offenders = [
        f"{path.relative_to(TESTS)}:{number}"
        for path in scanned
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "sys.path.insert" in line and path.name != Path(__file__).name
    ]
    assert not offenders, (
        f"sys.path.insert in {offenders}; declare the module as a py_library instead"
    )
