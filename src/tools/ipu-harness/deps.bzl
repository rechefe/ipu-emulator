"""Third-party packages this package imports at runtime.

Must equal `[project].dependencies` in `pyproject.toml`, minus the workspace
packages (`ipu-*`); `//tools:test_deps_consistency` checks it. Test packages
(torch, pytest, jsonschema) belong on the test targets, not here.
"""

RUNTIME_DEPS = ["numpy", "tomli"]
