"""Third-party packages this package imports at runtime.

Must equal `[project].dependencies` in `pyproject.toml`, minus the workspace
packages (`ipu-*`); `//tools:test_deps_consistency` checks it. Test and docs
packages belong on the test and docs targets, not here.
"""

RUNTIME_DEPS = ["numpy"]
