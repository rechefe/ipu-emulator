"""Third-party packages this package imports at runtime.

Must equal `[project].dependencies` in `pyproject.toml`, minus the workspace
packages (`ipu-*`); `//tools:deps_consistency_test` checks it. Test packages
(torch, pytest) belong on the test targets, not here.
"""

RUNTIME_DEPS = ["numpy"]
