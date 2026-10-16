"""Third-party packages this package imports at runtime.

Must equal `[project].dependencies` in `pyproject.toml`, minus the workspace
packages (`ipu-*`); `//tools:deps_consistency_test` checks it. Test and docs
packages belong on the test and docs targets, not here.
"""

RUNTIME_DEPS = ["jinja2", "numpy"]
