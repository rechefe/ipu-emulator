"""Third-party packages this package imports at runtime.

Must equal `[project].dependencies` in `pyproject.toml`, minus the workspace
packages (`ipu-*`). Test packages (pytest) belong on the test targets.
"""

RUNTIME_DEPS = ["numpy", "tomli"]
