"""``import/<package>``: wall time of ``python -c "import <module>"``, best of 5.

Each package is measured through the module its users import first: a bare
``import ipu_as`` or ``import ipu_apps`` loads an empty ``__init__``.
"""

from __future__ import annotations

from ipu_bench.isolated import python_wall_s
from ipu_bench.result import Result, tiny_budget

# Result name -> the module imported.
MODULES = {
    "import/ipu_as": "ipu_as.lark_tree",
    "import/ipu_emu": "ipu_emu",
    "import/ipu_apps": "ipu_apps.kernel_registry.registry",
}


def run(quick: bool) -> list[Result]:
    repeats = 1 if tiny_budget() else 2 if quick else 5
    startup = min(python_wall_s("pass") for _ in range(repeats))
    results = []
    for name, module in MODULES.items():
        best = min(python_wall_s(f"import {module}") for _ in range(repeats))
        results.append(
            Result(
                name,
                "s",
                best,
                extra=f"import {module}; best of {repeats}; interpreter startup {startup:.3f} s",
            )
        )
    return results
