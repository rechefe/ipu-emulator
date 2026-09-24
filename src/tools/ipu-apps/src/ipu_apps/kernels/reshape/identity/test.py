"""Identity case regression tests."""
from ipu_apps.kernel_registry import resolve
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__)


def test_registry_example_loads_runs_and_reads_memory():
    """The built-in boilerplate is executable, not only a discoverable spec."""
    verdict = resolve("identity", shape=(3, 128))
    assert verdict.supported and verdict.app_name == "identity"
    _, cycles = run_case(verdict.app_name, load_cases(verdict.app_name)["default"])
    assert cycles > 0
