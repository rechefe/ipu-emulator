"""Identity case regression tests."""
import pytest

from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.kernel_registry.identity.cases import CASES


@pytest.mark.parametrize("name", CASES)
def test_identity(name):
    _, cycles = run_case("identity", CASES[name])
    assert cycles > 0
