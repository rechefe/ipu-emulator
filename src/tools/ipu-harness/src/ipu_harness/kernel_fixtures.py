"""Pytest fixtures of a kernel's target (E06-13): ``kernel``, ``cases``, ``run_case``.

Loaded by :mod:`ipu_harness.kernel_test` as a plugin, so the generated tests and
a kernel folder's own ``test_*.py`` share them, and one assembly cache: a
program renders and assembles once per distinct case values.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, expand
from ipu_harness.kernel_test import KERNEL_ENV, TIER_ENV
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.runner import RunOptions, RunResult, run_kernel
from ipu_isa.formats import ElementFormat

RunCase = Callable[..., RunResult]


def _manifest() -> KernelManifest:
    return load_manifest(Path(os.environ[KERNEL_ENV]))


def _tier_cases(manifest: KernelManifest) -> tuple[Case, ...]:
    tier = os.environ.get(TIER_ENV, "default")
    return tuple(case for case in expand(manifest) if case.tier == tier)


@pytest.fixture(scope="session")
def kernel() -> KernelManifest:
    """The kernel of this target."""
    return _manifest()


@pytest.fixture(scope="session")
def cases(kernel: KernelManifest) -> tuple[Case, ...]:
    """The target's cases: its tier's, sweeps expanded."""
    return _tier_cases(kernel)


@pytest.fixture(scope="session")
def run_case(kernel: KernelManifest) -> RunCase:
    """``run_case(case, fmt, options=None)``: :func:`run_kernel` with the shared cache."""
    cache = AssemblyCache()

    def run(case: Case, fmt: ElementFormat, options: RunOptions | None = None) -> RunResult:
        return run_kernel(kernel, case, fmt, options, cache=cache)

    return run


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """``case_format``: each case of the tier in each format it runs in."""
    if "case_format" not in metafunc.fixturenames:
        return
    pairs = [(case, fmt) for case in _tier_cases(_manifest()) for fmt in case.formats]
    metafunc.parametrize(
        "case_format", pairs, ids=[f"{case.name}-{fmt.value}" for case, fmt in pairs]
    )


# Bazel sharding (shard_count): the rules_python_pytest shim passes the
# pytest-shard plugin's options; keep every num_shards-th test from shard_id.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--shard-id", type=int, default=0)
    parser.addoption("--num-shards", type=int, default=1)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard, shards = config.getoption("--shard-id"), config.getoption("--num-shards")
    if shards > 1:
        items[:] = [item for index, item in enumerate(items) if index % shards == shard]
