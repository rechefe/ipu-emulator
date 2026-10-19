"""Bazel test sharding for pytest, loaded with ``-p pytest_shard_plugin``.

With ``shard_count`` set, the rules_python_pytest shim passes ``--shard-id``
and ``--num-shards`` (the pytest-shard plugin's options); this plugin accepts
them and keeps every ``num_shards``-th collected test, starting at
``shard_id``. A kernel family's ``:tests`` loads it when the family sets
``tests_shard_count`` (``kernels/defs.bzl``). The timing probes need it because each probe builds a fresh
``IpuState``, whose 512 MiB XMEM costs about 0.25 s to allocate.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--shard-id", type=int, default=0)
    parser.addoption("--num-shards", type=int, default=1)


def pytest_collection_modifyitems(config, items):
    shard, shards = config.getoption("--shard-id"), config.getoption("--num-shards")
    if shards > 1:
        items[:] = [item for index, item in enumerate(items) if index % shards == shard]
