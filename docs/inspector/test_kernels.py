"""curated.toml agrees with this package's kernel_trace targets, and kernel_list.py's checks."""

from __future__ import annotations

import re
from pathlib import Path

import kernel_list
import pytest
import tomli

HERE = Path(__file__).parent
CURATED = tomli.loads((HERE / "curated.toml").read_text(encoding="utf-8"))
BUILD = (HERE / "BUILD.bazel").read_text(encoding="utf-8")


def _trace(kernel="k", case="default"):
    return {
        "kernel": kernel,
        "case": case,
        "source_path": "k.asm",
        "meta": {"format": "int8"},
        "total_cycles": 3,
    }


def _entry(id_="a"):
    return {"id": id_, "group": "native", "title": "T", "label": "INT8", "description": "D"}


def test_curated_toml_lists_the_build_files_traces_in_order():
    curated = BUILD.split("CURATED = [", 1)[1].split("\n]\n", 1)[0]
    targets = re.findall(r'^    \{"name": "([^"]+)"', curated, re.M)
    ids = [entry["id"] for entry in CURATED["kernel"]]
    assert targets == ids, "BUILD.bazel's CURATED and curated.toml list the same ids in order"


def test_every_curated_entry_is_complete():
    trace_ids = {entry["id"]: _trace() for entry in CURATED["kernel"]}
    assert len(kernel_list.kernel_list(CURATED, trace_ids)) == len(CURATED["kernel"])


def test_the_list_adds_each_trace_facts():
    (entry,) = kernel_list.kernel_list({"kernel": [_entry()]}, {"a": _trace("identity")})
    assert entry == dict(
        _entry(),
        kernel="identity",
        case="default",
        source_path="k.asm",
        format="int8",
        total_cycles=3,
    )


@pytest.mark.parametrize(
    ("curated", "traces", "message"),
    [
        ({"kernel": [_entry()]}, {}, "no trace"),
        ({"kernel": []}, {"a": _trace()}, "not listed"),
        ({"kernel": [_entry(), _entry()]}, {"a": _trace()}, "duplicate"),
        ({"kernel": [dict(_entry(), group="other")]}, {"a": _trace()}, "group"),
        ({"kernel": [dict(_entry(), extra=1)]}, {"a": _trace()}, "unknown"),
    ],
)
def test_a_mismatch_fails(curated, traces, message):
    with pytest.raises(ValueError, match=message):
        kernel_list.kernel_list(curated, traces)
