"""The Cycle Inspector page reads the versions the docs build writes (E08 final review).

The page (docs/content/inspector/inspector.js) refuses a trace or kernel list
whose ``schemaVersion`` is not its own. Bumping a writer's version without the
page, or back, would pass every other test and break the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import kernel_list
import pytest

from ipu_gen import inspector_trace

VIEWER = Path("docs/content/inspector/inspector.js").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("constant", "version"),
    [
        ("VIEWER_SCHEMA", inspector_trace.SCHEMA_VERSION),
        ("KERNELS_SCHEMA", kernel_list.SCHEMA_VERSION),
    ],
)
def test_the_page_reads_the_version_the_build_writes(constant, version):
    match = re.search(rf"^const {constant} = (\d+);", VIEWER, re.M)
    assert match, f"inspector.js declares no {constant}"
    assert int(match.group(1)) == version
