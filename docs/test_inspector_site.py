"""The built docs site serves the Cycle Inspector (E08-17).

Extracts ``site.tar`` and checks the page, its script and stylesheet, the
kernel list and every trace the list names, plus the ISA reference: each
exists, and each JSON file parses. Tagged ``docs``: it builds the site.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pytest

from ipu_gen import inspector_trace

SITE = Path("docs/site.tar")


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("site")
    with tarfile.open(SITE) as tar:
        # Our own build output; "data" refuses absolute paths and links out.
        tar.extractall(root, filter="data")
    return root


def _json(path: Path):
    assert path.is_file(), f"missing {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_page_and_its_assets_are_there(site):
    page = (site / "how-to" / "inspector" / "index.html").read_text(encoding="utf-8")
    assert "inspector.js" in page and "inspector.css" in page
    assert (site / "how-to" / "inspector" / "inspector.js").is_file()
    assert (site / "how-to" / "inspector" / "inspector.css").is_file()


def test_the_page_makes_no_third_party_request(site):
    for name in ("index.html", "inspector.css", "inspector.js"):
        text = (site / "how-to" / "inspector" / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f"{name} names an external URL"


def test_every_listed_trace_exists_and_parses(site):
    data = site / "how-to" / "inspector" / "data"
    kernels = _json(data / "kernels.json")["kernels"]
    assert kernels
    for entry in kernels:
        trace = _json(data / f"{entry['id']}.json")
        assert trace["schemaVersion"] == inspector_trace.SCHEMA_VERSION
        assert trace["kernel"] == entry["kernel"]


def test_the_isa_reference_is_there(site):
    isa = _json(site / "how-to" / "inspector" / "data" / "isa.json")
    assert isa["instructions"] and isa["slotOrder"]


def test_the_intro_page_links_the_viewer(site):
    page = (site / "how-to" / "step-through-a-kernel-run" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "inspector/" in page
