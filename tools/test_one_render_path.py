"""ruff rejects a second assembly render path (E05-12).

ruff.toml bans `jinja2.Template` and `SandboxedEnvironment` everywhere but
`ipu_as/template.py`; the ruff_test targets then fail on any use. This test
plants each use in a file and checks the repository's config catches it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.ruff_runner import ruff_executable

CONFIG = Path("ruff.toml").resolve()


def _banned(tmp_path: Path, relative: str, source: str) -> str:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [ruff_executable(), "check", "--config", str(CONFIG), "--select", "TID251", "."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "import jinja2\n\njinja2.Template('x').render()\n",
        "from jinja2 import Template\n\nTemplate('x')\n",
        "from jinja2.sandbox import SandboxedEnvironment\n\nSandboxedEnvironment()\n",
    ],
)
def test_a_planted_render_fails_the_lint(tmp_path, source):
    assert "TID251" in _banned(tmp_path, "src/tools/x/test/test_x.py", source)


def test_the_template_module_may_render(tmp_path):
    source = "from jinja2.sandbox import SandboxedEnvironment\n\nSandboxedEnvironment()\n"
    assert "TID251" not in _banned(tmp_path, "src/tools/ipu-as-py/src/ipu_as/template.py", source)
