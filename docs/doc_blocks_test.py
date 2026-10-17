"""Unit tests for docs.doc_blocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from docs.doc_blocks import DocBlockError, fenced_blocks


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "page.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_two_blocks_in_one_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "intro\n```asm\nNOP;;\n```\n\n```python\nx\n```\n```asm\nHALT;;\n```\n")
    blocks = fenced_blocks(path, "asm")
    assert [(b.line, b.body) for b in blocks] == [(2, "NOP;;\n"), (9, "HALT;;\n")]
    assert all(b.fragment_reason is None for b in blocks)


def test_indented_fence_in_list_item(tmp_path: Path) -> None:
    path = _write(tmp_path, "1. Step:\n\n    ```asm\n    loop:\n        NOP;;\n    ```\n")
    (block,) = fenced_blocks(path, "asm")
    assert block.body == "loop:\n    NOP;;\n"


def test_unclosed_fence_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "text\n```asm\nNOP;;\n")
    with pytest.raises(DocBlockError, match=r"page.md:2: fence has no closing line"):
        fenced_blocks(path, "asm")


def test_asm_fence_inside_other_fence_is_not_a_block(tmp_path: Path) -> None:
    path = _write(tmp_path, "~~~markdown\n```asm\nnot asm\n```\n~~~\n")
    assert fenced_blocks(path, "asm") == []


def test_fragment_marker(tmp_path: Path) -> None:
    path = _write(tmp_path, "<!-- asm-fragment: inner loop only -->\n```asm\n...\n```\n")
    (block,) = fenced_blocks(path, "asm")
    assert block.fragment_reason == "inner loop only"


@pytest.mark.parametrize("marker", ["<!-- asm-fragment -->", "<!-- asm-fragment:  -->"])
def test_fragment_marker_without_reason_is_an_error(tmp_path: Path, marker: str) -> None:
    path = _write(tmp_path, f"{marker}\n```asm\n...\n```\n")
    with pytest.raises(DocBlockError, match="needs a reason"):
        fenced_blocks(path, "asm")


@pytest.mark.parametrize(
    "text",
    [
        "<!-- asm-fragment: why -->\n\n```asm\nNOP;;\n```\n",
        "<!-- asm-fragment: why -->\n```text\nNOP;;\n```\n",
        "<!-- asm-fragment: why -->\n",
    ],
)
def test_fragment_marker_not_before_asm_fence_is_an_error(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(DocBlockError, match="not followed by"):
        fenced_blocks(path, "asm")


def test_kernel_marker(tmp_path: Path) -> None:
    path = _write(tmp_path, "<!-- asm-kernel: kernels/a/b -->\n```asm\nNOP;;\n```\n")
    (block,) = fenced_blocks(path, "asm")
    assert (block.kernel, block.fragment_reason) == ("kernels/a/b", None)


def test_kernel_marker_needs_a_folder(tmp_path: Path) -> None:
    path = _write(tmp_path, "<!-- asm-kernel -->\n```asm\nNOP;;\n```\n")
    with pytest.raises(DocBlockError, match="needs a kernel folder"):
        fenced_blocks(path, "asm")
