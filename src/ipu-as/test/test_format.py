"""`ipu-as format` (E08-24): the editor's formatter's rules, run by the assembler.

The expected outputs are the editor's formatter's (vscode-ipu-asm/language.js
`format`), recorded once by vscode-ipu-asm/test/record-format.js into
test/baselines/format/: in full for the extension's corpus, as SHA-256 digests for
the kernels. The small cases below are the editor's own unit cases
(vscode-ipu-asm/test/language.js), each written out by hand.

A kernel's digests hold only for the kernel as recorded (its `input` digest).
An edited kernel is skipped, not failed, until it is recorded again:

    bazel run //vscode-ipu-asm:gen_vscode
    node vscode-ipu-asm/test/record-format.js

The recorder is not in tools/update_baselines.sh because Bazel has no node
toolchain here. Re-recording after a kernel edit changes only that kernel's
entry in kernels.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ipu_as.cli import cli
from ipu_as.errors import AssemblerError, FormatError
from ipu_as.formatter import check_same_program, format_source
from ipu_as.jinja_tags import tag_spans

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[3]
GOLDEN = ROOT / "src" / "ipu-as" / "test" / "baselines" / "format"
CORPUS = json.loads((GOLDEN / "corpus.json").read_text(encoding="utf-8"))
KERNELS = json.loads((GOLDEN / "kernels.json").read_text(encoding="utf-8"))

B = "    HALT;;"
WORDS = "top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n    HALT;;"
FOR_LOOP = "HALT;;\n{% for i in range(2) %}\nHALT;;\n{% endfor %}\nHALT;;"
CALL_AFTER = "    HALT;; {{ block(0) }}\n    HALT;;"
CALL_ALONE = f"{B}\n\n    {{{{ block(0) }}}}\n\n{B}"
TAG_AFTER = '    HALT;; {% set s = "a\nb" %}\n    HALT;;'
BL = {"blank_line": True}


# (input, options, output; None: unchanged)
@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("  top:\nSET lr0 cr0;;", {}, "top:\n    SET lr0 cr0;;"),
        ("\t\tHALT;;", {"indent": "\t"}, "\tHALT;;"),
        ("HALT;; HALT;;", {}, f"{B}\n{B}"),
        ("a: HALT;; b: HALT;;", {}, f"a:\n{B}\nb:\n{B}"),
        (WORDS, BL, "top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n\n    HALT;;"),
        (WORDS, {}, None),
        ("HALT;;\n\n\nHALT;;", BL, f"{B}\n\n\n{B}"),
        (
            '{%- set r = "lr0" -%}\n# note\n    MULT.VE {{r}}   cr1 0 {{r}} cr15.l cr14.l ;     '
            "{#- aligned -#}\n    ACC.ADD.FIRST ;; // done",
            {},
            None,
        ),
        ("    HALT;; # HALT;; more", {}, None),
        ("    HALT;; {# x;; y #}", {}, None),
        ("{%- macro m(a,\n      b) -%}\n{{ block(0) }}\n{% endmacro %}", {}, None),
        # A range formats its own lines, and sees the word before it.
        ("  a:\nHALT;;\nHALT;;", {"line_range": (1, 1)}, f"  a:\n{B}\nHALT;;"),
        ("HALT;;\nHALT;;\nHALT;;", {**BL, "line_range": (1, 2)}, f"HALT;;\n\n{B}\n\n{B}"),
        ("HALT;;\nHALT;;\nHALT;;", {**BL, "line_range": (1, 1)}, f"HALT;;\n\n{B}\nHALT;;"),
        ("top: HALT;; HALT;;  \r\nHALT;;\r\n", {}, f"top:\r\n{B}\r\n{B}\r\n{B}\r\n"),
        ("  top:  # c \r\n", {}, "top:  # c\r\n"),
        ("a: HALT;;\r\nHALT;; HALT;;\nHALT;;", {}, f"a:\r\n{B}\r\n{B}\n{B}\n{B}"),
        # A comment or Jinja block goes with the word after it; a closing tag with the one before.
        ("HALT;;\n# next\nHALT;;", BL, f"{B}\n\n# next\n{B}"),
        (FOR_LOOP, BL, f"{B}\n\n{{% for i in range(2) %}}\n{B}\n{{% endfor %}}\n\n{B}"),
        ("HALT;;\n\n# c\nHALT;;", BL, f"{B}\n\n# c\n{B}"),
        ("HALT;;\n{{ lbl(0) }}:\nHALT;;\nHALT;;", BL, f"{B}\n\n{{{{ lbl(0) }}}}:\n{B}\n\n{B}"),
        ("    SET lr0\n        cr0;;", {}, None),  # a continuation keeps its deeper indent
        ("SET lr0\ncr0;;", {}, "    SET lr0\n    cr0;;"),
        ("SET lr0 cr0;\n        HALT;;", {}, f"    SET lr0 cr0;\n{B}"),
        (CALL_AFTER, BL, CALL_ALONE),  # a macro call split onto its own line ends its word...
        (CALL_ALONE, BL, None),  # ...so formatting is idempotent
        (TAG_AFTER, BL, f'    HALT;; {{% set s = "a\nb" %}}\n\n{B}'),  # no gap inside a tag
    ],
)
def test_the_editors_cases(text, options, expected):
    assert format_source(text, **options) == (text if expected is None else expected)


# --- The recorded output of the editor's formatter ---


def _mangle(raw: str) -> str:
    """vscode-ipu-asm/test/mangle.js: indentation stripped and consecutive words joined
    onto one line, where no comment or Jinja on the line could swallow what follows."""
    spans = tag_spans(raw)
    out: list[list] = []
    at = 0
    for line in raw.split("\n"):
        start, end = at, at + len(line)
        at = end + 1
        jinja = any(s.start < max(end, start + 1) and s.end > start for s in spans)
        simple = bool(line.strip()) and not jinja and "#" not in line and "//" not in line
        prev = out[-1] if out else None
        if simple and prev and prev[1] and prev[0].rstrip().endswith(";;"):
            prev[0] = f"{prev[0].rstrip()} {line.strip()}"
        else:
            out.append([line.strip() if simple else line, simple])
    return "\n".join(text for text, _ in out)


def _options(variant: str, text: str) -> dict:
    """record-format.js `VARIANTS`: the range is the middle third of the lines."""
    n = len(text.split("\n"))
    middle = (n // 3, (2 * n) // 3)
    return {
        "default": {},
        "blank-line": {"blank_line": True},
        "indent-tab": {"indent": "\t"},
        "indent-2-range": {"indent": "  ", "line_range": middle},
        "blank-line-range": {"blank_line": True, "line_range": middle},
    }[variant]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cases(recorded: dict) -> list:
    return [
        pytest.param(name, key, id=f"{name}:{key}")
        for name, entry in recorded.items()
        for key in entry
        if key not in ("input", "mangled")
    ]


def _raw(name: str) -> str:
    """The file's text; a kernel edited since it was recorded is skipped."""
    raw = (ROOT / name).read_text(encoding="utf-8")
    if name in KERNELS and _sha(raw) != KERNELS[name]["input"]:
        pytest.skip(
            f"{name} changed since its format was recorded; "
            "re-record with `node vscode-ipu-asm/test/record-format.js` (see this module's docstring)"
        )
    return raw


def _input(name: str, key: str) -> str:
    raw = _raw(name)
    return _mangle(raw) if key.startswith("mangled/") else raw


def _variant(key: str) -> str:
    return key.removeprefix("mangled/")


#: Where record-format.js finds the kernels (its DEFAULT_DIRS).
KERNEL_DIRS = ("kernels",)


def test_every_kernel_is_recorded():
    # A kernel added, renamed or deleted since the recording: record again.
    found = {p.relative_to(ROOT).as_posix() for d in KERNEL_DIRS for p in (ROOT / d).rglob("*.asm")}
    assert found == set(KERNELS), "re-record with `node vscode-ipu-asm/test/record-format.js`"


@pytest.mark.parametrize("name", sorted(CORPUS) + sorted(KERNELS))
def test_the_stripped_copy_is_the_editors(name):
    raw = _raw(name)
    if name in CORPUS:
        assert _mangle(raw) == CORPUS[name]["mangled"]
    else:
        assert _sha(_mangle(raw)) == KERNELS[name]["mangled"]


@pytest.mark.parametrize(("name", "key"), _cases(CORPUS))
def test_the_corpus_formats_as_the_editor_formats_it(name, key):
    text = _input(name, key)
    assert format_source(text, **_options(_variant(key), text)) == CORPUS[name][key]


@pytest.mark.parametrize(("name", "key"), _cases(KERNELS))
def test_every_kernel_formats_as_the_editor_formats_it(name, key):
    text = _input(name, key)
    assert _sha(format_source(text, **_options(_variant(key), text))) == KERNELS[name][key]


@pytest.mark.parametrize("name", sorted(CORPUS) + sorted(KERNELS))
@pytest.mark.parametrize("options", [{}, BL, {"indent": "\t"}], ids=["default", "blank", "tab"])
def test_formatting_is_idempotent(name, options):
    raw = (ROOT / name).read_text(encoding="utf-8")
    for text in (raw, _mangle(raw)):
        once = format_source(text, **options)
        assert format_source(once, **options) == once


# --- The safety check ---


def test_a_change_inside_a_jinja_tag_is_refused():
    # The editor's rules take the first `}}` for the tag's end, so they split
    # at the `;;` inside the string; Jinja reads one tag, which would change.
    text = 'HALT;; {{ "}};; HALT" }}'
    with pytest.raises(FormatError, match="would change the program") as info:
        format_source(text)
    assert isinstance(info.value, AssemblerError)
    assert info.value.stage == "format"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("SET lr0 cr0;;", "SET lr0cr0;;"),  # two tokens joined
        ("HALT;;\n# c\nHALT;;", "HALT;;\n# c HALT;;"),  # code moved into a comment
        ("SET lr{{ i }} cr0;;", "SET lr {{ i }} cr0;;"),  # a token parted from its tag
        ("SET lr0 {{ x }};;", "SET lr0 {{x}};;"),  # a tag's own text
        ("SET lr0 cr0 ;\n;", "SET lr0 cr0 ;;"),  # `;` `;` is not `;;`
        ("HALT;;", "HALT;; HALT;;"),
    ],
)
def test_a_change_beyond_layout_is_refused(before, after):
    with pytest.raises(FormatError):
        check_same_program(before, after)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("a: HALT;;b:HALT;;", "a:\n    HALT;;\nb:\n    HALT;;"),  # after `:` and `;;`
        ("  HALT;;  # c  \n", "    HALT;;  # c\n"),  # trailing whitespace of a comment
        ("{% if a %}HALT;;{% endif %}", "{% if a %}HALT;;\n{% endif %}"),
        ("FOO lr0;; HALT;;", "FOO lr0;;\nHALT;;"),  # does not assemble: tokens only
    ],
)
def test_a_layout_change_is_allowed(before, after):
    check_same_program(before, after)


def test_a_source_that_does_not_assemble_is_still_formatted():
    assert format_source("FOO lr0;; HALT;;") == f"    FOO lr0;;\n{B}"


def test_a_bad_range_is_refused():
    with pytest.raises(ValueError, match="line range"):
        format_source("HALT;;", line_range=(2, 1))


# --- The command line ---


def _run(*args: str, stdin: str | None = None):
    return CliRunner().invoke(cli, ["format", *args], input=stdin)


def test_stdin_to_stdout():
    result = _run("--input", "-", stdin="a: HALT;; HALT;;")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"a:\n{B}\n{B}"


def test_check_exit_codes(tmp_path):
    messy = tmp_path / "messy.asm"
    messy.write_text("HALT;; HALT;;\n")
    tidy = tmp_path / "tidy.asm"
    tidy.write_text(f"{B}\n{B}\n")
    assert _run("--check", "--input", str(messy)).exit_code == 1
    assert _run("--check", "--input", str(tidy)).exit_code == 0
    assert messy.read_text() == "HALT;; HALT;;\n"  # --check writes nothing


def test_in_place_keeps_crlf(tmp_path):
    path = tmp_path / "k.asm"
    path.write_bytes(b"top: HALT;;\r\nHALT;;\r\n")
    assert _run("--in-place", "--input", str(path)).exit_code == 0
    assert path.read_bytes() == b"top:\r\n    HALT;;\r\n    HALT;;\r\n"


def test_options_reach_the_formatter():
    result = _run(
        "--indent", "tab", "--blank-line", "--range", "2:3", "--input", "-",
        stdin="HALT;;\nHALT;;\nHALT;;",
    )  # fmt: skip
    assert result.stdout == "HALT;;\n\n\tHALT;;\n\n\tHALT;;"
    assert _run("--indent", "2", "--input", "-", stdin="HALT;;").stdout == "  HALT;;"


@pytest.mark.parametrize(
    "args",
    [
        ["--range", "3:2", "--input", "-"],
        ["--range", "0:2", "--input", "-"],
        ["--indent", "x", "--input", "-"],
        ["--in-place", "--input", "-"],
    ],
)
def test_bad_options_are_usage_errors(args):
    assert _run(*args, stdin="HALT;;").exit_code == 2


@pytest.mark.parametrize("mode", ["--in-place", "--check"])
def test_a_format_error_exits_3_and_writes_nothing(tmp_path, mode):
    path = tmp_path / "k.asm"
    text = 'HALT;; {{ "}};; HALT" }}'
    path.write_text(text)
    result = _run(mode, "--input", str(path))
    assert result.exit_code == 3
    assert "would change the program" in result.stderr
    assert path.read_text() == text


def test_an_input_not_in_utf8_exits_3(tmp_path):
    path = tmp_path / "k.asm"
    path.write_bytes(b"HALT;; # \xff\n")
    result = _run("--input", str(path))
    assert result.exit_code == 3
    assert "Error formatting" in result.stderr
