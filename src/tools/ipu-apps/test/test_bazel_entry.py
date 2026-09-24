"""The unified target preserves each command's arguments and exit status."""
import sys

import pytest

from ipu_apps.kernel_registry import bazel_entry, runner


@pytest.mark.parametrize("bazel_test", [False, True])
def test_run_selects_app_even_with_bazel_test_environment(monkeypatch, bazel_test):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", "/workspace")
    if bazel_test:
        monkeypatch.setenv("BAZEL_TEST", "1")
    else:
        monkeypatch.delenv("BAZEL_TEST", raising=False)
    calls = []
    monkeypatch.setattr(runner, "main", lambda args: calls.append(args) or 7)
    monkeypatch.setattr(bazel_entry.runpy, "run_path", lambda *a, **kw: pytest.fail("ran pytest"))
    assert bazel_entry.main(["identity", "test.py", "shim.py", "--case", "single_row"]) == 7
    assert calls == [["--kernel", "identity", "--case", "single_row"]]


@pytest.mark.parametrize("exit_code", [0, 1, 5])
def test_test_selects_pytest_and_preserves_status(monkeypatch, exit_code):
    monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)
    monkeypatch.setenv("BAZEL_TEST", "1")
    monkeypatch.setenv("XML_OUTPUT_FILE", "/tmp/results.xml")
    monkeypatch.setenv("TESTBRIDGE_TEST_ONLY", "single_row")
    previous = sys.argv
    monkeypatch.setattr(runner, "main", lambda *a: pytest.fail("ran app"))

    def run_shim(path, *, run_name):
        assert path == "shim.py" and run_name == "__main__"
        assert sys.argv == ["shim.py", "--capture=no", "test.py", "-v"]
        raise SystemExit(exit_code)

    monkeypatch.setattr(bazel_entry.runpy, "run_path", run_shim)
    with pytest.raises(SystemExit) as exc:
        bazel_entry.main(["identity", "test.py", "shim.py", "-v"])
    assert exc.value.code == exit_code
    assert sys.argv is previous


@pytest.mark.parametrize("cancel", [False, True])
def test_debug_terminal_restored_on_completion_and_cancellation(monkeypatch, cancel):
    from io import StringIO

    class Stream(StringIO):
        def __init__(self, fd, tty):
            super().__init__()
            self.fd, self.tty = fd, tty

        def fileno(self):
            return self.fd

        def isatty(self):
            return self.tty

    monkeypatch.setenv("IPU_DEBUG_TUI", "1")
    monkeypatch.setattr(sys, "stdin", Stream(0, True))
    monkeypatch.setattr(sys, "stdout", Stream(1, False))
    foreground = [200]
    descriptors = {0: "terminal", 1: "pipe"}
    signals = []
    monkeypatch.setattr(bazel_entry.os, "tcgetpgrp", lambda fd: foreground[0])
    monkeypatch.setattr(bazel_entry.os, "tcsetpgrp", lambda fd, group: foreground.__setitem__(0, group))
    monkeypatch.setattr(bazel_entry.os, "getpgrp", lambda: 100)
    monkeypatch.setattr(bazel_entry.os, "dup", lambda fd: descriptors.__setitem__(77, descriptors[fd]) or 77)
    monkeypatch.setattr(bazel_entry.os, "dup2", lambda src, dst: descriptors.__setitem__(dst, descriptors[src]))
    monkeypatch.setattr(bazel_entry.os, "close", lambda fd: descriptors.pop(fd))
    monkeypatch.setattr(bazel_entry.signal, "signal", lambda sig, handler: signals.append(handler) or "original")

    try:
        with bazel_entry.app_terminal():
            assert foreground[0] == 100
            assert descriptors[1] == "terminal"
            if cancel:
                raise SystemExit(0)
    except SystemExit as exc:
        assert cancel and exc.code == 0
    assert foreground[0] == 200
    assert descriptors == {0: "terminal", 1: "pipe"}
    assert signals == [bazel_entry.signal.SIG_IGN, "original"]


def test_noninteractive_debug_does_not_reopen_a_terminal(monkeypatch):
    from io import StringIO

    monkeypatch.setenv("IPU_DEBUG_TUI", "1")
    monkeypatch.setattr(sys, "stdin", StringIO())
    monkeypatch.setattr(bazel_entry.os, "dup", lambda fd: pytest.fail("redirected noninteractive stream"))
    with bazel_entry.app_terminal():
        assert not sys.stdin.isatty()
