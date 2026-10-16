"""Pytest helpers for kernel ``test.py`` files; imported only by tests.

Every kernel's ``test.py`` starts with::

    from ipu_apps.kernel_registry.testing import case_tests

    test_case = case_tests(__package__)

and adds only the tests that go beyond "each case runs and passes its check".
A sweep of the default case over other sizes is declared, not written::

    test_case = case_tests(__package__, sweep=[dict(rows=8, n=200), ...])

A kernel whose program contains an emulator-only instruction (``STR_ACC_REG``)
must emit ``DebugOnlyInstructionWarning`` in every case; any other kernel fails
if the warning appears. Tests that run kernels outside ``case_tests`` say which
they expect with :func:`debug_only_warning`.

After its check passes, every case is also compared with its record in the
kernel's ``baseline.json`` (see :mod:`ipu_apps.kernel_registry.baseline`).
"""

from contextlib import contextmanager
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import pytest

from ipu_apps.kernel_registry import baseline
from ipu_apps.kernel_registry.cases import (
    assemble_kernel,
    load_cases,
    options_label,
    package_kernel,
    run_case,
)
from ipu_emu.errors import DebugOnlyInstructionWarning
from ipu_emu.execute import emulator_only_instructions, load_binary_instructions


def uses_debug_only_instructions(inst_path) -> bool:
    """Whether an assembled program has a non-NOP emulator-only slot.

    Reads the instruction words, not the source, so it survives mnemonic renames.
    """
    words = load_binary_instructions(Path(inst_path).read_bytes())
    return bool(emulator_only_instructions(words))


@cache
def kernel_uses_debug_only_instructions(kernel: str) -> bool:
    """:func:`uses_debug_only_instructions` for a registered kernel's program."""
    with TemporaryDirectory(prefix="ipu-debug-only-") as tmp:
        return uses_debug_only_instructions(assemble_kernel(kernel, tmp))


@contextmanager
def debug_only_warning(expected: bool):
    """Assert ``DebugOnlyInstructionWarning`` fires inside the block, or never does.

    ``expected`` true wraps the block in ``pytest.warns``; false turns the
    warning into an error, so an unintended emulator-only instruction fails.
    """
    if expected:
        with pytest.warns(DebugOnlyInstructionWarning):
            yield
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DebugOnlyInstructionWarning)
            yield


def case_tests(package, sweep=()):
    """A pytest test running every case of the kernel declared in ``package``.

    ``sweep`` adds runs of the default case with option overrides -- dicts,
    like a benchmark's ``CONFIGS`` -- each checked by the case exactly as the
    named cases are. The kernel is assembled once and reused. ``run_case``
    raises if the kernel does not halt or a check fails. Each run asserts the
    ``DebugOnlyInstructionWarning`` exactly when the assembled program uses an
    emulator-only instruction (see :func:`debug_only_warning`), and is compared
    with its baseline record (see :func:`check_baseline`).
    """
    kernel = package_kernel(package)
    cases = load_cases(kernel)
    params = [(name, {}) for name in cases]
    ids = list(cases)
    for options in sweep:
        unknown = set(options) - set(cases["default"].defaults)
        if unknown:
            raise ValueError(
                f"{kernel}: sweep options {sorted(unknown)} are not options "
                f"of its default case {sorted(cases['default'].defaults)}"
            )
        params.append(("default", dict(options)))
        ids.append(options_label(options))
    assembled = []
    updated: dict = {}

    @pytest.mark.parametrize("case,options", params, ids=[f"{kernel}-{i}" for i in ids])
    def test_case(case, options, tmp_path_factory):
        if not assembled:
            inst_path = assemble_kernel(kernel, tmp_path_factory.mktemp(kernel))
            assembled.append((inst_path, uses_debug_only_instructions(inst_path)))
        inst_path, debug_only = assembled[0]
        recorder = baseline.Recorder()
        with debug_only_warning(debug_only):
            run_case(
                kernel,
                load_cases(kernel)[case],
                options=options,
                inst_path=inst_path,
                recorder=recorder,
            )
        case_id = options_label(options) if options else case
        check_baseline(kernel, case_id, recorder.record, updated)

    return test_case


def check_baseline(kernel, case_id, record, updated) -> None:
    """Compare a case's run with its baseline record, or, under the update
    command, collect it into ``updated`` and rewrite the kernel's file."""
    root = baseline.update_root()
    if root is None:
        baseline.check(kernel, case_id, record)
        return
    updated[case_id] = record
    baseline.write(baseline.source_path(root, kernel), updated)
