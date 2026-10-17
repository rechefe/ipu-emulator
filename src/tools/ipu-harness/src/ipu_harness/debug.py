"""Debugging and alias profiling as options of a run (E06-12).

:class:`DebugLaunch` says a run stops in the debugger; only the command line
(``ipu_harness.cli``) builds one, from ``--debug`` or the debug environment
variable.
The harness never reads the environment itself, and imports the debugger
(``ipu_debug``, and with it curses) only when a launch asks for its callback.

The reference-model calls these options need are reached through
:func:`debug_run` and :func:`attach_alias_profile` only: the tool escape hatch
of the reference-model API (``ReferenceModel.debug_run``) and its profiler
attachment, API 1.2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunResult

__all__ = [
    "DebugCallback",
    "DebugLaunch",
    "attach_alias_profile",
    "debug_run",
    "new_alias_profile",
]

#: The debugger's callback (``ipu_debug.launch``): called when the run stops,
#: it returns what the run does next. The harness only passes it on.
DebugCallback = Callable[..., Any]

#: Makes a kernel's debugger callback; the default opens the TUI.
CallbackFactory = Callable[[str], DebugCallback]


def _tui_callback(kernel: str) -> DebugCallback:
    from ipu_debug.launch import make_tui_debug_callback  # curses: only when debugging

    callback, _control = make_tui_debug_callback(kernel)
    return callback


@dataclass(frozen=True)
class DebugLaunch:
    """A run that stops in the debugger.

    Attributes:
        break_on_entry: Stop before the first word, as a debug launch does.
        factory: Makes the callback for a kernel's name; the TUI by default
            (a test passes a fake).
    """

    break_on_entry: bool = True
    factory: CallbackFactory = _tui_callback

    def callback(self, kernel: str) -> DebugCallback:
        """The debugger callback for ``kernel``'s run."""
        return self.factory(kernel)


#: The API version that has the calls below.
_API = "the reference-model API 1.2.0"


def debug_run(model: ReferenceModel, callback: DebugCallback, *, break_on_entry: bool) -> RunResult:
    """Run ``model`` to ``HALT`` or a fault, stopping in ``callback``.

    Raises:
        ConfigurationError: The model has no ``debug_run`` (before API 1.2.0).
    """
    run = getattr(model, "debug_run", None)
    if run is None:
        raise ConfigurationError(f"--debug needs ReferenceModel.debug_run ({_API})")
    result: RunResult = run(callback, break_on_entry=break_on_entry)
    return result


def new_alias_profile(case: str) -> Any:
    """An empty ``AliasProfile`` for one run, named after its case."""
    from ipu_emu.alias_profile import AliasProfile  # the profile the 1.2.0 API attaches

    return AliasProfile(metadata={"case": case})


def attach_alias_profile(model: ReferenceModel, profile: object) -> None:
    """Attach ``profile`` (an ``AliasProfile``) to ``model`` before its run.

    Raises:
        ConfigurationError: The model has no ``attach_alias_profile`` (before
            API 1.2.0).
    """
    attach = getattr(model, "attach_alias_profile", None)
    if attach is None:
        raise ConfigurationError(
            f"--profile-aliases needs ReferenceModel.attach_alias_profile ({_API})"
        )
    attach(profile)
