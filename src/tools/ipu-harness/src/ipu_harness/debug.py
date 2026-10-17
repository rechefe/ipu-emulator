"""Debugging and alias profiling as options of a run (E06-12).

:class:`DebugLaunch` says a run stops in the debugger; only the command line
(``ipu_harness.cli``) builds one, from ``--debug`` or the debug environment
variable. The harness never reads the environment itself, and imports the
debugger (``ipu_debug``, and with it curses) only when a launch asks for its
hook.

The reference-model calls these options need are reached through
:func:`debug_run` and :func:`attach_alias_profile` only: the tool escape hatch
of the reference-model API (``ReferenceModel.debug_run``) and its profiler
attachment, API 1.2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunResult

if TYPE_CHECKING:
    from ipu_emu.alias_profile import AliasProfile
    from ipu_emu.emulator import DebugCallback
    from ipu_emu.run_control import RunControl

__all__ = [
    "DebugHook",
    "DebugLaunch",
    "DebugOutcome",
    "attach_alias_profile",
    "debug_run",
    "new_alias_profile",
]


@dataclass(frozen=True)
class DebugHook:
    """What the debugger hands a run: the callback it stops in, and the control
    its breakpoints and run-to targets live in (the run honours only that one).
    """

    callback: DebugCallback
    control: RunControl | None = None


#: Makes a kernel's debug hook; the default opens the TUI.
HookFactory = Callable[[str], DebugHook]


def _tui_hook(kernel: str) -> DebugHook:
    from ipu_debug.launch import make_tui_debug_callback  # curses: only when debugging

    try:
        callback, control = make_tui_debug_callback(kernel)
    except RuntimeError as error:  # no terminal
        raise ConfigurationError(f"--debug needs an interactive terminal: {error}") from None
    return DebugHook(callback, control)


@dataclass(frozen=True)
class DebugLaunch:
    """A run that stops in the debugger.

    Attributes:
        break_on_entry: Stop before the first word, as a debug launch does.
        factory: Makes the hook for a kernel's name; the TUI by default
            (a test passes a fake).
    """

    break_on_entry: bool = True
    factory: HookFactory = _tui_hook

    def hook(self, kernel: str) -> DebugHook:
        """The debugger hook for ``kernel``'s run."""
        return self.factory(kernel)


@dataclass(frozen=True)
class DebugOutcome:
    """How a debug run ended.

    Attributes:
        run: The model's result, unless the user quit.
        quit: The user quit the debugger (or pressed Ctrl-C): the run has
            no result.
    """

    run: RunResult | None
    quit: bool


#: The API version that has the calls below.
_API = "the reference-model API 1.2.0"


def debug_run(
    model: ReferenceModel, hook: DebugHook, *, break_on_entry: bool, max_cycles: int | None
) -> DebugOutcome:
    """Run ``model`` to ``HALT``, a fault or a quit, stopping in ``hook``.

    Raises:
        ConfigurationError: The model has no ``debug_run`` (before API 1.2.0).
    """
    run = getattr(model, "debug_run", None)
    if run is None:
        raise ConfigurationError(f"--debug needs ReferenceModel.debug_run ({_API})")
    try:
        result: RunResult = run(
            hook.callback,
            control=hook.control,
            break_on_entry=break_on_entry,
            max_cycles=max_cycles,
        )
    except KeyboardInterrupt:
        return DebugOutcome(None, quit=True)
    quit_requested = bool(getattr(result, "quit", False)) or result.status.state.value == "quit"
    return DebugOutcome(None if quit_requested else result, quit=quit_requested)


def new_alias_profile(**metadata: str) -> AliasProfile:
    """An empty ``AliasProfile`` for one run, its ``metadata`` naming the run."""
    from ipu_emu.alias_profile import AliasProfile  # the profile the 1.2.0 API attaches

    return AliasProfile(metadata=dict(metadata))


def attach_alias_profile(model: ReferenceModel, profile: AliasProfile) -> None:
    """Attach ``profile`` to ``model`` before its run.

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
