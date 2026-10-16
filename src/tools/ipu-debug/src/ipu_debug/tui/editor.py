"""The editors panes open: a tab target, a value under the cursor, a scalar.

Each function returns the :class:`~ipu_debug.tui.pane.Reply` that opens one
:class:`~ipu_debug.tui.pane.Editor`; the editor's ``submit`` parses the text,
mutates through the given callables or the ``DebugSession``, and replies
with the status line text. The view's prompt (``ipu_debug.tui.prompt``) only
draws the editor and passes Enter on.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ipu_debug.session import DebugError, DebugSession
from ipu_debug.tui.keymap import PaneId
from ipu_debug.tui.pane import Editor, Reply
from ipu_debug.tui.tabs import TabSet
from ipu_emu.errors import EmulatorError

# Parses an editor's text into a tab, given the tab being edited (None when
# adding one): ``(tab, None)`` or ``(None, error text)``.
TabParser = Callable[[str, Any], tuple[Any, str | None]]

_SIGNED_FORMATS = ("int8", "int32")


@dataclass(frozen=True)
class TabTarget:
    """What one pane's tab editor asks for: its label, two hint lines, a parser."""

    label: str
    hints: tuple[str, str]
    parse: TabParser


@dataclass(frozen=True)
class EditedValue:
    """The value under a cursor: *size* bytes *raw* shown in format *fmt*."""

    label: str
    fmt: str
    raw: bytes
    size: int


def tab_editor(pane: PaneId, target: TabTarget, mode: str, tabs: TabSet[Any], text: str) -> Reply:
    """Open the editor that adds a tab to *tabs* or (``mode="edit"``) replaces the active one."""

    def submit(command: str) -> Reply:
        existing = tabs[tabs.active] if mode == "edit" and tabs else None
        tab, error = target.parse(command.strip(), existing)
        if error is not None or tab is None:
            return Reply(f"{pane} tab error: {error}", error=True)
        if mode == "add" or not tabs:
            tabs.add(tab)
        else:
            tabs.tabs[tabs.active] = tab
            tabs.select(tabs.active)
        action = "added" if mode == "add" else "updated"
        return Reply(f"{pane} tab {action}: {tab.title}")

    action = "Add" if mode == "add" else "Edit"
    editor = Editor(mode, pane, f"{action} {target.label} tab", target.hints, text, submit)
    return Reply(f"Enter a {pane} tab target; press Enter to apply or Esc to cancel", editor=editor)


def value_text(raw: bytes, fmt: str) -> str:
    """How the value editor first shows *raw* in format *fmt*."""
    if fmt == "f32":
        return repr(struct.unpack("<f", raw)[0])
    value = int.from_bytes(raw, "little", signed=fmt in _SIGNED_FORMATS)
    if fmt in ("hex", "cell16"):
        return hex(value)
    if fmt == "bits":
        return bin(value)
    return str(value)


def value_editor(
    session: DebugSession, pane: PaneId, value: EditedValue, write: Callable[[bytes], None]
) -> Reply:
    """Open the editor for *value*; *write* stores the new bytes."""
    label, fmt, size = value.label, value.fmt, value.size
    if len(value.raw) != size:
        return Reply("Selected value is out of range", error=True)

    def submit(command: str) -> Reply:
        try:
            if fmt == "f32":
                data = struct.pack("<f", float(command.strip()))
            else:
                try:
                    number = session.parse_int(command.strip())
                except DebugError:
                    raise ValueError("Enter a valid integer") from None
                data = number.to_bytes(size, "little", signed=fmt in _SIGNED_FORMATS)
            write(data)
        except (ValueError, OverflowError, struct.error, EmulatorError) as error:
            return Reply(f"Invalid value: {error}", error=True)
        return Reply(f"Updated {label}")

    hints = (
        f"Format: {fmt} / {size} bytes",
        "Enter a number; hex/binary prefixes accepted for integers",
    )
    editor = Editor("value", pane, f"Edit {label}", hints, value_text(value.raw, fmt), submit)
    return Reply(f"Editing {label}", editor=editor)


def scalar_editor(
    session: DebugSession,
    pane: PaneId,
    register: str,
    current: int,
    range_hint: str,
) -> Reply:
    """Open the editor that sets *register* through ``DebugSession.set_register``."""
    name = register.upper()

    def submit(command: str) -> Reply:
        try:
            value = session.parse_int(command.strip())
        except DebugError:
            return Reply("Invalid integer value", error=True)
        try:
            written = session.set_register(register, value)
        except DebugError as error:
            return Reply(str(error), error=True)
        return Reply(f"Set {name} = {written.value:#x}")

    hints = ("Integer value: decimal, 0x hex, or 0b binary", range_hint)
    editor = Editor(
        "scalar", pane, f"Edit {name} value", hints, hex(current), submit, register=register
    )
    return Reply(f"Enter an integer for {name}", editor=editor)
