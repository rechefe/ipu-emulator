"""Every key the TUI understands, the command it runs and how it is documented.

:data:`KEY_BINDINGS` is the single source: each row names the keys it binds,
the command each key runs, and the text the footer and the help overlay show.
:data:`KEYMAP` is derived from it, and so is :func:`help_lines`, so a key
cannot be bound without being documented (``test_debug_tui`` checks that every
keymap entry appears in the help overlay).

Mouse gestures have rows too (they are documented) but bind no keys: pointer
events arrive as ``MouseEvent`` and are handled by position, not by keymap.
Typing into the tab editor is text input, not a command, so printable
characters, Backspace and Delete are not in the keymap either.
"""

from __future__ import annotations

import enum
import textwrap
from dataclasses import dataclass
from typing import Literal

from ipu_debug.tui.screen import Key

PaneId = Literal["pipeline", "disassembly", "xmem", "registers"]
PANE_FOCUS_ORDER: tuple[PaneId, ...] = ("pipeline", "disassembly", "xmem", "registers")


class Focus(enum.Enum):
    """Who receives a key: the overlay or editor when one is open, else a pane.

    Pane-focus lookups fall back to ``GLOBAL``; the help overlay and the
    editor are modal and do not.
    """

    GLOBAL = "global"
    PIPELINE = "pipeline"
    DISASSEMBLY = "disassembly"
    XMEM = "xmem"
    REGISTERS = "registers"
    EDITOR = "editor"
    HELP = "help"


PANE_FOCUS: dict[str, Focus] = {pane: Focus(pane) for pane in PANE_FOCUS_ORDER}


class Command(enum.Enum):
    """What a key does. ``CursesDebugView`` dispatches each to one method."""

    QUIT = enum.auto()
    CONTINUE = enum.auto()
    STEP = enum.auto()
    SHOW_HELP = enum.auto()
    RESET_SPLITS = enum.auto()
    NEXT_PANE = enum.auto()
    FOCUS_PIPELINE = enum.auto()
    FOCUS_DISASSEMBLY = enum.auto()
    FOCUS_XMEM = enum.auto()
    FOCUS_REGISTERS = enum.auto()
    SPLIT_LEFT = enum.auto()
    SPLIT_RIGHT = enum.auto()
    SPLIT_UP = enum.auto()
    SPLIT_DOWN = enum.auto()
    # The focused pane.
    ADD_TAB = enum.auto()
    EDIT_TAB = enum.auto()
    CLOSE_TAB = enum.auto()
    NEXT_TAB = enum.auto()
    PREVIOUS_TAB = enum.auto()
    CYCLE_FORMAT = enum.auto()
    EDIT_VALUE = enum.auto()
    TOGGLE_BREAKPOINT = enum.auto()
    RUN_TO_CURSOR = enum.auto()
    MOVE_UP = enum.auto()
    MOVE_DOWN = enum.auto()
    MOVE_LEFT = enum.auto()
    MOVE_RIGHT = enum.auto()
    PAGE_UP = enum.auto()
    PAGE_DOWN = enum.auto()
    JUMP_FIRST = enum.auto()
    JUMP_LAST = enum.auto()
    CANCEL = enum.auto()
    # The tab editor.
    EDITOR_LEFT = enum.auto()
    EDITOR_RIGHT = enum.auto()
    EDITOR_HOME = enum.auto()
    EDITOR_END = enum.auto()
    EDITOR_SUBMIT = enum.auto()
    # The help overlay.
    CLOSE_HELP = enum.auto()
    HELP_UP = enum.auto()
    HELP_DOWN = enum.auto()
    HELP_PAGE_UP = enum.auto()
    HELP_PAGE_DOWN = enum.auto()
    HELP_TOP = enum.auto()
    HELP_BOTTOM = enum.auto()


# Commands that act on the focused pane's own content; the view hands them to
# that pane's ``Pane.handle``.
PANE_COMMANDS = frozenset(
    {
        Command.CYCLE_FORMAT,
        Command.EDIT_VALUE,
        Command.MOVE_UP,
        Command.MOVE_DOWN,
        Command.MOVE_LEFT,
        Command.MOVE_RIGHT,
        Command.PAGE_UP,
        Command.PAGE_DOWN,
        Command.JUMP_FIRST,
        Command.JUMP_LAST,
    }
)

HelpScope = Literal["global", "pane", "editor", "help"]
Binding = tuple["str | Key", Command]


@dataclass(frozen=True)
class KeyBinding:
    """One documented control, shown in the footer and the help overlay.

    ``keys`` are the labels shown; ``binds`` are the keys this row puts in
    :data:`KEYMAP` and the command each runs, for every focus the row's
    ``scope`` (and, for pane rows, ``panes``) covers. ``also`` adds focuses
    beyond those.
    """

    keys: tuple[str, ...]
    scope: HelpScope
    description: str
    footer: str | None = None
    footer_rank: int = 3
    panes: tuple[PaneId, ...] = PANE_FOCUS_ORDER
    binds: tuple[Binding, ...] = ()
    also: tuple[Focus, ...] = ()

    @property
    def focuses(self) -> tuple[Focus, ...]:
        if self.scope == "global":
            base: tuple[Focus, ...] = (Focus.GLOBAL,)
        elif self.scope == "pane":
            base = tuple(PANE_FOCUS[pane] for pane in self.panes)
        elif self.scope == "editor":
            base = (Focus.EDITOR,)
        else:
            base = (Focus.HELP,)
        return base + self.also


HELP_SCOPE_TITLES: tuple[tuple[HelpScope, str], ...] = (
    ("global", "Execution and layout"),
    ("pane", "Focused pane"),
    ("editor", "Tab editor"),
    ("help", "This overlay"),
)

# The footer puts execution controls first. ``docs/content/debugging.md``
# documents the same controls and is synchronised by hand.
KEY_BINDINGS: tuple[KeyBinding, ...] = (
    KeyBinding(("F2",), "pane", "Add a tab to the focused pane", "F2:Add", 3,
               binds=((Key.F2, Command.ADD_TAB),)),
    KeyBinding(("F3",), "pane", "Edit the active tab", "F3:Edit", 3,
               binds=((Key.F3, Command.EDIT_TAB),)),
    KeyBinding(("F4",), "pane", "Close the active tab", "F4:Close", 3,
               binds=((Key.F4, Command.CLOSE_TAB),)),
    KeyBinding(("f",), "pane", "Cycle the active tab's format", "f:Format", 2,
               binds=(("f", Command.CYCLE_FORMAT),)),
    KeyBinding(
        ("d", "a"),
        "pane",
        "Select the next / previous tab",
        "a:Prev d:Next",
        3,
        binds=(("d", Command.NEXT_TAB), ("a", Command.PREVIOUS_TAB)),
    ),
    KeyBinding(
        ("Arrows",),
        "pane",
        "Move the cursor inside the focused pane",
        "Arrows:Move",
        2,
        binds=(
            (Key.UP, Command.MOVE_UP),
            (Key.DOWN, Command.MOVE_DOWN),
            (Key.LEFT, Command.MOVE_LEFT),
            (Key.RIGHT, Command.MOVE_RIGHT),
        ),
    ),
    KeyBinding(("Tab",), "pane", "Select the next tab", "Tab:Next", 2,
               binds=((Key.TAB, Command.NEXT_TAB),)),
    KeyBinding(
        ("PgUp", "PgDn"),
        "pane",
        "Move the cursor by one page",
        binds=((Key.PAGE_UP, Command.PAGE_UP), (Key.PAGE_DOWN, Command.PAGE_DOWN)),
    ),
    KeyBinding(
        ("Home", "End"),
        "pane",
        "Jump to the first / last value",
        binds=((Key.HOME, Command.JUMP_FIRST), (Key.END, Command.JUMP_LAST)),
    ),
    KeyBinding(("Click",), "pane", "Select panes, tabs, values, and footer actions"),
    KeyBinding(("Wheel",), "pane", "Scroll panes, help, or tabs under the pointer"),
    KeyBinding(("Shift-wheel",), "pane", "Move sideways within a pane"),
    KeyBinding(("Double-click",), "pane", "Edit the selected register or memory value"),
    KeyBinding(("Code gutter",), "pane", "Click to toggle an instruction breakpoint"),
    KeyBinding(("Drag",), "pane", "Drag a pane divider or scrollbar"),
    KeyBinding(("Shift-Tab",), "global", "Focus the next pane", "S-Tab:Pane",
               binds=((Key.SHIFT_TAB, Command.NEXT_PANE),)),
    KeyBinding(
        ("1-4",),
        "global",
        "Focus a pane directly, left to right",
        None,
        2,
        binds=(
            ("1", Command.FOCUS_PIPELINE),
            ("2", Command.FOCUS_DISASSEMBLY),
            ("3", Command.FOCUS_XMEM),
            ("4", Command.FOCUS_REGISTERS),
        ),
    ),
    KeyBinding(("F5",), "global", "Continue to the next breakpoint", "F5:Run", 0,
               binds=((Key.F5, Command.CONTINUE),)),
    KeyBinding(("F8",), "global", "Execute one instruction", "F8:Step", 0,
               binds=((Key.F8, Command.STEP),)),
    KeyBinding(
        ("F9",),
        "pane",
        "Disassembly: toggle breakpoint",
        "F9:Break",
        1,
        panes=("disassembly",),
        binds=((Key.F9, Command.TOGGLE_BREAKPOINT),),
    ),
    KeyBinding(
        ("F10",),
        "pane",
        "Disassembly: run to cursor or next stop",
        "F10:Until",
        1,
        panes=("disassembly",),
        binds=((Key.F10, Command.RUN_TO_CURSOR),),
    ),
    KeyBinding(
        ("e",),
        "pane",
        "Edit the selected value (PC in Disassembly)",
        "e:Value",
        1,
        panes=("disassembly", "registers", "pipeline", "xmem"),
        binds=(("e", Command.EDIT_VALUE),),
    ),
    KeyBinding(
        ("Shift-Left", "Shift-Right"),
        "global",
        "Move the focused row's split",
        "S-Arrows:Size",
        4,
        binds=((Key.SHIFT_LEFT, Command.SPLIT_LEFT), (Key.SHIFT_RIGHT, Command.SPLIT_RIGHT)),
    ),
    KeyBinding(
        ("Shift-Up", "Shift-Down"),
        "global",
        "Move the top / bottom split",
        footer_rank=4,
        binds=((Key.SHIFT_UP, Command.SPLIT_UP), (Key.SHIFT_DOWN, Command.SPLIT_DOWN)),
    ),
    KeyBinding(("=",), "global", "Reset every split", "=:Reset", 4,
               binds=(("=", Command.RESET_SPLITS),)),
    KeyBinding(("?",), "global", "Show this help overlay",
               binds=(("?", Command.SHOW_HELP),)),
    KeyBinding(("q",), "global", "Halt execution and exit the debugger", "q Quit", 0,
               binds=(("q", Command.QUIT),)),
    # Esc also leaves the disassembly pane's stage view.
    KeyBinding(("Esc",), "editor", "Cancel the editor",
               binds=((Key.ESCAPE, Command.CANCEL),), also=(Focus.DISASSEMBLY,)),
    KeyBinding(("Enter",), "editor", "Apply the edited value",
               binds=((Key.ENTER, Command.EDITOR_SUBMIT),)),
    KeyBinding(
        ("Left", "Right", "Home", "End"),
        "editor",
        "Move within the edited text",
        binds=(
            (Key.LEFT, Command.EDITOR_LEFT),
            (Key.RIGHT, Command.EDITOR_RIGHT),
            (Key.HOME, Command.EDITOR_HOME),
            (Key.END, Command.EDITOR_END),
        ),
    ),
    KeyBinding(
        ("Esc", "?", "Enter"),
        "help",
        "Close this overlay",
        binds=(
            (Key.ESCAPE, Command.CLOSE_HELP),
            ("?", Command.CLOSE_HELP),
            (Key.ENTER, Command.CLOSE_HELP),
        ),
    ),
    KeyBinding(
        ("Up", "Down", "PgUp", "PgDn"),
        "help",
        "Scroll this overlay",
        binds=(
            (Key.UP, Command.HELP_UP),
            (Key.DOWN, Command.HELP_DOWN),
            (Key.PAGE_UP, Command.HELP_PAGE_UP),
            (Key.PAGE_DOWN, Command.HELP_PAGE_DOWN),
            # Home and End are documented with the pane keys of the same name.
            (Key.HOME, Command.HELP_TOP),
            (Key.END, Command.HELP_BOTTOM),
        ),
    ),
)


def _build_keymap() -> dict[tuple[Focus, str | Key], Command]:
    keymap: dict[tuple[Focus, str | Key], Command] = {}
    for binding in KEY_BINDINGS:
        for focus in binding.focuses:
            for key, command in binding.binds:
                entry = (focus, key)
                if entry in keymap:
                    raise ValueError(f"{key!r} is bound twice for {focus.value}")
                keymap[entry] = command
    return keymap


KEYMAP: dict[tuple[Focus, str | Key], Command] = _build_keymap()


def lookup(focus: Focus, key: str | Key) -> Command | None:
    """The command *key* runs under *focus*, or None.

    Letters match either case. A pane focus falls back to ``GLOBAL``; the
    modal focuses (help, editor) do not.
    """
    candidates = [key]
    if isinstance(key, str) and key.lower() != key:
        candidates.append(key.lower())
    focuses = [focus] if focus in (Focus.HELP, Focus.EDITOR) else [focus, Focus.GLOBAL]
    for each_focus in focuses:
        for candidate in candidates:
            command = KEYMAP.get((each_focus, candidate))
            if command is not None:
                return command
    return None


def key_label(key: str | Key) -> str:
    """How the help overlay and footer name *key*."""
    return key.label if isinstance(key, Key) else key


def help_lines(width: int = 72) -> list[tuple[str, int]]:
    """Every documented binding as ``(text, kind)``, where kind 1 marks a heading.

    ``q`` is left out: the footer, always visible under the overlay, shows it.
    """
    lines: list[tuple[str, int]] = []
    key_width = max(len(" / ".join(binding.keys)) for binding in KEY_BINDINGS)
    for scope, title in HELP_SCOPE_TITLES:
        bindings = [
            binding
            for binding in KEY_BINDINGS
            if binding.scope == scope and binding.keys != ("q",)
        ]
        if not bindings:
            continue
        if lines:
            lines.append(("", 0))
        lines.append((title, 1))
        for binding in bindings:
            keys = " / ".join(binding.keys)
            if width >= 60:
                keys = keys.ljust(key_width)
            indent = " " * (key_width + 2)
            for line in textwrap.wrap(
                f"{keys}  {binding.description}",
                width=width,
                subsequent_indent=indent if width >= 60 else "  ",
            ):
                lines.append((line, 0))
    return lines


# Execution, navigation, view/editing, layout, exit: the footer's groups.
FOOTER_GROUPS = (
    ("F5", "F8", "F9", "F10"),
    ("Arrows", "Tab", "d", "Shift-Tab"),
    ("f", "e", "F2", "F3", "F4"),
    ("Shift-Left", "Shift-Up", "="),
    ("q",),
)


def footer_text(focus: str, width: int) -> str:
    """Fit whole shortcut labels for *focus*, most important first, never clipping."""
    candidates = [
        binding
        for binding in KEY_BINDINGS
        if binding.footer and binding.scope in ("global", "pane") and focus in binding.panes
    ]

    def format_bindings(selected: list[KeyBinding]) -> str:
        parts = []
        for group in FOOTER_GROUPS:
            ordered = sorted(
                (binding for binding in selected if binding.keys[0] in group),
                key=lambda binding: group.index(binding.keys[0]),
            )
            if ordered:
                parts.append(" ".join(binding.footer or "" for binding in ordered))
        return " " + " | ".join(parts)

    selected: list[KeyBinding] = []
    for binding in sorted(candidates, key=lambda item: item.footer_rank):
        if len(format_bindings([*selected, binding])) <= width:
            selected.append(binding)
    return format_bindings(selected)[:width]


def footer_actions(binding: KeyBinding) -> list[tuple[str, str | Key]]:
    """The ``(footer label, key)`` pairs a click on *binding*'s footer presses."""
    if binding.footer is None:
        return []
    if binding.keys == ("d", "a"):
        return [("a:Prev", "a"), ("d:Next", "d")]
    if binding.keys[0] in ("Arrows", "1-4") or binding.keys[0].startswith("Shift-") and (
        binding.keys[0] != "Shift-Tab"
    ):
        return []
    return [(binding.footer, binding.binds[0][0])]
