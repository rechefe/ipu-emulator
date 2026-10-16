"""What the TUI keeps from one debugger stop to the next: tabs, focus, splits.

A :class:`DebugViewSession` belongs to whoever runs the debugger, which hands
it to every stop, so the tabs a user opened, the pane they focused, the
status line and the split sizes they chose all survive a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ipu_debug.session import DebugSession
from ipu_debug.tui.panes import disassembly as disassembly_pane
from ipu_debug.tui.panes import pipeline as pipeline_pane
from ipu_debug.tui.panes import registers as registers_pane
from ipu_debug.tui.panes import xmem as xmem_pane
from ipu_debug.tui.panes.disassembly import DisassemblyTab, DisassemblyTabs
from ipu_debug.tui.panes.pipeline import PipelineTab
from ipu_debug.tui.panes.registers import RegisterTab
from ipu_debug.tui.panes.xmem import XmemTab
from ipu_debug.tui.tabs import TabSet
from ipu_emu.ipu_state import IpuState


def _new_panes() -> dict[str, TabSet[Any]]:
    """One empty tab set per pane; closing a pane's last tab reopens a default
    one, except in XMEM, which may be empty."""
    return {
        "pipeline": TabSet(default=PipelineTab),
        "disassembly": DisassemblyTabs(default=DisassemblyTab),
        "xmem": TabSet(),
        "registers": TabSet(default=RegisterTab),
    }


@dataclass
class DebugViewSession:
    """State retained between debugger callbacks for one IPU state."""

    active: bool = False
    initialized: bool = False
    panes: dict[str, TabSet[Any]] = field(default_factory=_new_panes)
    focus: str = "xmem"
    message: str = "Ready"
    message_is_error: bool = False
    # Preferred split sizes survive temporary shrinking of the terminal.
    register_pane_columns: int | None = None
    pipeline_pane_columns: int | None = None
    top_pane_rows: int | None = None
    maximized: bool = False

    @classmethod
    def with_tabs(
        cls,
        *,
        xmem: list[XmemTab] | None = None,
        disassembly: list[DisassemblyTab] | None = None,
        registers: list[RegisterTab] | None = None,
        pipeline: list[PipelineTab] | None = None,
        **fields: Any,
    ) -> DebugViewSession:
        """A session whose panes start with the given tabs."""
        session = cls(**fields)
        for pane, tabs in (
            ("xmem", xmem),
            ("disassembly", disassembly),
            ("registers", registers),
            ("pipeline", pipeline),
        ):
            session.panes[pane].tabs.extend(tabs or [])
        return session

    # Typed views of ``panes``.
    @property
    def xmem(self) -> TabSet[XmemTab]:
        return self.panes["xmem"]

    @property
    def disassembly(self) -> DisassemblyTabs:
        tabs = self.panes["disassembly"]
        assert isinstance(tabs, DisassemblyTabs)
        return tabs

    @property
    def registers(self) -> TabSet[RegisterTab]:
        return self.panes["registers"]

    @property
    def pipeline(self) -> TabSet[PipelineTab]:
        return self.panes["pipeline"]

    def disassembly_tab(self) -> DisassemblyTab:
        """The active disassembly tab (the pane always has one)."""
        tab = self.disassembly.current()
        assert tab is not None
        return tab

    def register_tab(self) -> RegisterTab:
        """The active LR / CR tab (the pane always has one)."""
        tab = self.registers.current()
        assert tab is not None
        return tab

    def pipeline_tab(self) -> PipelineTab:
        """The active pipeline-register tab (the pane always has one)."""
        tab = self.pipeline.current()
        assert tab is not None
        return tab

    @property
    def stages_visible(self) -> bool:
        return self.disassembly_tab().fmt == "stages"

    @stages_visible.setter
    def stages_visible(self, visible: bool) -> None:
        self.disassembly_tab().fmt = "stages" if visible else "compact"

    def set_message(self, text: str, *, error: bool = False) -> None:
        self.message = text.replace("\n", " ").strip() or "Ready"
        self.message_is_error = error

    def ensure_default_tab(self, state: IpuState) -> None:
        fresh = not self.initialized
        if fresh and not self.xmem.tabs:
            self.xmem.tabs.extend(xmem_pane.default_tabs(state))
        if not self.disassembly.tabs:
            self.disassembly.tabs.extend(disassembly_pane.default_tabs(fresh))
        if not self.registers.tabs:
            self.registers.tabs.extend(registers_pane.default_tabs(fresh))
        if not self.pipeline.tabs:
            self.pipeline.tabs.extend(pipeline_pane.default_tabs(state, fresh))
        self.initialized = True

    def commit_baseline(self, state: IpuState) -> None:
        """Remember every tab's values, so the next stop marks what changed."""
        debug = DebugSession(state)
        registers_pane.commit_baselines(debug, self.registers)
        pipeline_pane.commit_baselines(debug, self.pipeline)
        xmem_pane.commit_baselines(debug, self.xmem)


def new_debug_view_session(state: IpuState) -> DebugViewSession:
    """A TUI session for *state*, with its default XMEM tab. Whoever runs the
    debugger keeps it and passes it to every stop, so the view persists."""
    session = DebugViewSession()
    session.ensure_default_tab(state)
    return session
