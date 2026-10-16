"""One pane's tabs: the list, the active tab and how far the tab bar scrolled."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

# How many tabs stay visible to the left of the active one when the tab bar
# scrolls to show it.
VISIBLE_TAB_LOOKBEHIND = 2


@dataclass
class TabSet(Generic[T]):
    """The tabs of one pane.

    ``active`` indexes the selected tab and ``offset`` the first tab the tab
    bar shows. *default* (when given) makes the tab that replaces the last one
    closed, so such a pane is never empty.
    """

    tabs: list[T] = field(default_factory=list)
    active: int = 0
    offset: int = 0
    default: Callable[[], T] | None = None

    def __len__(self) -> int:
        return len(self.tabs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.tabs)

    def __getitem__(self, index: int) -> T:
        return self.tabs[index]

    def add(self, tab: T) -> None:
        """Append *tab* and select it."""
        self.tabs.append(tab)
        self.select(len(self.tabs) - 1)

    def replace(self, tab: T) -> None:
        """Put *tab* in place of the selected tab (or add it to an empty set)."""
        if not self.tabs:
            self.add(tab)
            return
        self.tabs[self._clamped_active()] = tab

    def close(self, index: int) -> None:
        """Remove tab *index*; the tab now at that position (or the last) is selected."""
        if not 0 <= index < len(self.tabs):
            return
        del self.tabs[index]
        if not self.tabs and self.default is not None:
            self.tabs.append(self.default())
        self.active = min(index, max(0, len(self.tabs) - 1))
        self.offset = min(self.offset, max(0, len(self.tabs) - 1))

    def select(self, index: int) -> None:
        """Make tab *index* the active one and scroll the tab bar to it."""
        self.active = index
        if index < self.offset:
            self.offset = index
        elif index > self.offset + VISIBLE_TAB_LOOKBEHIND:
            self.offset = max(0, index - VISIBLE_TAB_LOOKBEHIND)

    def step(self, delta: int) -> T | None:
        """Select the tab *delta* places on, wrapping around; None when empty."""
        if not self.tabs:
            return None
        self.select((self.active + delta) % len(self.tabs))
        return self.tabs[self.active]

    def current(self) -> T | None:
        """The selected tab, or None when there are no tabs."""
        if not self.tabs:
            return None
        return self.tabs[self._clamped_active()]

    def _clamped_active(self) -> int:
        self.active = min(self.active, len(self.tabs) - 1)
        return self.active
