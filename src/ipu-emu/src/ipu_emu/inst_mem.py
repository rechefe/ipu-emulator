"""Instruction memory: frozen words, a version per entry and a write hook.

:class:`InstMem` is what ``IpuState.inst_mem`` holds. Every entry is either
``None`` (no instruction: an idle cycle) or a decoded word frozen as a
``MappingProxyType``, so a word fetched from IMEM cannot be edited in place.
Changing a word means writing a new one through ``inst_mem[pc] = word``, which
bumps that entry's version and tells every write observer. The cycle caches one
schedule per PC and drops it from that observer (E04-08), so a debugger edit
made at a break is never hidden behind a stale schedule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import overload

#: An instruction word as IMEM stores it: field name to value, read-only.
FrozenWord = Mapping[str, int]


class InstMem(Sequence[FrozenWord | None]):
    """Instruction memory of a fixed number of entries.

    Args:
        size: Number of entries, all empty (``None``) at first.
    """

    __slots__ = ("_next_key", "_observers", "_versions", "_words")

    def __init__(self, size: int) -> None:
        self._words: list[FrozenWord | None] = [None] * size
        self._versions: list[int] = [0] * size
        self._observers: dict[int, Callable[[int], None]] = {}
        self._next_key = 0

    def __len__(self) -> int:
        return len(self._words)

    @overload
    def __getitem__(self, index: int) -> FrozenWord | None: ...

    @overload
    def __getitem__(self, index: slice) -> list[FrozenWord | None]: ...

    def __getitem__(self, index):
        return self._words[index]

    def __iter__(self) -> Iterator[FrozenWord | None]:
        return iter(self._words)

    def __setitem__(self, pc: int, word: Mapping[str, int] | None) -> None:
        """Store *word* at *pc*, frozen, and tell the write observers.

        Args:
            pc: The entry; a negative index counts from the end, as for a list.
            word: A decoded instruction word (copied, then frozen), or ``None``
                to empty the entry.

        Raises:
            IndexError: *pc* is outside IMEM.
            TypeError: *pc* is not an integer (slices are not supported).
        """
        if not isinstance(pc, int):
            raise TypeError(f"IMEM index must be an integer, not {type(pc).__name__}")
        size = len(self._words)
        if pc < 0:
            pc += size
        if not 0 <= pc < size:
            raise IndexError(f"IMEM index {pc} out of range (size {size})")
        self._words[pc] = None if word is None else MappingProxyType(dict(word))
        self._versions[pc] += 1
        for observer in tuple(self._observers.values()):
            observer(pc)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstMem):
            return self._words == other._words
        if isinstance(other, (list, tuple)):
            return self._words == list(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # mutable, compared by value

    def __repr__(self) -> str:
        used = sum(word is not None for word in self._words)
        return f"InstMem(size={len(self._words)}, words={used})"

    def version(self, pc: int) -> int:
        """How many times entry *pc* has been written (0: never).

        Args:
            pc: The entry.

        Returns:
            The entry's write count.
        """
        return self._versions[pc]

    def add_write_observer(self, fn: Callable[[int], None]) -> Callable[[], None]:
        """Call ``fn(pc)`` after every write to IMEM.

        Args:
            fn: Called with the (non-negative) PC of each written entry.

        Returns:
            A function that removes the observer; calling it again does nothing.
        """
        key = self._next_key
        self._next_key += 1
        self._observers[key] = fn

        def remove() -> None:
            self._observers.pop(key, None)

        return remove
