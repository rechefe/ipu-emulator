"""IPU register file — fully generated from the descriptor schema.

Every named accessor (``get_lr``, ``set_r_acc_bytes``, ``get_r_cyclic_at``, …)
is generated at init-time from ``REGFILE_SCHEMA`` — no hand-coded per-register
methods exist.  The generation rules are driven by four descriptor criteria:

1. **is_vector** (``desc.is_vector``):
   ``False`` → scalar integer register (LR, CR).
   ``True``  → byte-blob register (R, R_ACC, …).

2. **cyclic** (``desc.cyclic``):
   Generates wrapping ``get_{name}_at`` / ``set_{name}_at`` accessors.

3. **size_bytes** (``desc.size_bytes``):
   Width of each element in bytes (same concept whether scalar or vector).

4. **word_view** (``desc.word_view``):
   Adds uint32 word-level accessors and uses a ``_bytes`` suffix for
   the byte-blob accessor to disambiguate.

Additionally, ``desc.count`` determines whether the byte-blob accessor
takes an index argument (multi-element) or not (single-element).
"""

from __future__ import annotations

import struct
import warnings
import weakref
from typing import Any, Callable

import numpy as np

from ipu_emu.descriptors import REGFILE_SCHEMA, RegDescriptor, RegDtype, RegKind
from ipu_emu.errors import (
    EmulatorError,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
)
from ipu_emu.ipu_config import (
    CR_READ_ONLY_INITIAL_VALUES,
    CR_REGISTER_NAME,
    LR_CR_SCALAR_VALUE_MASK,
    REGISTER_WORD_VALUE_MASK,
)


# -- input validation ------------------------------------------------------
# Real exceptions, not ``assert``: these checks must hold under ``python -O``.
# Without them a negative index addresses the end of the backing bytearray and
# a wrong-length slice assignment resizes a register's storage.

def _check_index(name: str, index: int, count: int, *, unit: str | None = None) -> None:
    if not 0 <= index < count:
        bound = f"{count} {unit}" if unit else f"count={count}"
        raise RegisterIndexError(f"{name}[{index}] out of range ({bound})")


def _check_size(name: str, data: bytes | bytearray, size: int) -> None:
    if len(data) != size:
        raise RegisterSizeError(f"{name}: expected {size} bytes, got {len(data)}")


def _check_scalar(name: str, desc: RegDescriptor) -> None:
    if desc.is_vector:
        raise RegisterKindError(f"{name} is not a scalar register")


#: ``fn(name, start, size)``: register *name* bytes ``[start, start+size)`` may change.
RegisterObserver = Callable[[str, int, int], None]


def _remover(owner, fn: Callable) -> Callable[[], None]:
    """A function that detaches *fn* from *owner*. It holds *owner* weakly, so
    an observer that keeps its remover does not form a reference cycle."""
    ref = weakref.ref(owner)

    def remove() -> None:
        target = ref()
        if target is not None:
            target._observers = _without(target._observers, fn)

    return remove


def _without(observers: tuple, fn: Callable) -> tuple:
    """*observers* with the first occurrence of *fn* removed (if any)."""
    for i, observer in enumerate(observers):
        if observer is fn:
            return observers[:i] + observers[i + 1:]
    return observers


class RegFile:
    """Register file whose layout is defined by ``REGFILE_SCHEMA``.

    Internal storage
    ~~~~~~~~~~~~~~~~
    Each register (or register array) is stored as a flat ``bytearray`` of
    ``descriptor.size_bytes * descriptor.count`` bytes.  Registers with
    ``word_view=True`` additionally expose a NumPy uint32 view.
    """

    def __init__(self, schema: list[RegDescriptor] | None = None) -> None:
        self._schema = schema or REGFILE_SCHEMA
        self._storage: dict[str, bytearray] = {}
        self._descriptors: dict[str, RegDescriptor] = {}
        self._alias_map: dict[str, str] = {}  # alias -> canonical name
        self._constant_one_bytes: dict[str, int] = {}
        self._mutable_exports: set[str] = set()
        self._observers: tuple[RegisterObserver, ...] = ()

        for desc in self._schema:
            total = desc.size_bytes * desc.count
            self._storage[desc.name] = bytearray(total)
            self._descriptors[desc.name] = desc
            # register aliases
            for alias in desc.debug_aliases:
                self._alias_map[alias] = desc.name
        
        # Attach dynamically generated accessor methods
        # (eliminates duplication between REGISTER_DEFINITIONS and hard-coded methods).
        # The default schema attaches them to the class once; only a custom
        # schema pays for per-instance generation.
        if self._schema is REGFILE_SCHEMA:
            _attach_default_accessors_to_class()
        else:
            _attach_dynamic_accessors(self)

        for index, value in CR_READ_ONLY_INITIAL_VALUES.items():
            self.set_scalar(
                CR_REGISTER_NAME,
                index,
                value,
                allow_read_only=True,
            )

        # R_MASK defaults to all-ones: bit=1 means lane active, bit=0 means deactivated.
        self._storage["r_mask"][:] = b"\xff" * len(self._storage["r_mask"])

    # -- observers ----------------------------------------------------------

    def add_observer(self, fn: RegisterObserver) -> Callable[[], None]:
        """Call ``fn(name, start, size)`` whenever a register range may change.

        ``name`` is the canonical register name and ``start``/``size`` the
        byte range written (the whole register for a mutable export). Returns
        a function that detaches ``fn``; calling it twice is harmless.
        """
        self._observers = self._observers + (fn,)
        return _remover(self, fn)

    # -- helpers ------------------------------------------------------------

    def _resolve(self, name: str) -> str:
        """Resolve an alias to the canonical register name."""
        return self._alias_map.get(name, name)

    def descriptor(self, name: str) -> RegDescriptor:
        """The descriptor of register *name* (or of the register it aliases)."""
        return self._descriptors[self._resolve(name)]

    # -- byte access ----------------------------------------------------------

    def view(self, name: str) -> memoryview:
        """Register *name*'s bytes, read-only: no observer is told, and
        constant-ones provenance is kept. Writing through it raises ``TypeError``."""
        return memoryview(self._storage[self._resolve(name)]).toreadonly()

    def mutable(self, name: str, start: int = 0, size: int | None = None) -> memoryview:
        """Bytes ``[start, start+size)`` of register *name* (to its end when
        *size* is None), writable. Observers are told that exactly that range
        may change, and its constant-ones provenance is forgotten."""
        canon = self._resolve(name)
        storage = self._storage[canon]
        if size is None:
            size = len(storage) - start
        if start < 0 or size < 0 or start + size > len(storage):
            raise RegisterIndexError(f"{canon}[{start}:{start + size}] out of range ({len(storage)} bytes)")
        self._forget_constant_ones(canon, start, size)
        return memoryview(storage)[start:start + size]

    def forget_provenance(self, name: str) -> None:
        """Forget register *name*'s constant-ones provenance. Only the emulator's
        reads call it, to keep alias statistics as they were (E04-24)."""
        self._constant_one_bytes.pop(self._resolve(name), None)

    def raw(self, name: str) -> bytearray:
        """Return the ``bytearray`` backing register *name*.

        Deprecated: use :meth:`view` to read and :meth:`mutable` to write. The
        whole buffer escapes, so its provenance is never trusted again.
        """
        warnings.warn("RegFile.raw() is deprecated: use view() to read or mutable() to write",
                      DeprecationWarning, stacklevel=2)
        canon = self._resolve(name)
        # A mutable escape may be written without going through an accessor.
        self.mutable(canon)
        self._mutable_exports.add(canon)
        return self._storage[canon]

    def raw_readonly(self, name: str) -> bytes:
        """Read storage without handing out a mutable alias or losing provenance."""
        return bytes(self._storage[self._resolve(name)])

    def mark_constant_ones(self, name: str, start: int, size: int) -> None:
        canon = self._resolve(name)
        if start < 0 or size <= 0 or start + size > len(self._storage[canon]):
            raise ValueError("constant ONES register range out of bounds")
        # A previously handed-out mutable handle can still overwrite this buffer.
        if canon in self._mutable_exports:
            return
        self._constant_one_bytes[canon] = self._constant_one_bytes.get(canon, 0) | (((1 << size) - 1) << start)

    def has_constant_ones(self, name: str, start: int, size: int) -> bool:
        canon = self._resolve(name)
        bits = self._constant_one_bytes.get(canon, 0)
        if not bits:
            return False
        total = len(self._storage[canon])
        start %= total
        first = min(size, total - start)
        mask = ((1 << first) - 1) << start
        if size > first:
            mask |= (1 << (size - first)) - 1
        return bits & mask == mask

    def _forget_constant_ones(self, name: str, start: int, size: int) -> None:
        for observer in self._observers:
            observer(name, start, size)
        if name in self._constant_one_bytes:
            self._constant_one_bytes[name] &= ~(((1 << size) - 1) << start)
            if not self._constant_one_bytes[name]:
                del self._constant_one_bytes[name]

    # -- scalar (LR / CR) access --------------------------------------------

    def get_scalar(self, name: str, index: int) -> int:
        """Read a 32-bit scalar register (LR / CR) at *index*."""
        canon = self._resolve(name)
        desc = self._descriptors[canon]
        _check_scalar(name, desc)
        _check_index(name, index, desc.count)
        offset = index * desc.size_bytes
        return struct.unpack_from("<I", self._storage[canon], offset)[0]

    def set_scalar(
        self,
        name: str,
        index: int,
        value: int,
        *,
        allow_read_only: bool = False,
    ) -> None:
        """Write a scalar register (LR / CR) at *index*."""
        canon = self._resolve(name)
        desc = self._descriptors[canon]
        _check_scalar(name, desc)
        _check_index(name, index, desc.count)

        if self._is_read_only_cr(desc, index) and not allow_read_only:
            raise EmulatorError(f"CR{index} is read-only")

        if desc.kind in (RegKind.CR, RegKind.LR):
            value &= LR_CR_SCALAR_VALUE_MASK
        else:
            value &= REGISTER_WORD_VALUE_MASK

        offset = index * desc.size_bytes
        struct.pack_into("<I", self._storage[canon], offset, value)

    def _is_read_only_cr(self, desc: RegDescriptor, index: int) -> bool:
        return desc.kind == RegKind.CR and index in CR_READ_ONLY_INITIAL_VALUES

    # -- All convenience accessors generated from schema --------------------
    # get_lr, set_lr, get_cr, set_cr, get_r, set_r, get_r_cyclic_at,
    # set_r_cyclic_at, get_r_mask, set_r_mask, get_r_acc_bytes,
    # set_r_acc_bytes, get_r_acc_words, get_r_acc_word, set_r_acc_word,
    # get_mult_res_bytes, set_mult_res_bytes, get_mult_res_words,
    # get_mem_bypass, set_mem_bypass, etc.
    #
    # See _create_accessor_methods() below for the generation rules.

    # -- generic get / set by name (for debug CLI) --------------------------

    def get_register_bytes(self, name: str, index: int = 0) -> bytearray:
        """Read a register element by canonical or alias name."""
        canon = self._resolve(name)
        desc = self._descriptors[canon]
        _check_index(name, index, desc.count)
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        return bytearray(self._storage[canon][start:end])

    def set_register_bytes(self, name: str, index: int, data: bytes | bytearray) -> None:
        """Write a register element by canonical or alias name."""
        canon = self._resolve(name)
        desc = self._descriptors[canon]
        _check_index(name, index, desc.count)
        _check_size(name, data, desc.size_bytes)
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        self._forget_constant_ones(canon, start, desc.size_bytes)
        self._storage[canon][start:end] = data

    # -- snapshot (cycle-start copy for VLIW) -------------------------------

    def snapshot(self) -> RegFile:
        """Return an independent copy of this register file.

        Used to implement VLIW read-before-write semantics: all sub-
        instructions within a cycle read from the snapshot while
        writes go to the live register file.

        Only the register *storage* is copied. The schema, the descriptors and
        the alias map describe the layout rather than the state — they are
        never mutated, so the copy shares them. This runs once per emulated
        cycle, and ``copy.deepcopy`` spent 384 us of it walking descriptors and
        per-instance bound accessors (73% of total emulator runtime) to produce
        a register file that differs only in its bytes.
        """
        new = object.__new__(type(self))
        new._schema = self._schema
        new._descriptors = self._descriptors
        new._alias_map = self._alias_map
        new._constant_one_bytes = self._constant_one_bytes.copy()
        new._observers = ()
        new._mutable_exports = set()
        new._storage = {name: bytearray(buf) for name, buf in self._storage.items()}
        if self._schema is not REGFILE_SCHEMA:
            # A custom schema keeps its accessors on the instance, so the copy
            # needs its own bound set (the default schema's live on the class).
            _attach_dynamic_accessors(new)
        return new

    def copy_into(self, other: RegFile) -> None:
        """Overwrite *other*'s register storage with this register file's bytes.

        The counterpart to :meth:`snapshot` for a snapshot buffer reused across
        cycles: same result as taking a fresh snapshot, without allocating one.
        ``other`` must have been produced by :meth:`snapshot` from a register
        file with this layout; the caller is responsible for that.
        """
        dst = other._storage
        other._constant_one_bytes = {name: bits for name, bits in self._constant_one_bytes.items()
                                    if name not in other._mutable_exports}
        for name, buf in self._storage.items():
            for observer in other._observers:
                observer(name, 0, len(buf))
            dst[name][:] = buf

    # -- iteration (for debug / serialisation) ------------------------------

    @property
    def schema(self) -> list[RegDescriptor]:
        return list(self._schema)

    @property
    def register_names(self) -> list[str]:
        return list(self._storage.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire register file to a JSON-friendly dict."""
        result: dict[str, Any] = {}
        for desc in self._schema:
            buf = self._storage[desc.name]
            if not desc.is_vector:
                # scalar array (LR / CR / etc.)
                values = []
                for i in range(desc.count):
                    values.append(struct.unpack_from("<I", buf, i * 4)[0])
                result[desc.name] = values
            else:
                # byte array — hex string for compactness
                if desc.count == 1:
                    result[desc.name] = buf.hex()
                else:
                    result[desc.name] = [
                        buf[i * desc.size_bytes : (i + 1) * desc.size_bytes].hex()
                        for i in range(desc.count)
                    ]
        return result

    def __repr__(self) -> str:
        parts = [f"{d.name}({d.size_bytes}B×{d.count})" for d in self._schema]
        return f"RegFile([{', '.join(parts)}])"


# ===========================================================================
# Dynamic Accessor Generation
# ===========================================================================
#
# ALL named accessors (get_lr, set_r_acc_bytes, get_r_cyclic_at, …) are
# generated here from the schema.  The generation rules depend on four
# descriptor criteria:
#
#   desc.is_vector   – scalar (int) vs vector (byte-blob)
#   desc.cyclic      – wrapping _at accessors
#   desc.size_bytes  – width of each element
#   desc.word_view   – uint32 word accessors + _bytes suffix
#   desc.count       – indexed vs single-element
#
# ===========================================================================


def _create_accessor_methods(schema: list[RegDescriptor]) -> dict[str, Any]:
    """Generate get/set methods for every register in the schema.

    Returns a dict of ``method_name → unbound method`` to be attached
    to ``RegFile`` instances by ``_attach_dynamic_accessors``.
    """
    methods: dict[str, Any] = {}

    for desc in schema:
        name = desc.name

        if not desc.is_vector:
            # ── Scalar register (LR, CR) ──────────────────────────
            # get_{name}(index) → int
            # set_{name}(index, value)
            _add_scalar_accessors(methods, name)
        else:
            # ── Vector (byte-blob) register ───────────────────────
            if desc.count > 1:
                # Multi-element: get_{name}(index), set_{name}(index, data)
                _add_indexed_vector_accessors(methods, name, desc.count, desc.size_bytes)
            elif desc.word_view:
                # Single blob with word view: use _bytes suffix
                _add_blob_accessors(methods, name, desc.size_bytes, suffix="_bytes")
            else:
                # Single blob, no word view: no suffix
                _add_blob_accessors(methods, name, desc.size_bytes, suffix="")

            # Cyclic wrapping accessors (additive)
            if desc.cyclic:
                _add_cyclic_accessors(methods, name, desc.size_bytes)

            # Word-level accessors (additive)
            if desc.word_view:
                _add_word_view_accessors(methods, name, desc.size_bytes)

    return methods


# ---------------------------------------------------------------------------
# Scalar accessors  (e.g. get_lr / set_lr)
# ---------------------------------------------------------------------------

def _add_scalar_accessors(methods: dict, name: str) -> None:
    def _make_get(n: str = name):
        def get_val(self, index: int) -> int:
            return self.get_scalar(n, index)
        get_val.__name__ = f"get_{n}"
        get_val.__doc__ = f"Get {n}[index] as a 32-bit integer."
        return get_val

    def _make_set(n: str = name):
        def set_val(self, index: int, value: int) -> None:
            self.set_scalar(n, index, value)
        set_val.__name__ = f"set_{n}"
        set_val.__doc__ = f"Set {n}[index] = value (32-bit)."
        return set_val

    methods[f"get_{name}"] = _make_get()
    methods[f"set_{name}"] = _make_set()


# ---------------------------------------------------------------------------
# Single-element blob accessors  (e.g. get_r_mask / set_r_mask,
#                                  or  get_r_acc_bytes / set_r_acc_bytes)
# ---------------------------------------------------------------------------

def _add_blob_accessors(methods: dict, name: str, size: int, *, suffix: str) -> None:
    def _make_get(n: str = name, sz: int = size):
        def get_blob(self) -> bytearray:
            return bytearray(self._storage[n])
        get_blob.__name__ = f"get_{n}{suffix}"
        get_blob.__doc__ = f"Return a copy of {n} ({sz} bytes)."
        return get_blob

    def _make_set(n: str = name, sz: int = size):
        def set_blob(self, data: bytes | bytearray) -> None:
            _check_size(n, data, sz)
            self._forget_constant_ones(n, 0, sz)
            self._storage[n][:] = data
        set_blob.__name__ = f"set_{n}{suffix}"
        set_blob.__doc__ = f"Set {n} ({sz} bytes)."
        return set_blob

    methods[f"get_{name}{suffix}"] = _make_get()
    methods[f"set_{name}{suffix}"] = _make_set()


# ---------------------------------------------------------------------------
# Multi-element indexed blob accessors  (e.g. get_r / set_r)
# ---------------------------------------------------------------------------

def _add_indexed_vector_accessors(
    methods: dict, name: str, count: int, elem_size: int
) -> None:
    def _make_get(n: str = name, cnt: int = count, esz: int = elem_size):
        def get_elem(self, index: int) -> bytearray:
            _check_index(n, index, cnt)
            start = index * esz
            return bytearray(self._storage[n][start : start + esz])
        get_elem.__name__ = f"get_{n}"
        get_elem.__doc__ = f"Get {n}[index] ({esz} bytes, {cnt} elements)."
        return get_elem

    def _make_set(n: str = name, cnt: int = count, esz: int = elem_size):
        def set_elem(self, index: int, data: bytes | bytearray) -> None:
            _check_index(n, index, cnt)
            _check_size(n, data, esz)
            start = index * esz
            self._forget_constant_ones(n, start, esz)
            self._storage[n][start : start + esz] = data
        set_elem.__name__ = f"set_{n}"
        set_elem.__doc__ = f"Set {n}[index] ({esz} bytes, {cnt} elements)."
        return set_elem

    methods[f"get_{name}"] = _make_get()
    methods[f"set_{name}"] = _make_set()


# ---------------------------------------------------------------------------
# Cyclic (wrapping) accessors  (e.g. get_r_cyclic_at / set_r_cyclic_at)
# ---------------------------------------------------------------------------

def _add_cyclic_accessors(methods: dict, name: str, total_size: int) -> None:
    """Generate wrapping get/set accessors for a cyclic register.

    ``RegFile`` has no concept of "mode", and since #180 it needs none: the two
    cyclic rings are separate registers -- ``r_cyclic`` (512 B, 1 B/element) and
    ``r_cyclic_wide_debug`` (2048 B, 4 B/element) -- so each wraps at its own
    allocation. Both hold 512 *elements*; only the byte count per element
    differs.

    ``wrap_size`` predates that split, when a single 2048 B ``r_cyclic`` had to
    wrap at 512 B in narrow mode and the mode-aware caller (``Ipu``) supplied
    the size on every call. It defaults to the register's full allocation
    (``total_size``), which is now always the right answer -- nothing in
    ``src/`` overrides it.
    """
    def _make_get(n: str = name, default_sz: int = total_size):
        def get_at(self, start_idx: int, length: int = 128, wrap_size: int | None = None) -> bytearray:
            buf = self._storage[n]
            sz = default_sz if wrap_size is None else wrap_size
            start_idx %= sz
            if start_idx + length <= sz:
                return bytearray(buf[start_idx : start_idx + length])
            first = buf[start_idx:sz]
            second = buf[: length - len(first)]
            return bytearray(first + second)
        get_at.__name__ = f"get_{n}_at"
        get_at.__doc__ = (
            f"Read *length* bytes from {n} starting at *start_idx*, wrapping at "
            f"*wrap_size* bytes (default: the full {n} allocation)."
        )
        return get_at

    def _make_set(n: str = name, default_sz: int = total_size):
        def set_at(self, start_idx: int, data: bytes | bytearray, wrap_size: int | None = None) -> None:
            buf = self._storage[n]
            sz = default_sz if wrap_size is None else wrap_size
            start_idx %= sz
            length = len(data)
            first = min(length, sz - start_idx)
            self._forget_constant_ones(n, start_idx, first)
            if length > first:
                self._forget_constant_ones(n, 0, length - first)
            if start_idx + length <= sz:
                buf[start_idx : start_idx + length] = data
            else:
                first_len = sz - start_idx
                buf[start_idx : start_idx + first_len] = data[:first_len]
                buf[: length - first_len] = data[first_len:]
        set_at.__name__ = f"set_{n}_at"
        set_at.__doc__ = (
            f"Write *data* into {n} at *start_idx*, wrapping at *wrap_size* "
            f"bytes (default: the full {n} allocation)."
        )
        return set_at

    methods[f"get_{name}_at"] = _make_get()
    methods[f"set_{name}_at"] = _make_set()


# ---------------------------------------------------------------------------
# Word-view accessors  (e.g. get_r_acc_words / get_r_acc_word / set_r_acc_word)
# ---------------------------------------------------------------------------

def _add_word_view_accessors(methods: dict, name: str, size: int) -> None:
    n_words = size // 4

    def _make_get_words(n: str = name):
        def get_words(self) -> np.ndarray:
            return np.frombuffer(self.view(n), dtype=np.uint32)
        get_words.__name__ = f"get_{n}_words"
        get_words.__doc__ = f"View {n} as uint32 words ({n_words} words), read-only."
        return get_words

    def _make_set_words(n: str = name, nw: int = n_words):
        def set_words(self, words) -> None:
            data = np.asarray(words, dtype=np.uint32)
            if data.shape != (nw,):
                raise RegisterSizeError(f"{n}: expected {nw} words, got {data.size}")
            self.mutable(n)[:] = data.astype("<u4").tobytes()
        set_words.__name__ = f"set_{n}_words"
        set_words.__doc__ = f"Set all of {n} from {n_words} uint32 words."
        return set_words

    def _make_get_word(n: str = name, nw: int = n_words):
        def get_word(self, index: int) -> int:
            _check_index(f"{n} word", index, nw, unit="words")
            return struct.unpack_from("<I", self._storage[n], index * 4)[0]
        get_word.__name__ = f"get_{n}_word"
        get_word.__doc__ = f"Get one uint32 word from {n}."
        return get_word

    def _make_set_word(n: str = name, nw: int = n_words):
        def set_word(self, index: int, value: int) -> None:
            _check_index(f"{n} word", index, nw, unit="words")
            self._forget_constant_ones(n, index * 4, 4)
            struct.pack_into(
                "<I",
                self._storage[n],
                index * 4,
                value & REGISTER_WORD_VALUE_MASK,
            )
        set_word.__name__ = f"set_{n}_word"
        set_word.__doc__ = f"Set one uint32 word in {n}."
        return set_word

    methods[f"get_{name}_words"] = _make_get_words()
    methods[f"set_{name}_words"] = _make_set_words()
    methods[f"get_{name}_word"] = _make_get_word()
    methods[f"set_{name}_word"] = _make_set_word()


# ---------------------------------------------------------------------------
# Attach generated accessors to a RegFile instance
# ---------------------------------------------------------------------------

def _attach_dynamic_accessors(regfile_instance: RegFile) -> None:
    """Attach dynamically generated accessor methods to a RegFile instance.

    Called from ``__init__`` only for a **custom** schema.  The default schema
    attaches its accessors to the class instead (see
    ``_attach_default_accessors_to_class``), because building ~50 closures and
    binding them per instance is pure overhead when every instance shares one
    schema — and the emulator builds a ``RegFile`` per snapshot.
    """
    methods = _create_accessor_methods(regfile_instance._schema)
    for method_name, method in methods.items():
        setattr(
            regfile_instance,
            method_name,
            method.__get__(regfile_instance, type(regfile_instance)),
        )


_DEFAULT_ACCESSORS_ATTACHED = False


def _attach_default_accessors_to_class() -> None:
    """Attach the default schema's accessors to ``RegFile`` itself, once.

    Instance attachment stays available for a custom schema, and those
    instance attributes shadow these class methods for every name they define.
    """
    global _DEFAULT_ACCESSORS_ATTACHED
    if _DEFAULT_ACCESSORS_ATTACHED:
        return
    for method_name, method in _create_accessor_methods(REGFILE_SCHEMA).items():
        setattr(RegFile, method_name, method)
    _DEFAULT_ACCESSORS_ATTACHED = True
