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

import copy
import struct
from typing import Any

import numpy as np

from ipu_emu.descriptors import REGFILE_SCHEMA, RegDescriptor, RegDtype, RegKind


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

        for desc in self._schema:
            total = desc.size_bytes * desc.count
            self._storage[desc.name] = bytearray(total)
            self._descriptors[desc.name] = desc
            # register aliases
            for alias in desc.debug_aliases:
                self._alias_map[alias] = desc.name
        
        # Attach dynamically generated accessor methods
        # (eliminates duplication between REGISTER_DEFINITIONS and hard-coded methods)
        _attach_dynamic_accessors(self)

    # -- helpers ------------------------------------------------------------

    def _resolve(self, name: str) -> str:
        """Resolve an alias to the canonical register name."""
        return self._alias_map.get(name, name)

    def _desc(self, name: str) -> RegDescriptor:
        canon = self._resolve(name)
        return self._descriptors[canon]

    # -- raw byte access ----------------------------------------------------

    def raw(self, name: str) -> bytearray:
        """Return the raw ``bytearray`` backing register *name*."""
        return self._storage[self._resolve(name)]

    # -- scalar (LR / CR) access --------------------------------------------

    def get_scalar(self, name: str, index: int) -> int:
        """Read a 32-bit scalar register (LR / CR) at *index*."""
        desc = self._desc(name)
        assert not desc.is_vector, f"{name} is not a scalar register"
        assert 0 <= index < desc.count, f"{name}[{index}] out of range (count={desc.count})"
        offset = index * desc.size_bytes
        buf = self._storage[self._resolve(name)]
        return struct.unpack_from("<I", buf, offset)[0]

    def set_scalar(self, name: str, index: int, value: int) -> None:
        """Write a 32-bit scalar register (LR / CR) at *index*."""
        desc = self._desc(name)
        assert not desc.is_vector, f"{name} is not a scalar register"
        assert 0 <= index < desc.count, f"{name}[{index}] out of range (count={desc.count})"
        offset = index * desc.size_bytes
        buf = self._storage[self._resolve(name)]
        struct.pack_into("<I", buf, offset, value & 0xFFFFFFFF)

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
        assert 0 <= index < desc.count, f"{name}[{index}] out of range"
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        return bytearray(self._storage[canon][start:end])

    def set_register_bytes(self, name: str, index: int, data: bytes | bytearray) -> None:
        """Write a register element by canonical or alias name."""
        canon = self._resolve(name)
        desc = self._descriptors[canon]
        assert 0 <= index < desc.count, f"{name}[{index}] out of range"
        assert len(data) == desc.size_bytes
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        self._storage[canon][start:end] = data

    # -- snapshot (deep copy for VLIW) --------------------------------------

    def snapshot(self) -> RegFile:
        """Return a deep copy of this register file.

        Used to implement VLIW read-before-write semantics: all sub-
        instructions within a cycle read from the snapshot while
        writes go to the live register file.
        """
        return copy.deepcopy(self)

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
            assert len(data) == sz, f"{n}: expected {sz} bytes, got {len(data)}"
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
            assert 0 <= index < cnt, f"{n}[{index}] out of range (count={cnt})"
            start = index * esz
            return bytearray(self._storage[n][start : start + esz])
        get_elem.__name__ = f"get_{n}"
        get_elem.__doc__ = f"Get {n}[index] ({esz} bytes, {cnt} elements)."
        return get_elem

    def _make_set(n: str = name, cnt: int = count, esz: int = elem_size):
        def set_elem(self, index: int, data: bytes | bytearray) -> None:
            assert 0 <= index < cnt, f"{n}[{index}] out of range (count={cnt})"
            assert len(data) == esz, f"{n}: expected {esz} bytes, got {len(data)}"
            start = index * esz
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
    def _make_get(n: str = name, sz: int = total_size):
        def get_at(self, start_idx: int, length: int = 128) -> bytearray:
            buf = self._storage[n]
            start_idx %= sz
            if start_idx + length <= sz:
                return bytearray(buf[start_idx : start_idx + length])
            first = buf[start_idx:]
            second = buf[: length - len(first)]
            return bytearray(first + second)
        get_at.__name__ = f"get_{n}_at"
        get_at.__doc__ = (
            f"Read *length* bytes from {n} starting at *start_idx* (wrapping)."
        )
        return get_at

    def _make_set(n: str = name, sz: int = total_size):
        def set_at(self, start_idx: int, data: bytes | bytearray) -> None:
            buf = self._storage[n]
            start_idx %= sz
            length = len(data)
            if start_idx + length <= sz:
                buf[start_idx : start_idx + length] = data
            else:
                first_len = sz - start_idx
                buf[start_idx:] = data[:first_len]
                buf[: length - first_len] = data[first_len:]
        set_at.__name__ = f"set_{n}_at"
        set_at.__doc__ = f"Write *data* into {n} at *start_idx* (wrapping)."
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
            return np.frombuffer(self._storage[n], dtype=np.uint32)
        get_words.__name__ = f"get_{n}_words"
        get_words.__doc__ = f"View {n} as uint32 words ({n_words} words)."
        return get_words

    def _make_get_word(n: str = name, nw: int = n_words):
        def get_word(self, index: int) -> int:
            assert 0 <= index < nw, f"{n} word[{index}] out of range ({nw} words)"
            return struct.unpack_from("<I", self._storage[n], index * 4)[0]
        get_word.__name__ = f"get_{n}_word"
        get_word.__doc__ = f"Get one uint32 word from {n}."
        return get_word

    def _make_set_word(n: str = name, nw: int = n_words):
        def set_word(self, index: int, value: int) -> None:
            assert 0 <= index < nw, f"{n} word[{index}] out of range ({nw} words)"
            struct.pack_into("<I", self._storage[n], index * 4, value & 0xFFFFFFFF)
        set_word.__name__ = f"set_{n}_word"
        set_word.__doc__ = f"Set one uint32 word in {n}."
        return set_word

    methods[f"get_{name}_words"] = _make_get_words()
    methods[f"get_{name}_word"] = _make_get_word()
    methods[f"set_{name}_word"] = _make_set_word()


# ---------------------------------------------------------------------------
# Attach generated accessors to a RegFile instance
# ---------------------------------------------------------------------------

def _attach_dynamic_accessors(regfile_instance: RegFile) -> None:
    """Attach dynamically generated accessor methods to a RegFile instance.

    Called in ``__init__``.  Every convenience method (``get_lr``,
    ``set_r_acc_bytes``, ``get_r_cyclic_at``, …) is created here — no
    hand-coded per-register methods exist on the class.
    """
    methods = _create_accessor_methods(regfile_instance._schema)
    for method_name, method in methods.items():
        setattr(
            regfile_instance,
            method_name,
            method.__get__(regfile_instance, type(regfile_instance)),
        )
