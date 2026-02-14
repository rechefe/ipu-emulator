"""IPU register file — built from the descriptor schema.

Provides:
- Byte-level storage for every register defined in ``REGFILE_SCHEMA``.
- Named accessors: ``regfile["lr"]``, ``regfile.get_r(0)``, etc.
- Cyclic (wraparound) read/write for ``r_cyclic``.
- Word-view (uint32) access for ``r_acc`` and ``mult_res``.
- Deep-copy snapshot for VLIW read-before-write semantics.
- JSON serialisation driven by descriptors.

DESIGN NOTE: Accessor methods (get_r, set_r_acc_bytes, etc.) are generated
from REGFILE_SCHEMA to eliminate duplication between REGISTER_DEFINITIONS
and hand-coded methods. See _attach_dynamic_accessors() below.
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
        assert desc.dtype in (RegDtype.UINT32, RegDtype.INT32)
        assert 0 <= index < desc.count, f"{name}[{index}] out of range (count={desc.count})"
        offset = index * desc.size_bytes
        buf = self._storage[self._resolve(name)]
        return struct.unpack_from("<I", buf, offset)[0]

    def set_scalar(self, name: str, index: int, value: int) -> None:
        """Write a 32-bit scalar register (LR / CR) at *index*."""
        desc = self._desc(name)
        assert desc.dtype in (RegDtype.UINT32, RegDtype.INT32)
        assert 0 <= index < desc.count, f"{name}[{index}] out of range (count={desc.count})"
        offset = index * desc.size_bytes
        buf = self._storage[self._resolve(name)]
        struct.pack_into("<I", buf, offset, value & 0xFFFFFFFF)

    # -- LR / CR convenience ------------------------------------------------

    def get_lr(self, index: int) -> int:
        return self.get_scalar("lr", index)

    def set_lr(self, index: int, value: int) -> None:
        self.set_scalar("lr", index, value)

    def get_cr(self, index: int) -> int:
        return self.get_scalar("cr", index)

    def set_cr(self, index: int, value: int) -> None:
        self.set_scalar("cr", index, value)

    # -- R register access (128-byte) ---------------------------------------

    def get_r(self, index: int) -> bytearray:
        """Return a *copy* of R register *index* (0 or 1)."""
        desc = self._desc("r")
        assert 0 <= index < desc.count
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        return bytearray(self._storage["r"][start:end])

    def set_r(self, index: int, data: bytes | bytearray) -> None:
        """Write 128 bytes into R register *index*."""
        desc = self._desc("r")
        assert 0 <= index < desc.count
        assert len(data) == desc.size_bytes
        start = index * desc.size_bytes
        end = start + desc.size_bytes
        self._storage["r"][start:end] = data

    # -- R cyclic (512-byte, wraparound) ------------------------------------

    def get_r_cyclic_at(self, start_idx: int, length: int = 128) -> bytearray:
        """Read *length* bytes from the cyclic register starting at *start_idx*.

        Wraps around the 512-byte buffer, mirroring ``ipu__get_r_cyclic_at_idx``.
        """
        buf = self._storage["r_cyclic"]
        size = len(buf)
        start_idx %= size
        if start_idx + length <= size:
            return bytearray(buf[start_idx : start_idx + length])
        # wrap
        first = buf[start_idx:]
        second = buf[: length - len(first)]
        return bytearray(first + second)

    def set_r_cyclic_at(self, start_idx: int, data: bytes | bytearray) -> None:
        """Write *data* into the cyclic register at *start_idx* (with wrap)."""
        buf = self._storage["r_cyclic"]
        size = len(buf)
        start_idx %= size
        length = len(data)
        if start_idx + length <= size:
            buf[start_idx : start_idx + length] = data
        else:
            first_len = size - start_idx
            buf[start_idx:] = data[:first_len]
            buf[: length - first_len] = data[first_len:]

    # -- R mask (128-byte) --------------------------------------------------

    def get_r_mask(self) -> bytearray:
        """Return a copy of the mask register."""
        return bytearray(self._storage["r_mask"])

    def set_r_mask(self, data: bytes | bytearray) -> None:
        assert len(data) == self._desc("r_mask").size_bytes
        self._storage["r_mask"][:] = data

    # -- Accumulator (512-byte, byte + word views) --------------------------

    def get_r_acc_bytes(self) -> bytearray:
        """Return a copy of the accumulator as bytes."""
        return bytearray(self._storage["r_acc"])

    def set_r_acc_bytes(self, data: bytes | bytearray) -> None:
        assert len(data) == self._desc("r_acc").size_bytes
        self._storage["r_acc"][:] = data

    def get_r_acc_words(self) -> np.ndarray:
        """Return a *view* of the accumulator as uint32 words (128 words)."""
        return np.frombuffer(self._storage["r_acc"], dtype=np.uint32)

    def set_r_acc_word(self, index: int, value: int) -> None:
        """Set one uint32 word in the accumulator."""
        desc = self._desc("r_acc")
        n_words = desc.size_bytes // 4
        assert 0 <= index < n_words
        struct.pack_into("<I", self._storage["r_acc"], index * 4, value & 0xFFFFFFFF)

    def get_r_acc_word(self, index: int) -> int:
        """Read one uint32 word from the accumulator."""
        desc = self._desc("r_acc")
        n_words = desc.size_bytes // 4
        assert 0 <= index < n_words
        return struct.unpack_from("<I", self._storage["r_acc"], index * 4)[0]

    # -- Misc forwarding registers ------------------------------------------

    def get_mult_res_bytes(self) -> bytearray:
        return bytearray(self._storage["mult_res"])

    def set_mult_res_bytes(self, data: bytes | bytearray) -> None:
        assert len(data) == self._desc("mult_res").size_bytes
        self._storage["mult_res"][:] = data

    def get_mult_res_words(self) -> np.ndarray:
        return np.frombuffer(self._storage["mult_res"], dtype=np.uint32)

    def get_mem_bypass(self) -> bytearray:
        return bytearray(self._storage["mem_bypass"])

    def set_mem_bypass(self, data: bytes | bytearray) -> None:
        assert len(data) == self._desc("mem_bypass").size_bytes
        self._storage["mem_bypass"][:] = data

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
            if desc.dtype in (RegDtype.UINT32, RegDtype.INT32):
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
# Instead of hard-coding get_r(), set_r_acc_bytes() for every register,
# we generate them from REGFILE_SCHEMA at runtime. This eliminates duplication
# while keeping backward compatibility.
# ===========================================================================

def _create_accessor_methods(schema: list[RegDescriptor]) -> dict[str, Any]:
    """Generate get_* and set_* methods for all registers in the schema.
    
    Returns a dict of method names → method objects that can be attached to
    RegFile instances. This eliminates duplication between REGISTER_DEFINITIONS
    and hand-coded accessor methods.
    
    For each register, generates appropriate accessors based on dtype and properties:
    - Scalar (UINT32/INT32): get_{name}(idx) / set_{name}(idx, value)
    - Vector (other dtype): get_{name}_bytes() / set_{name}_bytes()
    """
    methods = {}
    
    for desc in schema:
        reg_name = desc.name
        is_scalar = desc.dtype in (RegDtype.UINT32, RegDtype.INT32)
        
        if is_scalar:
            # For scalar registers like LR/CR: generate get(idx)/set(idx, value)
            def make_get_scalar(desc=desc, reg_name=reg_name):
                def get_scalar(self, index: int) -> int:
                    """Get scalar register value at index."""
                    return self.get_scalar(reg_name, index)
                get_scalar.__doc__ = f"Get {reg_name}[index]."
                get_scalar.__name__ = f"get_{reg_name}"
                return get_scalar
            
            def make_set_scalar(desc=desc, reg_name=reg_name):
                def set_scalar(self, index: int, value: int) -> None:
                    """Set scalar register value at index."""
                    self.set_scalar(reg_name, index, value)
                set_scalar.__doc__ = f"Set {reg_name}[index] = value."
                set_scalar.__name__ = f"set_{reg_name}"
                return set_scalar
            
            methods[f"get_{reg_name}"] = make_get_scalar()
            methods[f"set_{reg_name}"] = make_set_scalar()
        else:
            # For vector registers: generate get_bytes() / set_bytes()
            def make_get_bytes(desc=desc, reg_name=reg_name):
                def get_bytes(self) -> bytearray:
                    """Return raw bytes for register."""
                    return self.get_register_bytes(reg_name, 0)
                get_bytes.__doc__ = f"Return raw bytes from {reg_name}."
                get_bytes.__name__ = f"get_{reg_name}_bytes"
                return get_bytes
            
            def make_set_bytes(desc=desc, reg_name=reg_name):
                def set_bytes(self, data: bytes | bytearray) -> None:
                    """Set raw bytes for register."""
                    self.set_register_bytes(reg_name, 0, data)
                set_bytes.__doc__ = f"Set raw bytes in {reg_name}."
                set_bytes.__name__ = f"set_{reg_name}_bytes"
                return set_bytes
            
            methods[f"get_{reg_name}_bytes"] = make_get_bytes()
            methods[f"set_{reg_name}_bytes"] = make_set_bytes()
            
            # For multi-element vector registers, generate get/set with index
            if desc.count > 1:
                def make_get_indexed(desc=desc, reg_name=reg_name):
                    def get_indexed(self, index: int) -> bytearray:
                        """Get register element at index."""
                        return self.get_register_bytes(reg_name, index)
                    get_indexed.__doc__ = f"Get {reg_name}[index] ({desc.count} elements)."
                    get_indexed.__name__ = f"get_{reg_name}"
                    return get_indexed
                
                def make_set_indexed(desc=desc, reg_name=reg_name):
                    def set_indexed(self, index: int, data: bytes | bytearray) -> None:
                        """Set register element at index."""
                        self.set_register_bytes(reg_name, index, data)
                    set_indexed.__doc__ = f"Set {reg_name}[index] ({desc.count} elements)."
                    set_indexed.__name__ = f"set_{reg_name}"
                    return set_indexed
                
                methods[f"get_{reg_name}"] = make_get_indexed()
                methods[f"set_{reg_name}"] = make_set_indexed()
            
            # For word_view registers, generate word accessors
            if desc.word_view:
                def make_get_words(desc=desc, reg_name=reg_name):
                    def get_words(self) -> np.ndarray:
                        """Return register as uint32 words."""
                        return np.frombuffer(self._storage[reg_name], dtype=np.uint32)
                    get_words.__doc__ = f"View {reg_name} as uint32 words."
                    get_words.__name__ = f"get_{reg_name}_words"
                    return get_words
                
                def make_get_word(desc=desc, reg_name=reg_name):
                    def get_word(self, index: int) -> int:
                        """Get one word from register."""
                        return struct.unpack_from(
                            "<I", self._storage[reg_name], index * 4
                        )[0]
                    get_word.__doc__ = f"Get one uint32 word from {reg_name}."
                    get_word.__name__ = f"get_{reg_name}_word"
                    return get_word
                
                def make_set_word(desc=desc, reg_name=reg_name):
                    def set_word(self, index: int, value: int) -> None:
                        """Set one word in register."""
                        struct.pack_into(
                            "<I", self._storage[reg_name], index * 4, value & 0xFFFFFFFF
                        )
                    set_word.__doc__ = f"Set one uint32 word in {reg_name}."
                    set_word.__name__ = f"set_{reg_name}_word"
                    return set_word
                
                methods[f"get_{reg_name}_words"] = make_get_words()
                methods[f"get_{reg_name}_word"] = make_get_word()
                methods[f"set_{reg_name}_word"] = make_set_word()
    
    return methods


def _attach_dynamic_accessors(regfile_instance: RegFile) -> None:
    """Attach dynamically generated accessor methods to a RegFile instance.
    
    This is called in __init__ to ensure every RegFile has convenience methods
    for each register without repeating them in code.
    """
    methods = _create_accessor_methods(regfile_instance._schema)
    for method_name, method in methods.items():
        setattr(regfile_instance, method_name, method.__get__(regfile_instance, type(regfile_instance)))
