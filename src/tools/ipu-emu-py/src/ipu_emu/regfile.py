"""IPU register file — built from the descriptor schema.

Provides:
- Byte-level storage for every register defined in ``REGFILE_SCHEMA``.
- Named accessors: ``regfile["lr"]``, ``regfile.get_r(0)``, etc.
- Cyclic (wraparound) read/write for ``r_cyclic``.
- Word-view (uint32) access for ``r_acc`` and ``mult_res``.
- Deep-copy snapshot for VLIW read-before-write semantics.
- JSON serialisation driven by descriptors.
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
