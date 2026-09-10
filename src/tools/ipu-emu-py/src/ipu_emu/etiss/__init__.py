"""ETISS backend for the IPU emulator.

The Python emulator stays the reference implementation; this package runs the
same programs on a native ETISS architecture plugin and writes the results back
into an :class:`~ipu_emu.ipu_state.IpuState`, so the two can be compared
directly.  See ``docs/content/specs/etiss-integration.md``.
"""

from ipu_emu.etiss.runner import (
    EtissNotAvailable,
    EtissRunner,
    build_imem_image,
    deserialize_state,
    encode_instruction_word,
    is_available,
    runner_path,
    serialize_state,
)

__all__ = [
    "EtissNotAvailable",
    "EtissRunner",
    "build_imem_image",
    "deserialize_state",
    "encode_instruction_word",
    "is_available",
    "runner_path",
    "serialize_state",
]
