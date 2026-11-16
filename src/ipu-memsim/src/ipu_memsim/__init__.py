"""The memory what-if tool: record a kernel's XMEM access trace on the
emulator once (``ipu_memsim.record``), then replay it through a swappable
memory model (``ipu_memsim.replay``, ``ipu_memsim.models``,
``ipu_memsim.cache_unit``) to estimate stall cycles for a memory configuration.

It never changes what the emulator computes: the recorder only listens to the
reference model's public API. Estimated runtime is the emulator's cycle count
plus the stall cycles a model reports; every model except ``Ideal`` is a toy
model until the open questions in ``docs/questions.md`` are answered.

Depends one way on ``ipu_harness``, ``ipu_emu`` and ``ipu_isa``; none of them
imports it.
"""
