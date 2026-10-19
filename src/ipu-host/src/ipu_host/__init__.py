"""The IPU host register map and the software that drives it.

``rdl/`` holds the SystemRDL sources, the single source of the register
offsets and fields: ``ipu_host.rdl`` (the IPU host port) and
``proto_platform.rdl`` (the FPGA prototype platform). ``regmap`` compiles them
into a lookup map; ``generate`` writes the SystemVerilog register blocks and
the docs tables; ``driver`` drives an IPU through a ``Bus``; ``model`` is a
model of the host port over the reference model, for running host software
without a card.
"""
