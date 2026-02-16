"""Declarative register descriptors for the IPU register file.

This module now imports register metadata from ipu-common, the single source
of truth for all IPU register definitions. It maintains full backward
compatibility by re-exporting RegDtype, RegKind, RegDescriptor, and REGFILE_SCHEMA.

Each RegDescriptor defines a single register (or array of registers) in the
IPU.  The RegFile class uses REGFILE_SCHEMA to build the full register state
and the debug CLI auto-generates read/write commands from the same schema.
"""

from __future__ import annotations

# Import types from ipu-common
from ipu_common.types import RegDtype, RegKind, RegDescriptor

# Import register schema generator from ipu-common
from ipu_common.registers import create_regfile_schema

# Re-export for backward compatibility
__all__ = ["RegDtype", "RegKind", "RegDescriptor", "REGFILE_SCHEMA"]

# ---------------------------------------------------------------------------
# The single source of truth for the IPU register file layout.
# Now generated from ipu_common.registers, which is the master definition.
# Adding a register there automatically creates storage, debug commands, and
# JSON serialisation support.
# ---------------------------------------------------------------------------

REGFILE_SCHEMA: list[RegDescriptor] = create_regfile_schema()
