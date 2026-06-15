"""Shared emulator configuration register layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

CR_REGISTER_NAME = "cr"

CR_ZERO_REG_INDEX = 0
CR_ONE_REG_INDEX = 1
CR_DSTRUCTURE_REG_INDEX = 15

CR_ZERO_REG_VALUE = 0
CR_ONE_REG_VALUE = 1
CR_READ_ONLY_INITIAL_VALUES = {
    CR_ZERO_REG_INDEX: CR_ZERO_REG_VALUE,
    CR_ONE_REG_INDEX: CR_ONE_REG_VALUE,
}

REGISTER_WORD_BITS = 32
REGISTER_WORD_VALUE_MASK = (1 << REGISTER_WORD_BITS) - 1

LR_CR_SCALAR_BITS = 20
LR_CR_SCALAR_VALUE_MASK = (1 << LR_CR_SCALAR_BITS) - 1

DSTRUCTURE_VALID_ELEMENTS_BITS = 8
DSTRUCTURE_PARTITION_BITS = 5                                    # P16=16 requires 5 bits
DSTRUCTURE_VALID_ELEMENTS_MASK = (1 << DSTRUCTURE_VALID_ELEMENTS_BITS) - 1
DSTRUCTURE_PARTITION_MASK = (1 << DSTRUCTURE_PARTITION_BITS) - 1
DSTRUCTURE_PARTITION_SHIFT = DSTRUCTURE_VALID_ELEMENTS_BITS

DEFAULT_VALID_ELEMENTS = 128


class Partition(IntEnum):
    """Valid CR15.partition values. Controls how the 128 lanes are grouped for mask shifts."""
    P0  = 0   # no partitioning — all 128 lanes in one group
    P2  = 2   # 2 groups of 64 lanes
    P4  = 4   # 4 groups of 32 lanes
    P8  = 8   # 8 groups of 16 lanes
    P16 = 16  # 16 groups of 8 lanes


DEFAULT_PARTITION = Partition.P0


@dataclass(frozen=True)
class DStructureConfig:
    """Decoded CR15 dstructure configuration."""

    valid_elements: int = DEFAULT_VALID_ELEMENTS
    partition: Partition = DEFAULT_PARTITION

    def __iter__(self):
        """Allow tuple unpacking as (valid_elements, partition)."""
        yield self.valid_elements
        yield self.partition

    def to_register_value(self) -> int:
        """Pack this dstructure into the CR15 register word."""
        return encode_dstructure(
            valid_elements=self.valid_elements,
            partition=self.partition,
        )


DEFAULT_DSTRUCTURE = DStructureConfig()


def encode_dstructure(*, valid_elements: int, partition: Partition | int = DEFAULT_PARTITION) -> int:
    """Pack dstructure fields into the CR15 register value."""
    partition = Partition(partition)   # validates and converts; raises ValueError if invalid
    valid = int(valid_elements) & DSTRUCTURE_VALID_ELEMENTS_MASK
    part = int(partition) & DSTRUCTURE_PARTITION_MASK
    return valid | (part << DSTRUCTURE_PARTITION_SHIFT)


def decode_dstructure(value: int) -> DStructureConfig:
    """Decode a CR15 register value into named dstructure fields."""
    word = int(value) & LR_CR_SCALAR_VALUE_MASK
    return DStructureConfig(
        valid_elements=word & DSTRUCTURE_VALID_ELEMENTS_MASK,
        partition=Partition((word >> DSTRUCTURE_PARTITION_SHIFT) & DSTRUCTURE_PARTITION_MASK),
    )


def get_config_valid_elements(value: int) -> int:
    """Return the active lane-count field from an encoded dstructure value."""
    return decode_dstructure(value).valid_elements


def get_config_partition(value: int) -> Partition:
    """Return the partition field from an encoded dstructure value."""
    return decode_dstructure(value).partition
