"""Shared emulator configuration register layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

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
DSTRUCTURE_PARTITION_BITS = 4
DSTRUCTURE_VALID_ELEMENTS_MASK = (1 << DSTRUCTURE_VALID_ELEMENTS_BITS) - 1
DSTRUCTURE_PARTITION_MASK = (1 << DSTRUCTURE_PARTITION_BITS) - 1
DSTRUCTURE_PARTITION_SHIFT = DSTRUCTURE_VALID_ELEMENTS_BITS

DEFAULT_VALID_ELEMENTS = 128
DEFAULT_PARTITION = 0


@dataclass(frozen=True)
class DStructureConfig:
    """Decoded CR15 dstructure configuration."""

    valid_elements: int = DEFAULT_VALID_ELEMENTS
    partition: int = DEFAULT_PARTITION

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


def encode_dstructure(*, valid_elements: int, partition: int = DEFAULT_PARTITION) -> int:
    """Pack dstructure fields into the CR15 register value."""
    valid = int(valid_elements) & DSTRUCTURE_VALID_ELEMENTS_MASK
    part = int(partition) & DSTRUCTURE_PARTITION_MASK
    return valid | (part << DSTRUCTURE_PARTITION_SHIFT)


def decode_dstructure(value: int) -> DStructureConfig:
    """Decode a CR15 register value into named dstructure fields."""
    word = int(value) & LR_CR_SCALAR_VALUE_MASK
    return DStructureConfig(
        valid_elements=word & DSTRUCTURE_VALID_ELEMENTS_MASK,
        partition=(word >> DSTRUCTURE_PARTITION_SHIFT) & DSTRUCTURE_PARTITION_MASK,
    )


def get_config_valid_elements(value: int) -> int:
    """Return the active lane-count field from an encoded dstructure value."""
    return decode_dstructure(value).valid_elements


def get_config_partition(value: int) -> int:
    """Return the partition field from an encoded dstructure value."""
    return decode_dstructure(value).partition
