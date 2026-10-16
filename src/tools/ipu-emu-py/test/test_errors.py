"""The emulator's errors: one root under ``IpuError``; setup errors are ``ConfigurationError``."""

import inspect

import pytest
from emu_support import ElementFormat, new_state

import ipu_emu.errors as errors
from ipu_emu.alias_profile import AliasProfile
from ipu_emu.emulator import load_program
from ipu_emu.errors import ConfigurationError, EmulatorError, EmulatorFault
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState
from ipu_emu.prototyping import build_enabled, resolve_enabled
from ipu_isa.errors import IpuError


def test_every_emulator_error_is_an_ipu_error():
    classes = [
        cls
        for _, cls in inspect.getmembers(errors, inspect.isclass)
        if cls.__module__ == errors.__name__
    ]
    assert len(classes) >= 6
    assert all(issubclass(cls, IpuError) for cls in classes), classes


def test_the_two_kinds_share_one_root():
    assert issubclass(EmulatorFault, EmulatorError) and issubclass(
        ConfigurationError, EmulatorError
    )
    assert issubclass(ConfigurationError, ValueError), "setup checks raised ValueError before"
    assert not issubclass(EmulatorError, RuntimeError)


def test_a_program_too_large_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="Program too large"):
        load_program(IpuState(), [{}] * (INST_MEM_SIZE + 1))


@pytest.mark.skipif(build_enabled(), reason="this build allows prototyping")
def test_prototyping_without_its_build_flag_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="ipu_proto"):
        resolve_enabled(True)


def test_an_alias_profile_on_a_second_state_is_a_configuration_error():
    profile = AliasProfile()
    first = new_state(fmt=ElementFormat.INT8)
    first.alias_profile = profile
    Ipu(first)
    second = new_state(fmt=ElementFormat.INT8)
    second.alias_profile = profile
    with pytest.raises(ConfigurationError, match="single IpuState"):
        Ipu(second)


def test_a_handler_outside_a_cycle_raises_a_typed_error():
    with pytest.raises(EmulatorError, match="outside a cycle"):
        Ipu(IpuState()).execute_lr_inc(dest=0, imm=1)


def test_a_partition_that_is_not_the_enum_is_a_typed_error():
    with pytest.raises(EmulatorError, match="Partition"):
        Ipu._build_partition_vector(4)
