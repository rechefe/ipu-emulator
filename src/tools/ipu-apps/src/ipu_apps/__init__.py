"""IPU application test harnesses.

This package provides a framework for building IPU application harnesses
that load data, run assembled programs on the emulator, and validate
results.

Subclass :class:`IpuApp` to create a new application harness.
"""

from ipu_apps.base import IpuApp

__all__ = ["IpuApp"]
