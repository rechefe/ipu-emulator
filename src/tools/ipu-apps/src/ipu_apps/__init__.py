"""IPU application test harnesses.

This package provides a framework for building IPU application harnesses
that load data, run assembled programs on the emulator, and validate
results.

Subclass :class:`IpuApp` to create a new application harness.
"""

from ipu_apps.base import IpuApp
from ipu_apps.matmul_128x128 import MatMul128x128App
from ipu_apps.matmul_128x64x128 import MatMul128x64x128App
from ipu_apps.matmul_128x64x64 import MatMul128x64x64App
from ipu_apps.matmul_64x64x64 import MatMul64x64x64App
from ipu_apps.matmul_432x144_x128 import MatMul432x144x128App
from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
from ipu_apps.matmul_288x144_x128 import MatMul288x144x128App
from ipu_apps.matmul_144x288_x128 import MatMul144x288x128App
from ipu_apps.unfold_32x32x144 import Unfold32x32x144App

__all__ = [
    "IpuApp",
    "MatMul128x128App", "MatMul128x64x128App", "MatMul128x64x64App", "MatMul64x64x64App",
    "MatMul432x144x128App", "MatMul144x144x128App", "MatMul288x144x128App", "MatMul144x288x128App",
    "Unfold32x32x144App",
]
