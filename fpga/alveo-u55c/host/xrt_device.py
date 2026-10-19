"""The U55C card as an ``ipu_host.driver.Device``, through XRT's Python
binding (``pyxrt``, from the XRT install: ``source /opt/xilinx/xrt/setup.sh``).

Not built or tested by Bazel: CI has no card and no XRT. The calls follow the
XRT 2.16 Python API; check them on the first bring-up (fpga-prototype.md,
"Bring-up").
"""

from __future__ import annotations

from pathlib import Path

import pyxrt

KERNEL = "ipu_proto_kernel"
#: The memory bank v++ connects m_axi_hbm to (vitis/connectivity.cfg).
MEMORY_TAG = "HBM[0]"


class XrtDevice:
    """One U55C with the prototype's xclbin loaded."""

    def __init__(self, xclbin: Path, index: int = 0) -> None:
        self.device = pyxrt.device(index)
        image = pyxrt.xclbin(str(xclbin))
        uuid = self.device.load_xclbin(image)
        self.ip = pyxrt.ip(self.device, uuid, KERNEL)
        self.group = next(m.get_index() for m in image.get_mems() if m.get_tag() == MEMORY_TAG)
        self.buffers: dict[int, pyxrt.bo] = {}

    # -- Bus -----------------------------------------------------------------

    def read32(self, address: int) -> int:
        return self.ip.read_register(address) & 0xFFFF_FFFF

    def write32(self, address: int, value: int) -> None:
        self.ip.write_register(address, value & 0xFFFF_FFFF)

    # -- DeviceMemory --------------------------------------------------------

    def alloc(self, size: int) -> int:
        bo = pyxrt.bo(self.device, size, pyxrt.bo.normal, self.group)
        address = bo.address()
        self.buffers[address] = bo
        return address

    def _buffer(self, address: int, size: int) -> tuple[pyxrt.bo, int]:
        for base, bo in self.buffers.items():
            if base <= address and address + size <= base + bo.size():
                return bo, address - base
        raise ValueError(f"device address 0x{address:X} (+{size}) is not allocated")

    def write(self, address: int, data: bytes) -> None:
        bo, offset = self._buffer(address, len(data))
        bo.write(data, offset)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(data), offset)

    def read(self, address: int, size: int) -> bytes:
        bo, offset = self._buffer(address, size)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, offset)
        return bytes(bo.read(size, offset))
