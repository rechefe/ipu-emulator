"""Native vector tour: a curated conformance program for the Cycle Inspector.

``vector_tour.asm`` runs once over 128-element 8-bit vectors (INT8, E3, E4 or
E5; no emulator-only format) and exercises each MULT form with ACC.ADD, ACC.SUB,
ACC.MAX, masked multiplies and an AGG.SUM.FIRST matrix-vector loop.

The input file is the XMEM the program reads, :data:`INPUT_ROWS` rows of
128 bytes: the four rows of ``A``, then ``w``, ``c`` and the R_MASK row. The
output file is what the program stores from :data:`OUTPUT_ROW`: three R_ACC
stores of :data:`ACC_STORE_ROWS` rows each and, in INT8 only, the
POST_AAQ_REG store after ``ACTIVATE.QUANTIZE relu``.
"""

from __future__ import annotations

from pathlib import Path

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernel_registry.spec import ExecutionConfig, folder_spec, no, yes
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import ElementFormat, encode_element

#: The element formats the tour runs in: the 8-bit formats that hold 2 exactly.
FORMATS = (ElementFormat.INT8, ElementFormat.E3, ElementFormat.E4, ElementFormat.E5)

ROW_BYTES = 128
#: Rows of ``A`` the matrix-vector loop walks.
A_ROWS = 4
A_ROW, W_ROW, C_ROW, MASK_ROW = 0, A_ROWS, A_ROWS + 1, A_ROWS + 2
INPUT_ROWS = A_ROWS + 3
OUTPUT_ROW = 8
#: One ST.ACC or ST.AAQ writes 512 bytes: four 128-byte rows.
ACC_STORE_ROWS = 4
ACC_STORES = 3


def parse_dtype(dtype: str | ElementFormat) -> ElementFormat:
    """The :class:`ElementFormat` ``dtype`` names (``"int8"``, ``"e4"``, ...)."""
    try:
        fmt = ElementFormat(str(dtype).lower())
    except ValueError:
        fmt = None
    if fmt not in FORMATS:
        names = ", ".join(f.value for f in FORMATS)
        raise ValueError(f"vector_tour runs in {names}; got {dtype!r}")
    return fmt


def output_rows(fmt: ElementFormat) -> int:
    """Rows the program stores: the R_ACC stores, plus POST_AAQ_REG in INT8."""
    stores = ACC_STORES + (fmt is ElementFormat.INT8)
    return stores * ACC_STORE_ROWS


class VectorTourApp(IpuApp):
    """Load the tour's input rows into XMEM, set its CRs, and read back what it stored.

    Args:
        input_path: The :data:`INPUT_ROWS` input rows.
        output_path: Where the stored rows go.
        dtype: The element format (see :data:`FORMATS`).
    """

    def __init__(self, *, dtype: str | ElementFormat = "int8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        SPEC.guard(dtype=dtype)
        self.fmt = parse_dtype(dtype)

    def setup(self, state: IpuState) -> None:
        if state.fmt is not self.fmt:
            raise ConfigurationError(f"vector_tour in {self.fmt} needs a state in {self.fmt}")
        image = self.input_path.read_bytes()
        if len(image) != INPUT_ROWS * ROW_BYTES:
            raise ValueError(
                f"vector_tour input must be {INPUT_ROWS} rows of {ROW_BYTES} bytes; "
                f"got {len(image)} bytes"
            )
        state.xmem.write_address(0, image)
        regs = state.regfile
        regs.set_cr(2, A_ROW)
        regs.set_cr(3, W_ROW)
        regs.set_cr(4, C_ROW)
        regs.set_cr(5, MASK_ROW)
        regs.set_cr(6, 128)  # the second R_CYCLIC slot, in elements
        regs.set_cr(7, OUTPUT_ROW)
        regs.set_cr(8, ACC_STORE_ROWS)
        regs.set_cr(9, encode_element(2.0, self.fmt))
        regs.set_cr(11, A_ROWS - 1)
        regs.set_cr(12, int(self.fmt is ElementFormat.INT8))

    def teardown(self, state: IpuState) -> None:
        if self.output_path is not None:
            size = output_rows(self.fmt) * ROW_BYTES
            stored = state.xmem.read_address(OUTPUT_ROW * ROW_BYTES, size)
            self.output_path.write_bytes(bytes(stored))


def _supports(**params):
    try:
        parse_dtype(params["dtype"])
    except ValueError as exc:
        return no(str(exc))
    return yes()


SPEC = folder_spec(
    VectorTourApp,
    op="vector_tour",
    variant="native",
    requires=("dtype",),
    supports=_supports,
    build=lambda **p: {"dtype": p["dtype"]},
    explain=lambda **p: f"fixed 4 x 128 tour of the MULT and ACC slots in {p['dtype']}",
    execution=lambda app: ExecutionConfig(fmt=app.fmt),
    tags=("conformance",),
)
