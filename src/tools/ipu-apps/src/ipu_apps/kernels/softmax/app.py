"""Shared code for the softmax kernels: harness base and spec helpers.

The five softmax kernels answer the same query -- a 2-D shape plus the axis
being normalised -- and share one device layout, so the query unpacking, the
spec boilerplate and the load / unload / core CR map live here once.

**Spec helpers.** A query carries ``shape`` (any rank; flattened to 2-D around
``dim``) and ``dim`` (torch's convention: negative counts from the end).
Everything a kernel routes on -- ``rows``, ``n``, ``width`` -- is derived by
:func:`softmax_query`, so the kernels cannot disagree about what a
``(shape, dim)`` means. :func:`softmax_spec` builds each kernel's KernelSpec
from just its ``supports`` / ``explain`` / ``cost``.

**Harness.**

Every softmax kernel reads and writes a dense row-major ``(rows, cols)`` FP32
file. On device each logical row occupies a fixed-width *slot* -- the row
padded with ``fill`` to the kernel's on-device width. A slot of ``k * 128``
spans ``k`` XMEM rows (the chunked kernels); a narrower slot packs
``128 / slot`` logical rows into one XMEM row (the packed kernels), with the
row count padded to fill the last one.

The three big regions (input, numerators, output) sit back to back from
``REGION_BASE``, 64 KiB aligned and sized to the device rows. The resident
vectors follow: the ``log2(e)`` constant first, then whatever the kernel
allocates in ``_layout_resident`` (order matters where the .asm derives a row
as a neighbour's row + 1).

A subclass supplies ``_geometry`` (returns the slot width), and optionally
``_layout_resident``, ``_write_resident`` and ``_crs`` for its extra CRs. Load,
unload and the core CR map (CR2-CR7, CR10) live here.

**Framework adapters.** Softmax's framework-layer adapters live here too, so
the op-agnostic registry carries no softmax vocabulary. They register on
import, and every softmax kernel imports this module, so
:func:`~ipu_apps.kernel_registry.lookup_layer` sees them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ipu_emu.ipu import LANES

from ipu_apps.kernel_registry import (
    ExecutionConfig,
    ShapeBundle,
    UnsupportedLayer,
    flatten_to_matrix,
    folder_spec,
    kernel_folder,
    no,
    register_layer,
    yes,
)
from ipu_apps.kernel_registry.base import IpuApp

ROW_BYTES = LANES * 4
LOG2E = math.log2(math.e)
REGION_BASE = 0x10000
_ALIGN = 0x10000


def partition_size(n: int) -> int:
    """Next power of two >= n, clamped to [16, 128]."""
    if not 1 <= n <= LANES:
        raise ValueError(f"N must be in 1..{LANES}; got {n}")
    ps = 16
    while ps < n:
        ps *= 2
    return ps


class SoftmaxApp(IpuApp):
    """Softmax over a row-major ``(rows, cols)`` FP32 matrix.

    ``cols`` is ``n`` for the row kernels (``dim = 1``) and ``width`` for the
    column kernels (``dim = 0``).
    """

    dim = 1                   # 1: along each row; 0: down each column
    fill = 0.0                # value of on-device padding elements
    valid_elements = LANES    # CR15 dstructure

    def __init__(self, *, rows: int, n: int | None = None, width: int | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        from ipu_apps.kernel_registry.registry import _harness_spec

        # Row kernels take the row length ``n``, column kernels ``width``; the
        # other axis's argument is a mistake, not something to ignore.
        size, other = ("n", "width") if self.dim == 1 else ("width", "n")
        given = {"n": n, "width": width}
        if given[other] is not None:
            raise TypeError(f"{type(self).__name__} takes {size}=, not {other}=")
        if given[size] is None:
            raise TypeError(f"{type(self).__name__} requires {size}=")
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        self.cols = int(given[size])
        if self.dim == 1:
            self.n = self.cols
        else:
            self.width = self.cols
        _harness_spec(self).guard(shape=(self.rows, self.cols), dim=self.dim)
        self.slot = self._geometry()
        self.device_rows = -(-self.rows * self.slot // LANES)

        step = (self.device_rows * ROW_BYTES + _ALIGN - 1) & ~(_ALIGN - 1)
        self.input_base = REGION_BASE
        self.num_base = self.input_base + step
        self.output_base = self.num_base + step
        self._next_resident = self.output_base + step
        self.cvec_addr = self._alloc()
        self._layout_resident()

    # -- subclass hooks -----------------------------------------------------

    def _geometry(self) -> int:
        """Set any kernel geometry attributes; return the on-device slot width."""
        return LANES

    def _layout_resident(self) -> None:
        self.max_addr = self._alloc()
        self.rvec_addr = self._alloc()

    def _write_resident(self, state) -> None:
        pass

    def _crs(self) -> dict[int, int]:
        return {}

    # -- shared machinery ---------------------------------------------------

    def _alloc(self, rows: int = 1) -> int:
        addr = self._next_resident
        self._next_resident += rows * ROW_BYTES
        return addr

    def setup(self, state) -> None:
        # Checked here, not in __init__: apps are also constructed for routing
        # checks with no input file behind them.
        expected = self.rows * self.cols * 4
        actual = self.input_path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__}: input must hold {self.rows} x {self.cols} FP32 "
                f"values ({expected} bytes); {self.input_path} has {actual} bytes"
            )
        x = np.fromfile(self.input_path, dtype="<f4", count=self.rows * self.cols)
        device = np.full((self.device_rows * LANES // self.slot, self.slot), self.fill, dtype="<f4")
        device[: self.rows, : self.cols] = x.reshape(self.rows, self.cols)
        state.xmem.write_address(self.input_base, device.tobytes())
        state.xmem.write_address(self.cvec_addr, np.full(LANES, LOG2E, dtype="<f4").tobytes())
        self._write_resident(state)

        # CR0 == 0 and CR1 == 1 are read-only; CR1 doubles as the 1.0 scalar
        # and the +1 increment. .asm XMEM operands are row numbers.
        bases = {2: self.output_base, 3: self.cvec_addr, 4: self.num_base,
                 5: self.max_addr, 6: self.rvec_addr, 10: self.input_base}
        for register, addr in bases.items():
            state.regfile.set_cr(register, addr // ROW_BYTES)
        state.regfile.set_cr(7, 1)  # region stride: one XMEM row per step
        for register, value in self._crs().items():
            state.regfile.set_cr(register, value)
        state.set_cr_dstructure(valid_elements=self.valid_elements)

    def teardown(self, state) -> None:
        """Write the output in the same dense row-major layout as the input."""
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.device_rows * ROW_BYTES)
        device = np.frombuffer(bytes(raw), dtype="<f4").reshape(-1, self.slot)
        Path(self.output_path).write_bytes(
            np.ascontiguousarray(device[: self.rows, : self.cols]).tobytes())


# -- spec helpers ---------------------------------------------------------

SINGLE_GROUP_MAX_ROWS = 128  # rows whose per-row scalars fit one 128-element vector

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "build on exp2/reciprocal over an FP32 vector path and have no narrow "
    "(INT8/FP8) variant."
)


@dataclass(frozen=True)
class SoftmaxQuery:
    """A softmax query reduced to what the kernels route on.

    Attributes:
        rows:    Rows of the (possibly flattened) 2-D problem.
        cols:    Columns of that problem.
        along_rows: True when softmax runs along each row (torch ``dim=1`` on a
            2-D input), False when it runs down each column (``dim=0``).
        n:       Reduction length when reducing along rows, else None.
        width:   Independent columns when reducing down columns, else None.
        bundle:  The shape bundle, carrying any flatten note.
    """

    rows: int
    cols: int
    along_rows: bool
    bundle: ShapeBundle

    @property
    def n(self) -> int | None:
        return self.cols if self.along_rows else None

    @property
    def width(self) -> int | None:
        return None if self.along_rows else self.cols

    @property
    def reduction_length(self) -> int:
        """How many elements each softmax sums over."""
        return self.cols if self.along_rows else self.rows


def softmax_query(shape, dim: int) -> SoftmaxQuery:
    """Normalise ``(shape, dim)`` into the form every softmax kernel routes on.

    Raises:
        ValueError: if ``dim`` is out of range for ``shape``, or the shape is
            rank > 2 with an interior reduction axis (which cannot be flattened
            without transposing -- see ``flatten_to_matrix``).
    """
    bundle, dim_2d, shape_2d = softmax_bundle(tuple(int(d) for d in shape), int(dim))
    rows, cols = shape_2d
    return SoftmaxQuery(
        rows=rows, cols=cols, along_rows=(dim_2d == 1), bundle=bundle
    )


def positive_dims(q: SoftmaxQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.rows < 1:
        return f"rows ({q.rows}) must be >= 1"
    if q.cols < 1:
        return f"columns ({q.cols}) must be >= 1"
    return None


def softmax_spec(app_class, *, along_rows, supports, explain, cost,
                 caveats=lambda q: (), tags=()):
    """KernelSpec for a softmax kernel; every callback receives the SoftmaxQuery.

    ``supports(q)`` returns a refusal reason, or None to accept. The shared
    preconditions -- positive extents and the reduction axis -- are checked
    first, and ``build`` hands the harness ``rows`` plus ``n`` or ``width``.
    """
    def query(params):
        return softmax_query(params["shape"], params["dim"])

    def _supports(**params):
        q = query(params)
        reason = positive_dims(q)
        if reason is None and q.along_rows != along_rows:
            reason = ("reduces down columns, not along rows" if along_rows
                      else "reduces along rows, not down columns")
        if reason is None:
            reason = supports(q)
        return no(reason) if reason else yes()

    def _build(**params):
        # q.n / q.width is None for a query on the other axis, which `supports`
        # has already refused -- so a wrong-axis build is visibly empty.
        q = query(params)
        return {"rows": q.rows, "n": q.n} if along_rows else {"rows": q.rows, "width": q.width}

    return folder_spec(
        app_class,
        op="softmax",
        variant=kernel_folder(app_class).removeprefix("softmax_"),
        # Every callback indexes these, so the registry checks them first: an
        # omitted parameter is then a refusal that names what is missing.
        requires=("shape", "dim"),
        tags=("fp32-wide", *tags),
        supports=_supports,
        build=_build,
        explain=lambda **params: explain(query(params)),
        caveats=lambda **params: (WIDE_VECTOR_ONLY, *caveats(query(params))),
        bundle=lambda **params: query(params).bundle,
        cost=lambda **params: cost(query(params)),
        execution=ExecutionConfig(mode="fp32"),
    )


def softmax_bundle(shape, dim: int):
    """Build the shape bundle for a softmax query.

    Softmax is shape-preserving, so the output shape is derived and equal to
    the input. A rank > 2 input is flattened around ``dim`` (recorded as a note
    on the bundle, never silently).

    Returns:
        ``(bundle, dim_2d, shape_2d)``.
    """
    shape_2d, dim_2d, note = flatten_to_matrix(shape, dim)
    bundle = ShapeBundle.of(input=shape).with_shapes(
        derived={"output": shape},
        notes=(note,) if note else (),
    )
    return bundle, dim_2d, shape_2d


# -- framework-layer adapters -----------------------------------------------


@register_layer("Softmax")
def _softmax_layer(layer, input_shape):
    """``nn.Softmax(dim=...)`` -> the ``softmax`` operation.

    ``nn.Softmax`` created without an explicit ``dim`` has ``dim=None``, which
    torch itself treats as deprecated and resolves with a heuristic. Rather
    than replicate that heuristic (and risk disagreeing with the framework on
    which axis is normalised), it is refused.
    """
    if not hasattr(layer, "dim"):
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) dim; it "
            f"does not look like the layer this adapter was written for"
        )
    if layer.dim is None:
        raise UnsupportedLayer(
            "Softmax(dim=None) does not state which axis to normalise; torch "
            "resolves it with a deprecated heuristic. Construct the layer with "
            "an explicit dim."
        )
    return "softmax", {"dim": int(layer.dim), "shape": input_shape}


@register_layer("LogSoftmax", "Softmin")
def _unsupported_softmax_relatives(layer, input_shape):
    """Refuse near-neighbours of Softmax explicitly.

    These sit beside ``Softmax`` in ``torch.nn`` and share its signature, so a
    permissive adapter would route them to a softmax kernel and return
    confidently wrong numbers.
    """
    name = type(layer).__name__
    detail = {
        "LogSoftmax": "computes log(softmax(x)), not softmax(x)",
        "Softmin": "computes softmax(-x), not softmax(x)",
    }[name]
    raise UnsupportedLayer(
        f"{name} {detail}; no kernel implements it. Using a softmax kernel "
        f"here would return confidently wrong values."
    )
