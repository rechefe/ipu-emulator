"""Shared code for the multi-stream (P=4) projection kernels: harness, query, SPEC helper.

Twelve kernels: the all-stream (P=4) counterparts of the single-stream
``matmul_*_x128`` projection matmuls in :mod:`ipu_apps.kernels.matmul`, one
set per MobileViT layer (L3/L4/L5, shape suffix 144/192/240) and role
(``qkv``, ``outproj``, ``ffn1``, ``ffn2``). Each loops all 4 pixel-streams
internally, sharing one weight matrix across streams, instead of one host
round-trip per stream::

    C[p, tg, j, t] = act(sum_k W[j, k] * D[p, k, tg, t])
      p in [0, 4), j in [0, N_OUT), tg in [0, N_TG), t in [0, N_TOK)

``act`` is ``silu`` for the FFN1 kernels (``ACTIVATE.QUANTIZE silu`` at the
store) and the identity for every other role.

All twelve harnesses are the same code over four numbers -- ``K``, ``N_OUT``,
``N_TG`` and ``N_TOK`` -- so it lives here once, as :class:`ProjectionP4App`
parameterised by a :class:`ProjectionLayout`. L3 (144) has N=256 tokens per
stream, i.e. ``N_TG = 2`` token groups of 128; L4/L5 have a single group
(``N_TG = 1``, 64 / 16 tokens padded to LANES).

Query
-----
Every projection kernel answers the same query: a fixed input-channel count
``k`` contracted against a shared weight matrix to produce ``n_out`` output
channels, run over a fixed number of pixel-streams. Each kernel is FIXED-SHAPE
-- its ``.asm`` is written for exactly one (k, n_out) pair -- so ``supports``
is an *exact match*, not a range check.

``k``          input channels (contraction length)
``n_out``      output channels
``n_streams``  optional, pixel-streams processed per invocation (this family
               is always ``N_STREAM = 4``)
``activation`` optional, ``"none"`` (default) or ``"silu"``. The FFN1 kernels
               fuse ``silu`` into their store; routing a plain projection to
               one of them would return confidently wrong values, so the
               activation is part of the query.

There is deliberately no ``shape`` parameter carrying batch/token dimensions:
the token count and token-group layout are internal to each kernel's fixed
.asm and are not something a caller chooses -- they follow from (k, n_out)
picking a specific kernel.

Files
-----
``input_path``    the 4 streams' D blocks back to back, FP32. Each stream's
                  block is channel-major ``(N_TG, K, N_TOK)``: N_TG token-group
                  blocks of K channels x N_TOK tokens (for N_TG = 1 simply
                  ``(K, N_TOK)``). The whole file is ``(N_STREAM, N_TG, K, N_TOK)``.
``weights_path``  the shared W, output-major ``(N_OUT, K)`` FP32, stored
                  verbatim (no transpose).
``output_path``   raw XMEM rows ``(N_STREAM, N_TG, N_OUT, 128)`` FP32; the
                  first ``N_TOK`` lanes of each row are valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.kernel_registry import (
    OUTPUT, WEIGHT, ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes,
)
from ipu_apps.kernel_registry.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

OP = "projection"
ACTIVATIONS = ("none", "silu")
N_STREAM = 4  # pixel-streams every kernel in this family processes per call

# ---------------------------------------------------------------------------
# Wide-vector FP32 only -- see the matmul_*_x128 harnesses for the full
# rationale (elements are 4 B, a row is LANES*4 = 512 B, XMEM .asm operands
# are ROW numbers per issue #179, region bases are derived from row counts).
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512
FULL_BOUND = LANES - 2                       # 126: bound for any full width-128 chunk

OUTPUT_ROW_BYTES   = 512    # one store = one row = one output channel
OUTPUT_STRIDE_ROWS = 1


@dataclass(frozen=True)
class ProjectionLayout:
    """One projection kernel's dimensions and the XMEM map derived from them.

    D (4 streams) / W (shared) / C (4 streams) are packed back to back, in rows:

    * D[p]: interleaved channel-major, row (k, tg) at
      ``DATA_BASE_ROW + p*DATA_ROWS_PER_STREAM + k*N_TG + tg`` (N_TOK valid
      lanes + zero pad) -- the single-stream ancestor's row (k, tg) layout.
    * W: ``W_STRIDE_ROWS`` rows per output channel j, chunk c at
      ``WEIGHTS_BASE_ROW + j*W_STRIDE_ROWS + c``, zero-padded on the last
      (partial) chunk.
    * C[p]: grouped channel-major, row (j, tg) at
      ``OUTPUT_BASE_ROW + p*OUTPUT_ROWS_PER_STREAM + tg*N_OUT + j``.
    """

    k: int                # input channels
    n_out: int            # output channels
    n_tok: int            # tokens per group (padded to LANES in XMEM)
    n_tg: int = 1         # token groups (L3: 256 tokens = 2 x 128)

    @property
    def w_stride_rows(self) -> int:
        """Rows per output channel (ceil(K/128))."""
        return -(-self.k // LANES)

    @property
    def chunk_count(self) -> int:
        """ceil(K/128) chunks in the runtime chunk loop."""
        return self.w_stride_rows

    @property
    def tail_bound(self) -> int:
        """Do-while bound of the last (possibly partial) chunk: width - 2 (see .asm header)."""
        return self.k - LANES * (self.chunk_count - 1) - 2

    @property
    def data_rows_per_stream(self) -> int:
        return self.k * self.n_tg                   # one row per (k, tg)

    @property
    def weight_rows(self) -> int:
        return self.n_out * self.w_stride_rows

    @property
    def output_rows_per_stream(self) -> int:
        return self.n_tg * self.n_out * OUTPUT_STRIDE_ROWS

    @property
    def data_base_row(self) -> int:
        return 0

    @property
    def weights_base_row(self) -> int:
        return self.data_base_row + N_STREAM * self.data_rows_per_stream

    @property
    def output_base_row(self) -> int:
        return self.weights_base_row + self.weight_rows

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """The input file's FP32 array shape."""
        return (N_STREAM, self.n_tg, self.k, self.n_tok)

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """The output file's FP32 array shape (raw XMEM rows)."""
        return (N_STREAM, self.n_tg, self.n_out, LANES)


class ProjectionP4App(IpuApp):
    """Multi-stream (P=4) projection harness; a kernel's subclass sets ``layout``.

    Args:
        inst_path:    Assembled kernel binary.
        input_path:   All 4 streams' D blocks, ``(N_STREAM, N_TG, K, N_TOK)`` FP32.
        weights_path: Shared W, ``(N_OUT, K)`` FP32.
        output_path:  Optional; receives ``(N_STREAM, N_TG, N_OUT, 128)`` FP32 rows.
    """

    layout: ProjectionLayout | None = None

    def __init__(self, *, input_path, weights_path, **kwargs) -> None:
        if self.layout is None:
            raise TypeError(
                f"{type(self).__name__} has no layout; construct a kernel's App "
                f"(or use create_harness), not the shared family base class"
            )
        super().__init__(**kwargs)
        self.input_path = Path(input_path)
        self.weights_path = Path(weights_path)

    def _load_data(self, state: "IpuState") -> None:
        """Stage every stream's D block, tg-interleaved per channel and padded to whole rows."""
        lo = self.layout
        raw = self.input_path.read_bytes()
        expected = int(np.prod(lo.input_shape)) * ELEM_BYTES
        if len(raw) != expected:
            raise ValueError(
                f"{self.input_path}: expected {expected} B "
                f"({N_STREAM} streams x {lo.n_tg * lo.k * lo.n_tok} FP32), got {len(raw)}"
            )
        d = np.frombuffer(raw, dtype="<f4").reshape(lo.input_shape)
        rows = np.zeros((N_STREAM, lo.k, lo.n_tg, LANES), dtype="<f4")
        rows[..., :lo.n_tok] = d.transpose(0, 2, 1, 3)        # file (p, tg, k) -> row (p, k, tg)
        state.xmem.write_address(lo.data_base_row * ROW_BYTES, rows.tobytes())

    def _load_weights(self, state: "IpuState") -> None:
        """Stage the shared W, padding each output channel's K elements to whole rows."""
        lo = self.layout
        raw = self.weights_path.read_bytes()
        expected = lo.n_out * lo.k * ELEM_BYTES
        if len(raw) < expected:
            raise ValueError(f"{self.weights_path}: expected >= {expected} B, got {len(raw)}")
        w = np.frombuffer(raw[:expected], dtype="<f4").reshape(lo.n_out, lo.k)
        rows = np.zeros((lo.n_out, lo.w_stride_rows * LANES), dtype="<f4")
        rows[:, :lo.k] = w
        state.xmem.write_address(lo.weights_base_row * ROW_BYTES, rows.tobytes())

    def setup(self, state: "IpuState") -> None:
        lo = self.layout
        self._load_data(state)
        self._load_weights(state)

        # CR0 (=0) and CR1 (=1) are read-only hardwired constants -- writing
        # anything else raises EmulatorError (issue #230). ZERO
        # is already CR0's hardwired value, so no write is needed here.
        state.regfile.set_cr(2, lo.data_base_row)               # DATA_BASE (stream 0)
        state.regfile.set_cr(3, lo.weights_base_row)            # WEIGHTS_BASE (shared)
        state.regfile.set_cr(4, lo.output_base_row)             # OUTPUT_BASE (stream 0, tg=0 sub-block)
        state.regfile.set_cr(5, -1)                             # NEG_ONE
        state.regfile.set_cr(6, FULL_BOUND)                     # FULL_BOUND (126)
        state.regfile.set_cr(7, lo.tail_bound)                  # TAIL_BOUND (shape-specific)
        state.regfile.set_cr(8, lo.w_stride_rows)               # W_STRIDE (rows per output channel)
        state.regfile.set_cr(9, lo.n_out)                       # N_OUT_CR (j-loop limit, tg1 output offset)
        state.regfile.set_cr(10, lo.chunk_count)                # CHUNK_COUNT
        state.regfile.set_cr(11, lo.chunk_count - 1)            # LAST_CHUNK_IDX
        state.regfile.set_cr(12, lo.data_rows_per_stream)       # DATA_STREAM_STR
        state.regfile.set_cr(13, lo.output_rows_per_stream)     # OUT_STREAM_STR
        state.regfile.set_cr(14, N_STREAM)                      # STREAM_COUNT

    def teardown(self, state: "IpuState") -> None:
        # Every stream's C block is contiguous and the blocks are back to back,
        # so the output is one run of N_STREAM * N_TG * N_OUT rows.
        if self.output_path is not None:
            lo = self.layout
            dump_xmem_to_binary(
                state, self.output_path,
                lo.output_base_row * ROW_BYTES, OUTPUT_ROW_BYTES,
                N_STREAM * lo.output_rows_per_stream,
            )


# -- registry declaration ---------------------------------------------------


@dataclass(frozen=True)
class ProjectionQuery:
    """A projection query reduced to what the kernels route on.

    Attributes:
        k:          Input channels (contraction length).
        n_out:      Output channels.
        n_streams:  Pixel-streams requested per invocation.
        activation: Activation fused into the output store.
        bundle:     The shape bundle (weight + output), for reporting.
    """

    k: int
    n_out: int
    n_streams: int
    activation: str
    bundle: ShapeBundle


def projection_query(*, k: int, n_out: int, n_streams: int = N_STREAM,
                     activation: str = "none") -> ProjectionQuery:
    """Normalise the raw params into the form every projection kernel routes on.

    Raises:
        ValueError: if the activation is unknown.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}; got {activation!r}")
    bundle = ShapeBundle.of(**{WEIGHT: (int(n_out), int(k))}).with_shapes(
        derived={OUTPUT: (int(n_streams), int(n_out))}
    )
    return ProjectionQuery(
        k=int(k), n_out=int(n_out), n_streams=int(n_streams),
        activation=activation, bundle=bundle,
    )


def _query(params) -> ProjectionQuery:
    return projection_query(
        k=params["k"], n_out=params["n_out"],
        n_streams=params.get("n_streams", N_STREAM),
        activation=params.get("activation", "none"),
    )


def positive_dims(q: ProjectionQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.k < 1:
        return f"k ({q.k}) must be >= 1"
    if q.n_out < 1:
        return f"n_out ({q.n_out}) must be >= 1"
    if q.n_streams < 1:
        return f"n_streams ({q.n_streams}) must be >= 1"
    return None


def exact_shape_reason(q: ProjectionQuery, k: int, n_out: int) -> str | None:
    """Refusal reason unless ``q`` matches this kernel's fixed (k, n_out) shape.

    Every projection kernel is written for one exact (k, n_out) pair -- there
    is no padding/chunking fallback the way softmax_rows tolerates any row
    count. A mismatch here is always a routing miss, never a partial fit.
    """
    if q.k != k or q.n_out != n_out:
        return (
            f"handles exactly k={k}, n_out={n_out}; this query is "
            f"k={q.k}, n_out={q.n_out}"
        )
    if q.n_streams != N_STREAM:
        return (
            f"handles exactly {N_STREAM} pixel-streams per invocation; "
            f"this query asked for {q.n_streams}"
        )
    return None


FIXED_STREAMS_ONLY = (
    f"Multi-stream (P={N_STREAM}) transformer projection kernel: N_STREAM is "
    f"baked into the .asm and into the input/output file layouts, not a "
    f"runtime parameter."
)

_ROLE_NAMES = {"qkv": "QKV", "outproj": "OutProj", "ffn1": "FFN1", "ffn2": "FFN2"}
_LAYERS = {144: 3, 192: 4, 240: 5}


def projection_spec(app_class: type[ProjectionP4App], *, activation: str = "none"):
    """KernelSpec for a ``proj_<role>_<d>_p4`` kernel whose harness is ``app_class``.

    The role and layer (``d`` = 144/192/240 -> L3/L4/L5) come from the folder
    name; the fixed (k, n_out) from ``app_class.layout``. ``activation`` is the
    activation the kernel's ``.asm`` fuses into its store.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}; got {activation!r}")
    name = kernel_folder(app_class)
    variant = name.removeprefix("proj_").removesuffix("_p4")
    role, _, width = variant.rpartition("_")
    what = f"Layer {_LAYERS[int(width)]} {_ROLE_NAMES[role]}"
    lo = app_class.layout

    def supports(**params):
        q = _query(params)
        reason = positive_dims(q) or exact_shape_reason(q, lo.k, lo.n_out)
        if reason is None and q.activation != activation:
            reason = (f"fuses activation={activation!r} into its store; "
                      f"query asks for {q.activation!r}")
        return no(reason) if reason else yes()

    act = "" if activation == "none" else f" with fused {activation}"
    layout_note = (
        f"input_path holds the {N_STREAM} streams' D blocks back to back, "
        f"{lo.input_shape} FP32; output_path receives raw XMEM rows "
        f"{lo.output_shape} FP32 with the first {lo.n_tok} lanes valid."
    )
    return folder_spec(
        app_class,
        op=OP,
        variant=variant,
        requires=("k", "n_out"),
        tags=("fp32-wide", "multi-stream"),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: (
            f"k == {lo.k} and n_out == {lo.n_out} exactly: the {what} multi-stream "
            f"(P={N_STREAM}) projection kernel{act}."
        ),
        caveats=lambda **params: (FIXED_STREAMS_ONLY, layout_note),
        bundle=lambda **params: _query(params).bundle,
        # Exact-shape match: no padding, no chunking. Cheapest possible claim;
        # no two projection kernels share a (k, n_out, activation).
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
