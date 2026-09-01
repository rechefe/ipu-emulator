"""QKᵀ scores harness (Layer 5), one attention head.

Computes the query-major score matrix for a single attention head::

    S[i, s] = sum_{c=0..59} Q[i, c] * K[s, c]      for i, s in [0, 16)

Layer-5 shape: d=240, N_TOK=16 tokens per stream, h=4 heads, head_dim=60.
This is the L5 port of ``qk_scores_256x36``; the mapping is unchanged, only the
loop counts differ:

  * ``D`` 36 -> 60.  head_dim is a LOOP COUNT under this mapping (the
    contraction bound ``lr6 = D-2``), not a lane count, so 60 needs no padding.
  * ``N`` 256 -> 16.  256 tokens spanned two 128-lane key groups; 16 tokens fit
    in ONE group, so the g=1 half of the L3 kernel disappears entirely
    (``N_TG`` 2 -> 1).

Inputs Q, K are logically channel-major (head_dim D=60, N=16 tokens). K is
loaded channel-major verbatim; Q is staged query-major (a gather of its strided
channels) so one query's 60 head-channels load into r0 with a single
``LDR_MULT_REG`` — the matmul broadcast template (scalar = Q[i,c] from r0,
vector = K's channel-c column in r_cyclic, ``MULT.RC.VE``).

ONE CHANNEL PER ROW: N_TOK=16 means each K channel column and each staged Q row
holds 16 valid FP32 elements (64 B) inside a WHOLE 512-B XMEM row. Rows are
never shared between channels; the unused 112 lanes stay zero and the output is
cropped in :meth:`teardown`. (Packing two channels into one row at a 64-B
stride is a bug — see the N_TOK=16 matmul family.)

The score row goes through the standard quantize boundary
(``ACTIVATE.QUANTIZE identity`` + ``STR_POST_AAQ_REG``), like every other
kernel. No AGG.

Wide-vector FP32 only: elements are 4-byte FP32 and an XMEM row is 512 B.

Usage::

    from ipu_apps.attention.qk_scores_16x60 import QkScores16x60App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, ShapeBundle, no, yes
from ipu_apps.attention._spec_support import positive_dims, scores_query

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N    = 16           # tokens (queries = keys) per stream
D    = 60           # head_dim (contraction width)
N_TG = 1            # key groups: 16 keys fit in a single 128-lane group
N_TPG = N          # keys per group

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (issue #179). Region bases are DERIVED
# from row counts, not hardcoded bytes.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

# One channel per row: 16 tokens occupy 64 B of a 512-B row, and the row is
# still exclusively that channel's. ceil-div keeps this correct for any N <= 128.
K_STRIDE_ROWS    = max(1, N // LANES)        # 1: rows per K channel column
QROW_STRIDE_ROWS = 1                         # one staged query row per query
ACC_STORE_ROWS   = 1                         # one r_acc store = one row (wide)

K_ROWS    = D * K_STRIDE_ROWS
QROW_ROWS = N * QROW_STRIDE_ROWS
S_ROWS    = N * N_TG * ACC_STORE_ROWS

K_BASE_ROW    = 0
QROW_BASE_ROW = K_BASE_ROW + K_ROWS
S_BASE_ROW    = QROW_BASE_ROW + QROW_ROWS

K_BASE      = K_BASE_ROW * ROW_BYTES
QROW_BASE   = QROW_BASE_ROW * ROW_BYTES
S_BASE      = S_BASE_ROW * ROW_BYTES

QROW_STRIDE      = QROW_STRIDE_ROWS * ROW_BYTES
OUTPUT_ROW_BYTES = ROW_BYTES


class QkScores16x60App(IpuApp):
    """One-head QKᵀ → query-major scores application harness (wide FP32, L5)."""

    def __init__(
        self,
        *,
        k_base_row: int = K_BASE_ROW,
        qrow_base_row: int = QROW_BASE_ROW,
        s_base_row: int = S_BASE_ROW,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # query_path/key_path may be None to skip disk staging entirely and
        # use whatever is already in the shared IpuState's XMEM (e.g. written
        # there by an upstream kernel in the same chain). Existing callers
        # always pass real paths, so their behaviour is unchanged.
        self.query_path = Path(self.query_path) if self.query_path is not None else None
        self.key_path = Path(self.key_path) if self.key_path is not None else None

        # Base rows default to the module constants (K_BASE_ROW=0 etc.) so
        # every existing caller is unaffected; passing explicit values lets a
        # caller place this kernel's regions inside a shared IpuState instead
        # of the row-0 layout colliding with another kernel's data.
        #
        # k_base_row is NOT actually relocatable: the .asm reads K_BASE from
        # cr0, which the ISA hardwires to the constant 0 (any write is
        # silently dropped by RegFile.set_scalar). K therefore always lands
        # at row 0 regardless of this argument. It stays a constructor
        # parameter for interface symmetry with qrow_base_row/s_base_row,
        # but a non-default value would silently have no effect, so it is
        # rejected here instead.
        if k_base_row != K_BASE_ROW:
            raise ValueError(
                f"k_base_row={k_base_row} has no effect: qk_scores_16x60.asm "
                f"reads K_BASE from cr0, which the ISA hardwires to 0. K "
                f"always occupies rows [0, K_ROWS); place other regions "
                f"around it instead."
            )
        self.k_base_row = k_base_row
        self.qrow_base_row = qrow_base_row
        self.s_base_row = s_base_row
        self.k_base = k_base_row * ROW_BYTES
        self.qrow_base = qrow_base_row * ROW_BYTES
        self.s_base = s_base_row * ROW_BYTES

    # -- staging -------------------------------------------------------------

    def _stage_inputs(self, state: "IpuState") -> None:
        """Write K channel-major and Q query-major into XMEM.

        Input files are stored channel-major: element [token t, channel c] at
        (c*N + t)*ELEM_BYTES. A None path skips that input's staging (the
        caller has already placed it in XMEM at this app's base row).
        """
        if self.query_path is None and self.key_path is None:
            return
        q_raw = self.query_path.read_bytes()
        k_raw = self.key_path.read_bytes()

        # K: one channel column per WHOLE row. Column c is N contiguous elements
        #    in the file; it lands at k_base + c*K_STRIDE_ROWS*ROW_BYTES and the
        #    remaining lanes of that row stay zero.
        for c in range(D):
            col = k_raw[(c * N) * ELEM_BYTES : (c * N + N) * ELEM_BYTES]
            state.xmem.write_address(self.k_base + c * K_STRIDE_ROWS * ROW_BYTES, bytearray(col))

        # Q: gather the strided channels into contiguous query-major rows.
        #    QROW[i] = Q[i, 0..D-1] at qrow_base + i*QROW_STRIDE (rest zero-pad).
        for i in range(N):
            row = bytearray(QROW_STRIDE)
            for c in range(D):
                src = (c * N + i) * ELEM_BYTES
                row[c * ELEM_BYTES : (c + 1) * ELEM_BYTES] = q_raw[src : src + ELEM_BYTES]
            state.xmem.write_address(self.qrow_base + i * QROW_STRIDE, row)

    def setup(self, state: "IpuState") -> None:
        self._stage_inputs(state)

        # Only N of the 128 lanes carry real keys. valid_elements gates the
        # ACTIVATE.QUANTIZE window, so the stored row holds exactly N scores.
        state.set_cr_dstructure(valid_elements=N)

        # Startup skew, in rows. This is a lane/row count, so it carries no
        # element-width factor. One key group only: first live row = 0.
        g0_start_rows = -K_STRIDE_ROWS

        # CR1 (≡1) is read-only hardwired; QROW base lives on CR9.
        state.regfile.set_cr(0, self.k_base_row)         # data base
        state.regfile.set_cr(9, self.qrow_base_row)      # staged query rows
        state.regfile.set_cr(3, self.s_base_row)         # output base (single key group)
        state.regfile.set_cr(5, g0_start_rows)           # K-data startup (rows)
        state.regfile.set_cr(7, -1)                     # channel fixed_idx startup
        state.regfile.set_cr(8, D - 2)                  # contraction bound (58)

        state.regfile.set_lr(0, 0)                       # r_cyclic write-index / mask_shift
        state.regfile.set_lr(2, K_STRIDE_ROWS)           # K data stride per channel (rows)
        state.regfile.set_lr(3, N_TG * ACC_STORE_ROWS)   # output stride per query (rows)
        state.regfile.set_lr(6, D - 2)                   # contraction BLT bound
        state.regfile.set_lr(7, 0)                       # output query row offset
        state.regfile.set_lr(8, 0)                       # Q-row row offset
        state.regfile.set_lr(9, 0)                       # query counter
        state.regfile.set_lr(10, N)                      # query-loop limit
        state.regfile.set_lr(12, QROW_STRIDE_ROWS)       # Q-row stride per query (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N queries x N_TG groups x 512 B, query-major:
            #   row (i, g) at s_base_row + i*N_TG + g, in rows.
            # Each row is WHOLE even though only N of its 128 lanes are live;
            # the test crops to [:N].
            dump_xmem_to_binary(
                state, self.output_path,
                self.s_base, OUTPUT_ROW_BYTES, N * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's exact-match domain (n_tok=16,
# d=60). `k_base_row` is deliberately NOT folded into `supports`: it is not a
# shape the registry routes on, it is a hardware invariant (K_BASE is read from
# cr0, which the ISA hardwires to 0) that the constructor must keep enforcing
# regardless of how the app was reached. `build` therefore never passes
# k_base_row -- the constructor default already satisfies the invariant.


def _supports(**params):
    q = scores_query(n_tok=params["n_tok"], d=params["d"])
    bad = positive_dims(n_tok=q.n_tok, d=q.d)
    if bad:
        return no(bad)
    if q.n_tok != N:
        return no(f"handles exactly n_tok={N}; got {q.n_tok}")
    if q.d != D:
        return no(f"handles exactly d={D}; got {q.d}")
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"n_tok == {N} and d == {D} exactly: the L5 one-head QKT query-major "
        f"scores kernel (16 tokens fit a single 128-lane key group)."
    )


SPEC = KernelSpec(
    name="qk_scores_16x60",
    op="qk_scores",
    variant="16x60",
    app_class=QkScores16x60App,
    asm="qk_scores_16x60.asm",
    requires=("n_tok", "d"),
    tags=("fp32-wide", "query-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        query=(params["n_tok"], params["d"]), key=(params["n_tok"], params["d"])
    ).with_shapes(derived={"output": (params["n_tok"], params["n_tok"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
