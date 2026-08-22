"""Selective-scan (Mamba / MambaVision SSM) application harness, FP32 wide-vector.

Computes the core selective scan of the Mamba state-space block -- the one part
of ``MambaVisionMixer`` that is not a matmul or a pointwise op, because it
carries state across the sequence axis::

    Abar[d,n,t] = exp(delta[d,t] * A[d,n])
    h[d,n,t]    = Abar[d,n,t] * h[d,n,t-1] + delta[d,t] * B[n,t] * u[d,t]
    y[d,t]      = SUM_n C[n,t] * h[d,n,t] + D_skip[d] * u[d,t]

with ``h[.,.,-1] = 0``. Shapes per batch: ``u, delta: (D, L)``, ``A: (D, N)``,
``B, C: (N, L)``, ``D_skip: (D,)``, ``y: (D, L)``.

``B`` and ``C`` carry no channel index -- in MambaVision they come out of
``x_proj`` as ``(b, d_state, l)`` and broadcast across all channels. That is what
makes the STATE-MAJOR layout below work.

Layout: lane = channel ``d``; one 128-lane row per state index ``n``. This is
forced: multiply masks are disabled in wide mode (see
docs/content/wide-vector-debug-mode.md), so the alternative "state-minor" layout
(lane = d*N+n) cannot do its segmented ``SUM_n`` reduction at all. State-major
turns ``SUM_n C[n,t]*h[.,n,t]`` into N vector-times-scalar multiplies plus
ACC.ADD -- no reduction instruction, no masks, no partitions -- and works
unchanged for any N with 2N <= 128.

WHAT RUNS WHERE
---------------
Everything that depends on the *input* runs on the device, including the
``softplus`` of ``delta_softplus=True``: the kernel reads ``dt + delta_bias``
and applies the native ``softplus`` activation itself (one extra cycle per
timestep). The bias add is not device work being skipped -- it is the bias of
the preceding ``dt_proj`` linear layer, which ``fully_connected`` would already
have folded in; Mamba only adds it separately for numerical reasons.

The one host-side transform is ``cA[d,n] = log2(e) * A[d,n]``, and that is a
*weight* refold, not activation work. MambaVision stores ``A_log`` in the
checkpoint and derives ``A = -exp(A_log)``, so ``cA = -log2(e) * exp(A_log)``
is simply a different constant folding of the same parameter -- computed once
per model, never per inference, exactly like folding batchnorm into a conv
weight (see ``ca_from_a_log``). What it buys is a single native ``exp2`` for
``Abar``: ``exp(delta*A)`` has no native activation, ``exp2`` does.

INSTANCE FLATTENING
-------------------
The kernel walks a flat list of "instances". An instance is one (batch, channel
tile) pair -- a 128-channel scan over all L timesteps. With ``T = ceil(D/128)``
tiles there are ``batch * T`` instances, and instance ``i = b*T + tile`` indexes
every region uniformly::

    U / DT / BC / Y     ->  row  i*L + t
    CA / H / ABAR       ->  row  i*N + n
    DSKIP               ->  row  i
    SCRATCH             ->  row  0 (du_t), row 1 (delta_t)

``B``/``C`` do not depend on the tile, so ``BC`` rows are replicated across the
T tiles of a batch. That costs a little XMEM (8 MB is plentiful) and buys one
uniform addressing rule for every region, which keeps the kernel's CR budget
inside the 13 writable CRs.

``H`` is indexed per instance rather than shared, so each instance starts on
freshly zeroed XMEM. That is why the kernel needs no state-zeroing prologue:
``h[.,.,-1] = 0`` falls out of XMEM being zero-initialised.

Usage::

    app = SelectiveScanApp(
        inst_path="selective_scan.bin",
        u=u, dt=dt, A=A, B=B, C=C, D_skip=D_skip, delta_bias=bias,
    )
    state, cycles = app.run()
    y = app.read_output(state)          # (batch, D, L)
"""

from __future__ import annotations

import math

import numpy as np
from ipu_emu.emulator import dump_xmem_to_binary
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp

# -- Constants --------------------------------------------------------------

LANES = 128                  # channels per row (fixed by the 128-element datapath)
ROW_BYTES = LANES * 4        # 512 bytes per FP32 row

LOG2E = math.log2(math.e)    # c = 1.4426950408889634

# Base of the XMEM map. Region bases are computed per instance in _layout()
# because every region scales with (instances, L, N). Deliberately non-zero:
# the literal 0 is CR0, and a zero base would make an addressing bug invisible.
XMEM_BASE = 0x10000


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(x))``, matching torch's threshold trick."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def ca_from_a_log(A_log: np.ndarray) -> np.ndarray:
    """Fold a MambaVision ``A_log`` checkpoint parameter straight into ``cA``.

    ``MambaVisionMixer`` stores ``A_log`` and computes ``A = -exp(A_log)`` at
    every forward pass; the kernel wants ``cA = log2(e) * A``. Composing the two
    gives ``cA = -log2(e) * exp(A_log)``, a pure constant transform of a weight
    -- so the runtime never needs either ``exp`` or the ``log2(e)`` multiply.
    """
    return (-LOG2E * np.exp(np.asarray(A_log, dtype=np.float32))).astype(np.float32)


def selective_scan_reference(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D_skip: np.ndarray,
) -> np.ndarray:
    """Golden recurrence, one batch. ``delta`` is already softplus'd.

    u, delta: (D, L);  A: (D, N);  B, C: (N, L);  D_skip: (D,)  ->  y: (D, L)
    """
    D, L = u.shape
    N = A.shape[1]
    h = np.zeros((D, N), dtype=np.float64)
    y = np.zeros((D, L), dtype=np.float64)
    for t in range(L):
        d_t = delta[:, t, None].astype(np.float64)
        h = np.exp(d_t * A) * h + d_t * B[None, :, t] * u[:, t, None]
        y[:, t] = h @ C[:, t] + D_skip * u[:, t]
    return y


class SelectiveScanApp(IpuApp):
    """Selective scan over (batch, D, L) with N state elements per channel.

    Args:
        u:        (batch, D, L) input sequence.
        dt:       (batch, D, L) raw timestep logits (pre-softplus).
        A:        (D, N) state matrix (negative in Mamba; refolded to cA).
        A_log:    (D, N) alternative to ``A``: the raw checkpoint parameter, with
                  ``A = -exp(A_log)`` as MambaVisionMixer computes it.
        B:        (batch, N, L) input projection, channel-independent.
        C:        (batch, N, L) output projection, channel-independent.
        D_skip:   (D,) skip-connection gain.
        delta_bias:      optional (D,) bias added to ``dt`` on the host; the
                         softplus itself runs on the device.
        delta_softplus:  must be True -- the kernel hard-wires the softplus
                         activation, matching MambaVision's call.
    """

    def __init__(
        self,
        *,
        u: np.ndarray,
        dt: np.ndarray,
        A: np.ndarray | None = None,
        A_log: np.ndarray | None = None,
        B: np.ndarray,
        C: np.ndarray,
        D_skip: np.ndarray,
        delta_bias: np.ndarray | None = None,
        delta_softplus: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if (A is None) == (A_log is None):
            raise ValueError("pass exactly one of A or A_log")
        self.u = np.asarray(u, dtype=np.float32)
        # cA is what the kernel actually reads; A is kept for the reference.
        if A_log is not None:
            self.cA = ca_from_a_log(A_log)
            self.A = (self.cA / LOG2E).astype(np.float32)
        else:
            self.A = np.asarray(A, dtype=np.float32)
            self.cA = (LOG2E * self.A).astype(np.float32)
        self.B = np.asarray(B, dtype=np.float32)
        self.C = np.asarray(C, dtype=np.float32)
        self.D_skip = np.asarray(D_skip, dtype=np.float32)

        self.batch, self.d_model, self.seq_len = self.u.shape
        self.d_state = self.A.shape[1]

        if 2 * self.d_state > LANES:
            raise ValueError(
                f"d_state={self.d_state}: BC row packs B and C side by side, so "
                f"2*d_state must fit in {LANES} lanes"
            )
        if self.A.shape[0] != self.d_model:
            raise ValueError(f"A has {self.A.shape[0]} channels, expected {self.d_model}")
        if self.B.shape != (self.batch, self.d_state, self.seq_len):
            raise ValueError(f"B shape {self.B.shape} != {(self.batch, self.d_state, self.seq_len)}")
        if self.C.shape != (self.batch, self.d_state, self.seq_len):
            raise ValueError(f"C shape {self.C.shape} != {(self.batch, self.d_state, self.seq_len)}")

        if not delta_softplus:
            raise ValueError(
                "delta_softplus=False is not supported: the kernel applies the "
                "softplus activation unconditionally (MambaVision always passes "
                "delta_softplus=True)"
            )
        # The device sees dt + delta_bias and softpluses it itself. The bias add
        # belongs to the preceding dt_proj linear layer, so it is not scan work.
        self.dtb = np.asarray(dt, dtype=np.float32)
        if delta_bias is not None:
            self.dtb = self.dtb + np.asarray(delta_bias, dtype=np.float32)[None, :, None]
        self.dtb = self.dtb.astype(np.float32)
        # Host-side delta, for the reference only -- never written to XMEM.
        self.delta = softplus(self.dtb).astype(np.float32)

        self.tiles = (self.d_model + LANES - 1) // LANES
        self.instances = self.batch * self.tiles
        self._layout()

    # -- XMEM map -----------------------------------------------------------

    def _layout(self) -> None:
        """Place every region back-to-back from XMEM_BASE, 64 KiB-aligned.

        Regions scale with (instances, L, N), so a fixed spacing will not do.
        Alignment keeps the addresses readable and leaves slack; the row-count
        arithmetic below is what actually guarantees no overlap.
        """
        seq_rows = self.instances * self.seq_len   # U / DELTA / BC / Y
        st_rows = self.instances * self.d_state    # CA / H / ABAR

        def step(rows: int) -> int:
            return ((rows * ROW_BYTES) + 0xFFFF) & ~0xFFFF

        addr = XMEM_BASE
        self.u_base = addr
        addr += step(seq_rows)
        self.dt_base = addr
        addr += step(seq_rows)
        self.bc_base = addr
        addr += step(seq_rows)
        self.y_base = addr
        addr += step(seq_rows)
        self.ca_base = addr
        addr += step(st_rows)
        self.h_base = addr
        addr += step(st_rows)
        self.abar_base = addr
        addr += step(st_rows)
        self.dskip_base = addr
        addr += step(self.instances)
        self.scratch_base = addr        # row 0 = du_t, row 1 = delta_t
        addr += step(2)
        self.xmem_end = addr

    def _channels(self, tile: int) -> slice:
        lo = tile * LANES
        return slice(lo, min(lo + LANES, self.d_model))

    def _pack_rows(self, x: np.ndarray, tile: int) -> np.ndarray:
        """(D, L) -> (L, 128) for one channel tile, zero-padded past d_model."""
        sl = self._channels(tile)
        out = np.zeros((x.shape[1], LANES), dtype=np.float32)
        out[:, : sl.stop - sl.start] = x[sl, :].T
        return out

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32 through
        AAQ (AAQ is a no-op); ACTIVATE writes FP32 into POST_AAQ_REG and
        STR_POST_AAQ_REG drains the full 512 bytes.
        """
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        # dtype is unused on the FP32 wide path, but several emulator helpers
        # branch on it; INT8 matches the existing wide-vector apps.
        state.dtype = DType.INT8
        return state

    def setup(self, state: IpuState) -> None:
        L, N = self.seq_len, self.d_state
        cA = self.cA

        u_rows = np.zeros((self.instances * L, LANES), dtype=np.float32)
        dt_rows = np.zeros((self.instances * L, LANES), dtype=np.float32)
        bc_rows = np.zeros((self.instances * L, LANES), dtype=np.float32)
        ca_rows = np.zeros((self.instances * N, LANES), dtype=np.float32)
        ds_rows = np.zeros((self.instances, LANES), dtype=np.float32)

        for b in range(self.batch):
            for tile in range(self.tiles):
                i = b * self.tiles + tile
                sl = self._channels(tile)
                w = sl.stop - sl.start

                u_rows[i * L : (i + 1) * L] = self._pack_rows(self.u[b], tile)
                dt_rows[i * L : (i + 1) * L] = self._pack_rows(self.dtb[b], tile)

                # One BC row per timestep serves every B[n,t] and C[n,t]:
                # B at elements 0..N-1, C at elements N..2N-1. Replicated per
                # tile so the kernel's row arithmetic stays uniform.
                bc_rows[i * L : (i + 1) * L, 0:N] = self.B[b].T
                bc_rows[i * L : (i + 1) * L, N : 2 * N] = self.C[b].T

                ca_rows[i * N : (i + 1) * N, :w] = cA[sl, :].T
                ds_rows[i, :w] = self.D_skip[sl]

        state.xmem.write_address(self.u_base, u_rows.tobytes())
        state.xmem.write_address(self.dt_base, dt_rows.tobytes())
        state.xmem.write_address(self.bc_base, bc_rows.tobytes())
        state.xmem.write_address(self.ca_base, ca_rows.tobytes())
        state.xmem.write_address(self.dskip_base, ds_rows.tobytes())

        # CR0 == 0 and CR1 == 1 are READ-ONLY hardware constants; the kernel
        # uses CR0 as its zero source and CR1 as the row stride / counter step.
        # All writable CRs hold ROW NUMBERS or plain integers -- .asm XMEM
        # operands are rows, not byte addresses (issue #179).
        set_cr = state.regfile.set_cr
        set_cr(2, self.u_base // ROW_BYTES)
        set_cr(3, self.dt_base // ROW_BYTES)
        set_cr(4, self.ca_base // ROW_BYTES)
        set_cr(5, self.bc_base // ROW_BYTES)
        set_cr(6, self.dskip_base // ROW_BYTES)
        set_cr(7, self.h_base // ROW_BYTES)
        set_cr(8, self.abar_base // ROW_BYTES)
        set_cr(9, self.y_base // ROW_BYTES)
        set_cr(10, self.scratch_base // ROW_BYTES)
        set_cr(11, N)
        set_cr(12, L)
        set_cr(13, self.instances)
        set_cr(14, LANES)

        # AGG.* / ACTIVATE.QUANTIZE read the active-element count from the
        # dstructure CR named on the instruction; the kernel names CR15 on all
        # of them, so process the full 128-lane FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    # -- results ------------------------------------------------------------

    def read_output(self, state: IpuState) -> np.ndarray:
        """Read the Y region back as ``(batch, d_model, seq_len)``."""
        L = self.seq_len
        raw = state.xmem.read_address(self.y_base, self.instances * L * ROW_BYTES)
        rows = np.frombuffer(raw, dtype=np.float32).reshape(self.instances * L, LANES)
        y = np.zeros((self.batch, self.d_model, L), dtype=np.float32)
        for b in range(self.batch):
            for tile in range(self.tiles):
                i = b * self.tiles + tile
                sl = self._channels(tile)
                w = sl.stop - sl.start
                y[b, sl, :] = rows[i * L : (i + 1) * L, :w].T
        return y

    def reference(self) -> np.ndarray:
        """Golden output for this app's inputs, ``(batch, d_model, seq_len)``."""
        return np.stack(
            [
                selective_scan_reference(
                    self.u[b], self.delta[b], self.A, self.B[b], self.C[b], self.D_skip
                )
                for b in range(self.batch)
            ]
        ).astype(np.float32)

    def teardown(self, state: IpuState) -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state,
                self.output_path,
                self.y_base,
                ROW_BYTES,
                self.instances * self.seq_len,
            )

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
