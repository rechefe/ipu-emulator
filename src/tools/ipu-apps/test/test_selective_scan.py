"""Correctness tests for the selective-scan app (FP32 wide-vector).

Assembles selective_scan.asm once per module and compares the kernel output
against the numpy recurrence across state sizes, channel counts (exact /
partial / multi tile), sequence lengths and batch sizes.

Two of the configs are the real MambaVision shapes. Only stages 3 and 4 run a
scan (stages 1-2 are ConvBlocks), ``d_state=8`` and ``expand=1`` are hard-coded
at the ``MambaVisionMixer(...)`` call site, and the scan runs on ``d_inner//2 =
dim/2`` channels because ``in_proj`` splits into x and z and only x is scanned.
For MambaVision-T (dim=80, stage width dim*2**i) at 224x224 with
``window_size=[8, 8, 14, 7]`` -- one window per stage, so no window batching --
that is D=160/L=196 for stage 3 and D=320/L=49 for stage 4. The other configs
are deliberately synthetic tile shapes, labelled as such.

The stability cases at the bottom are where a scan kernel actually breaks: a
decaying Abar must not underflow to NaN, an Abar of 1 must not drift over a
long sequence, and L=1 has a closed form.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ipu_apps.selective_scan import SelectiveScanApp
from ipu_as.lark_tree import assemble_to_bin_file

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/selective_scan/selective_scan.asm"
)

MAX_CYCLES = 40_000_000


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selective_scan.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _inputs(batch, d_model, d_state, seq_len, seed, a_scale=1.0, u_scale=1.0):
    """Mamba-shaped random inputs. A is negative, as in a real S4/Mamba block."""
    rng = np.random.RandomState(seed)
    return {
        "u": (rng.randn(batch, d_model, seq_len) * u_scale).astype(np.float32),
        "dt": (rng.randn(batch, d_model, seq_len) * 0.5 - 1.0).astype(np.float32),
        "A": (-np.abs(rng.randn(d_model, d_state)) * a_scale).astype(np.float32),
        "B": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "C": (rng.randn(batch, d_state, seq_len) * 0.5).astype(np.float32),
        "D_skip": (rng.randn(d_model) * 0.5).astype(np.float32),
    }


def _run(inst_file, **kw):
    app = SelectiveScanApp(inst_path=inst_file, **kw)
    state, cycles = app.run(max_cycles=MAX_CYCLES)
    return app.read_output(state), app.reference(), cycles


@pytest.mark.parametrize(
    "batch,d_model,d_state,seq_len,seed",
    [
        (1, 128, 8, 8, 0),     # synthetic: one exact tile, N=8 as MambaVision
        (1, 64, 8, 8, 1),      # synthetic: partial tile (64 of 128 lanes real)
        (1, 128, 16, 8, 2),    # synthetic: N=16, vanilla Mamba's d_state
        (1, 320, 8, 4, 3),     # synthetic: 3 tiles, last one partial
        (1, 128, 8, 1, 4),     # synthetic: degenerate L=1
        (2, 64, 8, 8, 5),      # synthetic: batch > 1
        (2, 64, 16, 16, 6),    # synthetic: batch > 1, larger state
        (1, 160, 8, 196, 7),   # MambaVision-T stage 3 (14x14 window, dim 320/2)
        (1, 320, 8, 49, 8),    # MambaVision-T stage 4 (7x7 window, dim 640/2)
        (2, 160, 8, 196, 9),   # MambaVision-T stage 3, 2 images
        (1, 128, 63, 4, 10),   # synthetic: N=63, widest BC row with padding left
        (1, 128, 64, 4, 11),   # synthetic: N=64, BC row exactly full -- no padding
    ],
)
def test_matches_reference(inst_file, batch, d_model, d_state, seq_len, seed):
    out, ref, cycles = _run(
        inst_file, **_inputs(batch, d_model, d_state, seq_len, seed)
    )
    assert cycles > 0
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() < 1e-4


def test_strong_decay_does_not_produce_nan(inst_file):
    """Very negative A with large delta drives Abar -> 0; state must decay to
    zero, not underflow into NaN or Inf."""
    kw = _inputs(1, 128, 8, 16, 11, a_scale=40.0)
    kw["dt"] = np.full_like(kw["dt"], 3.0)      # softplus(3) ~ 3.05
    out, ref, _ = _run(inst_file, **kw)
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() < 1e-4


def test_zero_A_is_running_sum(inst_file):
    """A = 0 gives Abar = 1, so the state is a pure running sum -- checks that
    nothing drifts over a long sequence."""
    kw = _inputs(1, 160, 8, 196, 12)     # MambaVision-T stage 3 length
    kw["A"] = np.zeros_like(kw["A"])
    out, ref, _ = _run(inst_file, **kw)
    assert np.abs(out - ref).max() < 1e-4


def test_single_timestep_closed_form(inst_file):
    """L=1 has h = delta*B*u exactly, so y = C*(delta*B*u) + D_skip*u."""
    kw = _inputs(1, 128, 8, 1, 13)
    out, _, _ = _run(inst_file, **kw)

    app = SelectiveScanApp(inst_path=inst_file, **kw)
    delta = app.delta[0][:, 0]                      # (D,)
    u0 = kw["u"][0][:, 0]                           # (D,)
    h = delta[:, None] * kw["B"][0][:, 0][None, :] * u0[:, None]   # (D, N)
    expect = h @ kw["C"][0][:, 0] + kw["D_skip"] * u0
    assert np.abs(out[0][:, 0] - expect).max() < 1e-4


def test_zero_input_gives_zero_output(inst_file):
    """All-zero u makes every term zero, so y must be exactly zero."""
    kw = _inputs(1, 128, 8, 8, 14)
    kw["u"] = np.zeros_like(kw["u"])
    out, _, _ = _run(inst_file, **kw)
    assert np.array_equal(out, np.zeros_like(out))


def test_padding_lanes_do_not_leak(inst_file):
    """A partial tile leaves lanes 40..127 unused; real channels must be
    unaffected by whatever those lanes compute."""
    kw = _inputs(1, 40, 8, 8, 15)
    out, ref, _ = _run(inst_file, **kw)
    assert out.shape == (1, 40, 8)
    assert np.abs(out - ref).max() < 1e-4


def test_a_log_matches_a(inst_file):
    """``A_log`` is what MambaVision stores; ``A = -exp(A_log)``. Passing either
    must give the same result, since cA is only a constant refold of the weight."""
    kw = _inputs(1, 128, 8, 8, 16)
    d_model, d_state = 128, 8
    # MambaVisionMixer's actual init: A = repeat(arange(1, d_state+1)), A_log = log(A).
    A_log = np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float32), (d_model, 1)))
    kw["A"] = (-np.exp(A_log)).astype(np.float32)

    out_a, ref, _ = _run(inst_file, **kw)
    kw_log = {k: v for k, v in kw.items() if k != "A"}
    out_log, _, _ = _run(inst_file, A_log=A_log, **kw_log)

    assert np.array_equal(out_a, out_log)
    assert np.abs(out_a - ref).max() < 1e-4


def test_softplus_runs_on_device(inst_file):
    """The kernel softpluses dt itself, so a delta_bias big enough to move
    softplus out of its linear region must still match the reference."""
    kw = _inputs(1, 128, 8, 8, 17)
    kw["dt"] = np.full_like(kw["dt"], -4.0)          # softplus(-4 + bias)
    kw["delta_bias"] = np.linspace(-6.0, 8.0, 128).astype(np.float32)
    out, ref, _ = _run(inst_file, **kw)
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() < 1e-4


def test_delta_softplus_false_is_rejected(inst_file):
    """The kernel hard-wires softplus, so the flag must not silently lie."""
    kw = _inputs(1, 128, 8, 8, 18)
    with pytest.raises(ValueError, match="delta_softplus"):
        SelectiveScanApp(inst_path=inst_file, delta_softplus=False, **kw)


def test_max_state_leaves_no_padding_in_bc_row(inst_file):
    """N=64 packs B into R1[0:64] and C into R1[64:128] with NOTHING left over,
    so lr_csel reaches 128+2N = 256, which wraps mod 2*128 back onto R0.

    This is a boundary test, not a regression test: the wrap is unreachable
    because out_loop runs exactly N times. See test_cycle_count_pins_loop_trip
    for the check that actually pins that.
    """
    out, ref, _ = _run(inst_file, **_inputs(1, 128, 64, 6, 20))
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() < 1e-4


@pytest.mark.parametrize("d_state,seq_len", [(8, 12), (16, 12), (64, 6)])
def test_cycle_count_pins_loop_trip(inst_file, d_state, seq_len):
    """The kernel costs 15 + 4N cycles per timestep, and that number is the only
    direct evidence of how many times the loops actually run.

    BLT reads its operands from the start-of-cycle SNAPSHOT, so in a single-word
    loop the counter's own ADD is invisible to the branch beside it and the loop
    runs N+1 times unless the counter is seeded to 1. abar_loop, out_loop and the
    timestep epilogue all have that shape. An extra pass is nearly undetectable
    from the output -- it reads the next instance's still-zeroed H rows, so it
    accumulates zero -- but it shows up here as +1 cycle per timestep per loop.
    """
    app = SelectiveScanApp(inst_path=inst_file, **_inputs(1, 128, d_state, seq_len, 22))
    _, cycles = app.run(max_cycles=MAX_CYCLES)

    # 6 fixed words (5 of constant setup + BKPT), then per instance 1 word to
    # seed lr_t and 2 to close the instance loop, around L timesteps of 15 + 4N.
    expect = 6 + app.instances * (3 + app.seq_len * (15 + 4 * d_state))
    assert cycles == expect


def test_state_wider_than_the_bc_row_is_rejected(inst_file):
    """2N must fit in one 128-lane row, since B and C share it."""
    kw = _inputs(1, 128, 65, 4, 21)
    with pytest.raises(ValueError, match="d_state"):
        SelectiveScanApp(inst_path=inst_file, **kw)
