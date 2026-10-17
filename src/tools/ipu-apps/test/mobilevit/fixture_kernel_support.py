"""Shared plumbing for the MobileViT cross-family tests in this directory.

* :func:`kernel_inst` -- a registered kernel's assembled binary, assembled once
  per process through the registry (:func:`~ipu_apps.kernel_registry.cases.assemble_kernel`,
  which finds the ``.asm`` via ``importlib.resources`` -- no source-tree path).
* :func:`skip_load_data` -- run a harness's real ``setup()`` with its
  module-level ``_load_data`` disabled, so a test can place a producer's raw
  XMEM bytes into the consumer's DATA region itself (the "direct XMEM
  handoff" seams) while weights and CR/LR programming stay exactly the
  harness's own.
* :func:`kernel_program` -- a //kernels kernel's program words for the given
  parameters (the packed chains load the registered packed kernels, E07-23).
* :func:`poison` -- fill XMEM rows with a value no real computation produces,
  so an under-write survives into the output.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from fixture_normalize import run_layernorm_rows

from ipu_apps.kernel_registry.cases import assemble_kernel
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat

POISON = np.float32(1e3)
LANES = 128

_INST: dict[str, Path] = {}
_INST_DIR: Path | None = None


def kernel_inst(name: str) -> Path:
    """Assemble (once per process) and return registered kernel ``name``'s binary."""
    global _INST_DIR
    if name not in _INST:
        if _INST_DIR is None:
            _INST_DIR = Path(tempfile.mkdtemp(prefix="mobilevit_inst_"))
        _INST[name] = assemble_kernel(name, _INST_DIR)
    return _INST[name]


def kernel_program(name: str, **params: object) -> bytes:
    """The //kernels kernel ``name``'s program in FP32, as a ``.bin`` file holds
    it: its default case's parameters, ``params`` replacing them."""
    manifest = default_registry().kernel(name)
    (case,) = (c for c in expand(manifest) if c.name == "default")
    plan = plan_case(manifest, {**case.values, **params}, ElementFormat.FP32)
    return assemble(plan).to_bin()


def poison(state, base: int, n_rows: int, lanes: int = LANES) -> None:
    """Fill ``n_rows`` rows of ``lanes`` FP32 lanes at byte address ``base`` with POISON."""
    state.xmem.write_address(
        base, bytearray(np.full(n_rows * lanes, POISON, dtype=np.float32).tobytes())
    )


def run_layernorm_capture_xmem(
    name: str, *, x_rows: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> bytes:
    """Run the //kernels/normalize entry ``name`` and return its raw output rows.

    ``x_rows`` is one zero-padded ``LANES``-lane FP32 row per ``(ch, tg)``,
    written to XMEM as it is (fixture_normalize). The output is read straight
    from XMEM after poisoning it, and must hold no untouched row.
    """
    run = run_layernorm_rows(name, x_rows, gamma, beta, poison=float(POISON))
    assert run.cycles > 0
    return run.rows.tobytes()


def run_matmul_direct_xmem_handoff(
    module,
    app_class,
    *,
    data_raw: bytes,
    W: np.ndarray,
    tmp_path: Path,
    tag: str,
    output_file: bool = True,
) -> np.ndarray:
    """Feed ``data_raw`` straight into a ``matmul_*_x128`` kernel's DATA region.

    DATA and OUTPUT are poisoned, ``data_raw`` is written verbatim at
    ``DATA_BASE``, and the harness's real ``setup()`` runs with ``_load_data``
    disabled -- so no file staging or repacking touches the handoff; only the
    weights go through the normal file path. Returns the raw OUTPUT region as
    ``[rows, LANES]`` FP32, read from XMEM (never through ``teardown()``).
    """
    assert len(data_raw) == module.DATA_ROWS * module.ROW_BYTES, (
        "byte-length mismatch between the producer's raw output region and the "
        "matmul's raw DATA region -- direct handoff is not even byte-shape compatible"
    )
    out_rows = module.N_OUT * getattr(module, "N_TG", 1)

    weights_path = tmp_path / f"mm_w_{tag}.bin"
    weights_path.write_bytes(W.astype(np.float32).tobytes())
    app = app_class(
        inst_path=kernel_inst(module.__package__.rpartition(".")[2]),
        input_path="/dev/null",  # placeholder; _load_data is skipped below
        weights_path=weights_path,
        output_path=tmp_path / f"mm_out_{tag}.bin" if output_file else None,
    )
    state = app.make_state()
    # Poison data+output first so a silently-skipped write anywhere in the
    # handoff shows up as 1e3, not as unrelated garbage.
    poison(state, module.DATA_BASE, module.DATA_ROWS)
    poison(state, module.OUTPUT_BASE, out_rows)

    # THE ACTUAL SEAM UNDER TEST: raw bytes, verbatim, no repacking.
    state.xmem.write_address(module.DATA_BASE, bytearray(data_raw))

    with skip_load_data(module):
        state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = bytes(state.xmem.read_address(module.OUTPUT_BASE, out_rows * module.ROW_BYTES))
    rows = np.frombuffer(raw, dtype=np.float32).reshape(out_rows, LANES)
    assert not np.all(rows == POISON, axis=1).any(), "matmul left poisoned output rows untouched"
    return rows


@contextmanager
def skip_load_data(module):
    """Replace ``module._load_data`` with a no-op for the duration of the block.

    The matmul ``*_x128`` harnesses call their module-level ``_load_data`` from
    ``setup()``; with it disabled, the harness stages only its weights and
    registers, and whatever the test already wrote to the DATA region is what
    the kernel reads.
    """
    original = module._load_data
    module._load_data = lambda state, data_path: None
    try:
        yield
    finally:
        module._load_data = original
