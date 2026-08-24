"""Shared chain-running helpers for the full-transformer-layer tests
(`test_full_layer_l4.py`, `test_full_layer_l5.py`).

Every helper here runs a REAL kernel binary (never a numpy shortcut) and
follows the standing method for this round of work:
  * the destination XMEM region is POISONED (filled with a value no real
    computation produces) before the kernel runs, so an under-write shows up
    in the output rather than silently reading as an old zero-fill;
  * assertions run on the RAW STORED BYTES the kernel actually wrote
    (`teardown()`'s output file), not on some post-hoc cropped numpy view
    built by the test;
  * data crosses kernel boundaries by staging bytes directly into the next
    kernel's expected file layout (matching what a real chained-execution
    caller does: stage XMEM, not round-trip through `_load_data`'s
    tightly-packed-file assumption -- see kernel_docs/kernel_layer_map.md's
    "Not in this round" note on `_load_data`).

Q is pre-scaled by 1/sqrt(head_dim) once, right after the QKV projection and
before either attention chain, per kernel_docs/kernel_layer_map.md ("no
kernel applies the attention scale -- the design folds it into Q"). No kernel
in this repo does that fold today, so this harness does it on the host, in
the same place a real weight-folding generator eventually would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

_POISON = np.float32(1e3)


def relative_error_stats(got: np.ndarray, expected: np.ndarray) -> dict:
    """Per-element relative error of `got` against `expected`, reported both
    as a single worst-case scalar and as a distribution.

    Max-ABSOLUTE error (what earlier scale-experiment runs reported) is
    scale-blind -- it can't say whether 4e-3 is excellent or poor without
    knowing the tensor's magnitude, and it is a single worst-of-~thousands
    order statistic that a lone badly-conditioned element can dominate,
    making a chain-wide problem look identical to one outlier. Relative error
    fixes the first issue; reporting the full distribution (not just max)
    fixes the second, since the same max can come from one outlier or a
    tensor-wide drift.

    Two figures are returned:
      * "max_rel" -- max|got-expected| / max|expected| over the WHOLE tensor:
        the worst absolute deviation, normalized by the tensor's own scale.
        This is what the "expected 1e-5 to 1e-4" FP32 calibration in the task
        brief refers to.
      * "median"/"p99"/"max" under "elementwise" -- the distribution of
        |got-expected| / (|expected| + eps) computed PER ELEMENT, so a single
        bad element shows up as an outlier in the max/p99 rather than being
        indistinguishable from a tensor-wide drift (which would show up in
        the median too). `eps` guards elements where `expected` is near zero
        (e.g. softmax-adjacent values, or attention weights on masked-out
        positions) from producing a spurious huge or infinite ratio.
    """
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    abs_diff = np.abs(got - expected)
    denom_scale = float(np.max(np.abs(expected)))
    max_rel = float(np.max(abs_diff)) / denom_scale if denom_scale > 0 else float(np.max(abs_diff))

    eps = 1e-8 * max(denom_scale, 1.0)
    elementwise = abs_diff / (np.abs(expected) + eps)
    return {
        "max_rel": max_rel,
        "elementwise": {
            "median": float(np.median(elementwise)),
            "p99": float(np.percentile(elementwise, 99)),
            "max": float(np.max(elementwise)),
        },
    }


def format_relative_error_stats(stats: dict) -> str:
    e = stats["elementwise"]
    return (f"max_rel={stats['max_rel']:.3e}  "
            f"elementwise[median={e['median']:.3e} p99={e['p99']:.3e} max={e['max']:.3e}]")


def _new_state() -> IpuState:
    return IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)


def _poison_region(state: IpuState, base: int, n_rows: int, lanes: int) -> None:
    row = np.full(lanes, _POISON, dtype=np.float32).tobytes()
    for r in range(n_rows):
        state.xmem.write_address(base + r * lanes * 4, bytearray(row))


def _assert_no_poison(arr: np.ndarray, tag: str) -> None:
    assert not np.any(arr == _POISON), (
        f"{tag}: poison value {_POISON} survived into stored output -- "
        "producer under-wrote its declared output region"
    )


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

def run_layernorm(App, *, inst_bin: Path, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   n_ch: int, n_tok: int, row_bytes: int, tmp_path: Path, tag: str,
                   full_row_output: bool) -> np.ndarray:
    """Run a real layernorm_*x* kernel. `x` is [n_ch, n_tok] channel-major.

    Returns the valid [n_ch, n_tok] output (cropped here in the test, per the
    crop convention: this call site is always the final consumer of the raw
    file, whether or not the kernel's own teardown already cropped).
    """
    lanes = row_bytes // 4
    x_rows = np.zeros((n_ch, lanes), dtype=np.float32)
    x_rows[:, :n_tok] = x
    input_path = tmp_path / f"ln_x_{tag}.bin"
    input_path.write_bytes(x_rows.tobytes())
    gamma_path = tmp_path / f"ln_gamma_{tag}.bin"
    gamma_path.write_bytes(gamma.astype(np.float32).tobytes())
    beta_path = tmp_path / f"ln_beta_{tag}.bin"
    beta_path.write_bytes(beta.astype(np.float32).tobytes())
    output_path = tmp_path / f"ln_out_{tag}.bin"

    state = _new_state()
    app = App(inst_path=inst_bin, input_path=input_path, gamma_path=gamma_path,
              beta_path=beta_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    if full_row_output:
        got = raw.reshape(n_ch, lanes)[:, :n_tok]
    else:
        got = raw.reshape(n_ch, n_tok)
    return got, cycles, state


# ---------------------------------------------------------------------------
# QKV / OutProj / FFN1 / FFN2 projections (proj_*_p4, all 4 streams at once)
# ---------------------------------------------------------------------------

def run_proj_p4(App, *, inst_bin: Path, d_streams: list[np.ndarray], w: np.ndarray,
                 k: int, n_out: int, n_tok: int, n_stream: int, row_bytes: int,
                 tmp_path: Path, tag: str) -> tuple[list[np.ndarray], int, IpuState]:
    """Run a real proj_*_p4 kernel (QKV / OutProj / FFN1 / FFN2), all 4
    streams in one invocation. `d_streams[p]` is [k, n_tok] channel-major,
    tightly packed on disk (the kernel's own file contract -- see
    _load_stream_data in proj_qkv_192_p4/__init__.py). `w` is [n_out, k].

    Returns full uncropped [n_out, lanes] output per stream (the kernel's
    real teardown emits full rows; cropping to n_tok is the caller's job,
    done explicitly at each call site rather than inside this helper, so a
    caller staging this output verbatim into the next kernel's XMEM can
    choose not to crop).
    """
    lanes = row_bytes // 4
    input_paths = []
    for p in range(n_stream):
        p_path = tmp_path / f"proj_{tag}_d{p}.bin"
        p_path.write_bytes(d_streams[p].astype(np.float32).tobytes())
        input_paths.append(p_path)
    weights_path = tmp_path / f"proj_{tag}_w.bin"
    weights_path.write_bytes(w.astype(np.float32).tobytes())
    output_paths = [tmp_path / f"proj_{tag}_o{p}.bin" for p in range(n_stream)]

    state = _new_state()
    app = App(inst_path=inst_bin, input_paths=input_paths, weights_path=weights_path,
              output_paths=output_paths)
    _, cycles = app.run(max_cycles=30_000_000, state=state)
    assert cycles > 0

    outs = []
    for p in range(n_stream):
        raw = output_paths[p].read_bytes()
        assert len(raw) == n_out * row_bytes, (
            f"{tag} stream {p}: output is {len(raw)} B, expected {n_out * row_bytes} B"
        )
        outs.append(np.frombuffer(raw, dtype=np.float32).reshape(n_out, lanes))
    return outs, cycles, state


# ---------------------------------------------------------------------------
# Query-major attention chain: qk_scores_* -> softmax stub -> attn_v_*
# ---------------------------------------------------------------------------

def run_qk_scores_query_major_all_blocks(App, *, inst_bin: Path, q_blocks: np.ndarray,
                                          k_blocks: np.ndarray, n_block: int, d: int, n: int,
                                          row_bytes: int, tmp_path: Path, tag: str,
                                          poison_base: int | None = None,
                                          ) -> tuple[np.ndarray, int]:
    """L4 shape: ONE qk_scores_64x48 call scores all N_BLOCK (stream, head)
    blocks. q_blocks/k_blocks: [n_block, d, n] channel-major. Returns cropped
    [n_block, n, n] scores (query-major: row i = query i's scores over keys).
    """
    query_path = tmp_path / f"qk_{tag}_q.bin"
    query_path.write_bytes(q_blocks.astype(np.float32).tobytes())
    key_path = tmp_path / f"qk_{tag}_k.bin"
    key_path.write_bytes(k_blocks.astype(np.float32).tobytes())
    output_path = tmp_path / f"qk_{tag}_s.bin"

    state = _new_state()
    if poison_base is not None:
        _poison_region(state, poison_base, n_block * n, row_bytes // 4)

    app = App(inst_path=inst_bin, query_path=query_path, key_path=key_path,
              output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(n_block * n, n)
    _assert_no_poison(got, f"{tag} qk_scores query-major")
    return got.reshape(n_block, n, n), cycles


def run_qk_scores_query_major_one_head(App, *, inst_bin: Path, q_head: np.ndarray,
                                        k_head: np.ndarray, d: int, n: int, row_bytes: int,
                                        tmp_path: Path, tag: str) -> tuple[bytes, int]:
    """L5 shape: qk_scores_16x60 scores exactly ONE (stream, head) block per
    call, and its teardown emits FULL uncropped rows (unlike its L4 sibling).
    q_head/k_head: [d, n] channel-major. Returns the raw output bytes
    (N * ROW_BYTES), the layout attn_v_16x60 expects staged verbatim at
    PBASE + h*P_HEAD_STRIDE.
    """
    query_path = tmp_path / f"qk1_{tag}_q.bin"
    query_path.write_bytes(q_head.astype(np.float32).tobytes())
    key_path = tmp_path / f"qk1_{tag}_k.bin"
    key_path.write_bytes(k_head.astype(np.float32).tobytes())
    output_path = tmp_path / f"qk1_{tag}_s.bin"

    from ipu_apps.qk_scores_16x60 import S_BASE, OUTPUT_ROW_BYTES, N_TG

    state = _new_state()
    _poison_region(state, S_BASE, n * N_TG, row_bytes // 4)

    app = App(inst_path=inst_bin, query_path=query_path, key_path=key_path,
              output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    _assert_no_poison(arr.reshape(-1, row_bytes // 4)[:, :n], f"{tag} qk_scores_16x60")
    return raw, cycles


def run_attn_v_query_major_all_blocks(App, *, inst_bin: Path, p_blocks: np.ndarray,
                                       v_blocks: np.ndarray, n_block: int, d: int, n: int,
                                       row_bytes: int, tmp_path: Path, tag: str,
                                       ) -> tuple[np.ndarray, int]:
    """L4 shape: attn_v_64x48 consumes all N_BLOCK blocks' P (query-major,
    post-softmax) and V (channel-major) in one call. p_blocks: [n_block, n,
    n] (post-softmax, row i = query i). v_blocks: [n_block, d, n]
    channel-major.
    """
    lanes = row_bytes // 4
    p_rows = np.zeros((n_block * n, lanes), dtype=np.float32)
    p_rows[:, :n] = p_blocks.reshape(n_block * n, n)
    p_path = tmp_path / f"av_{tag}_p.bin"
    p_path.write_bytes(p_rows.tobytes())

    v_rows = np.zeros((n_block * d, lanes), dtype=np.float32)
    v_rows[:, :n] = v_blocks.reshape(n_block * d, n)
    v_path = tmp_path / f"av_{tag}_v.bin"
    v_path.write_bytes(v_rows.tobytes())

    output_path = tmp_path / f"av_{tag}_o.bin"

    from ipu_apps.attn_v_64x48 import OBASE, OUTPUT_ROW_BYTES as O_ROW_BYTES

    state = _new_state()
    _poison_region(state, OBASE, n_block * d, lanes)

    app = App(inst_path=inst_bin, p_path=p_path, v_path=v_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(n_block * d, n)
    _assert_no_poison(got, f"{tag} attn_v query-major")
    return got.reshape(n_block, d, n), cycles


def run_attn_v_query_major_one_stream(App, *, inst_bin: Path, p_bytes_per_head: list[bytes],
                                       v_stream: np.ndarray, n_head: int, d: int, n: int,
                                       row_bytes: int, tmp_path: Path, tag: str,
                                       ) -> tuple[np.ndarray, int]:
    """L5 shape: attn_v_16x60 consumes ONE stream's worth of P (all N_HEAD
    heads concatenated, each head's block built by 16 head-wise qk_scores
    calls upstream) and V (channel-major, all heads). v_stream: [n_head, d,
    n].
    """
    lanes = row_bytes // 4
    p_bytes = b"".join(p_bytes_per_head)
    p_path = tmp_path / f"av1_{tag}_p.bin"
    p_path.write_bytes(p_bytes)

    v_rows = np.zeros((n_head * d, lanes), dtype=np.float32)
    v_rows[:, :n] = v_stream.reshape(n_head * d, n)
    v_path = tmp_path / f"av1_{tag}_v.bin"
    v_path.write_bytes(v_rows.tobytes())

    output_path = tmp_path / f"av1_{tag}_o.bin"

    from ipu_apps.attn_v_16x60 import OBASE, O_CHAN_BYTES, N_CHAN

    state = _new_state()
    _poison_region(state, OBASE, N_CHAN, lanes)

    app = App(inst_path=inst_bin, p_path=p_path, v_path=v_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N_CHAN, lanes)[:, :n]
    _assert_no_poison(got, f"{tag} attn_v_16x60")
    return got.reshape(n_head, d, n), cycles


# ---------------------------------------------------------------------------
# Key-major attention chain: attn_scores_km_* -> softmax stub -> attn_v_bcast_*
# ---------------------------------------------------------------------------

def run_attn_scores_km_one_head(App, *, inst_bin: Path, q_all_heads: np.ndarray,
                                 k_all_heads: np.ndarray, head: int, n_block_or_stream: int,
                                 d: int, n: int, row_bytes: int, tmp_path: Path, tag: str,
                                 ) -> tuple[bytes, int]:
    """Both L4 and L5: attn_scores_km_* scores ONE head per call (L4: across
    all P streams -> P*N key rows; L5: single stream -> N key rows). Returns
    the raw uncropped output bytes (key-major, one key per whole row) --
    exactly what attn_v_bcast_* stages verbatim.
    """
    input_path = tmp_path / f"km_{tag}_h{head}_q.bin"
    input_path.write_bytes(q_all_heads.astype(np.float32).tobytes())
    weights_path = tmp_path / f"km_{tag}_h{head}_k.bin"
    weights_path.write_bytes(k_all_heads.astype(np.float32).tobytes())
    output_path = tmp_path / f"km_{tag}_h{head}_s.bin"

    state = _new_state()
    app = App(inst_path=inst_bin, input_path=input_path, weights_path=weights_path,
              output_path=output_path, head=head)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n_block_or_stream * n, row_bytes // 4)
    _assert_no_poison(arr[:, :n], f"{tag} attn_scores_km head {head}")
    return raw, cycles


def run_attn_v_bcast(App, *, inst_bin: Path, p_bytes: bytes, v_blocks: np.ndarray,
                      n_block: int, d: int, n: int, row_bytes: int, tmp_path: Path,
                      tag: str, obase_module, full_row_output: bool) -> tuple[np.ndarray, int]:
    """Both L4/L5 shape (attn_v_bcast_48 / attn_v_bcast_60): consumes
    key-major P staged verbatim from attn_scores_km_*'s raw output, plus
    channel-major V. `obase_module` is the imported attn_v_bcast module
    (for OBASE / O_CHAN_ROWS / N_CHAN / output row-byte constants).

    attn_v_bcast_48 (L4) crops its teardown output to N_TOK*ELEM_BYTES/row;
    attn_v_bcast_60 (L5) emits full uncropped ROW_BYTES/row -- pass
    full_row_output accordingly (see kernel_layer_map.md's crop-convention
    notes and this task's constructor-signature research).
    """
    lanes = row_bytes // 4
    p_path = tmp_path / f"bc_{tag}_p.bin"
    p_path.write_bytes(p_bytes)

    v_rows = np.zeros((n_block * d, lanes), dtype=np.float32)
    v_rows[:, :n] = v_blocks.reshape(n_block * d, n)
    v_path = tmp_path / f"bc_{tag}_v.bin"
    v_path.write_bytes(v_rows.tobytes())

    output_path = tmp_path / f"bc_{tag}_o.bin"

    state = _new_state()
    _poison_region(state, obase_module.OBASE, n_block * d, lanes)

    app = App(inst_path=inst_bin, p_path=p_path, v_path=v_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    if full_row_output:
        got = raw.reshape(n_block * d, lanes)[:, :n]
    else:
        got = raw.reshape(n_block * d, n)
    _assert_no_poison(got, f"{tag} attn_v_bcast")
    return got.reshape(n_block, d, n), cycles


# ---------------------------------------------------------------------------
# Residual add
# ---------------------------------------------------------------------------

def run_residual_add(App, *, inst_bin: Path, a: np.ndarray, b: np.ndarray, n_ch: int,
                      n_tok: int, row_bytes: int, tmp_path: Path, tag: str,
                      full_row_output: bool) -> tuple[np.ndarray, int]:
    """a, b: [n_ch, n_tok] channel-major. Returns cropped [n_ch, n_tok] sum."""
    lanes = row_bytes // 4
    a_rows = np.zeros((n_ch, lanes), dtype=np.float32)
    a_rows[:, :n_tok] = a
    b_rows = np.zeros((n_ch, lanes), dtype=np.float32)
    b_rows[:, :n_tok] = b

    a_path = tmp_path / f"res_{tag}_a.bin"
    a_path.write_bytes(a_rows.tobytes())
    b_path = tmp_path / f"res_{tag}_b.bin"
    b_path.write_bytes(b_rows.tobytes())
    output_path = tmp_path / f"res_{tag}_o.bin"

    state = _new_state()
    app = App(inst_path=inst_bin, input_a_path=a_path, input_b_path=b_path,
              output_path=output_path)
    _, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    if full_row_output:
        got = raw.reshape(n_ch, lanes)[:, :n_tok]
    else:
        got = raw.reshape(n_ch, n_tok)
    _assert_no_poison(got, f"{tag} residual_add")
    return got, cycles


# ---------------------------------------------------------------------------
# Real softmax kernels (softmax_rows_partial / softmax_columns_packed),
# drop-in replacements for fixture_host_softmax_fold_stubs.softmax_query_major
# / softmax_key_major. Assembled once per process (module-level cache) since
# the .asm is fixed and every call in a test run reuses the same binary --
# see test_softmax_kernels_vs_stub.py's _assemble for why reset_labels() is
# required before each assembly.
# ---------------------------------------------------------------------------

_REAL_SOFTMAX_INST: dict[str, Path] = {}


def _real_softmax_inst(name: str) -> Path:
    """Assemble (once) and cache softmax_rows_partial.asm / softmax_columns_packed.asm."""
    if name in _REAL_SOFTMAX_INST:
        return _REAL_SOFTMAX_INST[name]

    import tempfile
    from ipu_as.lark_tree import assemble_to_bin_file
    from ipu_as.label import reset_labels

    src_dir = Path(__file__).resolve().parents[1] / "src/ipu_apps/softmax"
    asm_rel = {
        "rows_partial": "softmax_rows_partial/softmax_rows_partial.asm",
        "columns_packed": "softmax_columns_packed/softmax_columns_packed.asm",
    }[name]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"real_softmax_{name}_"))
    inst_path = tmp_dir / f"{name}.bin"
    reset_labels()
    assemble_to_bin_file((src_dir / asm_rel).read_text(), str(inst_path))
    _REAL_SOFTMAX_INST[name] = inst_path
    return inst_path


def real_softmax_query_major(scores: np.ndarray, *, tmp_path: Path, tag: str) -> np.ndarray:
    """Real-kernel drop-in for fixture_host_softmax_fold_stubs.softmax_query_major.

    scores: [..., n_query, n_key], row i = query i's raw scores over all
    keys (n_query == n_key == N_TOK here; softmax_rows_partial's `rows`/`n`
    are exactly this shape). Reduces over the trailing axis, same contract
    as the stub. Leading axes (heads, blocks) are looped on the host since
    the kernel call itself only knows about one [rows, n] matrix at a time.
    """
    from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp

    inst_path = _real_softmax_inst("rows_partial")
    orig_shape = scores.shape
    n = orig_shape[-1]
    rows = orig_shape[-2]
    flat = scores.reshape(-1, rows, n).astype(np.float32)

    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        inp = tmp_path / f"rsm_{tag}_{i}_in.bin"
        outp = tmp_path / f"rsm_{tag}_{i}_out.bin"
        inp.write_bytes(flat[i].tobytes())
        app = SoftmaxRowsPartialApp(inst_path=inst_path, input_path=inp,
                                     output_path=outp, n=n, rows=rows)
        app.run(max_cycles=20_000_000)
        out[i] = np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(rows, n)

    return out.reshape(orig_shape).astype(np.float64)


def real_softmax_key_major(scores_km: np.ndarray, *, tmp_path: Path, tag: str) -> np.ndarray:
    """Real-kernel drop-in for fixture_host_softmax_fold_stubs.softmax_key_major.

    scores_km: [..., n_key, n_query], row s = key s's scores against every
    query (column q = query q). Reduces over the row (key) axis, same
    contract as the stub. softmax_columns_packed's `rows`/`width` map
    directly: rows=n_key, width=n_query (width <= 64 required -- true for
    both L4 (N_TOK=64) and L5 (N_TOK=16)).
    """
    from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

    inst_path = _real_softmax_inst("columns_packed")
    orig_shape = scores_km.shape
    width = orig_shape[-1]
    rows = orig_shape[-2]
    flat = scores_km.reshape(-1, rows, width).astype(np.float32)

    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        inp = tmp_path / f"ksm_{tag}_{i}_in.bin"
        outp = tmp_path / f"ksm_{tag}_{i}_out.bin"
        inp.write_bytes(flat[i].tobytes())
        app = SoftmaxColumnsPackedApp(inst_path=inst_path, input_path=inp,
                                       output_path=outp, rows=rows, width=width)
        app.run(max_cycles=20_000_000)
        out[i] = np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(rows, width)

    return out.reshape(orig_shape).astype(np.float64)
