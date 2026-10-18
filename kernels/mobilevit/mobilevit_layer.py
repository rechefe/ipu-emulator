"""One MobileViT transformer layer on one ``Chain`` (E07-24): the shared
steps of the full-layer chain tests, and their float64 reference.

    LayerNorm -> QKV projection -> scores -> softmax -> attn@V -> OutProj
      -> residual add -> LayerNorm -> FFN1 (SiLU) -> FFN2 -> residual add

Every step is a registered kernel run on the chain's model, so a step reads
what the step before it left in XMEM. Where a producer's stored rows are the
consumer's layout, the hand-off is a verbatim copy or a relabel of those rows
(:func:`relabel`, :func:`copy_to_row_0`); where no base or stride bridges the
two (the attention seams: a head's channels into a transposed block, scores
into the softmax kernels' slot layout), the host repacks what :meth:`Chain.read`
or a step's unpacked output gives into the next step's array input, as the
seam tests do.

The projections run all four streams at once. Stream 0 carries the layer
under test; the other three are filler (``rng("mobilevit", tag, "filler")``,
E06-08) at the two LayerNorm outputs, and from there each is what the chain
computes for it, so every attention block is real data.

The reference is the shared ``ipu_kernels.reference.ops`` chained in float64
(``ops.layernorm`` has no eps, Q57: the one LayerNorm semantic). Each step is
also held to its own golden where its tier holds on chained data (LayerNorm,
the projections, the residual adds, softmax), and the score and attn@V steps
to their datapath mirrors bit for bit (chained, cancelling sums exceed
FP32_ORDER's floor, as on the seams). A step that misses its golden is
recorded, not raised, so one run reports every miss (:attr:`LayerRun.misses`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ipu_harness.cases import rng
from ipu_harness.chain import Chain, ChainStep, Resident
from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference import ops
from ipu_kernels.reference.datapath import acc_fold, agg_fold

FP32 = ElementFormat.FP32
POISON = 1e3

#: The stage boundaries of one layer, in order; the keys of :func:`reference`.
STAGES = (
    "ln1",
    "qkv",
    "scores",
    "softmax",
    "attn_v",
    "out_proj",
    "resid1",
    "ln2",
    "ffn1",
    "ffn2",
    "resid2",
)


# -- the float64 reference ---------------------------------------------------------


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(scores, probs, out)`` of ``[.., d, n]`` channel-major Q, K, V:
    scores ``[.., query, key]`` (``ops.matmul``), their softmax over the keys
    (``ops.softmax``) and ``V @ P^T`` ``[.., d, query]`` (``ops.matmul_nt``)."""
    scores = ops.matmul(q.transpose(-1, -2), k)["y"]
    probs = ops.softmax(scores, dim=-1)["y"]
    return scores, probs, ops.matmul_nt(v, probs)["y"]


def reference(
    x: np.ndarray, w: Mapping[str, np.ndarray], *, heads: int, scale: float = 1.0
) -> dict[str, np.ndarray]:
    """One layer of one stream in float64, every stage boundary by name (:data:`STAGES`).

    Args:
        x: The layer input ``[channels, tokens]``.
        w: The layer's weights (:func:`weights`' keys).
        heads: Attention heads; the head dimension is ``channels / heads``.
        scale: The attention scale applied to Q (1: none, as the kernels run).

    Returns:
        ``scores`` and ``softmax`` ``[heads, query, key]``; every other stage
        ``[channels, tokens]`` (``qkv`` ``[3 * channels, tokens]``, ``ffn1``
        ``[hidden, tokens]``).
    """
    xt = _t(x)
    channels, tokens = x.shape
    hd = channels // heads
    out: dict[str, torch.Tensor] = {}
    out["ln1"] = ops.layernorm(xt, _t(w["ln1_gamma"]), _t(w["ln1_beta"]), dim=0)["y"]
    out["qkv"] = ops.linear(_t(w["w_qkv"]), out["ln1"])["y"]
    q, k, v = (
        out["qkv"][i * channels : (i + 1) * channels].reshape(heads, hd, tokens) for i in range(3)
    )
    out["scores"], out["softmax"], o = attention(q * scale, k, v)
    out["attn_v"] = o.reshape(channels, tokens)
    return {name: t.numpy() for name, t in out.items()} | reference_tail(
        out["attn_v"].numpy(), x, w
    )


def reference_tail(
    attn: np.ndarray, x: np.ndarray, w: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The stages from ``out_proj`` on (:func:`reference`), from the attention
    output ``attn`` and the layer input ``x`` (the first residual's skip)."""
    out: dict[str, torch.Tensor] = {}
    out["out_proj"] = ops.linear(_t(w["w_outproj"]), _t(attn))["y"]
    out["resid1"] = ops.residual_add(out["out_proj"], _t(x))["y"]
    out["ln2"] = ops.layernorm(out["resid1"], _t(w["ln2_gamma"]), _t(w["ln2_beta"]), dim=0)["y"]
    out["ffn1"] = ops.linear(_t(w["w_ffn1"]), out["ln2"], activation="silu")["y"]
    out["ffn2"] = ops.linear(_t(w["w_ffn2"]), out["ffn1"])["y"]
    out["resid2"] = ops.residual_add(out["ffn2"], out["resid1"])["y"]
    return {name: t.numpy() for name, t in out.items()}


def weights(
    rs: np.random.RandomState,
    channels: int,
    hidden: int,
    *,
    gamma: tuple[float, float],
    beta: tuple[float, float],
    w: tuple[float, float],
) -> dict[str, np.ndarray]:
    """One layer's weights, uniform in the given ranges, drawn in the old tests' order."""

    def u(bounds: tuple[float, float], shape: tuple[int, ...]) -> np.ndarray:
        return rs.uniform(*bounds, size=shape).astype(np.float32)

    return dict(
        ln1_gamma=u(gamma, (channels,)),
        ln1_beta=u(beta, (channels,)),
        w_qkv=u(w, (3 * channels, channels)),
        w_outproj=u(w, (channels, channels)),
        ln2_gamma=u(gamma, (channels,)),
        ln2_beta=u(beta, (channels,)),
        w_ffn1=u(w, (hidden, channels)),
        w_ffn2=u(w, (channels, hidden)),
    )


# -- chain plumbing --------------------------------------------------------------


def acc_mirror(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` as the ACC datapath rounds it (the score kernels, ``attn_v_bcast_*``)."""
    a3, b3 = a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])
    return acc_fold(np.swapaxes(b3, -1, -2), a3).reshape(*a.shape[:-1], b.shape[-1])


def agg_mirror(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b^T`` as the AGG datapath rounds it (``attn_v_*``: ``b`` = P, ``a`` = V)."""
    a3, b3 = a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])
    return agg_fold(b3, a3).reshape(*a.shape[:-1], b.shape[-2])


def fixed(name: str) -> dict:
    """A registry entry's parameter values: its fixed ones over the template's defaults."""
    manifest = default_registry().kernel(name)
    return {k: d.default for k, d in manifest.params.items()} | dict(manifest.fixed)


@dataclass
class LayerRun:
    """What a chain of layers recorded.

    Attributes:
        cycles: Cycles by kernel, summed over the steps.
        misses: Each step that missed its own golden: ``"<kernel>: <comparison>"``.
    """

    cycles: dict[str, int] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)

    def run(
        self,
        chain: Chain,
        kernel: str,
        params: Mapping[str, object],
        inputs: Mapping[str, np.ndarray | Resident],
        **options: Any,
    ) -> ChainStep:
        """One ``chain.run`` (``options``: its ``check``, ``at``, ``reset``),
        the output ``y`` poisoned first unless ``at`` places it or
        ``kernel.toml`` pins it; a fault fails the step, a missed golden is
        recorded."""
        # An entry's fixed parameters are given, not left out (a left-out
        # one takes the template's default, which the entry may not allow).
        manifest = default_registry().kernel(kernel)
        params = {**manifest.fixed, **params}
        targets = dict(options.pop("at", {}))
        if "y" not in targets and "y" not in plan_case(manifest, dict(params), FP32).pins:
            inputs = _pinned_first(chain, kernel, params, inputs)
            targets["y"] = poisoned(chain, kernel, params)
        step = chain.run(kernel, params, inputs, at=targets, **options)
        assert step.result.halted and step.result.fault is None, step.result.describe()
        self.misses += [
            f"{kernel}: {c.describe()}" for c in step.result.comparisons if not c.passed
        ]
        self.cycles[kernel] = self.cycles.get(kernel, 0) + step.result.cycles
        y = np.asarray(step.result.outputs["y"])
        assert not np.any(y == POISON), (
            f"{kernel} left part of its output unwritten (poison survived)"
        )
        return step


def _pinned_first(
    chain: Chain,
    kernel: str,
    params: Mapping[str, object],
    inputs: Mapping[str, np.ndarray | Resident],
) -> dict[str, np.ndarray | Resident]:
    """``inputs`` with each array input ``kernel.toml`` pins to a row written
    there first (a row per leading index, zero past the last axis, as the
    inputs pinned here are laid out), so a poisoned output placed after it
    keeps off its rows."""
    manifest = default_registry().kernel(kernel)
    plan = plan_case(manifest, dict(params), FP32)
    out = dict(inputs)
    for name, row in plan.pins.items():
        value = inputs.get(name)
        if value is None or isinstance(value, Resident):
            continue
        rows = stream_rows(np.asarray(value))
        assert rows.shape[0] == stored_rows(manifest.tensors[name], plan), f"{kernel}.{name}"
        out[name] = chain.write(rows, at=row, pack=manifest.tensors[name].pack)
    return out


def poisoned(chain: Chain, kernel: str, params: Mapping[str, object]) -> Resident:
    """Rows for ``kernel``'s output ``y``, laid out as it is, filled with poison
    in the lanes it stores (zero past them), for the step's ``at``."""
    manifest = default_registry().kernel(kernel)
    y = manifest.tensors["y"]
    plan = plan_case(manifest, dict(params), FP32)
    rows = np.full((stored_rows(y, plan), LANES), POISON)
    live = plan.shapes["y"][-1] if plan.shapes["y"][-1] < LANES else LANES
    rows[:, live:] = 0.0
    return chain.write(rows, pack=y.pack)


def relabel(chain: Chain, resident: Resident, pack: str | None) -> Resident:
    """``resident``'s rows, unchanged, as a tensor laid out with ``pack``: the
    consumer's name for the same layout (a verbatim rewrite in place)."""
    return chain.write(chain.read(resident), at=resident.base_row, pack=pack)


def copy_to_row_0(chain: Chain, resident: Resident, pack: str | None = None) -> Resident:
    """``resident``'s rows, verbatim, at row 0: the input of a program that
    reads its base through CR0."""
    return chain.write(chain.read(resident), at=0, pack=pack)


def stream_rows(values: np.ndarray) -> np.ndarray:
    """``[.., tokens]`` values as whole rows, zero past the tokens (as a pack lays them)."""
    flat = values.reshape(-1, values.shape[-1])
    rows = np.zeros((flat.shape[0], LANES))
    rows[:, : flat.shape[1]] = flat
    return rows


# -- one layer -------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """One MobileViT layer's shape and kernels.

    Attributes:
        name: ``"L4"`` or ``"L5"``.
        channels: d_model.
        tokens: Tokens per stream.
        hidden: FFN1's outputs.
        layernorm, qkv, outproj, ffn1, ffn2, residual: The entries.
        scores, values: The query-major attention kernels.
        scores_km, values_km: The key-major ones.
        score_blocks: (stream, head) blocks per query-major score step (16:
            all, in one step; 1: one block a step).
        km_streams: Streams per key-major score step (one head each).
        value_blocks: Blocks per attn@V step (16: all; 4: one stream's heads).
        heads, streams: 4 each.
    """

    name: str
    channels: int
    tokens: int
    hidden: int
    layernorm: str
    qkv: str
    outproj: str
    ffn1: str
    ffn2: str
    residual: str
    scores: str
    values: str
    scores_km: str
    values_km: str
    score_blocks: int
    km_streams: int
    value_blocks: int
    heads: int = 4
    streams: int = 4

    @property
    def head_dim(self) -> int:
        return self.channels // self.heads

    @property
    def blocks(self) -> int:
        return self.streams * self.heads

    @property
    def attention_params(self) -> dict[str, int]:
        return {"n_tok": self.tokens, "d": self.head_dim}


L4 = Layer(
    "L4", 192, 64, 384,
    "layernorm_64x192", "proj_qkv_192_p4", "proj_outproj_192_p4", "proj_ffn1_192_p4",
    "proj_ffn2_192_p4", "residual_add_64x192",
    "qk_scores_64x48", "attn_v_64x48", "attn_scores_km_64x48", "attn_v_bcast_48",
    score_blocks=16, km_streams=4, value_blocks=16,
)  # fmt: skip
L5 = Layer(
    "L5", 240, 16, 480,
    "layernorm_16x240", "proj_qkv_240_p4", "proj_outproj_240_p4", "proj_ffn1_240_p4",
    "proj_ffn2_240_p4", "residual_add_16x240",
    "qk_scores_16x60", "attn_v_16x60", "attn_scores_km_16x60", "attn_v_bcast_60",
    score_blocks=1, km_streams=1, value_blocks=4,
)  # fmt: skip


def _project(run: LayerRun, chain: Chain, kernel: str, data: Resident, w: np.ndarray) -> Resident:
    """A projection step (all streams) on ``data`` (pack ``channel_major``), held to its golden."""
    return run.run(chain, kernel, {}, {"b": data, "a": w}, check=True).resident["y"]


def _with_filler(
    chain: Chain, layer: Layer, rows: np.ndarray, filler: np.random.Generator
) -> Resident:
    """Stream 0's stored rows, verbatim, then three filler streams: a projection's ``b``."""
    k = rows.shape[0]
    other = filler.uniform(-1.0, 1.0, size=(layer.streams - 1, k, layer.tokens))
    data = np.concatenate([rows, stream_rows(other.astype(np.float32))])
    return chain.write(data, pack="channel_major")


def _query_major(
    run: LayerRun, chain: Chain, layer: Layer, q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, softmax (``softmax_rows_partial``) and attn@V over every
    block ``[blocks, d, n]``: ``(scores, probs)`` ``[blocks, query, key]`` and
    the output ``[blocks, d, query]``."""
    n, b = layer.tokens, layer.score_blocks
    params = layer.attention_params
    scores = []
    for first in range(0, layer.blocks, b):
        a_in = np.ascontiguousarray(np.swapaxes(q[first : first + b], -1, -2))
        b_in = k[first : first + b]
        if b == 1:
            a_in, b_in = a_in[0], b_in[0]
        step = run.run(chain, layer.scores, params, {"b": b_in, "a": a_in})
        got = np.asarray(step.result.outputs["y"], dtype=np.float32)
        np.testing.assert_array_equal(got, acc_mirror(a_in, b_in))
        scores.append(got.reshape(-1, n, n))
    s = np.concatenate(scores)
    sm = run.run(
        chain, "softmax_rows_partial", {"rows": layer.blocks * n, "n": n},
        {"x": s.reshape(-1, n)}, check=True,
    )  # fmt: skip
    p = np.asarray(sm.result.outputs["y"], dtype=np.float32).reshape(s.shape)
    out = []
    for first in range(0, layer.blocks, layer.value_blocks):
        pb, vb = p[first : first + layer.value_blocks], v[first : first + layer.value_blocks]
        step = run.run(chain, layer.values, params, {"b": pb, "a": vb})
        got = np.asarray(step.result.outputs["y"], dtype=np.float32)
        np.testing.assert_array_equal(got, agg_mirror(vb, pb))
        out.append(got)
    return s, p, np.concatenate(out)


def _key_major(
    run: LayerRun, chain: Chain, layer: Layer, q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The key-major chain over every block ``[blocks, d, n]`` (block
    ``p * heads + h``): ``attn_scores_km_*`` (one head a step, ``km_streams``
    streams), ``softmax_columns_packed`` down each key column (as many blocks
    side by side as fit 64 lanes), ``attn_v_bcast_*``. Returns scores and
    probabilities query-major ``[blocks, query, key]`` and the output."""
    n, heads, ps = layer.tokens, layer.heads, layer.km_streams
    params = layer.attention_params
    qb = q.reshape(layer.streams, heads, *q.shape[1:])
    kb = k.reshape(layer.streams, heads, *k.shape[1:])
    s_km = np.zeros((layer.streams, heads, n, n), np.float32)  # [p, h, key, query]
    for first in range(0, layer.streams, ps):
        for h in range(heads):
            b_in = qb[first : first + ps, h]
            a_in = np.ascontiguousarray(np.swapaxes(kb[first : first + ps, h], -1, -2))
            if ps == 1:
                b_in, a_in = b_in[0], a_in[0]
            step = run.run(chain, layer.scores_km, {**params, "head": h}, {"b": b_in, "a": a_in})
            got = np.asarray(step.result.outputs["y"], dtype=np.float32)
            np.testing.assert_array_equal(got, acc_mirror(a_in, b_in))
            s_km[first : first + ps, h] = got.reshape(-1, n, n)
    blocks = s_km.reshape(layer.blocks, n, n)
    side = max(1, 64 // n)  # softmax_columns_packed takes width <= 64
    p_km = np.zeros_like(blocks)
    for first in range(0, layer.blocks, side):
        group = blocks[first : first + side]
        x = np.concatenate(list(group), axis=1)  # [key, side * query]
        sm = run.run(
            chain, "softmax_columns_packed", {"rows": n, "width": x.shape[1]}, {"x": x},
            check=True,
        )  # fmt: skip
        y = np.asarray(sm.result.outputs["y"], dtype=np.float32)
        p_km[first : first + side] = np.stack(np.split(y, len(group), axis=1))
    out = []
    for first in range(0, layer.blocks, layer.value_blocks):
        pb, vb = p_km[first : first + layer.value_blocks], v[first : first + layer.value_blocks]
        step = run.run(chain, layer.values_km, params, {"b": pb, "a": vb})
        got = np.asarray(step.result.outputs["y"], dtype=np.float32)
        np.testing.assert_array_equal(got, acc_mirror(vb, pb))
        out.append(got)
    return np.swapaxes(blocks, -1, -2), np.swapaxes(p_km, -1, -2), np.concatenate(out)


#: The two attention mappings, never mixed: query-major (AGG) and key-major (ACC.ADD).
MAPPINGS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "query_major": _query_major,
    "key_major": _key_major,
}


def run_layer(  # noqa: PLR0913 -- a layer's chain, input and weights, and the probe's knobs
    run: LayerRun,
    chain: Chain,
    layer: Layer,
    x: np.ndarray,
    w: Mapping[str, np.ndarray],
    *,
    mapping: str,
    tag: str,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """One layer on ``chain``: x ``[channels, tokens]`` (stream 0) in, every
    stage boundary of stream 0 out (:data:`STAGES`, the reference's shapes).

    ``scale`` multiplies Q on the host before the score steps, as folding the
    attention scale into W_q would (no kernel applies it).
    """
    filler = rng("mobilevit", tag, "filler")
    c, n, hd = layer.channels, layer.tokens, layer.head_dim
    got: dict[str, np.ndarray] = {}

    def values(resident: Resident, rows: int | None = None) -> np.ndarray:
        stored = chain.read(resident)
        return stored[: rows or stored.shape[0], :n]

    # x at row 0: LayerNorm reads it through CR0.
    x0 = chain.write(stream_rows(x.astype(np.float32)), at=0)
    ln1 = run.run(
        chain, layer.layernorm, {}, {"x": x0, "gamma": w["ln1_gamma"], "beta": w["ln1_beta"]},
        check=True,
    ).resident["y"]  # fmt: skip
    got["ln1"] = values(ln1)
    qkv = _project(
        run, chain, layer.qkv, _with_filler(chain, layer, chain.read(ln1), filler), w["w_qkv"]
    )
    stored = values(qkv).reshape(layer.streams, 3, layer.heads, hd, n)
    got["qkv"] = stored[0].reshape(3 * c, n)
    q, k, v = (stored[:, i].reshape(layer.blocks, hd, n) for i in range(3))
    s, p, o = MAPPINGS[mapping](run, chain, layer, (q * np.float32(scale)).astype(np.float32), k, v)
    got["scores"], got["softmax"] = s[: layer.heads], p[: layer.heads]
    attn = o.reshape(layer.streams, 1, c, n)
    got["attn_v"] = attn[0, 0]
    proj = _project(
        run,
        chain,
        layer.outproj,
        chain.write(stream_rows(attn), pack="channel_major"),
        w["w_outproj"],
    )
    got["out_proj"] = values(proj, c)
    # resid1 = out_proj + x: x (the layer input) at row 0 as a, OutProj's stream 0 rows as b.
    proj0 = relabel(chain, proj.slice(0, c), None)
    resid1 = run.run(
        chain, layer.residual, {}, {"a": x.astype(np.float32), "b": proj0}, check=True
    ).resident["y"]
    got["resid1"] = values(resid1)
    skip = chain.hold(copy_to_row_0(chain, resid1))  # LayerNorm's x, then residual 2's a
    ln2 = run.run(
        chain, layer.layernorm, {}, {"x": skip, "gamma": w["ln2_gamma"], "beta": w["ln2_beta"]},
        check=True,
    ).resident["y"]  # fmt: skip
    got["ln2"] = values(ln2)
    ffn1 = _project(
        run, chain, layer.ffn1, _with_filler(chain, layer, chain.read(ln2), filler), w["w_ffn1"]
    )
    got["ffn1"] = values(ffn1, layer.hidden)
    ffn2 = _project(run, chain, layer.ffn2, relabel(chain, ffn1, "channel_major"), w["w_ffn2"])
    got["ffn2"] = values(ffn2, c)
    resid2 = run.run(
        chain, layer.residual, {}, {"a": skip, "b": relabel(chain, ffn2.slice(0, c), None)},
        check=True,
    ).resident["y"]  # fmt: skip
    chain.release(skip)
    got["resid2"] = values(resid2)
    return {name: a.astype(np.float64) for name, a in got.items()}


# -- the full-layer tests' drivers -----------------------------------------------


def layer_weights(rs: np.random.RandomState, layer: Layer) -> dict[str, np.ndarray]:
    """The unpacked tests' weights: gamma in [0.5, 1.5], beta in [-0.5, 0.5], W in [-0.3, 0.3]."""
    return weights(
        rs, layer.channels, layer.hidden, gamma=(0.5, 1.5), beta=(-0.5, 0.5), w=(-0.3, 0.3)
    )


def run_stack(
    layer: Layer, mapping: str, *, seed: int, layers: int, scale: float = 1.0
) -> tuple[list[float], LayerRun]:
    """``layers`` layers back to back on one chain, stream 0's output of one
    the next one's input, each with its own weights; each layer's output held
    to the float64 reference run from the reference's own previous output.

    Returns:
        Each layer's max abs error, and what the run recorded.
    """
    rs = np.random.RandomState(seed)
    x = rs.uniform(-1.0, 1.0, size=(layer.channels, layer.tokens)).astype(np.float32)
    x_ref = x.astype(np.float64)
    run, chain = LayerRun(), Chain(FP32)
    errors = []
    for i in range(layers):
        w = layer_weights(rs, layer)
        got = run_layer(
            run, chain, layer, x, w, mapping=mapping, tag=f"{mapping}_l{i}", scale=scale
        )
        want = reference(x_ref, w, heads=layer.heads, scale=scale)["resid2"]
        errors.append(float(np.max(np.abs(got["resid2"] - want))))
        print(f"{layer.name} {mapping} scale={scale:.6f} layer {i}: max abs error {errors[-1]:.3e}")
        x, x_ref = got["resid2"].astype(np.float32), want
    print(f"{layer.name} {mapping} cycles by kernel: {dict(sorted(run.cycles.items()))}")
    return errors, run


def stage_errors(
    layer: Layer, *, seed: int, tag: str, scale: float = 1.0
) -> tuple[dict[str, float], LayerRun]:
    """One query-major layer, each stage boundary of stream 0 against the
    float64 reference of that same stage (from the layer input, not from the
    kernels' previous stage): where error enters, stage by stage."""
    rs = np.random.RandomState(seed)
    x = rs.uniform(-1.0, 1.0, size=(layer.channels, layer.tokens)).astype(np.float32)
    w = layer_weights(rs, layer)
    run = LayerRun()
    got = run_layer(run, Chain(FP32), layer, x, w, mapping="query_major", tag=tag, scale=scale)
    want = reference(x, w, heads=layer.heads, scale=scale)
    errors = {name: float(np.max(np.abs(got[name] - want[name]))) for name in STAGES}
    for name, err in errors.items():
        print(f"{layer.name} scale={scale:.6f} {name:<9} max abs error {err:.3e}")
    return errors, run
