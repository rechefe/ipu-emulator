"""One MobileViT layer in the packed layout on one ``Chain`` (E07-24, the
E07-23 kernels): the shared steps of the packed full-layer chain tests.

    packed LayerNorm -> QKV (packed linear) -> unpack Q, K, V -> attention
      -> pack -> OutProj -> residual add -> packed LayerNorm -> FFN1 (SiLU)
      -> FFN2 -> residual add

Every packed tensor is declared as its rows (``[channels * tokens / 128,
128]``, row-major over the flattened ``(channels, tokens)`` tensor, pack
``None``), so one packed step's output is the next one's input as it stands,
as a ``Resident``: no repack, no relabel. A packed linear step writes one
packed row; a projection is one step per row (``chain.reserve`` of the
whole output, each step writing its slice, in row order) over the held data.
Attention leaves the packed layout (scores have no channel axis): the
unpacked Q, K, V go through the attention kernels as in
:mod:`mobilevit_layer`, and the pack kernel takes the result back.

The residual add reads ``a`` through CR0, so ``a`` sits at row 0: the layer
input X is written there and held until the first add; the second add's
``a`` (the first add's output) is copied there verbatim. The pack, unpack
and linear programs leave R_MASK one-hot and the residual add assumes all
ones: each add resets it (``reset=("r_mask",)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from mobilevit_layer import L4, L5, MAPPINGS, POISON, Layer, LayerRun, copy_to_row_0

from ipu_harness.chain import Chain, Resident
from ipu_isa.geometry import LANES


@dataclass(frozen=True)
class PackedLayer:
    """One layer's packed kernels.

    Attributes:
        layer: Its shape and attention kernels (:mod:`mobilevit_layer`), with
            the streams the attention step runs.
        real_streams: Streams of real data (1): the attention kernel's other
            blocks are zero.
        layernorm, unpack, pack, linear, linear_silu, residual: The E07-23 entries.
    """

    layer: Layer
    real_streams: int
    layernorm: str
    unpack: str
    pack: str
    linear: str
    linear_silu: str
    residual: str

    @property
    def channels(self) -> int:
        return self.layer.channels

    @property
    def tokens(self) -> int:
        return self.layer.tokens

    @property
    def per_row(self) -> int:
        """Channels a packed row holds; a linear step's outputs."""
        return LANES // self.tokens

    @property
    def rows(self) -> int:
        """Packed rows of a ``[channels, tokens]`` tensor."""
        return self.channels // self.per_row


#: L4's attention kernels take all 16 (stream, head) blocks: one stream is real.
PACKED_L4 = PackedLayer(
    L4, 1, "packed_layernorm_192x64", "packed_unpack_192x64", "packed_pack_192x64",
    "packed_output_linear_192", "packed_output_linear_silu_192", "packed_residual_add_192x64",
)  # fmt: skip
#: L5's run one block (scores) and one stream (attn@V) a step: one stream.
PACKED_L5 = PackedLayer(
    replace(L5, streams=1), 1, "packed_layernorm_240x16", "packed_unpack_240x16",
    "packed_pack_240x16", "packed_output_linear_240", "packed_output_linear_silu_240",
    "packed_residual_add_240x16",
)  # fmt: skip


def packed(x: np.ndarray) -> np.ndarray:
    """``[channels, tokens]`` as its packed rows."""
    return np.asarray(x, dtype=np.float32).reshape(-1, LANES)


def linear(
    run: LayerRun,
    chain: Chain,
    cfg: PackedLayer,
    kernel: str,
    data: Resident,
    w: np.ndarray,
) -> Resident:
    """``W @ data`` in the packed layout: one step per packed output row
    (``per_row`` outputs), each into its slice of one reserved tensor, in row
    order, over the held data. The output is poisoned first."""
    per = cfg.per_row
    held = chain.hold(data)
    whole = chain.reserve(w.shape[0] // per)
    chain.write(np.full((whole.rows, LANES), POISON), at=whole)
    for g in range(whole.rows):
        run.run(
            chain, kernel, {"k": w.shape[1]}, {"b": held, "a": w[g * per : (g + 1) * per]},
            check=True, at={"y": whole.slice(g, 1)},
        )  # fmt: skip
    chain.release(held)
    chain.release(whole)
    assert not np.any(chain.read(whole) == POISON), f"{kernel} left a packed output row unwritten"
    return whole


def layernorm(
    run: LayerRun,
    chain: Chain,
    cfg: PackedLayer,
    x: Resident | np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> Resident:
    """The packed LayerNorm step, held to its golden."""
    step = run.run(chain, cfg.layernorm, {}, {"x": x, "gamma": gamma, "beta": beta}, check=True)
    return step.resident["y"]


def unpack(run: LayerRun, chain: Chain, cfg: PackedLayer, x: Resident) -> np.ndarray:
    """The unpack step: ``[channels, tokens]``, one channel a row."""
    step = run.run(chain, cfg.unpack, {}, {"x": x}, check=True)
    return np.asarray(step.result.outputs["y"], dtype=np.float32)


def attention(
    run: LayerRun, chain: Chain, cfg: PackedLayer, q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> dict[str, np.ndarray]:
    """The query-major attention steps on the unpacked ``[channels, tokens]``
    Q, K, V of the real stream (the other blocks zero).

    Returns:
        The stages ``scores``, ``softmax`` ``[heads, query, key]`` and
        ``attn_v`` ``[channels, tokens]`` of the real stream.
    """
    layer = cfg.layer
    shape = (layer.blocks, layer.head_dim, layer.tokens)

    def blocks(a: np.ndarray) -> np.ndarray:
        out = np.zeros(shape, np.float32)
        out[: cfg.real_streams * layer.heads] = a.reshape(-1, *shape[1:])
        return out

    s, p, o = MAPPINGS["query_major"](run, chain, layer, blocks(q), blocks(k), blocks(v))
    real = cfg.real_streams * layer.heads
    return {
        "scores": s[:real].astype(np.float64),
        "softmax": p[:real].astype(np.float64),
        "attn_v": o[:real].reshape(cfg.channels, cfg.tokens).astype(np.float64),
    }


def pack(run: LayerRun, chain: Chain, cfg: PackedLayer, x: np.ndarray) -> Resident:
    """The pack step on ``[channels, tokens]`` values (one channel a row)."""
    return run.run(chain, cfg.pack, {}, {"x": x.astype(np.float32)}, check=True).resident["y"]


def residual_add(
    run: LayerRun, chain: Chain, cfg: PackedLayer, a0: Resident, b: Resident
) -> Resident:
    """``a0 + b``, ``a0`` at row 0 (CR0), R_MASK reset to all ones first."""
    assert a0.base_row == 0
    step = run.run(chain, cfg.residual, {}, {"a": a0, "b": b}, check=True, reset=("r_mask",))
    return step.resident["y"]


def unpacked(chain: Chain, cfg: PackedLayer, resident: Resident) -> np.ndarray:
    """A packed resident's values as ``[channels', tokens]`` float64."""
    return chain.read(resident).reshape(-1, cfg.tokens).astype(np.float64)


def front(
    run: LayerRun,
    chain: Chain,
    cfg: PackedLayer,
    x: Resident | np.ndarray,
    w: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """LayerNorm, QKV and the three unpack steps on the packed x: the
    stages ``ln1`` and ``qkv``, and the unpacked Q, K, V ``[channels, tokens]``."""
    ln1 = layernorm(run, chain, cfg, x, w["ln1_gamma"], w["ln1_beta"])
    got = {"ln1": unpacked(chain, cfg, ln1)}
    qkv = linear(run, chain, cfg, cfg.linear, ln1, w["w_qkv"])
    got["qkv"] = unpacked(chain, cfg, qkv)
    q, k, v = (unpack(run, chain, cfg, qkv.slice(j * cfg.rows, cfg.rows)) for j in range(3))
    return got, (q, k, v)


def back(
    run: LayerRun,
    chain: Chain,
    cfg: PackedLayer,
    attn: Resident,
    x0: Resident,
    w: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """OutProj on the packed attention output, the residual add with the held
    x0 (row 0, released after), LayerNorm, FFN1, FFN2 and the second residual
    add (its ``a``, the first add's output, copied to row 0 and held): the
    stages ``out_proj`` to ``resid2``."""
    proj = linear(run, chain, cfg, cfg.linear, attn, w["w_outproj"])
    got = {"out_proj": unpacked(chain, cfg, proj)}
    resid1 = residual_add(run, chain, cfg, x0, proj)
    chain.release(x0)
    got["resid1"] = unpacked(chain, cfg, resid1)
    # The second add's a, at row 0 (over x0) and held until it: the steps
    # between would otherwise place their regions there.
    skip = chain.hold(copy_to_row_0(chain, resid1))
    ln2 = layernorm(run, chain, cfg, skip, w["ln2_gamma"], w["ln2_beta"])
    got["ln2"] = unpacked(chain, cfg, ln2)
    ffn1 = linear(run, chain, cfg, cfg.linear_silu, ln2, w["w_ffn1"])
    got["ffn1"] = unpacked(chain, cfg, ffn1)
    ffn2 = linear(run, chain, cfg, cfg.linear, ffn1, w["w_ffn2"])
    got["ffn2"] = unpacked(chain, cfg, ffn2)
    resid2 = residual_add(run, chain, cfg, skip, ffn2)
    chain.release(skip)
    got["resid2"] = unpacked(chain, cfg, resid2)
    return got


def hold_at_row_0(chain: Chain, x: np.ndarray) -> Resident:
    """``x`` ``[channels, tokens]`` packed at row 0 and held: the first residual add's ``a``."""
    return chain.hold(chain.write(packed(x), at=0))


def run_layer(
    run: LayerRun, chain: Chain, cfg: PackedLayer, x: np.ndarray, w: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """One packed layer on ``chain``: x ``[channels, tokens]`` in, every stage
    boundary out (``mobilevit_layer.STAGES``, the reference's shapes, plus
    ``packed``: the pack step's rows).

    x goes to row 0 (the first residual add's ``a``) after the score steps,
    which pin their own inputs there at L5, and before the pack step."""
    got, (q, k, v) = front(run, chain, cfg, packed(x), w)
    got |= attention(run, chain, cfg, q, k, v)
    x0 = hold_at_row_0(chain, x)
    attn = pack(run, chain, cfg, got["attn_v"])
    got["packed"] = chain.read(attn).astype(np.float64)
    got |= back(run, chain, cfg, attn, x0, w)
    return got
