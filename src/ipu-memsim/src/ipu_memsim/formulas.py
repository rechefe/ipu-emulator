"""Closed-form working-set formulas: how many banks a kernel needs, from its
shape alone, without running it.

For every table of a kernel two numbers are written as formulas of the layer
shape:

- ``rows``: the table's size in XMEM rows (128 elements each);
- ``look_back``: the furthest the kernel ever goes behind the highest row it
  has reached in that table. ``look_back + 1`` is the window of rows that
  must stay on chip: this is the quantity the BSc project book gives for the
  INT8 convolutions ("a standard 3x3 conv needs 3*C_in - 1 input lines" is
  a look-back of ``3*C_in - 1``).

From these, the banks a table needs so that **no bank is ever loaded twice**
(:func:`banks_needed`), with ``R`` rows per bank::

    banks = min( ceil(rows / R),  ceil(look_back / R) + 1 )

The first term is "the whole table is resident". The second: a window of
``look_back + 1`` consecutive rows can straddle at most
``ceil(look_back / R) + 1`` banks. Example with R = 1024: a table streamed
front to back (look_back 0) needs 1 bank however long it is; a 1440-row
weight table that is re-read from its start (look_back 1439) needs
``min(2, 2 + 1) = 2`` banks, all of it.

This is the *least* a table can live with. One bank more per multi-page
input lets the DMA prefetch the next page while the IPU reads the current
one; the knee finder (``ipu_memsim.knee``) measures where stall actually
settles.

Where the formulas come from: ``rows`` from each kernel's ``kernel.toml``
(tensor shapes and ``[derived]``), CONFIRMED; ``look_back`` from reading the
traces of the MobileViT-S shapes and explaining the pattern, then checked
against every recorded trace (:func:`validate`; the table is in STATUS.md).
A look-back formula is therefore VALIDATED ON MobileViT-S SHAPES ONLY; using
it for another shape is an extrapolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ipu_memsim.cache_unit import CacheUnit
from ipu_memsim.dram import BankFillDram
from ipu_memsim.replay import replay
from ipu_memsim.trace import Trace

__all__ = [
    "FORMULAS",
    "TableNeed",
    "banks_needed",
    "kernel_banks",
    "measured_banks",
    "tables_for",
    "validate",
]

LANES = 128  # elements in one XMEM row


@dataclass(frozen=True)
class TableNeed:
    """One table of a kernel, as the formulas see it.

    Attributes:
        name: The region's name in ``kernel.toml``.
        role: ``input``, ``output`` or ``scratch``.
        rows: Its size in rows.
        look_back: The furthest behind its highest row the kernel reads or writes.
    """

    name: str
    role: str
    rows: int
    look_back: int


def banks_needed(rows: int, look_back: int, bank_rows: int = 1024) -> int:
    """Banks a table needs so that no bank is loaded twice (see the module)."""
    pages = math.ceil(rows / bank_rows)
    return max(1, min(pages, math.ceil(look_back / bank_rows) + 1))


def _all(rows: int) -> int:
    """The look-back of a table that is re-read from its first row: all of it."""
    return rows - 1


# -- one function per kernel family: shape -> its tables -----------------------


def matmul_x128(p: Mapping[str, Any]) -> list[TableNeed]:
    """y = act(a @ b): ``m`` tokens, ``k`` input features, ``n`` output features.

    Tokens are 128 to a row, in ``n_tg = ceil(m / 128)`` token groups; ``k`` is
    cut into ``c = ceil(k / 128)`` chunks. The data ``b`` (one row per channel
    and token group) is read once per output feature, and the weights ``a``
    (one row per output feature and chunk) once per token group, so both must
    stay resident. The output is written in order within a token group; with
    two groups the kernel returns to the first group's rows.
    """
    n_tg, c = math.ceil(p["m"] / LANES), math.ceil(p["k"] / LANES)
    y_rows = n_tg * p["n"]
    return [
        TableNeed("b", "input", n_tg * p["k"], _all(n_tg * p["k"])),
        TableNeed("a", "input", p["n"] * c, _all(p["n"] * c)),
        TableNeed("y", "output", y_rows, (n_tg - 1) * p["n"] - 1 if n_tg > 1 else 0),
    ]


def qk_scores(p: Mapping[str, Any]) -> list[TableNeed]:
    """Attention scores S = Q K^T for ``n_block`` heads of ``n_tok`` tokens, ``d`` channels.

    K (``b``) is one row per channel and 128 keys; a head's K rows are walked
    once per query, so one head's K must stay: a look-back of about ``d`` rows
    per key group. Q (``a``, one row per query) is re-read in full. S is
    written in order.
    """
    blocks, groups = p.get("n_block", 1), math.ceil(p["n_tok"] / LANES)
    b_rows = blocks * p["d"] * groups
    a_rows = blocks * p["n_tok"]
    return [
        TableNeed("b", "input", b_rows, min(_all(b_rows), p["d"] * groups)),
        TableNeed("a", "input", a_rows, _all(a_rows)),
        TableNeed("y", "output", blocks * p["n_tok"] * groups, 0),
    ]


def attn_v(p: Mapping[str, Any]) -> list[TableNeed]:
    """Attention output O = P V for ``n_head`` heads.

    P (``b``, one row per query and 128 keys) is walked a query's rows at a
    time and revisited across a head's channels: a look-back of one head's
    queries, ``n_tok * groups`` rows (+1 where two key groups interleave). V
    (``a``, one row per channel) is re-read in full; O is written in order.
    """
    heads, groups = p["n_head"], math.ceil(p["n_tok"] / LANES)
    b_rows = heads * p["n_tok"] * groups
    return [
        TableNeed("b", "input", b_rows, min(_all(b_rows), p["n_tok"] * groups + (groups - 1))),
        TableNeed("a", "input", heads * p["d"] * groups, _all(heads * p["d"] * groups)),
        TableNeed("y", "output", heads * p["d"] * groups, 0),
    ]


def softmax_rows(p: Mapping[str, Any]) -> list[TableNeed]:
    """Softmax over ``rows`` rows of 128: three passes over the input (max,
    exponent, normalise), with the numerators kept in a scratch table."""
    r = p["rows"]
    return [
        TableNeed("x", "input", r, _all(r)),
        TableNeed("y", "output", r, 0),
        TableNeed("num", "scratch", r, _all(r)),
        *(TableNeed(name, "scratch", 1, 0) for name in ("log2e", "max", "rvec")),
    ]


def layernorm(p: Mapping[str, Any]) -> list[TableNeed]:
    """LayerNorm over ``channels`` for ``tokens`` tokens (128 to a row).

    Two passes over the input (mean, then variance and output), the centred
    values kept in a scratch table of one row per channel, and six one-row
    constants. With more than one token group the output is revisited too.
    """
    groups = math.ceil(p["tokens"] / LANES)
    rows = p["channels"] * groups
    vector = math.ceil(p["channels"] / LANES)  # rows of gamma and of beta
    return [
        TableNeed("x", "input", rows, _all(rows) - (groups - 1)),
        TableNeed("gamma", "input", vector, vector - 1 if groups > 1 else 0),
        TableNeed("beta", "input", vector, vector - 1 if groups > 1 else 0),
        TableNeed("y", "output", rows, _all(rows) - 2 * (groups - 1) if groups > 1 else 0),
        TableNeed("centered", "scratch", p["channels"], _all(p["channels"])),
        *(
            TableNeed(name, "scratch", 1, 0)
            for name in ("ones", "neg_inv_n", "inv_n", "neg_mean", "temp", "invstd")
        ),
    ]


def residual_add(p: Mapping[str, Any]) -> list[TableNeed]:
    """y = a + b, one row per channel and token group, front to back. (The
    kernel reads ``b``'s first row once more at the end, which makes ``b``'s
    look-back the whole table.)"""
    rows = p["channels"] * math.ceil(p["tokens"] / LANES)
    return [
        TableNeed("a", "input", rows, 0),
        TableNeed("b", "input", rows, _all(rows)),
        TableNeed("y", "output", rows, 0),
    ]


def _spatial_rows(p: Mapping[str, Any]) -> int:
    """Packed spatial rows of a feature map: ``cols`` is the smallest of 16,
    32, 64, 128 that holds the width, and at least 256 elements are kept
    (``[derived]`` of the convolution kernels' ``kernel.toml``)."""
    cols = next(c for c in (16, 32, 64, 128) if c >= p["width"])
    return max(p["height"], math.ceil(256 / cols)) * cols // LANES


def pointwise_conv(p: Mapping[str, Any]) -> list[TableNeed]:
    """1x1 convolution. The input is one row per channel and packed spatial
    row; an output row needs all ``C_in`` channel rows of its spatial row: a
    look-back of ``C_in - 1``. The weights (``C_out`` rows per 128 input
    channels) are re-read for every spatial row."""
    cols = next(c for c in (16, 32, 64, 128) if c >= p["width"])
    s = max(1, p["height"] * cols // LANES)  # packed spatial rows (no 256-element minimum here)
    w_rows = p["out_channels"] * math.ceil(p["in_channels"] / LANES)
    return [
        TableNeed("x", "input", p["in_channels"] * s, p["in_channels"] - 1),
        TableNeed("w", "input", w_rows, _all(w_rows) if s > 1 else 0),
        TableNeed("y", "output", p["out_channels"] * s, 0),
        TableNeed("mask", "scratch", 1, 0),
    ]


def conv3x3(p: Mapping[str, Any]) -> list[TableNeed]:
    """Standard 3x3 convolution, stride 1. The input has one extra (padding)
    spatial row; an output row needs three spatial rows of all ``C_in``
    channels: a look-back of ``3 * C_in - 1``, the project book's INT8 result.
    The filters (2 rows per 28 input channels, per output channel) are re-read
    for every spatial row."""
    s = _spatial_rows(p)
    x_rows = p["in_channels"] * (s + 1)
    f_rows = p["out_channels"] * math.ceil(p["in_channels"] / 28) * 2
    return [
        TableNeed("x", "input", x_rows, min(_all(x_rows), 3 * p["in_channels"] - 1)),
        TableNeed("filters", "input", f_rows, _all(f_rows)),
        TableNeed("y", "output", p["out_channels"] * s, 0),
        TableNeed("border_mask", "scratch", 1, 0),
    ]


def depthwise_conv3x3(p: Mapping[str, Any]) -> list[TableNeed]:
    """Depthwise 3x3 convolution, stride 1: ``C`` channels, each on its own.
    The input is walked two spatial rows of all channels behind the front: a
    look-back of ``2 * C``. The filters are 2 rows per 28 channels, re-read."""
    c, s = p["in_channels"], _spatial_rows(p)
    f_rows = math.ceil(c / 28) * 2
    return [
        TableNeed("x", "input", c * (s + 1), min(_all(c * (s + 1)), 2 * c)),
        TableNeed("filters", "input", f_rows, _all(f_rows)),
        TableNeed("y", "output", c * s, 0),
        TableNeed("border_mask", "scratch", 1, 0),
    ]


#: Kernel-name prefix -> its formula. The first matching prefix is used.
FORMULAS: dict[str, Callable[[Mapping[str, Any]], list[TableNeed]]] = {
    "matmul_": matmul_x128,
    "qk_scores": qk_scores,
    "attn_v_": attn_v,
    "softmax_rows": softmax_rows,
    "layernorm": layernorm,
    "residual_add": residual_add,
    "pointwise_conv": pointwise_conv,
    "conv_universal": conv3x3,
    "depthwise_conv_universal": depthwise_conv3x3,
}


def tables_for(kernel: str, params: Mapping[str, Any]) -> list[TableNeed] | None:
    """The formula tables of ``kernel`` at ``params``; ``None`` when the
    kernel has no formula (it is then "not covered")."""
    for prefix, formula in FORMULAS.items():
        if kernel.startswith(prefix) and not (prefix == "attn_v_" and "bcast" in kernel):
            return formula(params)
    return None


def kernel_banks(tables: list[TableNeed], bank_rows: int = 1024) -> int:
    """Banks a kernel needs: the sum over its tables."""
    return sum(banks_needed(t.rows, t.look_back, bank_rows) for t in tables)


# -- validation against the replay ------------------------------------------------


def measured_banks(trace: Trace, bank_rows: int = 1024) -> dict[str, int]:
    """Per table of ``trace``, the smallest number of banks at which the
    ``cache_unit`` replay loads no bank of that table twice.

    Each table is measured on its own: the other tables get a bank for every
    page, so they cannot disturb it. "Loaded twice" is a refetch of an input
    page, or any fetch at all of an output or scratch page (those start on
    chip, so a fetch means the page was pushed out and came back).

    Prefetch is switched off for this measurement: it asks what must be
    resident, not how to schedule the DMA. (With prefetch on, a kernel that
    does not start at a table's first row can need one bank more: the DMA
    fills a free bank with a page ahead, and the kernel then wants that bank
    for an earlier page. The depthwise convolution's input does this.)
    """
    regions = [r for i, r in enumerate(trace.header["regions"]) if (trace.region == i).any()]
    pages = {r["name"]: math.ceil(r["rows"] / bank_rows) for r in regions}
    result = {}
    for region in regions:
        name = region["name"]
        for banks in range(1, pages[name] + 1):
            model = CacheUnit(
                num_banks=sum(pages.values()), bank_rows=bank_rows, dram=BankFillDram(0),
                bank_allocation={**pages, name: banks}, prefetch=False,
            )  # fmt: skip
            table = replay(trace, model).counters["tables"][name]
            reloads = table["refetches"] + (0 if region["role"] == "input" else table["fetches"])
            if reloads == 0:
                break
        result[name] = banks
    return result


def validate(
    trace: Trace, bank_rows_list: tuple[int, ...] = (1024, 256, 64)
) -> list[dict[str, Any]]:
    """Compare the formulas with one trace: one row per table and bank size.

    Columns: the formula's ``rows`` and ``look_back`` against the trace's, and
    the formula's banks against the replay's measured minimum. ``verdict`` is
    ``match``, ``safe`` (the formula asks for more banks than the replay
    needs: its window bound is not tight for this shape) or ``UNDER`` (the
    formula asks for fewer: it is wrong for this shape).
    """
    kernel, params = trace.header["kernel"], trace.header["params"]
    tables = tables_for(kernel, params)
    if tables is None:
        return []
    by_name = {t.name: t for t in tables}
    rows = []
    for bank_rows in bank_rows_list:
        measured = measured_banks(trace, bank_rows)
        for index, region in enumerate(trace.header["regions"]):
            mine = trace.row[trace.region == index]
            if len(mine) == 0 or region["name"] not in by_name:
                continue
            formula = by_name[region["name"]]
            look_back = int((np.maximum.accumulate(mine) - mine).max())
            predicted = banks_needed(formula.rows, formula.look_back, bank_rows)
            got = measured[region["name"]]
            rows.append(
                {
                    "kernel": kernel,
                    "table": region["name"],
                    "bank_rows": bank_rows,
                    "rows_formula": formula.rows,
                    "rows_trace": region["rows"],
                    "look_back_formula": formula.look_back,
                    "look_back_trace": look_back,
                    "banks_formula": predicted,
                    "banks_replay": got,
                    "verdict": "match"
                    if predicted == got
                    else ("safe" if predicted > got else "UNDER"),
                }
            )
    return rows
