"""Host-side NUMPY TEST FIXTURES for the softmax and fold stages of a full
MobileViT transformer layer -- NOT kernel implementations.

These exist only so `test_full_layer_l4.py` / `test_full_layer_l5.py` can
close a whole layer end to end while the real softmax and fold kernels are
owned and built elsewhere (see kernel_docs/kernel_layer_map.md and the
`pr-softmax-apps` branch, not merged here). Nothing in this module runs on
the IPU emulator; it is pure host numpy, deliberately named so it cannot be
mistaken for a kernel: no `App` class, no `setup`/`teardown`, no `.asm`, and
it does not live under `src/ipu_apps/`.

Layout contract (the property that actually matters for this stub):
    A softmax stub MUST return its output in the SAME layout it was handed --
    row axis in, row axis out, unchanged meaning. The two attention chains
    hand it different layouts:
      * query-major chain (`qk_scores_*` -> `attn_v_*`): one row PER QUERY,
        softmax reduces over the trailing (key) axis.
      * key-major chain (`attn_scores_km_*` -> `attn_v_bcast_*`): one row PER
        KEY, so an entire query's scores are scattered one-per-row across all
        key rows at a fixed column -- softmax here must reduce over that
        column (the query axis positioned across rows), not over a row.
    A softmax that silently transposed either input would still produce
    *some* numbers, and only an explicit layout-contract assertion would
    catch that it fed attn@V a mapping it never asked for -- that is exactly
    the class of bug this stub's assertions exist to make impossible to miss.
"""

from __future__ import annotations

import numpy as np


def softmax_query_major(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the trailing (key) axis, query-major layout.

    Input/output shape: [..., n_query, n_key]. Row i is query i's raw scores
    over all keys; softmax reduces axis=-1 (the key axis) and leaves every
    other axis (heads, streams, layers, ...) untouched. Output layout is
    IDENTICAL to input layout -- same shape, same axis meaning -- which is
    the one property a real softmax kernel must also preserve.
    """
    scores = np.asarray(scores, dtype=np.float64)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    out = e / np.sum(e, axis=-1, keepdims=True)
    assert out.shape == scores.shape, (
        "softmax_query_major changed shape -- a layout-preserving stub must not"
    )
    return out


def softmax_key_major(scores_km: np.ndarray) -> np.ndarray:
    """Softmax for KEY-MAJOR score storage: input row s holds key s's scores
    against every query (column q = query q). The reduction axis for a
    correct softmax is still the KEY axis -- but here that is axis=-2 (across
    rows), not the trailing axis, because key-major storage put queries in
    columns and keys in rows.

    Input/output shape: [..., n_key, n_query]. Output layout is IDENTICAL to
    input layout (still key-major: row s, column q) -- a stub that flipped to
    query-major here would hide exactly the kind of chain incompatibility
    this fixture is designed to expose.
    """
    scores_km = np.asarray(scores_km, dtype=np.float64)
    m = np.max(scores_km, axis=-2, keepdims=True)
    e = np.exp(scores_km - m)
    out = e / np.sum(e, axis=-2, keepdims=True)
    assert out.shape == scores_km.shape, (
        "softmax_key_major changed shape -- a layout-preserving stub must not"
    )
    return out


def fold_stub(x: np.ndarray, *, out_channels: int) -> np.ndarray:
    """Minimal space-to-depth INVERSE (fold) stand-in for the block boundary.

    This sits at the boundary of the transformer block, not inside the
    layer chain under test here, so it does not need to model the real
    fold's specific channel/window arrangement -- it only needs to close the
    shape so a full-layer numpy reference can run without a real fold kernel.
    Averages spatial groups of size (x.shape[-1] // out_channels) along the
    LAST axis; deliberately the simplest operation that reduces width without
    inventing a channel ordering that could be mistaken for the real kernel's
    contract.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    assert n % out_channels == 0, (
        f"fold_stub: last axis {n} not divisible by out_channels {out_channels}"
    )
    group = n // out_channels
    return x.reshape(*x.shape[:-1], out_channels, group).mean(axis=-1)
