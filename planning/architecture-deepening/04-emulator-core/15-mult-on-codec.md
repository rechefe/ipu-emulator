# MULT handlers written once against the codec

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-15 · Size M · Depends on: E04-13, E04-14 · Behaviour change: none

## Goal

Each MULT handler becomes one path: decode lanes through the active codec, compute, store to MULT_RES, apply the mask (Q48). No `if wide:` and no `WideVectorArithmetic` checks (Q43). Narrow MULT becomes vectorised.

## Background

- MULT handlers split early on `self._wide_vector_active()` (for example `execute_mult_rc_vv`, ipu.py:1253-1258) and have a narrow per-lane loop calling `ipu_mult` and `pack_into` per lane.
- `MULT.EE` broadcasts with 128 `pack_into` calls in narrow mode.
- The LR/CR scalar is resolved by two functions, `_mult_resolve_lcr_scalar` (ipu.py:1223-1238) and `_mult_resolve_lcr_scalar_wide` (ipu.py:1240-1251).
- The wide CR scalar is the CR's low byte as a signed int (`_wide_cr_scalar_byte_as_int32`, ipu.py:693-696), converted to float in FP32 mode (ipu.py:1249-1250). E04-28 changes that rule as a behaviour change; this issue keeps it, inside the codec's `decode_scalar` for FP32/INT32.

## Implementation

1. Add `self._codec = codec_for(fmt)` in `Ipu`, where `fmt` is derived from today's `dtype`/`wide_vector_*` flags by one function, `active_format(state)`. (E04-17 replaces that function with the configured format.)
2. Rewrite `mult.py` handlers one at a time, in this order: `MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`, then the remaining variants. Each becomes:
   ```python
   a = codec.decode(view.read(...))      # or decode_scalar
   b = codec.decode(...)
   out = codec.mult(a, b)
   codec.store_acc(mult_res, out)
   apply_mask(...)
   ```
3. Merge the two scalar resolvers into one that returns `codec.decode_scalar(...)`.
4. Delete the narrow per-lane loops and the wide branches in `mult.py`.

## Tests

- [ ] Output-identical proof: E04-13 exhaustive codec tests; `test_execute.py` MULT tests; `test_wide_vector_debug.py`; `test_prototyping_equivalence.py` under `--define=ipu_proto=1`; E02 kernel output digests (all formats the kernels use today).
- [ ] New: each MULT handler run under INT8, E4 and FP32 against a numpy reference of the same codec (one test per handler × format).
- [ ] E04-02 benchmark before/after (narrow and wide).

## Acceptance Criteria

- [ ] `mult.py` contains no `_wide_vector_active`, `WideVectorArithmetic` or `DType` reference.
- [ ] No per-lane Python loop in MULT handlers.
- [ ] Narrow MULT cycles/s recorded; expected several times faster (tracked toward the Q71 matmul target).
