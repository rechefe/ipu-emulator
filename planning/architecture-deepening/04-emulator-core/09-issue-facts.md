# Issue step produces one `IssueFacts` record

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-09 · Size M · Depends on: E04-07 · Behaviour change: none

## Goal

Operand issue (decode, resolve, MAC-identity probe, alias attribution, stat bump) happens once per slot op and produces one immutable fact record. The handler, the statistics and the observers read that record. Mutable side-channel fields on `Ipu` disappear.

## Background

- The MAC-identity block is written twice: `dispatch_instruction` (ipu.py:1940-1947) and `_run_plan` (ipu.py:2216-2223).
- The identity verdict travels through three mutable fields, `_mult_identity`, `_mult_vector_identity`, `_mult_cr_identity` (ipu.py:595-597). `_count_mult_lanes` (ipu.py:926-934) and `alias_observer.py:145` read them later.
- `_mult_vector_is_one` (ipu.py:1960-1982) decodes the word from `inst_mem` again to recover the encoded index of `ra`, which the issuer already had.

## Implementation

New module `ipu_emu/issue.py`:

```python
@dataclass(frozen=True, slots=True)
class MultIdentity:
    cr: bool
    vector: bool
    @property
    def any(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class IssueFacts:
    slot: str
    plan: Plan
    encoded: Mapping[str, int]          # raw operand fields from the word
    operands: Mapping[str, Any]         # resolved values passed to the handler
    mult_identity: MultIdentity | None  # None for non-MULT ops

class Issuer:
    def issue(self, op: IssuedOp) -> IssueFacts      # resolve + identity + stat bump
    def execute(self, facts: IssueFacts) -> Any      # plan.fn(ipu, **facts.operands)
```

Steps:

1. Move the identity probe into `Issuer.issue`. `_mult_vector_is_one` takes `encoded` and stops reading `inst_mem`.
2. `Cycle` keeps the current op's `IssueFacts` in `cycle.current`. `_count_mult_lanes` reads `cycle.current.mult_identity`. Delete the three mutable fields.
3. `Ipu.dispatch_instruction(slot, inst)` stays as a delegate (`issue` + `execute`) so the mobilevit measuring fixture keeps working until E04-11 moves it to an observer.
4. The handler signature is unchanged: keyword-only arguments named after the spec operands (CLAUDE.md invariant).

## Tests

- [ ] New `test_issue.py`: `IssueFacts` for `MULT.RC.VV` carries the identity verdict without reading `inst_mem` (replace `inst_mem` with a sentinel that raises on access).
- [ ] Output-identical proof: `test_stats.py` (identity and lane counts), `test_alias_profile.py`, `test_alias_tracking.py`, `test_prototyping_equivalence.py` under `--define=ipu_proto=1`, E02 kernel output digests.
- [ ] E04-02 benchmark: `IssueFacts` allocation costs no more than 5 % (it uses `slots=True`; schedules from E04-08 hold pre-built `encoded` mappings).

## Acceptance Criteria

- [ ] One identity block in the code base.
- [ ] `_mult_identity`, `_mult_vector_identity` and `_mult_cr_identity` are deleted.
- [ ] No handler or helper reads `inst_mem`.
