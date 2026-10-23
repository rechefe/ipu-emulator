# Handlers read implicit registers through `CycleView`

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-12 · Size M · Depends on: E04-06, E04-10 · Behaviour change: none

## Goal

Every handler reads registers through one call that applies the view declared in the ISA timing contract (Q32): cycle-start ("snapshot") or post-producer ("forwarded"). After this issue, changing the timing of a register read is a one-line edit in `ipu_isa` plus tests. That is what makes E04-25, E04-26 and E04-27 small.

## Background

Handlers choose the view themselves today:

- snapshot reads: `self._snapshot_buffer.raw_readonly("r")` for R0/R1 (ipu.py:1235, 1363, 1392), `self._snapshot_buffer.raw("r_acc")` (ipu.py:1421, 1448, 1476, 1668, 1701), `raw("mult_res")` in RESHAPE (ipu.py:1603), LRD `source`/`dest` from the snapshot (ipu.py:1589-1590);
- forwarded reads: `self.state.regfile.raw("r_mask")` (ipu.py:909), `self.state.regfile.get_lr(...)` for RESHAPE `reshape_mask` (ipu.py:1594), `mult_res` in ACC.ADD/SUB/MAX/STRIDE and AGG, `r_acc` in `ACTIVATE.QUANTIZE`.

E03 declares, per instruction, `implicit_reads` with a view for every non-operand register, plus `SLOT_EXECUTION_ORDER`, and validates them at import. At this point the declaration matches today's behaviour exactly; E04-25 to E04-27 flip it.

## Implementation

```python
class CycleView:
    snapshot: RegFileReader      # cycle-start, read-only
    live: RegFile                # the file written this cycle
    def read(self, register: str) -> memoryview
        # looks up cycle.current.plan.implicit_reads[register] -> snapshot or live view;
        # raises UndeclaredRead (an EmulatorError) when the register is not declared
```

1. Add `CycleView` to `cycle.py`; handlers get it as `self._view` (set by `Cycle` before each op).
2. Replace each direct snapshot/live read in handlers with `self._view.read(name)`. One commit per slot. Writes keep going to `live` through `RegFile.mutable` (E04-04).
3. LR/CR operand reads go through the resolver (E04-10), which applies the operand's declared `read` view.
4. Generate a conformance test from the spec: for every instruction and every declared implicit read, a two-bundle probe program shows the declared view is honoured (same-cycle producer co-issued, value distinguishable). This replaces the hand-written E02 characterization tests as the long-term guard; keep both until E04-25.
5. In tests, an undeclared read raises `UndeclaredRead`.

## Tests

- [ ] Generated `test_timing_conformance.py`: one probe per (instruction, implicit read), faithful and prototyping.
- [ ] New: a handler that reads an undeclared register raises `UndeclaredRead`.
- [ ] Output-identical proof: E02 timing characterization tests and kernel output digests; `test_execute.py` (`test_agg_and_activate_quantize_same_cycle_sees_live_r_acc` at test_execute.py:1559, `test_reshape_mult_res_reads_from_snapshot` at test_execute.py:1886); `test_prototyping_equivalence.py` under `--define=ipu_proto=1`.

## Acceptance Criteria

- [ ] `grep -n "_snapshot_buffer\|state.regfile" ` inside handler bodies finds no reads (writes via `mutable` only).
- [ ] Every register read in a handler is either an operand or a declared implicit read.
- [ ] The generated conformance test covers every declared implicit read.
