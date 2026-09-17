# Prototyping mode

The emulator models what the hardware does, slot by slot: every slot is
dispatched even when it holds a `NOP`, every multiply applies its element mask,
every cycle takes a full register snapshot. That fidelity is the point — it is
what makes the emulator answerable about the chip.

**Prototyping mode** trades some of that fidelity for speed on long runs, where
the question is *"what does this kernel compute"* rather than *"what does the
chip do"*. It is off by default.

## Enabling it

```bash
bazel test --define=ipu_proto=1 //...
bazel run --define=ipu_proto=1 //src/tools/ipu-apps:fully_connected -- ...
```

The Bazel flag is mandatory. Runtime controls can select or disable the mode
only inside a build produced with that flag:

| Level | How | Use |
|---|---|---|
| Build gate | `--define=ipu_proto=1` | Required capability; prototype mode defaults on |
| Per state | `IpuState(prototyping=True/False)` | Select a mode inside a flagged build |
| Per process | `IPU_EMU_PROTOTYPING=1/0` | Select the default inside a flagged build |

A default Bazel build and a plain checkout cannot activate the fast paths.
Requesting `True` in either raises `ValueError` instead of silently bypassing
the build boundary.

## What it guarantees, and what it does not

**Guaranteed:** the state after every cycle is *bit-identical* to the faithful
path — every register, all of XMEM, the cycle count, and the run statistics.
`src/tools/ipu-emu-py/test/test_prototyping_equivalence.py` runs each program
in lockstep, once per mode, and compares every register, all 8 MiB of XMEM,
cycle outcomes, snapshots, and run statistics after every cycle. The default
build verifies that the gate cannot be bypassed; the flagged build runs the
two-mode equivalence cases.

**Not guaranteed:** that the work the emulator did resembles the work the chip
would do. A skipped `NOP` slot still issues in the real machine, and a
short-circuited mask is still applied there. Trust the numbers a prototyping run
produces; do not reason from it about what a cycle costs the hardware.

## The shortcuts

| Shortcut | Why it is safe | Why it is not faithful |
|---|---|---|
| Slots holding a `NOP` are not dispatched | Their handlers do nothing | The machine still issues the slot |
| Per-PC dispatch plans are cached and revalidated against the current instruction fields | Which slots are live, and the raw operands they read, are properties of the instruction word; debugger edits rebuild the plan | Real decode happens every cycle |
| An unshifted all-ones mask skips the mask pipeline | It deactivates no lane, so `MULT_RES` is unchanged | The mask is applied on every multiply in hardware |
| Cached LR/CR reads use their packed uint32 storage directly | The values and invalid-index exceptions match the generated accessors | It bypasses the normal register-access machinery |
| Full-row FP32 `identity`, `exp2`, and `reciprocal` activations batch row framing and function selection | They retain the scalar arithmetic, FP32 store boundary, exceptions, and partial writes | The hardware activation pipeline operates lane by lane |

One `NOP` is deliberately *not* skipped: the cond slot's, because
`execute_cond_nop` advances the program counter.

## What it is worth

The representative workload is FP32 `softmax_rows` with 128 rows. Assembly and
input preparation are outside the timed region. Each version received two
independent warm-ups followed by 14 measured `app.run()` calls in total:

| Version | Median time | Throughput | Gain vs master |
|---|---:|---:|---:|
| `master` (`ea7613e`) | 0.7197 s | 3,228 cycles/s | baseline |
| `performance-fix` (`b072bb3`) | 0.1145 s | 20,280 cycles/s | 6.28x |
| Prototyping mode | 0.07118 s | 32,636 cycles/s | 10.11x |

That is **1.61x over `performance-fix`**, or 37.9% less wall time. All versions
executed 2,323 cycles and produced the same output SHA-256:

`16c5777b09191f41d5f965097b0a29372cb42eb1d608d1450045989bcd6549ae`

These figures describe this machine and workload, not a universal speedup.
Kernels with partial activation rows or different activation functions fall
back to the scalar activation path; other instruction mixes benefit in
different proportions.
