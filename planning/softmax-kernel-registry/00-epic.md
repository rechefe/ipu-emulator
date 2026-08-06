# Softmax kernels, and a registry that keeps app coverage honest

### The vision

The emulator is accumulating applications faster than anyone can remember what
they do. Choosing one for a computation means reading five docstrings and their
constructor guards, and hoping the one you pick handles your shape. There is no
way to ask "can this machine do X?" and get a trustworthy answer.

This effort delivers two things: the **softmax kernel family** — five kernels
covering every 1-D softmax shape — and the **mechanism** that makes application
coverage self-describing.

The mechanism matters more than the kernels. Softmax is the first family to use
it; it is built so that convolution, linear and matmul fit without changing the
core.

### The problem

A rule written twice drifts, and the second copy is the one that rots. This
codebase already had an instance: `softmax_columns_packed` told callers to
*"use softmax_columns for width >= 128"* long after that boundary had moved to
65. Nothing broke, because nothing read the message — which is exactly how
coverage documentation decays without anyone noticing.

Any registry that restates each app's constraints centrally reproduces that
failure with more ceremony. So the design constraint is stronger than "add a
router".

### The design, decided

1. **Kernels declare themselves where they live.** Each app package exports a
   module-level `SPEC` beside its `.asm`; discovery walks the app tree and
   collects them, at any nesting depth. No central file lists the kernels, so
   adding one is purely additive.

2. **One source of truth per kernel.** `SPEC.supports` is the kernel's domain,
   and the app's own constructor guard delegates to it. The guard and the
   router cannot disagree because there is only one statement of the rule.

3. **`supports` says what a kernel CAN do; `cost` says which SHOULD win.**
   Overlapping claims are normal and are resolved by declared cost, never by
   discovery order — an order-dependent registry changes its answer when a file
   is renamed.

4. **Coverage is generated, never written.** Routing tables are produced by
   probing kernels across a parameter range, so they cannot describe behaviour
   the kernels lack.

5. **Queries carry configuration and shapes together.** A framework layer holds
   config (`dim`, `in_channels`) but never shape, and shape is what selects a
   kernel. Shapes travel as a role-keyed bundle (input/weight/bias/output), so
   multi-tensor operations — a matmul's two independent inputs, a convolution's
   weight — fit without changing the query envelope.

6. **Nothing is reinterpreted silently.** A rank > 2 input is flattened and the
   reshape is stated in the verdict; shapes the registry derived are marked as
   derived; an interior reduction axis is refused rather than transposed.

7. **Discovery tolerates broken packages.** A module that fails to import is
   recorded and reported, never raised. A registry that dies on an unrelated
   half-finished app would fail exactly when it is most needed.

### Acceptance criterion

This issue closes when **every 1-D softmax shape routes to a kernel that
provably computes it, and no coverage claim in the repository is written by
hand.**

"Provably" is the operative word: a generic conformance suite resolves a kernel
for a shape, assembles it, runs it, and compares against a reference — so a
kernel that over-claims its domain fails CI rather than a user's job, and a
newly added kernel inherits that check by registering.

### Order of resolve

Bottom-up, each PR a layer of the finished product rather than a step in how it
was built — so nothing is introduced and then corrected, and each branch passes
its own tests standing alone.

**First the kernels**, which are a complete, working feature on their own.
**Then the benchmarks and the layout contract**, which need kernels to measure
and to constrain. **Then the registry**, whose conformance suite runs the
kernels, so they must exist first. **Then the docs**, which describe the
registry.

### Children

- [ ] #A — softmax: five FP32 wide-vector kernels
- [ ] #B — softmax: benchmarks and a layout contract test
- [ ] #C — kernel registry: kernels declare themselves, coverage is generated
- [ ] #D — docs: application coverage, and a guide for adding applications

### Out of scope

Registering the convolution and fully-connected families. The query envelope is
sized for them — that is why shapes are a role-keyed bundle rather than a single
`input_shape` — but this effort registers softmax only.
