# Hardware Units Overview

## General Information

**Purpose:** High-level overview of the IPU hardware units and how they interact.

**Role:** Describes the overall architecture of the IPU's functional units and their interconnections.

**Integration:** Top-level view of the IPU pipeline — how data flows between the Control Unit, Multiplication Unit, Accumulation Unit, Vector Unit, and Cache Unit.

**Main Use Cases:**

- Understanding the IPU architecture before diving into individual units
- Identifying which unit is responsible for a given operation
- Understanding inter-unit data paths and dependencies

---

### Black Box Diagram

```
          ┌─────────────────────────────────────────┐
          │                  IPU                     │
 xmem --> │  ┌──────┐  ┌──────┐  ┌──────┐          │ --> results
 prog --> │  │Cache │->│ Mult │->│Accum │          │
 ctrl --> │  └──────┘  └──────┘  └──────┘          │
          │       ┌──────────┐  ┌──────┐            │
          │       │  Vector  │  │  CU  │            │
          │       └──────────┘  └──────┘            │
          └─────────────────────────────────────────┘
```

---

## Units Summary

| Unit | File | Responsibility |
|------|------|----------------|
| Control Unit | [control-unit.md](control-unit.md) | Instruction fetch, decode, and dispatch |
| Multiplication Unit | [mult-unit.md](mult-unit.md) | Matrix and vector multiply operations |
| Accumulation Unit | [accum-unit.md](accum-unit.md) | Accumulate partial results |
| Vector Unit | [vector-unit.md](vector-unit.md) | Element-wise vector operations |
| Cache Unit | [cache-unit.md](cache-unit.md) | Local data buffering and XMEM interface |
