# Cache Unit

## General Information

**Purpose:** TODO

**Role:** TODO

**Integration:** TODO

**Main Use Cases:**

- TODO

---

### Black Box Diagram

```
           ┌──────────────────────┐
  xmem <-> │                      │ <-> data_bus
  addr --> │      CACHE UNIT      │
    wr --> │                      │
           └──────────────────────┘
```

---

## Interfaces

| Name | Type and Direction | Description |
|------|--------------------|-------------|
| `TODO` | `input logic [N:0]` | TODO |

### Parameters

| Name | Default | Description |
|------|---------|-------------|
| `TODO` | `0` | TODO |

---

## Assumptions

- TODO

---

## Operation Logic

### Logic Flow

TODO

```mermaid
flowchart TD
    A[Address request] --> B{Hit?}
    B -- Yes --> C[Return cached data]
    B -- No --> D[Fetch from XMEM]
    D --> E[Fill cache line]
    E --> C
```

### Configuration

TODO

### Required TP and Latency

| Metric | Requirement | Notes |
|--------|-------------|-------|
| Throughput | TODO | |
| Latency | TODO | |
