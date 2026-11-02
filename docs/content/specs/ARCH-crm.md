# CRM

## 1. Overview

The CRM (Cache replacment menagment) performs the following tasks:

- **Data prefetch:** Read weights/activations from memory to cache.
- **Proactive write-back:** Write activations from cache to memory.
- **Cache snooping:** Track IPU accesses to free cache banks.

**Internal structure:**

- External memory controller
- DMA state machine
- Double buffer configuration

> **Note:** Does not act like standard cache that uses fetch upon miss and
> write-back upon invalidate.

## Table of Contents

1. [Overview](#1-overview)
2. [Block Diagram](#2-block-diagram)
3. [Interfaces](#3-interfaces)
    - 3.1 [Table Layout](#31-table-layout)
    - 3.2 [Table Types](#32-table-types)
    - 3.3 [Bank State](#33-bank-state)
4. [CRM Operations](#4-crm-operations)
    - 4.1 [IPU Read](#41-ipu-read)
    - 4.2 [IPU Write](#42-ipu-write)

## 2. Block Diagram

```mermaid
%%{init: {"flowchart": {"curve": "stepBefore", "nodeSpacing": 120, "rankSpacing": 30, "padding": 4}}}%%
flowchart TB
    RISC["RISC Configuration"]
    CACHE["Cache interface"]
    IPU["IPU handshake"]
    subgraph CRM ["<span style='display:inline-block;width:640px;text-align:left;font-weight:bold'>CRM</span>"]
        DMA["<span style='display:inline-block;width:160px;padding:26px 0'>DMA state machine</span>"]
        EMC["<span style='display:inline-block;width:160px;padding:16px 0'>External Memory<br>Controller</span>"]
    end
    EMI["External Memory Interface"]

    RISC --> DMA
    CACHE <--> DMA
    IPU <--> DMA
    DMA <--> EMC
    EMC <--> EMI
    linkStyle 1,3,4 interpolate linear

    classDef io fill:none,stroke:none
    classDef blk fill:#2e9e8c,stroke:#1a7060,color:#fff
    class RISC,CACHE,IPU,EMI io
    class DMA,EMC blk
    style CRM fill:none,stroke:#888,stroke-width:2px
```

## 3. Interfaces

| Name | Direction | Description |
|------|-----------|-------------|
| TBD | TBD | TBD. |

### 3.1 Table Layout

> **TBD:** State the number of tables and how the host writes them.

| Bits | Field | Description |
|------|-------|-------------|
| TBD | TBD | TBD. |

### 3.2 Table Types

> **TBD:** Give the behavior of each table type.

| Value | Type | Behavior |
|-------|------|----------|
| TBD | TBD | TBD. |

### 3.3 Bank State

> **TBD:** Give the per-bank state that the CRM reads and writes.

| Bits | Field | Description |
|------|-------|-------------|
| TBD | TBD | TBD. |

## 4. CRM Operations

> **TBD:** State which request starts each operation.

**Operations in this section:**

| Operation | Purpose |
|-----------|---------|
| [IPU Read](#41-ipu-read) | TBD. |
| [IPU Write](#42-ipu-write) | TBD. |

---

### 4.1 IPU Read

> **TBD:** Give what the operation does, in one to three sentences.

#### Inputs

| Name | Width | Details |
|------|-------|---------|
| TBD | TBD | TBD. |

#### Operation

```text
TBD
```

#### Example

```text
TBD
```

---

### 4.2 IPU Write

> **TBD:** Give what the operation does, in one to three sentences.

#### Inputs

| Name | Width | Details |
|------|-------|---------|
| TBD | TBD | TBD. |

#### Operation

```text
TBD
```

#### Example

```text
TBD
```
