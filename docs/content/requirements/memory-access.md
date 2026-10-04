# Memory access

The IPU reads and writes XMEM rows. The cache unit fills XMEM banks from DRAM and drains them to DRAM. The cache agent decides, for each word, whether its memory accesses can proceed.

## Address

A `LOAD` or `STORE` slot names a base register (CR) and an offset register (LR). CTRL computes the row address:

```
row_address = CR[base] + LR[offset]      // LR value after the LR writes of the same word
```

The row address identifies a table and an offset in that table.

| Field | Bits | Meaning |
|---|---|---|
| `table_id` | 4 | Table that holds the array. Position in the row address: [OQ-8](open-questions.md). |
| `page` | `offset[19:10]` | Page of the table. One page maps to one XMEM bank. |
| `row` | `offset[9:0]` | Row in the XMEM bank |

The tag of an access is `{ctx, table_id, page}`, where `ctx` is `active_ctx`. An access hits when the XMEM bank for that page is valid and holds the same tag.

## XMEM bank states

| State | Meaning | Hit |
|---|---|---|
| Free | The bank holds no tag. | No |
| Valid | The bank holds a tag. The IPU can load from it or store to it. | Yes, on a tag match |
| Draining | The IPU flushed the bank. The DMA writes it to DRAM. | No |

Each table has a `jump_back` parameter. When an access reaches row `jump_back` or beyond in a bank, the kernel does not access the previous bank of the table again. That access is the jump-back point of the previous bank. The cache unit specification, sections 9.1 and 9.2, defines the rule.

## Cache agent

| ID | Requirement |
|---|---|
| IPU-MEM-1 | The cache agent shall decide hit or miss for the `LOAD` and the `STORE` of a word in the cycle CTRL presents the word. See [OQ-16](open-questions.md). |
| IPU-MEM-2 | The cache agent shall decide from state it holds: the table view of the active context and the state and tag of each XMEM bank. |
| IPU-MEM-3 | The cache agent shall accept a word only when every memory access of the word hits and the word raises no fault. |
| IPU-MEM-4 | The cache agent shall accept a word that has no memory access without delay. |
| IPU-MEM-5 | On a miss, the cache agent shall stall CTRL and send one `bank_req` for each missing tag. |
| IPU-MEM-6 | The cache agent shall send at most one `bank_req` per cycle. The request for the load goes first. |
| IPU-MEM-7 | The cache agent shall accept the stalled word when the cache unit has published every bank the word needs. |
| IPU-MEM-8 | For a load that hits, the cache agent shall issue the XMEM read of the accepted word. The `MULT` slot of the same word uses the loaded data. See [OQ-9](open-questions.md). |
| IPU-MEM-9 | The cache agent shall not flush or release an XMEM bank while an accepted word has an unwritten store to it. |
| IPU-MEM-10 | The cache agent shall hit only on tags whose `ctx` bit equals `active_ctx`. |
| IPU-MEM-11 | The cache agent shall never report a hit for a bank that is not valid. |
| IPU-MEM-12 | The cache agent may report a miss for a bank that became valid in the last few cycles. |

A load of a row that an older word in the pipeline stores to is open: [OQ-17](open-questions.md).

## Bank state ownership

The cache agent and the cache unit each hold the state of the 16 XMEM banks. The cache unit makes the changes that make a bank available. The cache agent makes the changes that give a bank back. Each side reports its changes to the other side.

| Change | Trigger | Made by | Reported with |
|---|---|---|---|
| Free to valid: the DMA filled the bank | DMA fill complete | Cache unit | `bank_pub` |
| Free to valid: the cache unit claimed the bank for a store | `bank_req` with `bank_req_write` | Cache unit | `bank_pub` |
| Valid to free: the IPU loaded past the jump-back point of a read bank | IPU load | Cache agent | `bank_release` |
| Valid to draining: the IPU stored past the jump-back point of a written bank | IPU store | Cache agent | `bank_flush` |
| Draining to free: the DMA wrote the bank to DRAM | DMA drain complete | Cache unit | `bank_free` |
| Every bank of a context leaves valid | Kernel done or abort | Both, in the same cycle | `kernel_done`, `kernel_abort` |

| ID | Requirement |
|---|---|
| IPU-MEM-20 | The cache agent shall compute the bank state changes that IPU accesses trigger, from the accesses of accepted words and the table view. |
| IPU-MEM-21 | The cache agent shall report each such change to the cache unit with `bank_release` or `bank_flush`. |
| IPU-MEM-22 | The cache agent shall apply each `bank_pub` and `bank_free` event to its bank state. |
| IPU-MEM-23 | On `kernel_done` and on `kernel_abort`, the cache agent shall mark every bank tagged with the context of that kernel as not valid. |

## Expected from the cache unit

| ID | Expected behaviour |
|---|---|
| CU-20 | The cache unit never takes a bank out of the valid state on its own. A bank leaves valid only through `bank_release`, `bank_flush`, `kernel_done` or `kernel_abort`. |
| CU-21 | The cache unit publishes a read bank only after the DMA has filled the whole bank. |
| CU-22 | On `bank_req`, the cache unit makes a bank valid for the requested tag and publishes it. |
| CU-23 | The cache unit ignores a `bank_req` for a tag that is valid or that the DMA is filling. |
| CU-24 | On `bank_flush`, the cache unit drains the bank to DRAM and then sends `bank_free`. |
| CU-25 | On `bank_release`, the cache unit treats the bank as free. |
| CU-26 | Every bank tag carries the context bit. |

With CU-20, a late event costs a stall cycle and never a wrong result. The cache agent can learn late that a bank became valid. A bank that the cache agent holds as valid is always valid.
