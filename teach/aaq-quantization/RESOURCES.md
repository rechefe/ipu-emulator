# AaQ Quantization Resources

## Knowledge

- [Spec: AaQ Stage (`docs/content/specs/stage-aaq-str.md`)](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/specs/stage-aaq-str.md)
  The hardware behaviour of ACTIVATE.QUANTIZE, the 1033-bit payload, the 9-bit `format`, the quantizer (§5.1), dynamic mode (§5.1.1) and the LUT (§5.2). Use for: anything AaQ produces.
- [Spec: Cache Unit (`docs/content/specs/cache-unit.md`)](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/specs/cache-unit.md)
  XMEM banks, rows, the 16-bit per-row metadata (§4, §6) and table properties including `format` (§5). Use for: where a row's metadata and format live.
- [Open questions (`docs/content/requirements/open-questions.md`)](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/requirements/open-questions.md)
  OQ-6, OQ-7, OQ-20 to OQ-27 list what the AaQ spec leaves undecided. Use for: the gaps.
- [Requirements: memory access](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/requirements/memory-access.md)
  Row address fields, bank states, the cache agent's IPU-MEM requirements. Use for: how a load or store finds its table and bank.
- [Requirements: execution flow](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/requirements/execution-flow.md)
  Contexts, table sets, ARM and commit, CU-10 to CU-18. Use for: how many table sets exist and who writes them.
- [Requirements: interfaces](https://github.com/rechefe/ipu-emulator/blob/master/docs/content/requirements/interfaces.md)
  IPU ports to the cache unit: `tbl_view`, `tables_valid`, `bank_req`. Use for: what the IPU sees of the tables.
- [ADR 0003: Element formats replace wide mode](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0003-element-formats-replace-wide-mode.md)
  How the emulator models INT8, E1–E7, FP32 today: one global format per run. Use for: contrast with per-table AaQ formats.
- [OCP announcement: Microscaling (MX) Formats Specification v1.0](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai)
  The industry standard for block-scaled 8-bit and narrower formats (MXFP8, MXFP6, MXFP4, MXINT8). Use for: the general idea AaQ follows.
- [Paper: Rouhani et al., "Microscaling Data Formats for Deep Learning" (arXiv:2310.10537)](https://arxiv.org/abs/2310.10537)
  The research behind MX: shared scale per block plus narrow elements. Use for: why block scaling works.
- [NVIDIA CUTLASS: block-scaled GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/006_block_scaled_gemm.html)
  E4M3 elements with one E8M0 scale per 32 elements, and how a matrix multiply consumes them. Use for: how a MULT stage could expand scaled data.

## Wisdom (Communities)

- The AaQ spec owner and the downstream RTL team.
  Use for: every OQ row marked "Unassigned", and for whether hardware can afford a design choice.

## Gaps

- No MULT-stage spec: nothing defines how a scaled row is expanded to FP32 before multiplication.
- The OCP MX spec PDF itself could not be fetched from this environment; MX facts cite secondary sources.
