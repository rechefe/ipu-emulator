# Epic: Kernel migration onto `ipu_harness`, family by family

ID E07 · Phase: after E06 · Last epic in the program order

## Goal

Move all 89 registered kernels, plus the 12 MobileViT packed programs, from the
5-file `ipu_apps` harness format to the E06 format: `<name>.asm` +
`kernel.toml` + optional `reference.py`, with family `registry.toml` files for
shape variants. Every migration proves it changed nothing, using identical
binaries and identical FP32 outputs against the E02 digests, **before** it adds
a format. The pilot family, projections, also runs in INT8 and E4 (Q50). When
E07 is done, `src/tools/ipu-apps` is gone.

## Background

Measured at HEAD 694a8a8, under `src/tools/ipu-apps/src/ipu_apps/kernels/`:

| Family | Kernels | Python lines | asm lines | Copy groups (identical bodies once comments are stripped) |
|--------|---------|--------------|-----------|---------------------------------|
| projections | 12 | 1,532 | 2,854 | 4 distinct bodies over 12 kernels |
| reshape | 11 | 2,175 | 2,091 | `concat` x3 identical; `fold`/`unfold` similar but distinct |
| elementwise | 4 | 780 | 458 | `residual_add_{16x240,64x192,256x144}` identical |
| matmul | 16 | 2,836 | 2,181 | 4 small ones identical; `*_x128` x12 in 2-3 groups (0.85-1.00 similar) |
| normalize | 5 | 1,054 | 1,413 | `layernorm_*` x4 (0.87-0.96 similar) |
| softmax | 5 | 1,652 | 1,017 | none |
| attention | 12 | 3,998 | 1,953 | shape triples 0.62-0.85 similar; no identical pairs |
| pooling | 5 | 364 | 711 | `maxpool2d_nms7`/`nms9` 0.98 |
| sampling | 2 | 474 | 152 | none |
| detect | 2 | 108 | 201 | none |
| convolutions | 14 | 5,197 | 4,966 | `conv_universal` twin 0.97, pointwise twin 0.94, depthwise twin 0.89 |
| linear | 1 | 355 | 57 | 9 opaque `.bin` goldens |
| mobilevit (test tree) | 12 programs | 3,898 (4 full-layer tests + 5 fixtures) | 2,743 | `residual_add` pair 1.00, `output_linear` generic/silu 0.99, layernorm 0.88, pack 0.90, unpack 0.85 |

Other facts every family issue deals with:
- **Lower-case mnemonics.** These are concentrated in convolutions (about 660
  lines), plus a few lines in softmax, elementwise, pooling, reshape and detect.
- **Design notes.** They are spread across the `app.py` docstrings, `cases.py`
  docstrings and `.asm` headers.
- **Units.** Some kernels still pass byte addresses in CRs (L16).
- **Input sizes.** These are inconsistent: `matmul_*_x128` truncates oversize
  input (`kernels/matmul/matmul_144x288_x128/app.py:76`), while projections
  require the exact size (L15).
- **MemoryApp.** 15 kernels use the whole-image path that E06-20 deletes.

## Decisions applied

| ID | Decision |
|----|----------|
| Q50 | Pilot family projections in FP32, INT8 and E4. Kernels are format-independent, and goldens are per format. |
| Q76 | Family `registry.toml` for shape variants. Kernel names and Bazel target names stay the same. |
| Q78 | `fully_connected` goldens are generated from the reference per format, and the 9 `.bin` files are deleted. |
| Q79 | Design notes live only in the `.asm` (or template) header. |
| Q45, CLAUDE.md | Upper-case mnemonics, applied per family in its migration PR. |
| Q57 | One LayerNorm reference matching the kernel (no eps). |
| Q34 | `fully_connected` `fp8_e4m3`/`fp8_e5m2` become `fp8_e4`/`fp8_e5` (if E03/E00 has not already done it). |
| Q83 | No whole-image kernels after migration. |
| D3 | Each migration is output-identical. New variants, new formats and input-policy changes are separate, labelled issues. |
| L13 | MobileViT experiments are promoted if the L4/L5 chains use them, otherwise deleted. All 12 programs are used by `test_full_layer_l{4,5}_packed.py`, so all 12 are promoted. |
| L15 | Exact input sizes. |
| L16 | Byte-unit kernels are marked `unit = "bytes"` and frozen, not converted. |

## The per-family checklist (applies to every migration issue)

Each family issue copies this list into its PR description. E07-01 provides
the tools.

1. Record the baseline: the E02 asm digests and the E02 FP32 kernel output
   digests for every case of the family, taken at the parent commit.
2. Write `kernel.toml` for each kernel, or a template plus `registry.toml` for
   a copy group.
3. Express the old harness layout through placement, `pack` transforms and
   `[abi]`. Prove that the CR values and the XMEM image after setup are
   identical to the old harness, using `compare_setup` from E07-01.
4. Port the cases. Old seeds and old input generators are kept through a
   `generator` hook where needed, so FP32 outputs can be compared bit for bit.
   Moving to the E06 seeded generators is a separate commit, after the proof.
5. Choose the reference: the shared `op` reference, or a local `reference.py`.
   Choose a tolerance tier per format.
6. Move all design notes into the `.asm` header. Delete the docstrings that held
   them.
7. Upper-case the mnemonics with `tools/upper_mnemonics.py` from E07-01. This must be
   binary-identical, which the digest proves.
8. Mark byte-unit CRs `unit = "bytes"` (L16). Make input sizes exact (L15). If
   a kernel used to truncate, that change is its own issue.
9. Move the family `test.py` tests that go beyond case x format, such as router
   and contract tests, into `test_*.py` in the template or family folder. Delete
   the tests the framework now covers.
10. Delete the old folders (`app.py`, `cases.py`, `test.py`, `__init__.py`). The
    old Bazel targets under `//src/tools/ipu-apps` disappear, and the new
    targets under `//kernels/<family>` keep the same names.
11. Run the proof: `assemble_<name>` digests equal the E02 asm digests, and the
    FP32 outputs of every case equal the E02 output digests. Paste the
    `compare_legacy` report into the PR.
12. `//kernels:query` gives the same routing answers as the old `:query` for
    the family's ops.

## Scope

- Every family under `kernels/`, the 12 MobileViT packed programs, and the
  MobileViT chain tests.
- INT8 and E4 for the pilot family.
- One new conv variant: bias without ReLU.
- `fully_connected` goldens from the reference.
- Retiring `ipu-apps`.

## Out of scope

- INT8/E4 (or other formats) for the families after the pilot. Each family gets
  one follow-up issue, filed after its migration lands and modelled on E07-03.
  The format work depends on E04 handlers and on AaQ behaviour, which is out of
  scope (Q25/Q35).
- Converting byte-unit kernels to rows (L16 freezes them).
- The timing-contract behaviour fix for the 3 `attn_v_*` kernels (Q32). That is
  E04's behaviour PR. E07-13 is ordered after it so the digests are taken once.
- Re-seeding test data beyond what each migration needs.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|----|------|-------|------|------------|-------------------|
| E07-01 | [01-migration-tooling.md](01-migration-tooling.md) | Migration tooling: `compare_legacy`, `compare_setup`, upper-case mnemonics | S | E02 (asm + output digests), E06-13 | none |
| E07-02 | [02-projections-fp32.md](02-projections-fp32.md) | Pilot: projections onto `ipu_harness` (FP32, 12 `registry.toml` entries over 3 templates) | M | E07-01, E06-17 | none |
| E07-03 | [03-projections-int8-e4.md](03-projections-int8-e4.md) | Pilot: projections in INT8 and E4 | M | E07-02, E04 (format-independent handlers, Q49) | yes (new formats and goldens) |
| E07-04 | [04-reshape-identity-concat.md](04-reshape-identity-concat.md) | Reshape (1/2): `identity`, `depth_to_space`, `concat` x3 | M | E07-02 | none |
| E07-05 | [05-reshape-fold-unfold.md](05-reshape-fold-unfold.md) | Reshape (2/2): `fold` x3 and `unfold` x3 | M | E07-04 | none |
| E07-06 | [06-elementwise.md](06-elementwise.md) | Elementwise: `residual_add` x4 | S | E07-02 | none |
| E07-07 | [07-matmul-small.md](07-matmul-small.md) | Matmul (1/2): the four small matmuls as one template | S | E07-02 | none |
| E07-08 | [08-matmul-x128.md](08-matmul-x128.md) | Matmul (2/2): the `*_x128` group onto shared layouts | M | E07-07 | none |
| E07-09 | [09-matmul-exact-input-size.md](09-matmul-exact-input-size.md) | Matmul: reject oversize inputs instead of truncating (L15) | S | E07-08 | yes (oversize input rejected) |
| E07-10 | [10-normalize.md](10-normalize.md) | Normalize: `layernorm_*` x4 and `l2_normalize_channels` | M | E07-02 | none |
| E07-11 | [11-softmax.md](11-softmax.md) | Softmax: 5 kernels and their benchmarks | M | E07-02, E06-11 (benchmarks question) | none |
| E07-12 | [12-attention-scores.md](12-attention-scores.md) | Attention (1/2): `qk_scores` x3 and `attn_scores_km` x3 | M | E07-11 | none |
| E07-13 | [13-attention-values.md](13-attention-values.md) | Attention (2/2): `attn_v` x3 and `attn_v_bcast` x3 | M | E07-12, E04 (attn_v timing behaviour PR) | none |
| E07-14 | [14-pooling.md](14-pooling.md) | Pooling: 5 `maxpool2d_*` kernels | M | E07-02 | none |
| E07-15 | [15-sampling-detect.md](15-sampling-detect.md) | Sampling and detect: 4 whole-image kernels | M | E07-02 | none |
| E07-16 | [16-conv-simple.md](16-conv-simple.md) | Convolutions (1/4): `conv1x1`, `conv3x3_relu`, `conv3x3_relu_cin1`, `conv_first_layer` | M | E07-02 | none |
| E07-17 | [17-conv-universal-merge.md](17-conv-universal-merge.md) | Convolutions (2/4): merge the `conv_universal` twins into one template | M | E07-16 | none |
| E07-18 | [18-conv-pointwise-depthwise-merge.md](18-conv-pointwise-depthwise-merge.md) | Convolutions (3/4): merge the pointwise and depthwise twins | M | E07-17 | none |
| E07-19 | [19-conv-multistage-and-render-per-run.md](19-conv-multistage-and-render-per-run.md) | Convolutions (4/4): `conv_universal_wide384` and `depthwise_conv_stride2_*` on params and stages | M | E07-18, E06-14 | none |
| E07-20 | [20-conv-bias-no-relu.md](20-conv-bias-no-relu.md) | Add the missing conv variant: bias without ReLU | S | E07-17, E07-18 | yes (new kernel entries) |
| E07-21 | [21-fully-connected.md](21-fully-connected.md) | `fully_connected`: goldens from the reference per format; delete the 9 `.bin` | M | E07-02, E03 (codecs) | yes (golden source changes) |
| E07-22 | [22-mobilevit-packed-kernels.md](22-mobilevit-packed-kernels.md) | Promote the 12 MobileViT packed programs to registered kernels | M | E07-06, E07-10 | none |
| E07-23 | [23-mobilevit-chain-tests.md](23-mobilevit-chain-tests.md) | Port the MobileViT chain tests to the harness chaining API | M | E07-22, E07-02, E07-13, E06-15, E06-19 | none |
| E07-24 | [24-retire-ipu-apps.md](24-retire-ipu-apps.md) | Retire `src/tools/ipu-apps` and its Bazel macros | S | all of E07, E06-20 | none |

## Suggested order

```
E07-01 ──▶ E07-02 (pilot) ──▶ E07-03 (pilot formats)
              │
              ├──▶ E07-04 ──▶ E07-05                    (reshape)
              ├──▶ E07-06                               (elementwise)
              ├──▶ E07-07 ──▶ E07-08 ──▶ E07-09         (matmul)
              ├──▶ E07-10                               (normalize)
              ├──▶ E07-11 ──▶ E07-12 ──▶ E07-13         (softmax, attention)
              ├──▶ E07-14, E07-15                       (pooling, sampling, detect)
              └──▶ E07-16 ──▶ E07-17 ──▶ E07-18 ──▶ E07-19
                                  └────────┴──▶ E07-20   (convolutions last)
E07-02 ──▶ E07-21                                       (linear)
E07-06 + E07-10 ──▶ E07-22 ──▶ E07-23
MemoryApp users (E07-04, -10, -14, -15, -16) ──▶ E06-20
everything ──▶ E07-24
```

The order follows the decided sequence: pilot, then reshape, elementwise,
matmul, normalize, softmax, attention, pooling/sampling/detect, convolutions
last, then linear. `fully_connected` (E07-21) needs nothing from the other
families, so it is drawn off the pilot, but it is scheduled last. Families with no arrow between them can be worked in
parallel by different people once the pilot has landed. Convolutions go last
because they have the most code, the most lower-case mnemonics, and both
escape hatches (params per run and stages).
