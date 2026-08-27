# Benchmark: depthwise_conv_stride2_narrow

| config | cycles | mult util % | correct |
|---|---:|---:|:---:|
| 8x64x2 | 273 | 71.1% | ✓ |
| 8x64x16 | 1309 | 106.7% | ✓ |
| 16x32x4 | 421 | 87.2% | ✓ |
| 16x32x24 | 1901 | 109.6% | ✓ |
| 32x16x3 | 347 | 80.9% | ✓ |
| 32x16x16 | 1309 | 106.7% | ✓ |
| **TOTAL** | 5560 | 102.9% | ✓ |
