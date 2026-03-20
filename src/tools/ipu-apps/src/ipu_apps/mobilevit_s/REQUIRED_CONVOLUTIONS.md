# MobileViT-S — Required Convolution Configurations

All convolutions use 8-bit data types (INT8 / FP8).

---

## 1. Standard Convolution (3×3)

### Stride 2 (downsampling)

| Layer | Input | Output | Notes |
|---|---|---|---|
| Stem | 256×256×3 → 128×128×16 | in=3, out=16 | First layer of the network |

### Stride 1

| Layer | Input | Output | Notes |
|---|---|---|---|
| Stage 1 Local | 32×32×96 → 32×32×96 | in=96, out=96 | |
| Stage 1 Fusion | 32×32×192 → 32×32×96 | in=192, out=96 | Input doubled by concatenation |
| Stage 2 Local | 16×16×128 → 16×16×128 | in=128, out=128 | |
| Stage 2 Fusion | 16×16×256 → 16×16×128 | in=256, out=128 | Input doubled by concatenation |
| Stage 3 Local | 8×8×160 → 8×8×160 | in=160, out=160 | |
| Stage 3 Fusion | 8×8×320 → 8×8×160 | in=320, out=160 | Input doubled by concatenation |

---

## 2. Depthwise Convolution (3×3)

### Stride 2 (downsampling)

| Layer | Input | Output | Notes |
|---|---|---|---|
| Downsample 1 | 128×128×128 → 64×64×128 | ch=128 | |
| Downsample 2 | 64×64×256 → 32×32×256 | ch=256 | |
| Downsample 3 | 32×32×384 → 16×16×384 | ch=384 | |
| Downsample 4 | 16×16×512 → 8×8×512 | ch=512 | |

### Stride 1

| Layer | Input | Output | Notes |
|---|---|---|---|
| Early Stage | 128×128×64 → 128×128×64 | ch=64 | |
| Mid Stage (×2) | 64×64×256 → 64×64×256 | ch=256 | Repeated twice |

---

## 3. Pointwise Convolution (1×1, stride 1)

### Expansions (increasing channels before depthwise)

| Layer | Input | Output | Notes |
|---|---|---|---|
| Expand 1 | 128×128×16 → 128×128×64 | in=16, out=64 | |
| Expand 2 | 128×128×32 → 128×128×128 | in=32, out=128 | |
| Expand 3 | 64×64×64 → 64×64×256 | in=64, out=256 | |
| Expand 4 | 32×32×96 → 32×32×384 | in=96, out=384 | |
| Expand 5 | 16×16×128 → 16×16×512 | in=128, out=512 | |

### Projections (reducing channels after depthwise or before transformers)

| Layer | Input | Output | Notes |
|---|---|---|---|
| Project 1 | 128×128×64 → 128×128×32 | in=64, out=32 | |
| Project 2 | 64×64×128 → 64×64×64 | in=128, out=64 | |
| Project 3 | 32×32×256 → 32×32×96 | in=256, out=96 | |
| Project 4 | 16×16×384 → 16×16×128 | in=384, out=128 | |
| Project 5 | 8×8×512 → 8×8×160 | in=512, out=160 | |

### Final expansion (before global pooling)

| Layer | Input | Output | Notes |
|---|---|---|---|
| Final | 8×8×160 → 8×8×640 | in=160, out=640 | |
