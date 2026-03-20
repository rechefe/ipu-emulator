import ipu.nn as nn


class MobileViTBlock(nn.Module):
    """Local-global processing: CNN extracts spatial features,
    Transformer captures long-range dependencies."""

    def __init__(self, inp, dim, heads, depth):
        super().__init__()
        self.local_conv = nn.Conv2d(inp, dim, kernel_size=3)
        self.transformers = nn.Sequential(*[nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiHeadAttention(dim, heads),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2, activation="gelu"),
            nn.Linear(dim * 2, dim),
        ) for _ in range(depth)])
        self.project = nn.PointwiseConv2d(dim, inp)

    def forward(self, x):
        out = self.local_conv(x)
        out = self.transformers(out)
        return self.project(out) + x


class MobileViT_S(nn.Module):
    """MobileViT-S: hybrid CNN-Transformer for mobile vision."""

    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, activation="silu")
        # MobileNetV2 inverted-residual stages
        self.stage1 = nn.MV2Block(16, 32, stride=2, expand_ratio=4)
        self.stage2 = nn.MV2Block(32, 64, stride=2, expand_ratio=4)
        # MobileViT blocks (local conv + transformer + projection)
        self.mvit1  = MobileViTBlock(64, dim=144, heads=1, depth=2)
        self.stage3 = nn.MV2Block(64, 96, stride=2, expand_ratio=4)
        self.mvit2  = MobileViTBlock(96, dim=192, heads=3, depth=4)
        self.stage4 = nn.MV2Block(96, 128, stride=2, expand_ratio=4)
        self.mvit3  = MobileViTBlock(128, dim=240, heads=4, depth=3)
        # Classification head
        self.head = nn.Sequential(nn.GlobalAvgPool2d(),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.mvit1(x)
        x = self.stage3(x)
        x = self.mvit2(x)
        x = self.stage4(x)
        x = self.mvit3(x)
        return self.head(x)
