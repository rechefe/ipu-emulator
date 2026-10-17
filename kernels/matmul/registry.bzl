"""Generated from registry.toml by //kernels:update_registry_bzl. Do not edit."""

ENTRIES = [
    {"name": "matmul_64x64x64", "template": "matmul_small"},
    {"name": "matmul_128x64x64", "template": "matmul_small"},
    {"name": "matmul_128x64x128", "template": "matmul_small"},
    {"name": "matmul_128x128", "template": "matmul_small"},
    {"name": "matmul_144x144_x128", "template": "matmul_x128_2tg_c2"},
    {"name": "matmul_432x144_x128", "template": "matmul_x128_2tg_c2"},
    {"name": "matmul_288x144_x128", "template": "matmul_x128_2tg_c2"},
    {"name": "matmul_144x288_x128", "template": "matmul_x128_2tg_c3"},
    {"name": "matmul_192x192_x128", "template": "matmul_x128_c2"},
    {"name": "matmul_384x192_x128", "template": "matmul_x128_c2"},
    {"name": "matmul_240x240_x128", "template": "matmul_x128_c2"},
    {"name": "matmul_480x240_x128", "template": "matmul_x128_c2"},
    {"name": "matmul_720x240_x128", "template": "matmul_x128_c2"},
    {"name": "matmul_576x192_x128", "template": "matmul_x128_c2_qkv"},
    {"name": "matmul_192x384_x128", "template": "matmul_x128_c3"},
    {"name": "matmul_240x480_x128", "template": "matmul_x128_c4"},
]
