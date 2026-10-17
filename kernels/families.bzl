"""The kernel families, their kernels, and the only declared cross-family use (E06-18).

    load("//kernels:families.bzl", "FAMILY_DEPS", "KERNELS")

A family is one Bazel package, //kernels/<family>, declared with
ipu_kernel_family(); each kernel is a subpackage, //kernels/<family>/<name>,
declared with ipu_kernel(). Bazel cannot glob into subpackages, so KERNELS
names them; //kernels:test_families checks it against the kernel.toml folders.
"""

#: Each family's kernels, by folder name.
KERNELS = {
    "attention": [
        "attn_scores_km_16x60",
        "attn_scores_km_256x36",
        "attn_scores_km_64x48",
        "attn_v_16x60",
        "attn_v_256x36",
        "attn_v_64x48",
        "attn_v_bcast_36",
        "attn_v_bcast_48",
        "attn_v_bcast_60",
        "qk_scores_16x60",
        "qk_scores_256x36",
        "qk_scores_64x48",
    ],
    "conformance": ["vector_tour"],
    "convolutions": [
        "conv1x1",
        "conv3x3_relu",
        "conv3x3_relu_cin1",
        "conv_first_layer",
        "conv_universal_tpl",
    ],
    "detect": ["channel_peak", "score_threshold"],
    "elementwise": ["residual_add", "residual_add_rows", "scale"],
    "linear": ["fully_connected"],
    "matmul": [
        "matmul_small",
        "matmul_x128_2tg_c2",
        "matmul_x128_2tg_c3",
        "matmul_x128_c2",
        "matmul_x128_c2_qkv",
        "matmul_x128_c3",
        "matmul_x128_c4",
    ],
    "mobilevit": [],
    "normalize": ["l2_normalize_channels", "layernorm"],
    "pooling": [
        "maxpool2d_nms",
        "maxpool2d_stride2",
        "maxpool2d_stride2_tail",
        "maxpool2d_window",
    ],
    "projections": ["proj_p4", "proj_p4_2tg", "proj_p4_2tg_silu"],
    "reshape": [
        "concat",
        "depth_to_space",
        "fold_16x16x192",
        "fold_32x32x144",
        "fold_8x8x240",
        "identity",
        "unfold_16x16x192",
        "unfold_32x32x144",
        "unfold_8x8x240",
    ],
    "sampling": ["sample_descriptors", "sample_descriptors_separable"],
    "softmax": [
        "softmax_columns",
        "softmax_columns_packed",
        "softmax_rows",
        "softmax_rows_long",
        "softmax_rows_partial",
    ],
}

#: The families each family may use: import from (ipu_kernels.<other>) or refer
#: to the files of (kernels/<other>/...). A use not listed here fails
#: //kernels:test_families. E07 adds the edges its migrations need, for
#: example "attention": ["softmax"].
FAMILY_DEPS = {
    "attention": ["softmax"],
}
