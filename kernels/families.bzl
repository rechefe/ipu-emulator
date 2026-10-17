"""The kernel families, their kernels, and the only declared cross-family use (E06-18).

    load("//kernels:families.bzl", "FAMILY_DEPS", "KERNELS")

A family is one Bazel package, //kernels/<family>, declared with
ipu_kernel_family(); each kernel is a subpackage, //kernels/<family>/<name>,
declared with ipu_kernel(). Bazel cannot glob into subpackages, so KERNELS
names them; //kernels:test_families checks it against the kernel.toml folders.
"""

#: Each family's kernels, by folder name.
KERNELS = {
    "detect": ["channel_peak", "score_threshold"],
    "elementwise": ["scale"],
    "pooling": [
        "maxpool2d_nms",
        "maxpool2d_stride2",
        "maxpool2d_stride2_tail",
        "maxpool2d_window",
    ],
    "projections": ["proj_p4", "proj_p4_2tg", "proj_p4_2tg_silu"],
    "reshape": ["concat", "depth_to_space", "identity"],
    "sampling": ["sample_descriptors", "sample_descriptors_separable"],
}

#: The families each family may use: import from (ipu_kernels.<other>) or refer
#: to the files of (kernels/<other>/...). A use not listed here fails
#: //kernels:test_families. E07 adds the edges its migrations need, for
#: example "attention": ["softmax"].
FAMILY_DEPS = {}
