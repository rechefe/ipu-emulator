"""depthwise_conv_stride2_128's two programs (E06-14).

Stage 1 is depthwise_conv_tpl's program, unseeded (has_bias and apply_relu are
false), at full resolution, with the kernel's [abi] (the template's table:
CR15 holds the partitioned DS and the case's QF); it writes the scratch y.
Stage 2 is this folder's program: it reads y through CR3 (a stage's table binds
a region at its base) and writes out, with CR15 its stage2_ds (ds(128) and the
case's QF), as the old harness's second setup wrote the default dstructure;
the other CRs and the LRs carry over from stage 1. load_program resets the PC
between them.
"""

STAGE1 = "../depthwise_conv_tpl/depthwise_conv_tpl.asm"


def run(ctx):
    """Stage 1, then stage 2 on the same machine state."""
    ctx.run_stage(STAGE1)
    ctx.run_stage(
        "depthwise_conv_stride2_128.asm",
        abi={
            "y": "CR3",
            "in_channels": "CR4",
            "out": "CR5",
            "num_row_pairs": "CR6",
            "stage2_ds": "CR15",
        },
    )
