"""depthwise_conv_stride2_narrow's two programs (E06-14).

Stage 1 is depthwise_conv_tpl's program, unseeded (has_bias and apply_relu are
false), at full resolution, with the kernel's [abi] (the template's table, so y
at offset -1 and CR15 the partitioned dstructure); it writes the scratch y.
Stage 2 is this folder's program: it reads y through CR3 (a stage's table binds
a region at its base) and writes out, with CR15 the default dstructure (128
valid elements, P0), as the old harness's second setup wrote it; the other CRs
and the LRs carry over from stage 1. load_program resets the PC between them.
"""

STAGE1 = "../depthwise_conv_tpl/depthwise_conv_tpl.asm"


def run(ctx):
    """Stage 1, then stage 2 on the same machine state."""
    ctx.run_stage(STAGE1)
    ctx.run_stage(
        "depthwise_conv_stride2_narrow.asm",
        abi={"y": "CR3", "in_channels": "CR4", "out": "CR5", "num_out_groups": "CR6"},
    )
