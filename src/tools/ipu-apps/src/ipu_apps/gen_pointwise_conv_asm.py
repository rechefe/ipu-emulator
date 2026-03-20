"""Generate pointwise (1x1) convolution assembly for the IPU.

Produces a fully-pipelined VLIW assembly file for an arbitrary
pointwise convolution: (ROWS x COLS x IN_CH) -> (ROWS x COLS x OUT_CH).

Spatial dimensions must be powers of 2 in [16..128].  Channel counts
must satisfy:
  - IN_CH divides 128 (so an integer number of oc fit in one r register)
  - IN_CH >= 4 (required for r_cyclic 4-slot pipeline)
  - OUT_CH is divisible by 2 * (128 // IN_CH)

Usage::

    python -m ipu_apps.gen_pointwise_conv_asm \\
        --rows 128 --cols 128 --in-channels 64 --out-channels 16 \\
        --output pointwise_conv_128x128_64to16.asm

    # Or print to stdout:
    python -m ipu_apps.gen_pointwise_conv_asm \\
        --rows 64 --cols 64 --in-channels 16 --out-channels 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# -- Cyclic slot register mapping -------------------------------------------
# S0=lr7(0), S1=lr6(128), S2=lr8(256), S3=lr9(384)
_SLOT_LR = {0: "lr7", 1: "lr6", 2: "lr8", 3: "lr9"}
_SLOT_NAME = {0: "S0", 1: "S1", 2: "S2", 3: "S3"}


def generate_pointwise_conv_asm(
    rows: int,
    cols: int,
    in_channels: int,
    out_channels: int,
) -> str:
    """Return the full assembly text for a pointwise convolution kernel.

    Parameters
    ----------
    rows, cols : int
        Spatial dimensions (powers of 2, 16–128).
    in_channels : int
        Number of input channels (must divide 128, >= 4).
    out_channels : int
        Number of output channels.

    Returns
    -------
    str
        Complete assembly source code.
    """
    # -- Validate ------------------------------------------------------------
    valid_spatial = {16, 32, 64, 128}
    if rows not in valid_spatial:
        raise ValueError(f"rows must be a power of 2 in {valid_spatial}, got {rows}")
    if cols not in valid_spatial:
        raise ValueError(f"cols must be a power of 2 in {valid_spatial}, got {cols}")
    if 128 % in_channels != 0:
        raise ValueError(f"in_channels ({in_channels}) must divide 128")
    if in_channels < 4:
        raise ValueError(f"in_channels ({in_channels}) must be >= 4")
    oc_per_reg = 128 // in_channels
    if out_channels % (2 * oc_per_reg) != 0:
        raise ValueError(
            f"out_channels ({out_channels}) must be divisible by "
            f"{2 * oc_per_reg} (2 * 128/{in_channels})"
        )

    # -- Derived constants ---------------------------------------------------
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    row_group_stride = in_channels * 128
    kernel_groups = out_channels // (2 * oc_per_reg)
    kernel_bytes_total = in_channels * out_channels
    num_pipelined = in_channels - 4  # pipeline body length
    words_per_oc = in_channels + 4  # total VLIW words per output channel
    input_bytes = row_groups * in_channels * 128
    output_bytes = row_groups * out_channels * 128 * 4

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    def gen_inner_loop(label: str, ra: str) -> None:
        """Emit inner loop body for one register half."""
        emit(f"{label}:")
        emit("")
        emit("    # W1: Reset accumulator")
        emit("    reset_acc;;")
        emit("")
        emit("    # W2: mult ich0 (S0), kernel[lr4], accumulate")
        emit(f"    mult.ve             {ra} lr7 lr7 lr7 lr4;")
        emit("    acc;;")
        emit("")

        # Pipelined body: load ich(4+k) while multiplying ich(1+k)
        for k in range(num_pipelined):
            ich_load = 4 + k
            ich_mult = 1 + k
            w_num = 3 + k
            load_lr = _SLOT_LR[k % 4]
            mult_lr = _SLOT_LR[(1 + k) % 4]
            load_sn = _SLOT_NAME[k % 4]
            mult_sn = _SLOT_NAME[(1 + k) % 4]

            emit(f"    # W{w_num}: incr lr4; load ich{ich_load}->{load_sn};"
                 f" mult ich{ich_mult} ({mult_sn}); acc")
            emit("    incr                lr4 1;")
            if k == 0:
                emit("    add                 lr14 lr0 lr5;")
            else:
                emit("    add                 lr14 lr14 lr6;")
            emit(f"    ldr_cyclic_mult_reg lr14 cr0 {load_lr};")
            emit(f"    mult.ve             {ra} {mult_lr} lr7 lr7 lr4;")
            emit("    acc;;")
            emit("")

        # Reload ich0-2 while computing last 3 mults
        w_base = 3 + num_pipelined
        emit(f"    # W{w_base}: incr lr4; reload ich0->S0;"
             f" mult ich{in_channels - 3} (S1); acc")
        emit("    incr                lr4 1;")
        emit("    ldr_cyclic_mult_reg lr0 cr0 lr7;")
        emit(f"    mult.ve             {ra} lr6 lr7 lr7 lr4;")
        emit("    acc;;")
        emit("")

        emit(f"    # W{w_base + 1}: incr lr4; reload ich1->S1;"
             f" mult ich{in_channels - 2} (S2); acc")
        emit("    incr                lr4 1;")
        emit("    add                 lr14 lr0 lr6;")
        emit("    ldr_cyclic_mult_reg lr14 cr0 lr6;")
        emit(f"    mult.ve             {ra} lr8 lr7 lr7 lr4;")
        emit("    acc;;")
        emit("")

        emit(f"    # W{w_base + 2}: incr lr4; reload ich2->S2;"
             f" mult ich{in_channels - 1} (S3); acc")
        emit("    incr                lr4 1;")
        emit("    add                 lr14 lr0 lr8;")
        emit("    ldr_cyclic_mult_reg lr14 cr0 lr8;")
        emit(f"    mult.ve             {ra} lr9 lr7 lr7 lr4;")
        emit("    acc;;")
        emit("")

        emit(f"    # W{w_base + 3}: incr lr4; reload ich3->S3")
        emit("    incr                lr4 1;")
        emit("    add                 lr14 lr0 lr9;")
        emit("    ldr_cyclic_mult_reg lr14 cr0 lr9;;")
        emit("")

        emit(f"    # W{w_base + 4}: advance output pointer (+512),"
             " store accumulator, incr oc counter")
        emit("    add                 lr1 lr1 lr5;")
        emit("    incr                lr3 1;")
        emit("    str_acc_reg         lr1 cr3;;")
        emit("")

        emit(f"    # W{w_base + 5}: branch if more output channels in this half")
        emit(f"    blt                 lr3 lr13 {label};;")
        emit("")

    # === Header =============================================================
    emit(f"# Pointwise convolution: {rows}x{cols}x{in_channels} input,"
         f" 1x1x{in_channels}x{out_channels} kernel,"
         f" {rows}x{cols}x{out_channels} output.")
    emit("#")
    emit(f"# 1x1 convolution with {in_channels} input channels"
         f" and {out_channels} output channels.")
    emit("# No spatial kernel -- pure channel mixing (pointwise).")
    emit("#")
    if cols == 128:
        emit("# Since spatial width (128) = SIMD width (128), 1 spatial row is packed")
        emit("# per 128-byte chunk.")
    else:
        emit(f"# Spatial width ({cols}) < SIMD width (128),"
             f" so {rows_per_chunk} spatial rows pack")
        emit("# into each 128-byte chunk.")
    emit(f"# The outer loop iterates over {row_groups} row groups"
         f" ({rows} rows / {rows_per_chunk} per group).")
    emit("#")
    emit(f"# Kernel has {in_channels}x{out_channels} ="
         f" {kernel_bytes_total} weights.")
    emit(f"# Each mult register (r0, r1) holds 128 bytes ="
         f" {oc_per_reg} output channels x {in_channels} inputs.")
    emit(f"# kernel_group_loop iterates {kernel_groups} times,"
         f" each loading r0+r1 (256 bytes)"
         f" = {2 * oc_per_reg} output channels per group.")
    for g in range(kernel_groups):
        oc_lo = g * 2 * oc_per_reg
        oc_mid = oc_lo + oc_per_reg
        oc_hi = oc_mid + oc_per_reg
        koff = g * 256
        emit(f"#   Group {g}: r0=kernel[{koff}..{koff + 127}]"
             f" (oc {oc_lo}-{oc_mid - 1}),"
             f" r1=kernel[{koff + 128}..{koff + 255}]"
             f" (oc {oc_mid}-{oc_hi - 1})")
    emit("#")
    emit("# Pipelining strategy:")
    emit("#   r_cyclic has 4 slots (S0-S3, 128 bytes each, 512 total).")
    emit(f"#   {in_channels} input channels: first 4 pre-loaded,"
         f" remaining {num_pipelined} loaded during computation.")
    emit(f"#   Each output channel requires {words_per_oc} VLIW words"
         f" ({in_channels} mult+acc + pipeline overhead).")
    emit("#")
    emit("# Memory layout (set via CR registers):")
    emit(f"#   cr0 = input  base  ({row_groups} row-groups"
         f" x {in_channels} ch x 128 bytes = {input_bytes} bytes)")
    emit(f"#   cr1 = kernel base  ({kernel_bytes_total} bytes,"
         f" split into {kernel_groups} groups of 256 bytes)")
    emit("#   cr2 = mask   base  (128 bytes: all zeros, no masking needed)")
    emit(f"#   cr3 = output base  ({row_groups} row-groups"
         f" x {out_channels} ch x 512 bytes = {output_bytes} bytes)")
    emit("#")
    emit("# Input interleaving (per row group):")
    emit(f"#   Channel ic, row-group rg: offset ="
         f" rg * {row_group_stride} + ic * 128")
    emit("#")
    emit("# Output interleaving (per row group):")
    emit(f"#   Output channel oc, row-group rg: offset ="
         f" (rg * {out_channels} + oc) * 512")
    emit("#")
    emit("# Kernel layout:")
    emit(f"#   kernel[oc * {in_channels} + ic]"
         f" for oc=0..{out_channels - 1}, ic=0..{in_channels - 1}")
    emit(f"#   Within each r register (128 bytes):"
         f" {oc_per_reg} output channels x {in_channels} input channels")
    emit("#")
    emit("# Register allocation:")
    emit(f"#   lr0  = input row-group base address"
         f" (rg * {row_group_stride})")
    emit("#   lr1  = output pointer (pre-offset by -512,"
         " auto-advances by 512 each store)")
    emit(f"#   lr2  = row-group counter (0..{row_groups - 1})")
    emit(f"#   lr3  = output channel counter (0..{out_channels - 1})")
    emit(f"#   lr4  = kernel index (0..{in_channels - 1} per oc,"
         " advances by 1 per mult.ve)")
    emit("#   lr5  = 512  (output stride)")
    emit("#   lr6  = 128  (cyclic S1 offset / channel stride)")
    emit("#   lr7  = 0    (cyclic S0 offset / no-mask constant)")
    emit("#   lr8  = 256  (cyclic S2 offset / kernel group stride)")
    emit("#   lr9  = 384  (cyclic S3 offset)")
    emit(f"#   lr10 = {oc_per_reg}    (output channels per kernel half)")
    emit(f"#   lr11 = {row_groups}    (row-group loop limit)")
    emit("#   lr12 = kernel memory offset"
         " (0, 256, 512, ... -- advances per kernel group)")
    emit("#   lr13 = inner loop output channel limit"
         " (lr3 + lr10, computed dynamically)")
    emit("#   lr14 = temp (address computation)")
    emit(f"#   lr15 = {out_channels}   (total output channel limit)")
    emit("")

    # === Initialization =====================================================
    emit("# ===========================================================================")
    emit("# Initialization")
    emit("# ===========================================================================")
    emit("")
    emit("    # Load mask data (all zeros -- no masking)")
    emit("    # (lr7 defaults to 0 -- used as zero offset)")
    emit("    ldr_mult_mask_reg   lr7 cr2;")
    emit("    set                 lr5 512;")
    emit("    set                 lr6 128;;")
    emit("")
    emit("    set                 lr8 256;")
    emit("    set                 lr9 384;;")
    emit("")
    emit(f"    set                 lr10 {oc_per_reg};")
    emit(f"    set                 lr15 {out_channels};;")
    emit("")
    emit(f"    set                 lr11 {row_groups};")
    emit("    set                 lr1 -512;;")
    emit("")

    # === Row-group loop =====================================================
    emit("# ===========================================================================")
    emit(f"# Row-group loop ({row_groups} row groups,"
         f" {rows_per_chunk} spatial row(s) each)")
    emit("# ===========================================================================")
    emit("")
    emit("row_loop:")
    emit("")
    emit("    # Load initial 4 input channels (Group A) into r_cyclic S0-S3")
    emit("    ldr_cyclic_mult_reg lr0 cr0 lr7;;")
    emit("")
    emit("    add                 lr14 lr0 lr6;")
    emit("    ldr_cyclic_mult_reg lr14 cr0 lr6;;")
    emit("")
    emit("    add                 lr14 lr0 lr8;")
    emit("    ldr_cyclic_mult_reg lr14 cr0 lr8;;")
    emit("")
    emit("    add                 lr14 lr0 lr9;")
    emit("    ldr_cyclic_mult_reg lr14 cr0 lr9;;")
    emit("")
    emit("    set                 lr3 0;")
    emit("    set                 lr12 0;;")
    emit("")

    # === Kernel-group loop ==================================================
    emit("# ===========================================================================")
    emit(f"# Kernel-group loop ({kernel_groups} groups of"
         f" {2 * oc_per_reg} output channels each)")
    emit("# ===========================================================================")
    emit("")
    emit("kernel_group_loop:")
    emit("")
    emit("    # Load kernel pair: r0 = kernel[lr12..lr12+127],"
         " r1 = kernel[lr12+128..lr12+255]")
    emit("    ldr_mult_reg        r0 lr12 cr1;;")
    emit("")
    emit("    add                 lr14 lr12 lr6;")
    emit("    ldr_mult_reg        r1 lr14 cr1;;")
    emit("")
    emit("    set                 lr4 0;")
    emit("    add                 lr13 lr3 lr10;;")
    emit("")

    # === Inner loop A (r0) ==================================================
    emit("# ---------------------------------------------------------------------------")
    emit(f"# Inner loop A: {oc_per_reg} output channels from r0"
         f" ({words_per_oc} cycles each)")
    emit("# ---------------------------------------------------------------------------")
    emit("")
    gen_inner_loop("LOOP_OCH_A", "r0")

    # === Transition =========================================================
    emit("# ---------------------------------------------------------------------------")
    emit("# Transition: reset kernel index, compute new limit for second half")
    emit("# ---------------------------------------------------------------------------")
    emit("")
    emit("    set                 lr4 0;")
    emit("    add                 lr13 lr3 lr10;;")
    emit("")

    # === Inner loop B (r1) ==================================================
    emit("# ---------------------------------------------------------------------------")
    emit(f"# Inner loop B: {oc_per_reg} output channels from r1"
         f" ({words_per_oc} cycles each)")
    emit("# ---------------------------------------------------------------------------")
    emit("")
    gen_inner_loop("LOOP_OCH_B", "r1")

    # === Kernel group advance ===============================================
    emit("# ---------------------------------------------------------------------------")
    emit("# Advance to next kernel group")
    emit("# ---------------------------------------------------------------------------")
    emit("")
    emit("    add                 lr12 lr12 lr8;")
    emit("    blt                 lr3 lr15 kernel_group_loop;;")
    emit("")

    # === Row-group advance ==================================================
    emit("# ---------------------------------------------------------------------------")
    emit("# Advance to next row group")
    emit("# ---------------------------------------------------------------------------")
    emit("")
    emit(f"    incr                lr0 {row_group_stride};")
    emit("    incr                lr2 1;;")
    emit("")
    emit("    blt                 lr2 lr11 row_loop;;")
    emit("")
    emit("end:")
    emit("    bkpt;;")

    return "\n".join(lines) + "\n"


# -- CLI ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pointwise (1x1) convolution assembly for the IPU.",
    )
    parser.add_argument("--rows", type=int, default=128,
                        help="Spatial height (power of 2, 16-128)")
    parser.add_argument("--cols", type=int, default=128,
                        help="Spatial width (power of 2, 16-128)")
    parser.add_argument("--in-channels", type=int, required=True,
                        help="Number of input channels (must divide 128, >= 4)")
    parser.add_argument("--out-channels", type=int, required=True,
                        help="Number of output channels")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output .asm file (default: stdout)")
    args = parser.parse_args()

    asm = generate_pointwise_conv_asm(
        rows=args.rows,
        cols=args.cols,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(asm, newline="\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(asm)


if __name__ == "__main__":
    main()
