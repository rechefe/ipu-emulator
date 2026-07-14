"""Debug runner for MambaVision Stage 3 depthwise Conv1D + SiLU.

Usage:

    bazel run //src/tools/ipu-apps:mambavision_stage3_depthwise_conv1d_silu -- \
      --output /tmp/conv1d_silu_output.bin
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ipu_apps.mambavision_stage3_depthwise_conv1d_silu import (
    MambavisionStage3DepthwiseConv1dSiluApp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MambaVision Stage 3 Conv1D+SiLU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=5_000_000)
    args = parser.parse_args()

    data_dir = Path(os.environ["CONV1D_SILU_DATA_DIR"]) / "fp32"

    app = MambavisionStage3DepthwiseConv1dSiluApp(
        inst_path=Path(os.environ["CONV1D_SILU_INST_BIN"]),
        inputs_padded_path=data_dir / "inputs_padded_fp32.bin",
        tap_minus1_path=data_dir / "tap_minus1_fp32.bin",
        tap_zero_path=data_dir / "tap_zero_fp32.bin",
        tap_plus1_path=data_dir / "tap_plus1_fp32.bin",
        output_path=args.output,
    )

    _state, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Total cycles: {cycles}")


if __name__ == "__main__":
    main()
