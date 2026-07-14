"""Debug runner for the MambaVision Stage 3 Layer Norm app.

Usage:

    bazel run //src/tools/ipu-apps:mambavision_stage3_layer_norm -- \
      --output /tmp/layer_norm_output.bin
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ipu_apps.mambavision_stage3_layer_norm import MambavisionStage3LayerNormApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MambaVision Stage 3 Layer Norm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=5_000_000)
    args = parser.parse_args()

    data_dir = Path(os.environ["LAYER_NORM_DATA_DIR"]) / "fp32"

    app = MambavisionStage3LayerNormApp(
        inst_path=Path(os.environ["LAYER_NORM_INST_BIN"]),
        inputs_path=data_dir / "inputs_fp32.bin",
        gamma_path=data_dir / "gamma_fp32.bin",
        beta_path=data_dir / "beta_fp32.bin",
        const_path=data_dir / "const_fp32.bin",
        output_path=args.output,
    )

    _state, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Total cycles: {cycles}")


if __name__ == "__main__":
    main()
