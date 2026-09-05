"""Fully connected fixtures and checks, shared by run and test."""
from pathlib import Path

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, check_output_bytes

DATA = Path(__file__).with_name("test_data_format")


class MissingInputFixture(FileNotFoundError):
    """An optional input or weight fixture is unavailable."""


def prepare(workspace, *, dtype, wide_mode=False):
    name = dtype.lower()
    if name not in ("int8", "fp8_e4m3", "fp8_e5m2"):
        raise ValueError("dtype must be INT8, FP8_E4M3, or FP8_E5M2")
    directory = DATA / name
    inputs = directory / f"inputs_{name}.bin"
    weights = directory / f"weights_{name}.bin"
    for path in (inputs, weights):
        if not path.is_file():
            raise MissingInputFixture(f"missing fully connected fixture: {path}")
    out = workspace / "output.bin"
    suffix = "int32" if name == "int8" else "fp32"
    golden = directory / f"out_{name}_acc_{suffix}.bin"

    def check():
        check_output_bytes(out, golden)

    return PreparedCase({"dtype": dtype, "wide_mode": wide_mode},
                        {"inputs_path": inputs, "weights_path": weights, "output_path": out}, check)


def prepare_default(workspace, *, dtype):
    # Retain the existing INT8 runner's wide arithmetic configuration.
    return prepare(workspace, dtype=dtype, wide_mode=dtype.upper() == "INT8")


CASES = {
    "default": KernelCase(prepare_default, {"dtype": "INT8"}, 2_000_000),
    "int8": KernelCase(prepare, {"dtype": "INT8"}, 2_000_000),
    "fp8_e4m3": KernelCase(prepare, {"dtype": "FP8_E4M3"}, 2_000_000),
    "fp8_e5m2": KernelCase(prepare, {"dtype": "FP8_E5M2"}, 2_000_000),
}
