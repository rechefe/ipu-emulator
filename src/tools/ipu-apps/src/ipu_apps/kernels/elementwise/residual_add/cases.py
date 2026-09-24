"""Runnable cases for residual_add: two random FP32 tensors, checked bit-exact
against their NumPy float32 sum."""
import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase

LANES = 128


def prepare(workspace, *, num_channels, seed):
    rng = np.random.RandomState(seed)
    a = rng.uniform(-4.0, 4.0, size=(num_channels, LANES)).astype("<f4")
    b = rng.uniform(-6.0, 6.0, size=(num_channels, LANES)).astype("<f4")
    expected = (a + b).astype("<f4")
    paths = {name: workspace / f"{name}.bin" for name in ("input_a", "input_b", "output")}
    paths["input_a"].write_bytes(a.tobytes())
    paths["input_b"].write_bytes(b.tobytes())

    def check():
        actual = np.fromfile(paths["output"], dtype="<f4")
        if actual.size != expected.size:
            raise ValueError(f"output has {actual.size} FP32 values, expected {expected.size}")
        mismatches = np.flatnonzero(actual != expected.ravel())
        if mismatches.size:
            ch, lane = divmod(int(mismatches[0]), LANES)
            raise ValueError(
                f"{mismatches.size} mismatches; first at ch={ch} lane={lane}: "
                f"got {actual[mismatches[0]]}, expected {expected[ch, lane]}")

    return PreparedCase({"num_channels": num_channels},
                        {f"{name}_path": path for name, path in paths.items()}, check)


CASES = {
    "default": KernelCase(prepare, {"num_channels": 64, "seed": 42}, max_cycles=1_000_000),
}
