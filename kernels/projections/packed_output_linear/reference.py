import torch

from ipu_kernels.reference.ops import linear


def reference(
    a: torch.Tensor, b: torch.Tensor, *, k: int, activation: str, **_params: object
) -> dict[str, torch.Tensor]:
    """ops.linear on the (k, tokens) data whose packed rows are b; y is the
    (n_out, tokens) result as its one packed row."""
    y = linear(a, b.reshape(k, -1), activation=activation)["y"]
    return {"y": y.reshape(1, 128)}
