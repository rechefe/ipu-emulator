import torch

from ipu_kernels.reference.ops import layernorm


def reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    channels: int,
    tokens: int,
    **_params: object,
) -> dict[str, torch.Tensor]:
    """ops.layernorm over the channels of the (channels, tokens) tensor whose
    packed rows are x; y is the result's packed rows."""
    y = layernorm(x.reshape(channels, tokens), gamma, beta, dim=0)["y"]
    return {"y": y.reshape(-1, 128)}
