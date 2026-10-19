import torch


def reference(
    x: torch.Tensor, *, channels: int, tokens: int, **_params: object
) -> dict[str, torch.Tensor]:
    """The packed rows x as the ``(channels, tokens)`` tensor, one channel a row."""
    return {"y": x.reshape(channels, tokens).clone()}
