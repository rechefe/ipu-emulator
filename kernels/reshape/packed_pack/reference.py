import torch


def reference(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """The ``(channels, tokens)`` tensor x as its packed rows: row-major over
    the flattened tensor, ``128 // tokens`` channels a row."""
    return {"y": x.reshape(-1, 128).clone()}
