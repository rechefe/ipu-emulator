import torch


def reference(x: torch.Tensor, *, rows: int, s: float) -> dict[str, torch.Tensor]:
    """y = x * s. The framework passes inputs decoded from the case's variant
    and rounds the result to it before comparing."""
    return {"y": x * s}
