import torch


def reference(x: torch.Tensor, *, rows: int, s: float) -> dict[str, torch.Tensor]:
    """y = x * s. The framework passes inputs decoded from the active format
    and rounds the result to that format before comparing."""
    return {"y": x * s}
