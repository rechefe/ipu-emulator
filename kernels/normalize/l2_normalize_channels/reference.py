import torch


def reference(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """y = x / ||x||_2 down the channels (dim 0), per column.

    As the kernel computes it: y = x * rsqrt(sum over channels of x^2), and the
    rsqrt activation maps a non-positive input to 0, so an all-zero column
    gives 0 (not NaN or inf)."""
    ss = (x * x).sum(dim=0, keepdim=True)
    scale = torch.where(ss > 0, torch.rsqrt(ss), torch.zeros_like(ss))
    return {"y": x * scale}
