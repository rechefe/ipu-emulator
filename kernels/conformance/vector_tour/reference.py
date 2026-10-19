import torch

#: The R0 element the program reads as a scalar (LR4).
SCALAR_LANE = 4


def reference(
    a: torch.Tensor, w: torch.Tensor, c: torch.Tensor, *, two: float, int8_tail: bool
) -> dict[str, torch.Tensor]:
    """The three R_ACC stores, as the program's comments describe them, and in
    INT8 the ST.AAQ store of relu(R_ACC) (the third store's vector).

    The data is exact in every format, so float64 arithmetic gives each
    accumulator value exactly."""
    a, w, c = a.double(), w.double(), c.double()
    a0 = a[0]
    lanes = torch.arange(a0.numel())
    first = torch.maximum(two * a0 + w * a0 - c * c, c * a0[SCALAR_LANE])
    zero = torch.zeros_like(a0)
    second = torch.where(lanes % 2 == 0, w * a0, zero) + torch.where(
        lanes < a0.numel() // 2, two * a0[SCALAR_LANE], zero
    )
    rows = a.shape[0]
    third = second.clone()
    third[:rows] = (a * w).sum(dim=1)
    third[rows] = (a[rows - 1] * w).max()
    out = {"acc": torch.stack((first, second, third))}
    if int8_tail:
        relu = torch.zeros(4, a0.numel(), dtype=torch.float64)
        relu[0] = third.clamp(0, 127)
        out["relu"] = relu
    return out
