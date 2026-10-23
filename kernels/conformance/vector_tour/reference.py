import torch

#: The R0 element the program reads as a scalar (LR4).
SCALAR_LANE = 4


def reference(
    a: torch.Tensor, w: torch.Tensor, c: torch.Tensor, *, two: float
) -> dict[str, torch.Tensor]:
    """The three R_ACC stores, as the program's comments describe them, and
    the relu of the third store's vector.

    The data is exact in every variant, so float64 arithmetic gives each
    accumulator value exactly; the harness quantizes each row like the
    program's ACTIVATE.QUANTIZE."""
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
    return {"acc": torch.stack((first, second, third)), "relu": third.clamp(min=0)}
