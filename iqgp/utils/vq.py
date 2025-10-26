"""Vector quantization utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vector_quantize(inputs: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``inputs`` with the nearest entries in ``codebook``.

    Args:
        inputs: Tensor of shape (..., dim).
        codebook: Tensor of shape (num_codes, dim).

    Returns:
        Tuple of (quantized, codes) where ``quantized`` matches the shape of
        ``inputs`` and ``codes`` has shape (...) with integer indices.
    """
    flat_inputs = inputs.view(-1, inputs.size(-1))
    distances = (
        flat_inputs.pow(2).sum(dim=1, keepdim=True)
        - 2 * flat_inputs @ codebook.t()
        + codebook.pow(2).sum(dim=1)
    )
    codes = distances.argmin(dim=1)
    quantized = codebook.index_select(0, codes).view_as(inputs)
    quantized = inputs + (quantized - inputs).detach()
    codes = codes.view(inputs.shape[:-1])
    return quantized, codes


def commitment_loss(inputs: torch.Tensor, quantized: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    """Compute the VQ commitment loss."""
    return F.mse_loss(quantized.detach(), inputs) + beta * F.mse_loss(quantized, inputs.detach())
