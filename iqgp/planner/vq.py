"""Vector quantization with EMA updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return squared L2 distance between every pair of rows."""

    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_sq = (y ** 2).sum(dim=-1).unsqueeze(0)
    distance = x_sq + y_sq - 2 * torch.matmul(x, y.t())
    return distance


@dataclass
class VQOutput:
    """Outputs of the vector quantizer."""

    quantized: torch.Tensor
    codes: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor

    @property
    def vq_loss(self) -> torch.Tensor:
        return self.commitment_loss + self.codebook_loss


class VectorQuantizerEMA(nn.Module):
    """Exponential moving average Vector Quantizer (VQ-VAE style)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_embeddings <= 0:
            raise ValueError("embedding_dim and num_embeddings must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())

    def forward(self, inputs: torch.Tensor) -> VQOutput:
        orig_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        distances = _compute_l2_distance(flat_inputs, self.embedding)
        encoding_indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)
        quantized = torch.matmul(encodings, self.embedding).view(orig_shape)

        # EMA updates
        if self.training:
            updated_cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size.mul_(self.decay).add_(updated_cluster_size, alpha=1 - self.decay)
            dw = torch.matmul(encodings.t(), flat_inputs)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            normalized_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.copy_(self.ema_w / normalized_cluster_size.unsqueeze(1))

        # Losses
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        codes = encoding_indices.view(*orig_shape[:-1])

        return VQOutput(
            quantized=quantized,
            codes=codes,
            commitment_loss=self.commitment_cost * commitment_loss,
            codebook_loss=codebook_loss,
        )


__all__ = ["VectorQuantizerEMA", "VQOutput"]
