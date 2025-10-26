"""Language adversary utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class LanguageAdversary(nn.Module):
    """Predict the language from planner states (used with gradient reversal)."""

    def __init__(self, hidden_size: int, num_languages: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_languages),
        )

    def forward(self, states: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        reversed_states = grad_reverse(states, lambda_)
        return self.classifier(reversed_states)


__all__ = ["LanguageAdversary", "grad_reverse"]
