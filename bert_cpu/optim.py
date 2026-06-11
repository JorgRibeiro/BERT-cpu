"""Parameter optimisers."""

from __future__ import annotations

from typing import List

from bert_cpu.engine import Tensor


class Optimizer:
    """Base optimiser interface."""

    def __init__(self, parameters: List[Tensor]) -> None:
        raise NotImplementedError

    def step(self) -> None:
        """Apply one update to every parameter from its current gradient."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(
        self, parameters: List[Tensor], lr: float = 1e-3, momentum: float = 0.0
    ) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class Adam(Optimizer):
    """Adam optimiser (the usual choice for Transformer training)."""

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError
