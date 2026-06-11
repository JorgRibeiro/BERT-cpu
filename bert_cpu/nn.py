"""Neural-network building blocks built on top of the ``Tensor`` engine.

Mirrors micrograd's ``nn.py`` (``Module`` base + composable layers) but the
primitives are now tensor-valued: ``Linear``, ``Embedding``, ``LayerNorm``,
``Dropout`` and container modules.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from bert_cpu.engine import Tensor


class Parameter(Tensor):
    """A ``Tensor`` that is registered as a learnable parameter of a module."""


class Module:
    """Base class for all neural-network modules.

    Subclasses implement ``forward`` (invoked via ``__call__``) and expose their
    learnable tensors through ``parameters``.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        """Invoke ``forward``."""
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute the module's output. Must be overridden by subclasses."""
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Return the flat list of learnable parameters in this module."""
        raise NotImplementedError

    def named_parameters(self) -> Iterator[tuple]:
        """Yield ``(name, parameter)`` pairs for this module."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Zero the gradient of every parameter."""
        raise NotImplementedError

    def train(self, mode: bool = True) -> "Module":
        """Set training mode (affects e.g. Dropout)."""
        raise NotImplementedError

    def eval(self) -> "Module":
        """Set evaluation mode."""
        raise NotImplementedError


class Linear(Module):
    """Affine transformation ``y = x @ W + b``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Embedding(Module):
    """Lookup table mapping integer ids to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        raise NotImplementedError

    def forward(self, idx) -> Tensor:
        """Gather rows of the table for the given integer ids."""
        raise NotImplementedError


class LayerNorm(Module):
    """Layer normalisation over the last dimension with learnable affine."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Dropout(Module):
    """Inverted dropout; a no-op when in eval mode."""

    def __init__(self, p: float = 0.1) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Sequential(Module):
    """Container that chains modules end to end."""

    def __init__(self, *modules: Module) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Weight initialisation
# ---------------------------------------------------------------------- #
def xavier_uniform(in_features: int, out_features: int) -> Tensor:
    """Glorot/Xavier uniform initialisation for a weight matrix."""
    raise NotImplementedError


def normal_(shape, mean: float = 0.0, std: float = 0.02) -> Tensor:
    """BERT-style truncated-ish normal initialisation."""
    raise NotImplementedError
