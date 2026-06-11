"""Loss functions."""

from __future__ import annotations

from typing import Optional

from bert_cpu.engine import Tensor


def cross_entropy(logits: Tensor, targets, ignore_index: int = -100) -> Tensor:
    """Mean softmax cross-entropy over a batch of token predictions.

    Parameters
    ----------
    logits : Tensor
        Unnormalised scores of shape ``(..., num_classes)``.
    targets : array-like of int
        Ground-truth class ids, broadcastable to ``logits`` minus the class axis.
    ignore_index : int
        Target value that should not contribute to the loss (e.g. non-masked
        tokens during MLM training).
    """
    raise NotImplementedError


def masked_lm_loss(logits: Tensor, labels, ignore_index: int = -100) -> Tensor:
    """Cross-entropy restricted to masked positions for MLM pretraining."""
    raise NotImplementedError
