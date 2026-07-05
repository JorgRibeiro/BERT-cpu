"""Exercise 05 — Binary classification on the Adult dataset (the baseline).

This is the **capstone**: the first exercise that runs the *whole* pipeline end
to end on real data. Everything the earlier files built in isolation — the
autograd engine, a `Linear` layer, an activation, an optimiser, a loss — is
assembled here into a working income classifier and **actually trained**.

    datasets.load_adult  ->  X, y (model-ready NumPy)
              |
         nn.Linear + relu + nn.Linear     (the model — a small MLP)
              |
         cross_entropy(logits, y)         (scalar loss; one graph back to W)
              |
         loss.backward()                  (fills every parameter's .grad)
              |
         optim.Adam(...).step()           (nudges each parameter downhill)

It is the *baseline*: complete, runnable, nothing to fill in. (The other
exercises, q01–q04, deliberately never train — they gradient-check a `forward`.
q05 is the sanctioned exception, because a classifier only means something once
it is trained.) A later pass will turn parts of this into student TODOs and add
guided questions and tests.

The whole Adult training set fits in memory, so there is no need for
mini-batches: each epoch is one **full-batch** step over *all* training rows at
once — the four-line cycle from the ``loss.py`` / ``optim.py`` docstrings::

    opt.zero_grad()                 # clear last step's gradients
    loss = cross_entropy(model(X).T, y)
    loss.backward()                 # one graph: loss -> ... -> every weight
    opt.step()                      # p.data -= lr * (adam update)

A slice of the training base is held out as a **validation** set. At the end of
each epoch we also measure the loss there (a forward pass only, no gradient
step): if the training loss keeps falling while the validation loss turns back
up, the model is starting to over-fit.

Conventions
-----------
Column-oriented like the rest of the library: features run down axis 0 and
samples across axis 1, so ``X`` is ``(n_features, n_samples)`` and ``nn.Linear``
consumes it directly. The model outputs logits ``(2, batch)``; we transpose to
``(batch, 2)`` for ``cross_entropy`` (whose class axis is last). Binary income
(``<=50K`` / ``>50K``) is framed as a **2-class** problem so the existing
``cross_entropy`` is reused unchanged.

HOW TO RUN
==========
    python -m exercises.q05_binary_classification
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the script runnable *directly* (``python exercises/q05_binary_classification.py``)
# as well as via ``python -m exercises.q05_binary_classification``. Running a file
# directly only puts its own folder (``exercises/``) on the import path, so we add
# the repo root so ``datasets`` and ``bert_cpu`` resolve either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu import optim
from bert_cpu.loss import cross_entropy


# ============================================================================ #
# The model — a small multilayer perceptron
# ============================================================================ #
class AdultMLP(nn.Module):
    """``Linear -> ReLU -> Linear`` classifier over the Adult features.

    Two learnable layers (each a ``nn.Linear`` with its bias folded into the
    weight, the project's bias trick). The hidden ReLU gives the model the
    non-linearity it needs to beat a plain logistic regression; the final layer
    produces two logits, one per income class.
    """

    def __init__(self, n_features: int, hidden: int = 64) -> None:
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # x: (n_features, batch)  ->  logits: (2, batch), column-oriented.
        h = self.fc1(x).relu()
        return self.fc2(h)


# ============================================================================ #
# Evaluation
# ============================================================================ #
def accuracy(model: AdultMLP, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples whose arg-max logit matches the label.

    A pure forward pass (no graph needed for a metric); ``model`` predicts the
    class with the larger logit for every column of ``X``.
    """
    logits = model(cpu.Tensor(X)).data          # (2, n_samples)
    preds = logits.argmax(axis=0)               # class per sample
    return float((preds == y).mean())


# ============================================================================ #
# Training
# ============================================================================ #
def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2):
    """Carve a validation set out of the training base (column-oriented split).

    Shuffles the sample indices once and holds out ``val_frac`` of them for
    validation. Returns ``(X_tr, y_tr, X_val, y_val)``.
    """
    n = X.shape[1]                              # samples across axis 1
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return X[:, tr_idx], y[tr_idx], X[:, val_idx], y[val_idx]


def train(
    model: AdultMLP,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-2,
) -> None:
    """Full-batch training with Adam; report train and validation loss per epoch.

    The whole training split is one batch, so each epoch is a single
    forward/backward over every row. The input tensors are built once and marked
    ``requires_grad=False`` (we need gradients on the *parameters*, not the data),
    which also avoids uselessly accumulating gradient into the inputs.
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    Xt = cpu.Tensor(X_tr, requires_grad=False)      # (n_features, n_train), a constant
    Xv = cpu.Tensor(X_val, requires_grad=False)     # (n_features, n_val),   a constant

    total_flops = 0
    for epoch in range(1, epochs + 1):
        cpu.reset_flops()                           # start this epoch's FLOP tally

        opt.zero_grad()
        loss = cross_entropy(model(Xt).T, y_tr)     # full-batch training loss
        loss.backward()                             # one graph -> every weight's grad
        opt.step()

        # Validation loss: a forward pass only (no backward, no step) on the
        # held-out slice of the training base.
        val_loss = float(cross_entropy(model(Xv).T, y_val).data)

        # FLOPs the engine executed this epoch (train forward + backward + the
        # validation forward). It is the same every epoch — the graph is fixed.
        epoch_flops = cpu.flop_count()
        total_flops += epoch_flops

        print(f"  epoch {epoch:3d}/{epochs}   train loss = {float(loss.data):.4f}"
              f"   val loss = {val_loss:.4f}   FLOPs = {epoch_flops:,}")

    print(f"\nTotal FLOPs over {epochs} epochs: {total_flops:,}"
          f"   (~{total_flops / 1e9:.2f} GFLOP)")


# ============================================================================ #
# Entry point
# ============================================================================ #
def main() -> None:
    print("=" * 70)
    print("Adult income classification — end-to-end baseline on bert_cpu")
    print("=" * 70)

    cpu.set_seed(0)                             # reproducible init + shuffling

    train_ds = datasets.load_adult("train")
    test_ds = datasets.load_adult("test")
    print(f"\nData: {train_ds}   {test_ds}")
    print(f"Features per sample: {train_ds.n_features}  (standardised + one-hot)")

    # Hold out 20% of the training base for validation.
    X_tr, y_tr, X_val, y_val = train_val_split(train_ds.X, train_ds.y, val_frac=0.2)
    print(f"Train/val split: {X_tr.shape[1]} train / {X_val.shape[1]} val (20%)\n")

    model = AdultMLP(train_ds.n_features, hidden=64)
    print(f"Model: Linear({train_ds.n_features}, 64) -> ReLU -> Linear(64, 2)")
    print(f"Trainable parameter tensors: {len(model.parameters())}\n")

    print("Training (full-batch Adam):")
    train(model, X_tr, y_tr, X_val, y_val, epochs=100, lr=1e-2)

    train_acc = accuracy(model, X_tr, y_tr)
    val_acc = accuracy(model, X_val, y_val)
    test_acc = accuracy(model, test_ds.X, test_ds.y)
    print(f"\nFinal accuracy   train = {train_acc:.4f}   val = {val_acc:.4f}   test = {test_acc:.4f}")
    # A majority-class baseline (always predict <=50K) scores ~0.76 on test;
    # this MLP should comfortably clear that, landing around 0.84–0.85.


if __name__ == "__main__":
    main()
