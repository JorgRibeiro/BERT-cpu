"""Binary classification on Adult for the controlled q01 experiments.

The model keeps the original column-oriented architecture ``108 -> 64 -> 2``.
For Variable 1, only the hidden activation changes: ReLU (the default),
Sigmoid, Swish or Softplus. Training is full-batch Adam and validation is a
fixed hold-out from the official Adult training file.

Variable 3 uses a separate ``AdultLinearClassifier`` so the activation studies
remain unchanged. It directly composes one, two or three affine layers without
creating an artificial Identity operation between them.

The official test file is deliberately opt-in. A normal invocation uses only
training and validation data::

    python -m exercises.task_binary_classification --activation relu

Use ``--evaluate-test`` only in the previously approved final evaluation phase.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# Also support ``python exercises/task_binary_classification.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu import optim
from bert_cpu.loss import cross_entropy
from exercises.q01_activations import ExTensor


ACTIVATIONS = ("relu", "sigmoid", "swish", "softplus")
SUPPORTED_ACTIVATIONS = ACTIVATIONS + ("softplus_beta",)
SUPPORTED_LINEAR_DEPTHS = (1, 2, 3)


# ============================================================================ #
# Model
# ============================================================================ #
class AdultMLP(nn.Module):
    """``Linear -> activation -> Linear`` classifier for Adult.

    ReLU remains the default so existing callers keep the historical baseline.
    The q01 methods are called as unbound methods on ``z``. This preserves the
    graph created by ``fc1``; wrapping ``z`` in a new ``ExTensor`` would copy
    only its data and disconnect the first layer from the loss.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        activation: str = "relu",
        activation_beta: float | None = None,
    ) -> None:
        if activation not in SUPPORTED_ACTIVATIONS:
            choices = ", ".join(SUPPORTED_ACTIVATIONS)
            raise ValueError(f"unknown activation {activation!r}; choose one of: {choices}")
        if activation == "softplus_beta":
            if activation_beta is None:
                raise ValueError("softplus_beta requires activation_beta")
            if isinstance(activation_beta, (bool, np.bool_)) or not np.isscalar(
                activation_beta
            ):
                raise ValueError("activation_beta must be a finite positive scalar")
            try:
                activation_beta = float(activation_beta)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "activation_beta must be a finite positive scalar"
                ) from error
            if not np.isfinite(activation_beta) or activation_beta <= 0.0:
                raise ValueError("activation_beta must be a finite positive scalar")
        elif activation_beta is not None:
            raise ValueError("activation_beta is only valid with softplus_beta")
        self.n_features = n_features
        self.hidden = hidden
        self.activation = activation
        self.activation_beta = activation_beta
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def _activate(self, z: cpu.Tensor) -> cpu.Tensor:
        if self.activation == "relu":
            return z.relu()
        if self.activation == "sigmoid":
            return ExTensor.sigmoid(z)
        if self.activation == "swish":
            return ExTensor.swish(z)
        if self.activation == "softplus":
            return ExTensor.softplus(z)
        return ExTensor.softplus_beta(z, self.activation_beta)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        """Return logits shaped ``(2, batch)`` without detaching ``fc1``."""
        z = self.fc1(x)
        return self.fc2(self._activate(z))


class AdultLinearClassifier(nn.Module):
    """Adult classifier made only of directly composed affine layers.

    ``depth=1`` is ``108 -> 2``; ``depth=2`` is ``108 -> 64 -> 2``; and
    ``depth=3`` is ``108 -> 64 -> 64 -> 2``. Outputs are passed directly from
    one ``Linear`` to the next, so there is no Identity node or corresponding
    elementwise FLOP.
    """

    def __init__(
        self,
        n_features: int,
        depth: int,
        hidden: int = 64,
    ) -> None:
        if (
            isinstance(n_features, (bool, np.bool_))
            or not isinstance(n_features, (int, np.integer))
            or isinstance(hidden, (bool, np.bool_))
            or not isinstance(hidden, (int, np.integer))
        ):
            raise ValueError("n_features and hidden must be positive integers")
        n_features = int(n_features)
        hidden = int(hidden)
        if isinstance(depth, (bool, np.bool_)) or not isinstance(
            depth, (int, np.integer)
        ):
            raise ValueError("depth must be one of: 1, 2, 3")
        depth = int(depth)
        if depth not in SUPPORTED_LINEAR_DEPTHS:
            raise ValueError("depth must be one of: 1, 2, 3")
        if n_features <= 0 or hidden <= 0:
            raise ValueError("n_features and hidden must be positive integers")

        self.n_features = n_features
        self.hidden = hidden
        self.depth = depth
        sizes = {
            1: (n_features, 2),
            2: (n_features, hidden, 2),
            3: (n_features, hidden, hidden, 2),
        }[depth]
        self.layer_sizes = sizes
        self.layers = tuple(
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(sizes, sizes[1:])
        )

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        """Return ``(2, batch)`` logits using only the configured affine maps."""
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


def build_model(
    n_features: int,
    activation: str = "relu",
    model_seed: int = 0,
    *,
    activation_beta: float | None = None,
) -> AdultMLP:
    """Seed immediately before model construction, independently of the split."""
    cpu.set_seed(model_seed)
    return AdultMLP(
        n_features,
        hidden=64,
        activation=activation,
        activation_beta=activation_beta,
    )


def build_linear_model(
    n_features: int,
    depth: int,
    model_seed: int = 0,
    *,
    hidden: int = 64,
) -> AdultLinearClassifier:
    """Seed immediately before constructing one Variable 3 architecture."""
    cpu.set_seed(model_seed)
    return AdultLinearClassifier(n_features, depth=depth, hidden=hidden)


def collapse_affine(
    model: AdultLinearClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single matrix and bias equivalent to the composed model.

    The returned arrays satisfy ``model(x) == matrix @ x + bias[:, None]`` up to
    floating-point roundoff. This is a mechanical verification helper; it does
    not change the trained model or its parameterization.
    """
    if not model.layers or any(not layer.bias for layer in model.layers):
        raise ValueError("collapse_affine requires biased Linear layers")
    dtype = model.layers[0].weight.data.dtype
    matrix = np.eye(model.n_features, dtype=dtype)
    bias = np.zeros(model.n_features, dtype=dtype)
    for layer in model.layers:
        affine_matrix = layer.weight.data[1:].T
        affine_bias = layer.weight.data[0]
        matrix = affine_matrix @ matrix
        bias = affine_matrix @ bias + affine_bias
    return matrix, bias


def parameter_count(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return int(sum(parameter.data.size for parameter in model.parameters()))


# ============================================================================ #
# Fixed training/validation split
# ============================================================================ #
def train_val_indices(
    n_samples: int,
    val_frac: float = 0.2,
    split_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed ``(train_indices, validation_indices)``.

    ``RandomState`` intentionally uses NumPy's legacy MT19937 sequence. Thus
    seed 0 reproduces the permutation generated by the previous global
    ``np.random.seed(0); np.random.permutation(...)`` implementation, without
    consuming or changing the random state used for model initialization.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")

    permutation = np.random.RandomState(split_seed).permutation(n_samples)
    n_val = int(n_samples * val_frac)
    if n_val == 0:
        raise ValueError("validation split would be empty")
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    return train_idx, val_idx


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    split_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split column-oriented arrays using indices fixed by ``split_seed``."""
    if X.ndim != 2:
        raise ValueError("X must have shape (features, samples)")
    if X.shape[1] != len(y):
        raise ValueError("X and y must contain the same number of samples")
    train_idx, val_idx = train_val_indices(X.shape[1], val_frac, split_seed)
    return X[:, train_idx], y[train_idx], X[:, val_idx], y[val_idx]


# ============================================================================ #
# Evaluation and structured training output
# ============================================================================ #
def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples whose largest logit matches the label."""
    logits = model(cpu.Tensor(X, requires_grad=False)).data
    predictions = logits.argmax(axis=0)
    return float((predictions == y).mean())


@dataclass(frozen=True)
class EpochMetrics:
    """Measurements from one epoch; accuracies are computed after the update."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    flops: int


@dataclass(frozen=True)
class TrainingResult:
    """Complete in-memory record returned by ``train``."""

    history: tuple[EpochMetrics, ...]
    total_flops: int
    final_train_accuracy: float
    final_val_accuracy: float


@dataclass(frozen=True)
class ExperimentResult:
    """Model, configuration and metrics from one CLI/programmatic run."""

    model: AdultMLP
    training: TrainingResult
    activation: str
    model_seed: int
    split_seed: int
    test_accuracy: float | None = None
    activation_beta: float | None = None


def train(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-2,
    *,
    verbose: bool = True,
    epoch_callback: Callable[[EpochMetrics, nn.Module], None] | None = None,
) -> TrainingResult:
    """Train full-batch with Adam and return every loss, accuracy and FLOP count.

    The measured FLOP window is unchanged: training forward/loss/backward and
    the post-step validation forward/loss. Accuracy calls and the optional
    callback happen only after the epoch counter is read, so they do not enter
    the stored epoch cost. The next epoch always resets the global counter.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_tensor = cpu.Tensor(X_tr, requires_grad=False)
    val_tensor = cpu.Tensor(X_val, requires_grad=False)

    history: list[EpochMetrics] = []
    total_flops = 0
    for epoch in range(1, epochs + 1):
        cpu.reset_flops()

        optimizer.zero_grad()
        loss = cross_entropy(model(train_tensor).T, y_tr)
        loss.backward()
        optimizer.step()

        val_loss = float(cross_entropy(model(val_tensor).T, y_val).data)
        epoch_flops = cpu.flop_count()
        total_flops += epoch_flops

        # These diagnostics are deliberately outside the measured FLOP window.
        train_accuracy = accuracy(model, X_tr, y_tr)
        val_accuracy = accuracy(model, X_val, y_val)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(loss.data),
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            flops=epoch_flops,
        )
        history.append(metrics)

        if epoch_callback is not None:
            epoch_callback(metrics, model)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs}"
                f"   train loss = {metrics.train_loss:.4f}"
                f"   val loss = {metrics.val_loss:.4f}"
                f"   train acc = {metrics.train_accuracy:.4f}"
                f"   val acc = {metrics.val_accuracy:.4f}"
                f"   FLOPs = {epoch_flops:,}"
            )

    result = TrainingResult(
        history=tuple(history),
        total_flops=total_flops,
        final_train_accuracy=history[-1].train_accuracy,
        final_val_accuracy=history[-1].val_accuracy,
    )
    if verbose:
        print(
            f"\nTotal FLOPs over {epochs} epochs: {total_flops:,}"
            f"   (~{total_flops / 1e9:.2f} GFLOP)"
        )
    return result


def run_experiment(
    *,
    activation: str = "relu",
    activation_beta: float | None = None,
    epochs: int = 100,
    model_seed: int = 0,
    split_seed: int = 0,
    lr: float = 1e-2,
    evaluate_test: bool = False,
    verbose: bool = True,
    epoch_callback: Callable[[EpochMetrics, AdultMLP], None] | None = None,
) -> ExperimentResult:
    """Run one V1 configuration; load the official test only when requested."""
    train_dataset = datasets.load_adult("train")
    X_tr, y_tr, X_val, y_val = train_val_split(
        train_dataset.X,
        train_dataset.y,
        val_frac=0.2,
        split_seed=split_seed,
    )

    # This must remain immediately before construction for comparable weights.
    model = build_model(
        train_dataset.n_features,
        activation=activation,
        model_seed=model_seed,
        activation_beta=activation_beta,
    )

    if verbose:
        print("=" * 70)
        print("Adult income classification — q01 activation family")
        print("=" * 70)
        print(
            f"Data: {train_dataset}"
            f"\nTrain/val: {X_tr.shape[1]} / {X_val.shape[1]}"
            f"   split seed = {split_seed}"
        )
        print(
            f"Model: Linear({train_dataset.n_features}, 64) -> {activation}"
            f"{'' if activation_beta is None else f'(beta={activation_beta:g})'}"
            f" -> Linear(64, 2)   parameters = {parameter_count(model):,}"
            f"   model seed = {model_seed}\n"
        )
        print("Training (full-batch Adam):")

    training = train(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
        epoch_callback=epoch_callback,
    )

    test_accuracy = None
    if evaluate_test:
        test_dataset = datasets.load_adult("test")
        test_accuracy = accuracy(model, test_dataset.X, test_dataset.y)

    if verbose:
        message = (
            "\nFinal accuracy"
            f"   train = {training.final_train_accuracy:.4f}"
            f"   val = {training.final_val_accuracy:.4f}"
        )
        if test_accuracy is not None:
            message += f"   test = {test_accuracy:.4f}"
        print(message)

    return ExperimentResult(
        model=model,
        training=training,
        activation=activation,
        activation_beta=activation_beta,
        model_seed=model_seed,
        split_seed=split_seed,
        test_accuracy=test_accuracy,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Adult MLP with a q01 activation configuration.",
    )
    parser.add_argument("--activation", choices=SUPPORTED_ACTIVATIONS, default="relu")
    parser.add_argument("--activation-beta", type=float)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="opt in to the official Adult test set (final evaluation only)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> ExperimentResult:
    """Parse CLI arguments and return the structured experiment result."""
    args = _build_parser().parse_args(argv)
    return run_experiment(
        activation=args.activation,
        activation_beta=args.activation_beta,
        epochs=args.epochs,
        model_seed=args.model_seed,
        split_seed=args.split_seed,
        lr=args.lr,
        evaluate_test=args.evaluate_test,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
