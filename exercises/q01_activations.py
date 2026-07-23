"""Exercise 01 — new activation functions, from scratch.

YOUR TASK
=========
Subclass the engine's ``Tensor`` and add three activation functions that the
engine does not provide: ``sigmoid``, ``swish`` and ``softplus``. For each one
you implement both the forward value *and* the local ``_backward`` rule (the
derivative), exactly like the activations already in ``bert_cpu/engine.py``.

Why a subclass? Because nothing is removed from the engine — you *extend* it.
The existing ``Tensor`` (with `+`, `*`, `@`, `tanh`, ...) is your foundation;
you are adding new leaves on top.

The maths you need (let s = sigmoid):

    sigmoid(x)  = 1 / (1 + e^{-x})           sigmoid'(x)  = s(x) (1 - s(x))
    swish(x)    = x * sigmoid(x)             swish'(x)    = s(x) + x s(x)(1 - s(x))
    softplus(x) = ln(1 + e^{x})              softplus'(x) = s(x)

HOW TO WORK
===========
1. Fill in the three methods below (remove the ``raise NotImplementedError``).
2. Plot them to *see* your functions and — overlaid — the derivative your
   autograd produces:   ``python -m exercises.q01_activations``
3. Run the focused tests, which compare your gradients against finite
   differences:   ``pytest -q test/test_q01_activations.py``

TIP: an activation is just another op. Look at how ``tanh`` / ``relu`` are
written in ``bert_cpu/engine.py``: compute the forward array, build the output
``Tensor`` with ``self`` as its only child, then set ``out._backward`` to a
closure that accumulates ``(local derivative) * out.grad`` into ``self.grad``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bert_cpu.engine import Tensor


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid without overflowing for large negative inputs.

    For non-negative values, ``exp(-x)`` is safe.  For negative values we use
    the equivalent ``exp(x) / (1 + exp(x))``, so the exponential never receives
    a large positive argument.
    """
    result = np.empty_like(values)
    non_negative = values >= 0.0

    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exp_x = np.exp(values[~non_negative])
    result[~non_negative] = exp_x / (1.0 + exp_x)
    return result


class ExTensor(Tensor):
    """A ``Tensor`` extended with extra activation functions."""

    def sigmoid(self) -> Tensor:
        """Logistic sigmoid, 1 / (1 + e^{-x})."""
        sigmoid = _stable_sigmoid(self.data)
        out = Tensor(
            data=sigmoid,
            _children=(self,),
            _op="sigmoid",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sigmoid * (1.0 - sigmoid) * out.grad

        out._backward = _backward
        return out

    def swish(self) -> Tensor:
        """Swish / SiLU, x * sigmoid(x)."""
        sigmoid = _stable_sigmoid(self.data)
        out = Tensor(
            data=self.data * sigmoid,
            _children=(self,),
            _op="swish",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                local_derivative = sigmoid + self.data * sigmoid * (1.0 - sigmoid)
                self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def softplus(self) -> Tensor:
        """Softplus, ln(1 + e^{x}) (a smooth ReLU)."""
        # logaddexp(0, x) is algebraically equal to log(1 + exp(x)), while
        # remaining finite for large positive inputs.
        out = Tensor(
            data=np.logaddexp(0.0, self.data),
            _children=(self,),
            _op="softplus",
            requires_grad=self.requires_grad,
        )
        sigmoid = _stable_sigmoid(self.data)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sigmoid * out.grad

        out._backward = _backward
        return out

    def softplus_beta(self, beta: float) -> Tensor:
        """Softplus with fixed positive curvature ``beta``.

        ``logaddexp`` preserves the stable value of
        ``log(1 + exp(beta*x)) / beta``.  The constant ``beta`` is not a
        trainable parameter, and its local derivative is ``sigmoid(beta*x)``.
        """
        if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta):
            raise ValueError("beta must be a finite positive scalar")
        try:
            beta_value = float(beta)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("beta must be a finite positive scalar") from error
        if not np.isfinite(beta_value) or beta_value <= 0.0:
            raise ValueError("beta must be a finite positive scalar")

        scaled = beta_value * self.data
        out = Tensor(
            data=np.logaddexp(0.0, scaled) / beta_value,
            _children=(self,),
            _op="softplus_beta",
            requires_grad=self.requires_grad,
        )
        sigmoid = _stable_sigmoid(scaled)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sigmoid * out.grad

        out._backward = _backward
        return out


# --------------------------------------------------------------------- #
# The three GIVEN equations, of increasing complexity. You do not edit
# these — once your activations work, the checker computes and verifies
# their gradients. Each is a function of one or more leaf tensors.
# --------------------------------------------------------------------- #
EQUATIONS = [
    (
        "g1(x) = mean( sigmoid(x) )",
        lambda x: x.sigmoid().mean(),
        [("x", (4,))],
    ),
    (
        "g2(x) = sum( swish(x) + softplus(x) )",
        lambda x: (x.swish() + x.softplus()).sum(),
        [("x", (4,))],
    ),
    (
        "g3(x, w, b) = sum( softplus(x @ w + b) )   # a tiny neuron",
        lambda x, w, b: ExTensor.softplus(x @ w + b).sum(),
        [("x", (3,)), ("w", (3,)), ("b", ())],
    ),
]
# Note on g3: ``x @ w + b`` is produced by the base engine, so it is a plain
# ``Tensor`` — calling ``ExTensor.softplus(z)`` applies your method to it (a
# method is just a function of a tensor).


def plot_activations(path: str = "experiments/plots/q01_activations.png") -> None:
    """Plot each activation and, overlaid, the derivative your autograd gives.

    The derivative curve is *not* hand-coded here: for an elementwise function,
    ``y.sum().backward()`` leaves ``x.grad[i] = f'(x_i)``, so we literally draw
    the gradient your ``_backward`` produced.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, no display needed
    import matplotlib.pyplot as plt

    xs = np.linspace(-6.0, 6.0, 400)
    names = ["sigmoid", "swish", "softplus"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, names):
        x = ExTensor(xs.copy())
        y = getattr(x, name)()       # forward (your implementation)
        y.sum().backward()           # backward -> x.grad holds f'(x)
        ax.plot(xs, y.data, label=f"{name}(x)")
        ax.plot(xs, x.grad, "--", label=f"{name}'(x)  [autograd]")
        ax.set_title(name)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    try:
        plot_activations()
    except NotImplementedError as exc:
        print(f"Implement the activations first ({exc}).")
