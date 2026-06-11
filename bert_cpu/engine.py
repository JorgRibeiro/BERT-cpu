"""Tensor-based reverse-mode autodiff engine.

This is the NumPy-backed successor to micrograd's scalar ``Value``. A
``Tensor`` wraps an ``np.ndarray`` and records the operations applied to it so
that gradients can be computed by reverse-mode automatic differentiation.

The autograd machinery (topological sort + per-node ``_backward`` closures) is
identical in spirit to micrograd; the difference is that every node now holds an
array instead of a scalar, so each ``_backward`` must respect NumPy broadcasting
when accumulating gradients back into its parents.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int]
Axis = Optional[Union[int, Tuple[int, ...]]]


def _expand_reduced(
    grad: np.ndarray, axis: Axis, keepdims: bool, target_shape: Tuple[int, ...]
) -> np.ndarray:
    """Broadcast a reduced gradient back to the pre-reduction shape.

    Reductions (``sum``/``mean``/``var``/``max``) collapse one or more axes; on
    the way back the upstream gradient must be re-expanded along those axes and
    broadcast to the original input shape.
    """
    if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis)
    return np.broadcast_to(grad, target_shape)


class Tensor:
    """An n-dimensional array node in the autograd graph.

    Attributes
    ----------
    data : np.ndarray
        The forward value held by this node.
    grad : np.ndarray
        The accumulated gradient of the final scalar output w.r.t. ``data``.
        Same shape as ``data``; initialised to zeros.
    requires_grad : bool
        If False, this node is treated as a constant and no gradient is
        accumulated into it.
    _backward : Callable[[], None]
        Closure that propagates ``self.grad`` into the gradients of the parents.
    _prev : set[Tensor]
        The parent nodes that produced this tensor.
    _op : str
        Label of the producing op (for debugging / graph visualisation).
    """

    def __init__(
        self,
        data: ArrayLike,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
        requires_grad: bool = True,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self.requires_grad: bool = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: set = set(_children)
        self._op: str = _op

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_tensor(x: ArrayLike) -> "Tensor":
        """Wrap a raw value as a non-grad constant tensor (or pass through)."""
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        """Transpose of the last two axes (matrix transpose)."""
        if self.data.ndim < 2:
            return self
        axes = list(range(self.data.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    # ------------------------------------------------------------------ #
    # Elementwise binary ops
    # ------------------------------------------------------------------ #
    def __add__(self, other: ArrayLike) -> "Tensor":
        """Elementwise addition with broadcasting."""
        other = self._as_tensor(other)
        out = Tensor(
            self.data + other.data,
            (self, other),
            "+",
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        """Elementwise multiplication with broadcasting."""
        other = self._as_tensor(other)
        out = Tensor(
            self.data * other.data,
            (self, other),
            "*",
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += self._unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> "Tensor":
        """Elementwise power by a constant exponent."""
        assert isinstance(other, (int, float)), "exponent must be a scalar constant"
        out = Tensor(self.data ** other, (self,), f"**{other}", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Batched matrix multiplication (``@``)."""
        other = self._as_tensor(other)
        out = Tensor(
            self.data @ other.data,
            (self, other),
            "@",
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += self._unbroadcast(grad, self.shape)
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += self._unbroadcast(grad, other.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Unary / activation ops
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        """Elementwise natural exponential."""
        out = Tensor(np.exp(self.data), (self,), "exp", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Elementwise natural logarithm."""
        out = Tensor(np.log(self.data), (self,), "log", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        """Elementwise square root."""
        out = Tensor(np.sqrt(self.data), (self,), "sqrt", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (0.5 / out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """Elementwise hyperbolic tangent."""
        out = Tensor(np.tanh(self.data), (self,), "tanh", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Elementwise rectified linear unit."""
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit (tanh approximation, as used by BERT)."""
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1.0 + t), (self,), "gelu", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                dinner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
                dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner
                self.grad += dgelu * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Shape ops
    # ------------------------------------------------------------------ #
    def reshape(self, *shape: int) -> "Tensor":
        """Return a view of the tensor with a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        """Permute the axes of the tensor."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes), (self,), "transpose", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                inv = tuple(np.argsort(axes))
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions (must broadcast the upstream grad back on the way down)
    # ------------------------------------------------------------------ #
    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Sum over ``axis``."""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _expand_reduced(out.grad, axis, keepdims, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Arithmetic mean over ``axis``."""
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            (self,),
            "mean",
            self.requires_grad,
        )
        # Number of elements collapsed into each output entry.
        n = self.data.size / out.data.size

        def _backward() -> None:
            if self.requires_grad:
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += g / n

        out._backward = _backward
        return out

    def var(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Variance over ``axis`` (used by LayerNorm). Uses population (ddof=0)."""
        out = Tensor(
            self.data.var(axis=axis, keepdims=keepdims),
            (self,),
            "var",
            self.requires_grad,
        )
        mu = self.data.mean(axis=axis, keepdims=True)
        n = self.data.size / out.data.size

        def _backward() -> None:
            if self.requires_grad:
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += (2.0 / n) * (self.data - mu) * g

        out._backward = _backward
        return out

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Maximum over ``axis`` (used for numerically stable softmax)."""
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            (self,),
            "max",
            self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                md = self.data.max(axis=axis, keepdims=True)
                mask = (self.data == md).astype(self.data.dtype)
                # Split the gradient evenly across tied maxima.
                mask /= mask.sum(axis=axis, keepdims=True)
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += mask * g

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Composite ops
    # ------------------------------------------------------------------ #
    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax along ``axis``."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        sm = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(sm, (self,), "softmax", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                dot = (out.grad * sm).sum(axis=axis, keepdims=True)
                self.grad += sm * (out.grad - dot)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Autograd
    # ------------------------------------------------------------------ #
    def backward(self) -> None:
        """Run reverse-mode autodiff from this (scalar) tensor.

        Builds the topological order of the graph, seeds ``self.grad`` with
        ones, then invokes each node's ``_backward`` in reverse order.
        """
        topo: list = []
        visited: set = set()

        def build(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros."""
        self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------ #
    # Broadcasting helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Sum ``grad`` so that it matches ``shape``.

        Reverses NumPy broadcasting: any axis that was expanded during the
        forward pass is summed out in the backward pass.
        """
        # Collapse the leading axes that broadcasting prepended.
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # Collapse axes that were size-1 in the original but expanded.
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ------------------------------------------------------------------ #
    # Reflected / derived operators
    # ------------------------------------------------------------------ #
    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._as_tensor(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return (-self) + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self * (self._as_tensor(other) ** -1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return (self ** -1.0) * other

    def __getitem__(self, idx) -> "Tensor":
        """Indexing / slicing (used for embedding lookups)."""
        out = Tensor(self.data[idx], (self,), "getitem", self.requires_grad)

        def _backward() -> None:
            if self.requires_grad:
                # ``add.at`` accumulates correctly when ``idx`` repeats rows.
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, "
            f"data={self.data!r})"
        )


# ---------------------------------------------------------------------- #
# Tensor constructors
# ---------------------------------------------------------------------- #
def zeros(*shape: int, requires_grad: bool = True) -> Tensor:
    """Create a tensor of zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = True) -> Tensor:
    """Create a tensor of ones."""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = True) -> Tensor:
    """Create a tensor of standard-normal samples."""
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
